#!/usr/bin/env python3
"""Record a live Interactive Brokers option-chain session for Option Scope.

Connects to a running TWS or IB Gateway, samples the option chain (underlying
price + model greeks + volume/open-interest) on an interval through the session,
and writes snapshots in the schema the web app reads. Each day lands in
``public/data/sessions/session-<date>.json`` and is registered in
``public/data/manifest.json`` (which the app uses to auto-load the latest day),
so running this once per trading day gives you fresh real data every day.

Quick start
-----------
    pip install ib_insync
    # Start TWS or IB Gateway with the API enabled (paper 7497 / live 7496)
    python tools/record_ibkr_session.py --symbol SPY --auto-rth

`--auto-rth` records from now until the 16:00 US/Eastern close. To run it
unattended every trading day, see tools/scheduling/ (cron / launchd / Task
Scheduler) and tools/run_daily.sh.

The file is written incrementally (after every snapshot) so you can replay
"today so far" while it's still recording.

    python tools/record_ibkr_session.py --self-test   # validate plumbing, no TWS
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Python 3.14+ requires an explicit event loop for ib_insync.
try:
    _loop = asyncio.get_running_loop()
except RuntimeError:
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

ET = ZoneInfo("America/New_York")
DATA_DIR = Path(__file__).resolve().parent.parent / "public" / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
MANIFEST = DATA_DIR / "manifest.json"

# Generic tick list -> 100: option volume, 101: option open interest,
# 106: option implied volatility. Model greeks arrive on ticker.modelGreeks.
GENERIC_TICKS = "100,101,106"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record an IBKR option session for Option Scope.")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--sec-type", default="STK", choices=["STK", "IND"],
                   help="STK for ETFs/stocks (SPY, AAPL), IND for indices (SPX, VIX).")
    p.add_argument("--exchange", default="SMART")
    p.add_argument("--currency", default="USD")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497, help="7497 paper TWS, 7496 live, 4001/4002 gateway.")
    p.add_argument("--client-id", type=int, default=17)
    p.add_argument("--num-expiries", type=int, default=7)
    p.add_argument("--strikes-each-side", type=int, default=10, help="strikes above/below spot.")
    p.add_argument("--interval-min", type=float, default=10.0)
    p.add_argument("--duration-min", type=float, default=390.0, help="total recording length.")
    p.add_argument("--auto-rth", action="store_true",
                   help="record from now until the 16:00 ET close (overrides --duration-min).")
    p.add_argument("--out", default=None, help="explicit output file (defaults to sessions/session-<date>.json).")
    p.add_argument("--self-test", action="store_true",
                   help="write a small synthetic IBKR-labelled day to validate file/manifest plumbing.")
    return p.parse_args()


# --------------------------------------------------------------------------- IO

def write_session(out_path: Path, session: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(session))
    tmp.replace(out_path)  # atomic so the app never reads a half-written file


def update_manifest(date: str, symbol: str, source: str, rel_file: str, snapshots: int) -> None:
    """Upsert this session into manifest.json and recompute `latest`."""
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"symbol": symbol, "sessions": []}
    manifest["sessions"] = [s for s in manifest.get("sessions", []) if s.get("file") != rel_file]
    tag = "● IBKR" if source == "IBKR" else "○ sample"
    manifest["sessions"].append({
        "date": date, "symbol": symbol, "source": source,
        "file": rel_file, "snapshots": snapshots,
        "label": f"{date}  {symbol}  {tag}",
    })
    manifest["sessions"].sort(key=lambda s: s["date"])
    manifest["latest"] = manifest["sessions"][-1]["date"]
    manifest["updated"] = datetime.now(ET).isoformat()
    MANIFEST.write_text(json.dumps(manifest, indent=2))


def rel_to_data(path: Path) -> str:
    return path.relative_to(DATA_DIR).as_posix()


# ------------------------------------------------------------------- self-test

def run_self_test(args) -> None:
    """Emit a tiny synthetic session labelled IBKR to exercise the pipeline."""
    date = datetime.now(ET).strftime("%Y-%m-%d")
    strikes = [round(500 + i * 5) for i in range(-args.strikes_each_side, args.strikes_each_side + 1)]
    NS = len(strikes)
    expiries = [{"date": (datetime.now(ET) + timedelta(days=d)).strftime("%Y-%m-%d"),
                 "days": d, "T": d / 365} for d in (7, 14, 30, 60, 90, 180, 365)]
    NE = len(expiries)
    out = Path(args.out) if args.out else SESSIONS_DIR / f"session-{date}.json"

    snapshots = []
    for step in range(6):
        minutes = 9 * 60 + 30 + step * 60
        spot = 500 + math.sin(step) * 2
        n = NE * NS

        def grid(fn):
            return [round(fn(e, s), 5) for e in range(NE) for s in range(NS)]

        snapshots.append({
            "t": f"{date}T{minutes//60:02d}:{minutes%60:02d}:00-04:00",
            "tLabel": f"{minutes//60:02d}:{minutes%60:02d}", "minutes": minutes,
            "underlying": round(spot, 2), "rate": 0.045,
            "iv": grid(lambda e, s: 0.15 + abs(strikes[s] - spot) / 4000 + e * 0.002),
            "callDelta": grid(lambda e, s: max(0, min(1, 0.5 - (strikes[s] - spot) / 60))),
            "putDelta": grid(lambda e, s: max(-1, min(0, -0.5 - (strikes[s] - spot) / 60))),
            "gamma": grid(lambda e, s: 0.02 * math.exp(-((strikes[s] - spot) ** 2) / 200)),
            "vega": grid(lambda e, s: 0.1 * math.exp(-((strikes[s] - spot) ** 2) / 400)),
            "callTheta": grid(lambda e, s: -0.05 - e * 0.001),
            "putTheta": grid(lambda e, s: -0.05 - e * 0.001),
            "callOI": grid(lambda e, s: int(5000 + step * 100)),
            "putOI": grid(lambda e, s: int(5200 + step * 100)),
            "callVol": grid(lambda e, s: int(300 + step * 20)),
            "putVol": grid(lambda e, s: int(320 + step * 20)),
        })
        session = {
            "meta": {"symbol": args.symbol, "source": "IBKR", "feed": "self-test",
                     "sessionDate": date, "timezone": "America/New_York",
                     "intervalMin": 60, "snapshotCount": len(snapshots),
                     "generatedBy": "tools/record_ibkr_session.py --self-test"},
            "strikes": strikes, "expiries": expiries,
            "layout": "row-major: index = expiryIndex * strikes.length + strikeIndex",
            "snapshots": snapshots,
        }
        write_session(out, session)
        update_manifest(date, args.symbol, "IBKR", rel_to_data(out), len(snapshots))
    print(f"✓ self-test wrote {out} and updated {MANIFEST}")


# ------------------------------------------------------------------- live record

def underlying_contract(args, ibm):
    Stock, Index = ibm.Stock, ibm.Index
    if args.sec_type == "IND":
        return Index(args.symbol, args.exchange, args.currency)
    return Stock(args.symbol, args.exchange, args.currency)


def pick_chain(ib, und, args):
    [ticker] = ib.reqTickers(und)
    spot = ticker.marketPrice()
    if not spot or math.isnan(spot):
        spot = ticker.close
    print(f"  spot {args.symbol} = {spot:.2f}")

    chains = ib.reqSecDefOptParams(und.symbol, "", und.secType, und.conId)
    chain = next((c for c in chains if c.exchange == "SMART"), chains[0])
    expiries = sorted(chain.expirations)[: args.num_expiries]

    strikes = sorted(chain.strikes)
    nearest = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
    lo = max(0, nearest - args.strikes_each_side)
    hi = min(len(strikes), nearest + args.strikes_each_side + 1)
    strikes = strikes[lo:hi]

    print(f"  {len(expiries)} expiries × {len(strikes)} strikes")
    return expiries, strikes, chain.tradingClass


def build_options(args, expiries, strikes, trading_class, Option):
    contracts = {}
    for e_idx, exp in enumerate(expiries):
        for s_idx, strike in enumerate(strikes):
            for right in ("C", "P"):
                contracts[(e_idx, s_idx, right)] = Option(
                    args.symbol, exp, strike, right, args.exchange,
                    tradingClass=trading_class, currency=args.currency)
    return contracts


def snapshot(ticker_by_key, NE, NS, und_ticker, when, expiries_iso) -> dict:
    n = NE * NS

    def blank():
        return [0.0] * n

    def num(x, d=0.0):
        return d if x is None or (isinstance(x, float) and math.isnan(x)) else x

    iv = blank(); call_delta = blank(); put_delta = blank()
    gamma = blank(); vega = blank(); call_theta = blank(); put_theta = blank()
    call_oi = blank(); put_oi = blank(); call_vol = blank(); put_vol = blank()

    for (e, s, right), tk in ticker_by_key.items():
        i = e * NS + s
        g = tk.modelGreeks
        if right == "C":
            if g:
                iv[i] = round(num(g.impliedVol), 4)
                call_delta[i] = round(num(g.delta), 4)
                gamma[i] = round(num(g.gamma), 5)
                vega[i] = round(num(g.vega) / 100.0, 4)
                call_theta[i] = round(num(g.theta) / 365.0, 4)
            call_oi[i] = int(num(tk.callOpenInterest))
            call_vol[i] = int(num(tk.volume) * 100) if tk.volume else 0
        else:
            if g:
                put_delta[i] = round(num(g.delta), 4)
                put_theta[i] = round(num(g.theta) / 365.0, 4)
            put_oi[i] = int(num(tk.putOpenInterest))
            put_vol[i] = int(num(tk.volume) * 100) if tk.volume else 0

    return {
        "t": when.isoformat(), "tLabel": when.strftime("%H:%M"),
        "minutes": when.hour * 60 + when.minute,
        "underlying": round(num(und_ticker.marketPrice() or und_ticker.close), 2),
        "rate": 0.045,
        "iv": iv, "callDelta": call_delta, "putDelta": put_delta,
        "gamma": gamma, "vega": vega, "callTheta": call_theta, "putTheta": put_theta,
        "callOI": call_oi, "putOI": put_oi, "callVol": call_vol, "putVol": put_vol,
    }


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test(args)
        return

    import ib_insync as ibm  # imported lazily so --self-test needs no dependency

    ib = ibm.IB()
    print(f"Connecting to IBKR at {args.host}:{args.port} …")
    ib.connect(args.host, args.port, clientId=args.client_id)

    und = underlying_contract(args, ibm)
    ib.qualifyContracts(und)
    und_ticker = ib.reqMktData(und, "", False, False)

    expiries, strikes, trading_class = pick_chain(ib, und, args)
    NE, NS = len(expiries), len(strikes)
    contracts = build_options(args, expiries, strikes, trading_class, ibm.Option)
    ib.qualifyContracts(*contracts.values())

    print("Subscribing to market data …")
    ticker_by_key = {key: ib.reqMktData(c, GENERIC_TICKS, False, False)
                     for key, c in contracts.items()}
    ib.sleep(3)

    now = datetime.now(ET)
    date = now.strftime("%Y-%m-%d")
    if args.auto_rth:
        close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        duration_min = max(args.interval_min, (close - now).total_seconds() / 60)
    else:
        duration_min = args.duration_min
    n_steps = int(duration_min // args.interval_min) + 1

    out = Path(args.out) if args.out else SESSIONS_DIR / f"session-{date}.json"
    expiries_meta = [{"date": e, "days": None, "T": None} for e in expiries]
    snapshots = []
    print(f"Recording {n_steps} snapshots every {args.interval_min} min → {out}")

    for step in range(n_steps):
        ib.sleep(2)
        snap = snapshot(ticker_by_key, NE, NS, und_ticker, datetime.now(ET), expiries)
        snapshots.append(snap)
        session = {
            "meta": {"symbol": args.symbol, "source": "IBKR", "feed": "live",
                     "sessionDate": date, "timezone": "America/New_York",
                     "intervalMin": args.interval_min, "snapshotCount": len(snapshots),
                     "generatedBy": "tools/record_ibkr_session.py"},
            "strikes": strikes, "expiries": expiries_meta,
            "layout": "row-major: index = expiryIndex * strikes.length + strikeIndex",
            "snapshots": snapshots,
        }
        write_session(out, session)                # incremental, atomic
        update_manifest(date, args.symbol, "IBKR", rel_to_data(out), len(snapshots))
        print(f"  [{step+1}/{n_steps}] {snap['tLabel']}  S={snap['underlying']}")
        if step < n_steps - 1:
            ib.sleep(args.interval_min * 60 - 2)

    print(f"✓ done — {len(snapshots)} snapshots in {out}")
    ib.disconnect()


if __name__ == "__main__":
    main()
