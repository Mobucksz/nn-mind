"""Live IBKR option-chain manager for nn-mind.

Connects to a running IB Gateway / TWS over the socket API (exactly like
data/record_session.py — the proven, shared pipeline) and streams the option
chain, writing incremental session files that data.loader picks up and pushing
`ibkr.snapshot` events so the dashboard's "live" badge means what it says.

IBKR's API has no username/password: you log into IB Gateway/TWS yourself and
this connects to it at host:port (7497 paper / 7496 live / 4001 gateway).

A clearly-labelled `host="selftest"` mode generates a synthetic drifting
surface WITHOUT a broker, so the streaming UI can be exercised end-to-end.
That feed is always tagged source="test" / feed="selftest" — it is never
presented as real IBKR data.

Threading model: ib_insync needs all its calls on one thread that owns an
asyncio loop, so everything runs on a single dedicated worker thread driven by
a command queue. Public methods just enqueue commands and read status.
"""

import math
import time
import json
import queue
import threading
from datetime import date, datetime
from pathlib import Path

import numpy as np

from data.loader import LOCAL_RECORDINGS

ET_FMT = "%Y-%m-%dT%H:%M:%S"


def _expiry_meta(exp_str):
    """IBKR 'YYYYMMDD' -> {date, days, T} with a usable T for pricing."""
    y, m, d = int(exp_str[:4]), int(exp_str[4:6]), int(exp_str[6:8])
    days = max((date(y, m, d) - date.today()).days, 0)
    return {"date": f"{y:04d}-{m:02d}-{d:02d}", "days": days, "T": round(days / 365.0, 6)}


class IBKRManager:
    def __init__(self, emit):
        self.emit = emit
        self._cmd = queue.Queue()
        self._thread = None
        self._lock = threading.Lock()
        self._status = {
            "connected": False, "streaming": False, "mode": None,
            "host": None, "port": None, "symbol": None,
            "snapshots": 0, "spot": None, "error": None, "last": None,
        }
        # streaming params
        self._stream = None
        self._next_sample = 0.0
        self._ib = None
        self._session = None        # accumulating session dict
        self._out_path = None

    # ── public API (called from sidecar dispatch threads) ────────
    def status_dict(self):
        with self._lock:
            return dict(self._status)

    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)

    def connect(self, host="127.0.0.1", port=7497, client_id=17):
        if self._thread and self._thread.is_alive():
            return {"ok": False, "error": "already connected/connecting", **self.status_dict()}
        self._set(host=host, port=port, error=None,
                  mode="selftest" if host == "selftest" else "ibkr")
        result = {"event": threading.Event(), "value": None}
        self._cmd = queue.Queue()
        self._cmd.put(("connect", {"host": host, "port": port, "client_id": client_id,
                                    "result": result}))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        result["event"].wait(timeout=12)
        return result["value"] or {"ok": False, "error": "connect timed out", **self.status_dict()}

    def start_stream(self, params):
        if not self._status["connected"]:
            return {"ok": False, "error": "not connected"}
        self._cmd.put(("stream_start", params))
        return {"ok": True}

    def stop_stream(self):
        self._cmd.put(("stream_stop", {}))
        return {"ok": True}

    def disconnect(self):
        self._cmd.put(("disconnect", {}))
        return {"ok": True}

    # ── worker thread ────────────────────────────────────────────
    def _run(self):
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        selftest = self._status["mode"] == "selftest"
        ibm = None
        if not selftest:
            try:
                import ib_insync as ibm
            except Exception as e:
                self._finish_connect(None, ok=False, error=f"ib_insync not installed: {e}")
                return

        # The connect command is always the first thing on the queue (enqueued
        # by connect() right before this thread was started).
        cmd, arg = self._cmd.get()
        host = arg["host"]; port = arg["port"]; client_id = arg["client_id"]
        if selftest:
            self._ib = None
            self._set(connected=True, error=None)
            self._finish_connect(arg["result"], ok=True)
        else:
            try:
                self._ib = ibm.IB()
                self._ib.connect(host, port, clientId=client_id, timeout=8)
                self._set(connected=True, error=None)
                self._finish_connect(arg["result"], ok=True)
            except Exception as e:
                self._set(connected=False, error=str(e))
                self._finish_connect(arg["result"], ok=False, error=str(e))
                return

        # main cooperative loop
        running = True
        while running:
            try:
                cmd, arg = self._cmd.get_nowait()
            except queue.Empty:
                cmd, arg = None, None
            if cmd == "stream_start":
                self._begin_stream(ibm, arg)
            elif cmd == "stream_stop":
                self._set(streaming=False)
                self.emit("ibkr.status", self.status_dict())
            elif cmd == "disconnect":
                running = False
                break

            if self._status["streaming"] and time.time() >= self._next_sample:
                try:
                    self._take_snapshot(selftest)
                except Exception as e:
                    self._set(error=f"snapshot failed: {e}")
                    self.emit("ibkr.status", self.status_dict())
                self._next_sample = time.time() + self._stream["interval"]

            if selftest:
                time.sleep(0.2)
            else:
                self._ib.sleep(0.25)   # pumps the ib_insync event loop

        if self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception:
                pass
        self._set(connected=False, streaming=False)
        self.emit("ibkr.status", self.status_dict())

    def _finish_connect(self, result, ok, error=None):
        if result is not None:
            result["value"] = {"ok": ok, "error": error, **self.status_dict()}
            result["event"].set()
        self.emit("ibkr.status", self.status_dict())

    # ── streaming setup ──────────────────────────────────────────
    def _begin_stream(self, ibm, params):
        symbol = params.get("symbol", "SPY")
        sec_type = params.get("sec_type", "STK")
        n_exp = int(params.get("num_expiries", 5))
        n_strk = int(params.get("strikes_each_side", 10))
        interval = float(params.get("interval_sec", 5))

        if self._status["mode"] == "selftest":
            self._setup_selftest(symbol, n_exp, n_strk)
        else:
            self._setup_ibkr(ibm, symbol, sec_type, n_exp, n_strk)

        self._stream = {"interval": interval, "symbol": symbol}
        self._session_init(symbol, "selftest" if self._status["mode"] == "selftest" else "live")
        self._set(streaming=True, symbol=symbol, snapshots=0, error=None)
        self._next_sample = 0.0   # sample immediately
        self.emit("ibkr.status", self.status_dict())

    def _setup_selftest(self, symbol, n_exp, n_strk):
        self._spot0 = 530.0
        self._spot = self._spot0
        step = 5.0
        center = round(self._spot0 / step) * step
        self._strikes = [float(center + (i - n_strk) * step) for i in range(2 * n_strk + 1)]
        days = [7, 14, 30, 45, 60, 90, 120][:n_exp]
        self._expiries = [{
            "date": (date.fromordinal(date.today().toordinal() + d)).isoformat(),
            "days": d, "T": round(d / 365.0, 6)} for d in days]

    def _setup_ibkr(self, ibm, symbol, sec_type, n_exp, n_strk):
        ib = self._ib
        Stock, Index, Option = ibm.Stock, ibm.Index, ibm.Option
        und = Index(symbol, "SMART", "USD") if sec_type == "IND" else Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(und)
        self._und_ticker = ib.reqMktData(und, "", False, False)
        ib.sleep(1.5)
        spot = self._und_ticker.marketPrice()
        if not spot or math.isnan(spot):
            spot = self._und_ticker.close
        chains = ib.reqSecDefOptParams(und.symbol, "", und.secType, und.conId)
        chain = next((c for c in chains if c.exchange == "SMART"), chains[0])
        expiries = sorted(chain.expirations)[:n_exp]
        strikes = sorted(chain.strikes)
        nearest = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
        lo = max(0, nearest - n_strk); hi = min(len(strikes), nearest + n_strk + 1)
        strikes = strikes[lo:hi]

        self._strikes = [float(s) for s in strikes]
        self._expiries = [_expiry_meta(e) for e in expiries]
        self._tickers = {}
        for ei, exp in enumerate(expiries):
            for si, strike in enumerate(strikes):
                for right in ("C", "P"):
                    c = Option(symbol, exp, strike, right, "SMART",
                               tradingClass=chain.tradingClass, currency="USD")
                    self._tickers[(ei, si, right)] = c
        ib.qualifyContracts(*self._tickers.values())
        self._tickers = {k: ib.reqMktData(c, "100,101,106", False, False)
                         for k, c in self._tickers.items()}
        ib.sleep(3)

    # ── snapshot building ────────────────────────────────────────
    def _take_snapshot(self, selftest):
        if selftest:
            snap, spot = self._snapshot_selftest()
        else:
            snap, spot = self._snapshot_ibkr()
        self._session["snapshots"].append(snap)
        self._session["meta"]["snapshotCount"] = len(self._session["snapshots"])
        self._write_session()
        n = len(self._session["snapshots"])
        self._set(snapshots=n, spot=round(float(spot), 2),
                  last=datetime.now().strftime("%H:%M:%S"))
        self.emit("ibkr.snapshot", {
            "symbol": self._stream["symbol"], "spot": round(float(spot), 2),
            "snapshots": n, "t": snap["t"], "source": self._session["meta"]["source"],
        })

    def _snapshot_selftest(self):
        self._spot += float(np.random.normal(0, 0.5))
        S = self._spot
        NE, NS = len(self._expiries), len(self._strikes)
        iv = [0.0] * (NE * NS); cd = [0.0] * (NE * NS); gm = [0.0] * (NE * NS)
        vg = [0.0] * (NE * NS); th = [0.0] * (NE * NS)
        coi = [0] * (NE * NS); poi = [0] * (NE * NS); cv = [0] * (NE * NS); pv = [0] * (NE * NS)
        for ei, e in enumerate(self._expiries):
            T = max(e["T"], 1e-4)
            for si, K in enumerate(self._strikes):
                i = ei * NS + si
                m = math.log(K / S)
                sigma = max(0.16 + 0.04 * math.sqrt(T) - 0.55 * m + 2.2 * m * m, 0.05)
                d1 = (math.log(S / K) + (0.045 + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                from math import erf
                Nd1 = 0.5 * (1 + erf(d1 / math.sqrt(2)))
                pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
                iv[i] = round(sigma, 5)
                cd[i] = round(Nd1, 5)
                gm[i] = round(pdf / (S * sigma * math.sqrt(T)), 6)
                vg[i] = round(S * pdf * math.sqrt(T) / 100, 5)
                th[i] = round(-(S * pdf * sigma) / (2 * math.sqrt(T)) / 365, 6)
                atm = math.exp(-((K - S) / 30) ** 2)
                coi[i] = int(2000 * atm); poi[i] = int(2400 * atm)
                cv[i] = int(500 * atm); pv[i] = int(600 * atm)
        snap = {"t": datetime.now().strftime(ET_FMT), "underlying": round(S, 2),
                "iv": iv, "callDelta": cd, "gamma": gm, "vega": vg, "callTheta": th,
                "callOI": coi, "putOI": poi, "callVol": cv, "putVol": pv}
        return snap, S

    def _snapshot_ibkr(self):
        NE, NS = len(self._expiries), len(self._strikes)
        n = NE * NS

        def num(x, d=0.0):
            return d if x is None or (isinstance(x, float) and math.isnan(x)) else x

        iv = [0.0] * n; cd = [0.0] * n; gm = [0.0] * n; vg = [0.0] * n; th = [0.0] * n
        coi = [0] * n; poi = [0] * n; cv = [0] * n; pv = [0] * n
        for (e, s, right), tk in self._tickers.items():
            i = e * NS + s
            g = tk.modelGreeks
            if right == "C":
                if g:
                    iv[i] = round(num(g.impliedVol), 4); cd[i] = round(num(g.delta), 4)
                    gm[i] = round(num(g.gamma), 5); vg[i] = round(num(g.vega) / 100.0, 4)
                    th[i] = round(num(g.theta) / 365.0, 4)
                coi[i] = int(num(tk.callOpenInterest))
                cv[i] = int(num(tk.volume) * 100) if tk.volume else 0
            else:
                poi[i] = int(num(tk.putOpenInterest))
                pv[i] = int(num(tk.volume) * 100) if tk.volume else 0
        spot = num(self._und_ticker.marketPrice() or self._und_ticker.close)
        snap = {"t": datetime.now().strftime(ET_FMT), "underlying": round(spot, 2),
                "iv": iv, "callDelta": cd, "gamma": gm, "vega": vg, "callTheta": th,
                "callOI": coi, "putOI": poi, "callVol": cv, "putVol": pv}
        return snap, spot

    # ── session file (loader reads this) ─────────────────────────
    def _session_init(self, symbol, feed):
        today = date.today().isoformat()
        self._session = {
            "meta": {"symbol": symbol, "source": "IBKR" if feed == "live" else "test",
                     "feed": feed, "sessionDate": today, "snapshotCount": 0,
                     "generatedBy": "backend/ibkr_live.py"},
            "strikes": self._strikes, "expiries": self._expiries, "snapshots": [],
        }
        LOCAL_RECORDINGS.mkdir(parents=True, exist_ok=True)
        self._out_path = LOCAL_RECORDINGS / f"session-{today}.json"

    def _write_session(self):
        tmp = self._out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._session))
        tmp.replace(self._out_path)
