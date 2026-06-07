"""Background training worker with job queue and status reporting."""

import threading
import uuid
import time
from datetime import datetime
from typing import Optional, Callable

from models import create_pricer
from models.trainer import train_model
from data.loader import latest_session, load_session, list_sessions


# In-memory job queue and status store
_jobs = {}
_lock = threading.Lock()

# Model instances in memory
_model_cache = {}

# All trained model data lives here (checkpoints, logs, sessions)
import os
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
SESSION_DIR = os.path.join(DATA_DIR, "sessions")
LOG_DIR = os.path.join(DATA_DIR, "logs")
for d in [MODEL_DIR, SESSION_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)


def load_saved_models():
    """Load all saved model checkpoints from data/models/ into cache."""
    if not os.path.isdir(MODEL_DIR):
        return
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(".latest.pt"):
            model_type = fname[:-len(".latest.pt")]
            try:
                pricer = create_pricer(model_type)
                ckpt_path = os.path.join(MODEL_DIR, fname)
                pricer.load(ckpt_path)
                pricer.is_trained = True
                _model_cache[model_type] = pricer
            except Exception as e:
                print(f"Failed to load {fname}: {e}")


def _get_or_create_model(model_type: str):
    """Get cached model or create a new one."""
    if model_type not in _model_cache:
        _model_cache[model_type] = create_pricer(model_type)
    return _model_cache[model_type]


def start_training(model_type: str, epochs: int = 50, lr: float = 0.001,
                   session_path: Optional[str] = None) -> str:
    """Start a training job in a background thread.

    Args:
        model_type: "mlp", "lstm", "transformer", "ensemble"
        epochs: number of epochs
        lr: learning rate
        session_path: path to recorded session, or None for latest

    Returns:
        job_id: string UUID for status polling
    """
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "model_type": model_type,
        "status": "queued",
        "progress": 0,
        "epochs": epochs,
        "current_epoch": 0,
        "loss": None,
        "val_loss": None,
        "lr": lr,
        "history": [],
        "error": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
    }

    with _lock:
        _jobs[job_id] = job

    def _status_cb(epoch, loss, val_loss, lr):
        with _lock:
            job = _jobs[job_id]
            job["status"] = "running"
            job["progress"] = epoch / epochs * 100
            job["current_epoch"] = epoch
            job["loss"] = loss
            job["val_loss"] = val_loss
            job["lr"] = lr

    def _run():
        try:
            with _lock:
                _jobs[job_id]["status"] = "loading_data"
            # Load session data
            if session_path:
                sd = load_session(session_path)
            else:
                sd = latest_session()
            if sd is None:
                raise ValueError("No session data available")

            with _lock:
                _jobs[job_id]["status"] = "training"

            # Create model
            pricer = _get_or_create_model(model_type)

            # Train
            history = train_model(pricer, sd, epochs=epochs, lr=lr,
                                  status_callback=_status_cb)

            # Save checkpoint to data/models/ (all trained models live here)
            # Versioned: keeps the latest, archives previous as .v{N}.pt
            base_path = os.path.join(MODEL_DIR, f"{model_type}")
            latest_path = base_path + ".latest.pt"
            if os.path.exists(latest_path):
                # Rotate: find next version number
                v = 1
                while os.path.exists(f"{base_path}.v{v}.pt"):
                    v += 1
                os.rename(latest_path, f"{base_path}.v{v}.pt")
            ckpt_path = latest_path
            pricer.save(ckpt_path)

            with _lock:
                job = _jobs[job_id]
                job["status"] = "completed"
                job["progress"] = 100
                job["current_epoch"] = epochs
                job["history"] = history
                job["completed_at"] = datetime.now().isoformat()
                job["checkpoint"] = ckpt_path

        except Exception as e:
            import traceback
            with _lock:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = str(e)
                _jobs[job_id]["completed_at"] = datetime.now().isoformat()
            traceback.print_exc()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return job_id


def get_job_status(job_id: str) -> Optional[dict]:
    """Get current status of a training job."""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        return dict(job)


def list_jobs() -> list:
    """List all training jobs."""
    with _lock:
        return [dict(j) for j in _jobs.values()]


def get_model(model_type: str):
    """Get the cached model instance."""
    return _model_cache.get(model_type)


def list_available_models() -> list:
    """List all models that are trained/available."""
    result = []
    for name, model in _model_cache.items():
        result.append({
            "name": name,
            "type": name,
            "trained": getattr(model, "is_trained", False),
            "has_attention": hasattr(model, "attention"),
        })
    return result


def compute_surface(model_type: str, strikes, expiries, S):
    """Compute full surface for a given model."""
    model = _model_cache.get(model_type)
    if model is None:
        return None
    return model.surface(strikes, expiries, S)


def compute_compare_surfaces(model_types: list, strikes, expiries, S):
    """Compute surfaces for multiple models and return difference."""
    surfaces = {}
    for mt in model_types:
        surf = compute_surface(mt, strikes, expiries, S)
        if surf is not None:
            surfaces[mt] = surf.tolist()
    return surfaces

# Auto-load any saved model checkpoints on import
load_saved_models()
