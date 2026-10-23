"""
Main training runner wrapper script with centralized lifecycle management,
live checkpointing, auto-resuming, health diagnostics, and dataset tracking.

Time-limited sessions (e.g. 6 x 5 minutes with real-life tests in between):
    python run_training.py --method bc --minutes 5 --data a --name bc_final --device cuda

  - Only training time counts (data loading/setup is not part of the limit).
  - If the time runs out in the middle of an iteration, that unfinished iteration
    is thrown away, so the saved model is exactly "N completed iterations".
  - Running the same command again resumes: weights, optimizer state and the
    iteration counter continue where the last session stopped.
  - Every session saves a snapshot temp_models/<name>_session<N>.pt and logs to
    training_logs/<name>_sessions.csv (one row per session) and
    training_logs/<name>_iterations.csv (one row per iteration).
"""

import _thread
import argparse
import csv
import importlib
import itertools
import pickle
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch

from training_common import describe_device, resolve_device, to_cpu

# ==============================================================================
# DEFAULT CONFIGURATION
# ==============================================================================
DEFAULT_METHOD = "bc"
DEFAULT_EPOCHS = 30  # Each epoch is one full pass over the training data
DEFAULT_OUTPUT_DIR = "temp_models"
DEFAULT_CHECKPOINT_DIR = "temp_modelparts"
DEFAULT_LOG_DIR = "training_logs"
DEFAULT_DATA_DIR = "./server/data"
DEFAULT_CACHE_DIR = "./Train_BC"         # Moved cache directory inside ./Train_BC
DEFAULT_USED_DATA_FILE = "used_data.txt"

# Safety and diagnostic thresholds
STAGNATION_PCT_THRESHOLD = 20.0
STAGNATION_MIN_ITERATIONS = 10
EXPLODING_LOSS_THRESHOLD = 1e5

# Maps training keys to their module import path (folder.file)
TRAINING_MODULES = {
    "bc": "Train_BC.Train_BC",
    "semisl": "Train_SemiSL.train_semisl",
    "rl": "Train_RL.train_rl",
}

# Methods that learn from the recorded dataset (RL makes its own data)
DATASET_METHODS = {"bc", "semisl"}


def prompt_user_config(args) -> tuple[str, float | None, int | None, str, str, str]:
    """Fills in method, time limit, epochs, dataset mode, model filename and device.

    Anything given on the command line is used as-is; the rest is asked for.
    """
    print("=" * 60)
    print(" TRAINING RUNNER CONFIGURATION")
    print("=" * 60)

    method = args.method
    if method is None:
        method_input = input(f"1. Enter training method {list(TRAINING_MODULES)} (default: {DEFAULT_METHOD}): ").strip().lower()
        method = method_input if method_input else DEFAULT_METHOD

    minutes = args.minutes
    if minutes is None:
        minutes_input = input("2. Time limit in minutes (blank = no limit): ").strip()
        minutes = float(minutes_input) if minutes_input else 0.0
    minutes = minutes if minutes > 0 else None

    epochs = args.epochs
    if epochs is None and args.minutes is not None and minutes is not None:
        pass  # --minutes given on the command line: run until the time runs out
    elif epochs is None:
        if minutes is not None:
            epochs_input = input("3. Max number of epochs (RL: iterations) (blank = until the time runs out): ").strip()
            epochs = int(epochs_input) if epochs_input else None
        else:
            epochs_input = input(f"3. Enter number of epochs (RL: iterations) (default: {DEFAULT_EPOCHS}): ").strip()
            epochs = int(epochs_input) if epochs_input else DEFAULT_EPOCHS

    cache_input = args.data
    if method not in DATASET_METHODS:
        cache_input = "i"
    elif cache_input is None:
        print("4. Dataset selection:")
        print("   [a]ll recordings (default)")
        print("   [n]ew recordings only (not yet in used_data.txt, for fine-tuning a resumed model)")
        print("   [r]eset log (clear used_data.txt, then train on all recordings)")
        print("   [i]gnore log (train on all recordings, don't read or write used_data.txt)")
        cache_input = input("   Select option [a/n/r/i]: ").strip().lower()

    if cache_input.startswith("n"):
        cache_mode = "new"
    elif cache_input.startswith("r"):
        cache_mode = "reset"
    elif cache_input.startswith("i"):
        cache_mode = "ignore"
    else:
        cache_mode = "all"

    filename = args.name
    if filename is None:
        default_filename = f"{method}_model.pt"
        filename_input = input(f"5. Enter model filename to save/resume (default: {default_filename}): ").strip()
        filename = filename_input if filename_input else default_filename
    if not filename.endswith(".pt"):
        filename += ".pt"

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            default_device, found = "cuda", f"GPU found: {torch.cuda.get_device_name(0)}"
        else:
            default_device, found = "cpu", f"no CUDA GPU found, torch {torch.__version__}"
        device_input = input(f"6. Train on [cpu/cuda] (default: {default_device}, {found}): ").strip().lower()
        device = device_input if device_input else default_device

    print("=" * 60 + "\n")
    return method, minutes, epochs, cache_mode, filename, device


def select_dataset_files(data_dir: Path, cache_file: Path, cache_mode: str) -> tuple[list[str], list[str]]:
    """Scans the data directory and picks files based on the selected mode.

    Returns (files_to_train_on, new_files_to_log_after_training).
    """
    if not data_dir.exists():
        print(f"[DATA WARNING] Data directory '{data_dir}' does not exist yet.")
        return [], []

    all_files = sorted([str(p.resolve()) for p in data_dir.glob("*") if p.is_file()])

    if cache_mode == "ignore":
        print(f"[DATA TRACKING] Log ignored. Training on all {len(all_files)} files.")
        return all_files, []

    if cache_mode == "reset" and cache_file.exists():
        cache_file.unlink()
        print(f"[DATA TRACKING] Log reset at '{cache_file}'.")

    used_files = set()
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            used_files = set(line.strip() for line in f if line.strip())

    new_files = [f for f in all_files if f not in used_files]

    print(f"[DATA TRACKING] Found {len(all_files)} total files in '{data_dir}'.")
    print(f"[DATA TRACKING] {len(new_files)} new files, {len(all_files) - len(new_files)} previously used files.")

    if cache_mode == "new":
        print(f"[DATA TRACKING] Training on the {len(new_files)} new files only.")
        return new_files, new_files

    print(f"[DATA TRACKING] Training on all {len(all_files)} files.")
    return all_files, new_files


def mark_files_used(cache_file: Path, files: list[str]) -> None:
    """Appends files to the used-data log once training has run on them."""
    if not files:
        return
    with open(cache_file, "a", encoding="utf-8") as f:
        for file_path in files:
            f.write(f"{file_path}\n")
    print(f"[DATA TRACKING] Logged {len(files)} new files as used in '{cache_file}'.")


def analyze_training_health(stats: dict) -> list[str]:
    warnings = []
    start_loss = stats.get("start_loss", 0.0)
    final_loss = stats.get("final_loss", 0.0)
    iterations = stats.get("iterations", 1)

    diff = start_loss - final_loss
    pct = (diff / start_loss) * 100 if start_loss > 0 else 0

    if pct < STAGNATION_PCT_THRESHOLD and iterations >= STAGNATION_MIN_ITERATIONS:
        warnings.append(f"[CRITICAL] Training stalled on a plateau! Less than {STAGNATION_PCT_THRESHOLD}% loss reduction.")

    if final_loss != final_loss or final_loss > EXPLODING_LOSS_THRESHOLD:
        warnings.append("[CRITICAL] Loss exploded or resulted in NaN! Lower your learning rate.")

    return warnings


def display_results(stats: dict) -> None:
    start_loss = stats.get('start_loss', 0.0)
    final_loss = stats.get('final_loss', 0.0)
    iterations = stats.get('iterations', 0)

    diff = start_loss - final_loss
    pct = (diff / start_loss) * 100 if start_loss > 0 else 0

    warnings = analyze_training_health(stats)
    status_str = "SUCCESS" if not warnings else "STALLED / WARNINGS DETECTED"

    print("\n" + "=" * 55)
    print(f" TRAINING METHOD : {stats.get('name', 'N/A')}")
    print(f" STATUS          : {status_str}")
    print("=" * 55)
    print(f" Session         : {stats.get('session', 'N/A')} (stopped by: {stats.get('stopped_by', 'N/A')})")
    print(f" Iterations      : {iterations} this session, {stats.get('total_iterations', iterations)} in total")
    print(f" Training Time   : {stats.get('session_seconds', 0.0) / 60:.2f} min this session, "
          f"{stats.get('total_seconds', 0.0) / 60:.2f} min in total")
    print(f" Start Loss      : {start_loss:.6f}")
    print(f" Final Loss      : {final_loss:.6f}")
    print(f" Improvement     : {diff:.6f} ({pct:.1f}%)")
    print(f" Partial Weights : {stats.get('checkpoint_path', 'N/A')}")
    print(f" Final Model     : {stats.get('save_path', 'N/A')}")
    print(f" Session Snapshot: {stats.get('snapshot_path', 'N/A')}")
    print(f" Logs            : {stats.get('sessions_log', 'N/A')}")
    print(f"                   {stats.get('iterations_log', 'N/A')}")

    if warnings:
        print("\n" + "-" * 55)
        print(" SAFETY & DIAGNOSTIC ALERTS:")
        for warning in warnings:
            print(f"   * {warning}")
    print("=" * 55 + "\n")


# ==============================================================================
# SAVING, RESUMING & LOGGING
# ==============================================================================
def atomic_save(obj, target_path) -> None:
    """Saves safely using Python pickle, bypassing PyTorch zip bugs entirely."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")

    # Write using standard pickle to a temp file, then atomically replace
    with open(temp_path, "wb") as f:
        pickle.dump(obj, f)
    temp_path.replace(target_path)


def load_pickle(path):
    # Bypass torch.load zip reader and load via pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def training_state_path(checkpoint_path: Path) -> Path:
    """Optimizer state, iteration counter etc. live next to the checkpoint weights."""
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_trainstate.pkl")


def resume_weights(model, checkpoint_path: Path, save_path: Path) -> bool:
    for path in [checkpoint_path, save_path]:
        if path.exists():
            print(f"[RESUME] Found existing weights at '{path}'. Loading via pickle...")
            try:
                model.load_state_dict(load_pickle(path))
                return True
            except Exception as e:
                # Don't silently train from scratch and overwrite the old file
                print(f"[RESUME ERROR] Failed to load '{path}': {e}")
                print("[RESUME ERROR] The weights are probably from an older architecture. "
                      "Choose a new model filename to train from scratch.")
                sys.exit(1)
    return False


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def append_csv_row(path: Path, row: dict) -> None:
    """Appends a row; if the row has new columns, the file is rewritten with a wider header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, fieldnames = [], []
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            if any(key not in fieldnames for key in row):
                rows = list(reader)

    new_keys = [key for key in row if key not in fieldnames]
    if fieldnames and not new_keys:
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames, restval="").writerow(row)
        return

    fieldnames += new_keys
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(row)


def normalize_metrics(result) -> dict:
    """step_fn may return a float (the loss) or a dict of metrics containing "loss"."""
    metrics = dict(result) if isinstance(result, dict) else {"loss": result}
    return {key: (round(value, 6) if isinstance(value, float) else value) for key, value in metrics.items()}


class TimeLimit:
    """Interrupts the main thread (like Ctrl+C) once the time limit is reached.

    Code inside `protected()` (saving files) is never interrupted halfway;
    the interrupt then happens right after it.
    """
    def __init__(self, seconds: float | None):
        self.seconds = seconds
        self.reached = threading.Event()
        self._lock = threading.Lock()
        self._protected = False
        self._pending = False
        self._timer = None

    def start(self) -> None:
        if self.seconds is None:
            return
        self._timer = threading.Timer(self.seconds, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def cancel(self) -> None:
        if self._timer is not None:
            self._timer.cancel()

    def _fire(self) -> None:
        with self._lock:
            self.reached.set()
            if self._protected:
                self._pending = True
            else:
                _thread.interrupt_main()

    @contextmanager
    def protected(self):
        with self._lock:
            self._protected = True
        try:
            yield
        finally:
            with self._lock:
                self._protected = False
                pending, self._pending = self._pending, False
            if pending:
                raise KeyboardInterrupt


def run_session(model, step_fn, method_name, save_path, checkpoint_path,
                epochs=None, time_limit_minutes=None, log_dir=DEFAULT_LOG_DIR, device_name="") -> dict:
    """Runs one training session: resume, train until epochs or time limit, save, snapshot, log.

    epochs=None means "no epoch limit" (stop on the time limit or Ctrl+C).
    Weights are always saved as CPU tensors, so models trained on a GPU load anywhere.
    """
    save_path, checkpoint_path, log_dir = Path(save_path), Path(checkpoint_path), Path(log_dir)
    state_path = training_state_path(checkpoint_path)

    resumed = resume_weights(model, checkpoint_path, save_path)
    base_iterations, base_seconds = 0, 0.0
    if resumed and state_path.exists():
        saved = load_pickle(state_path)
        base_iterations = saved.get("iterations", 0)
        base_seconds = saved.get("training_seconds", 0.0)
        if saved.get("module_state") is not None and hasattr(step_fn, "set_state"):
            step_fn.set_state(saved["module_state"])
        print(f"[RESUME] Continuing after iteration {base_iterations} "
              f"({base_seconds / 60:.2f} min trained so far).")

    stem = save_path.stem
    sessions_log = log_dir / f"{stem}_sessions.csv"
    iterations_log = log_dir / f"{stem}_iterations.csv"
    session = count_csv_rows(sessions_log) + 1
    started_at = datetime.now().isoformat(timespec="seconds")

    limit = TimeLimit(time_limit_minutes * 60 if time_limit_minutes else None)
    last_completed_weights = to_cpu(model.state_dict())
    history = []
    completed_seconds = 0.0
    stopped_by = "epochs"

    limit_text = f"{time_limit_minutes:g} min" if time_limit_minutes else "no time limit"
    epochs_text = f"{epochs} iterations" if epochs is not None else "no iteration limit"
    print(f"[TRAIN] Session {session}: {limit_text}, {epochs_text}...")

    session_start = time.perf_counter()
    try:
        try:
            limit.start()
            for session_iteration in (itertools.count(1) if epochs is None else range(1, epochs + 1)):
                iteration = base_iterations + session_iteration
                iteration_start = time.perf_counter()
                metrics = normalize_metrics(step_fn(iteration))
                now = time.perf_counter()

                with limit.protected():
                    completed_seconds = now - session_start
                    row = {
                        "session": session,
                        "iteration": iteration,
                        "session_iteration": session_iteration,
                        "iteration_seconds": round(now - iteration_start, 3),
                        "session_seconds": round(completed_seconds, 3),
                        "total_training_seconds": round(base_seconds + completed_seconds, 3),
                        **metrics,
                    }
                    history.append(row)
                    append_csv_row(iterations_log, row)

                    last_completed_weights = to_cpu(model.state_dict())
                    atomic_save(last_completed_weights, checkpoint_path)
                    atomic_save({
                        "iterations": iteration,
                        "training_seconds": base_seconds + completed_seconds,
                        "module_state": step_fn.get_state() if hasattr(step_fn, "get_state") else None,
                    }, state_path)
        finally:
            limit.cancel()
    except KeyboardInterrupt:
        stopped_by = "time_limit" if limit.reached.is_set() else "user"
        print(f"\n[STOPPED] {'Time limit reached' if stopped_by == 'time_limit' else 'KeyboardInterrupt'}. "
              f"Discarding the unfinished iteration; keeping the model after iteration "
              f"{base_iterations + len(history)}.")

    wall_seconds = time.perf_counter() - session_start
    model.load_state_dict(last_completed_weights)

    atomic_save(last_completed_weights, save_path)
    snapshot_path = save_path.with_name(f"{stem}_session{session}.pt")
    shutil.copy2(save_path, snapshot_path)

    losses = [row["loss"] for row in history]
    session_row = {
        "session": session,
        "started_at": started_at,
        "method": method_name,
        "device": device_name,
        "stopped_by": stopped_by,
        "time_limit_minutes": time_limit_minutes or "",
        "iterations_this_session": len(history),
        "first_iteration": history[0]["iteration"] if history else "",
        "last_iteration": history[-1]["iteration"] if history else "",
        "total_iterations": base_iterations + len(history),
        "session_training_seconds": round(completed_seconds, 3),
        "session_wall_seconds": round(wall_seconds, 3),
        "total_training_seconds": round(base_seconds + completed_seconds, 3),
        "start_loss": losses[0] if losses else "",
        "final_loss": losses[-1] if losses else "",
        "snapshot": str(snapshot_path),
    }
    if history:
        for key, value in history[-1].items():
            if key not in ("session", "iteration", "session_iteration", "iteration_seconds",
                           "session_seconds", "total_training_seconds", "loss"):
                session_row[f"final_{key}"] = value
    append_csv_row(sessions_log, session_row)

    return {
        "name": f"{method_name}{' [Resumed]' if resumed else ''}",
        "session": session,
        "stopped_by": stopped_by,
        "iterations": len(history),
        "total_iterations": base_iterations + len(history),
        "session_seconds": completed_seconds,
        "total_seconds": base_seconds + completed_seconds,
        "start_loss": losses[0] if losses else 0.0,
        "final_loss": losses[-1] if losses else 0.0,
        "save_path": str(save_path),
        "checkpoint_path": str(checkpoint_path),
        "snapshot_path": str(snapshot_path),
        "sessions_log": str(sessions_log),
        "iterations_log": str(iterations_log),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training methods safely")
    parser.add_argument("--method", choices=list(TRAINING_MODULES), default=None, help="Training method (asked if omitted)")
    parser.add_argument("--minutes", type=float, default=None, help="Time limit per session in minutes, 0 = no limit (asked if omitted)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs/iterations (asked if omitted)")
    parser.add_argument("--data", choices=["a", "n", "r", "i"], default=None, help="Dataset selection: all/new/reset/ignore (asked if omitted)")
    parser.add_argument("--name", type=str, default=None, help="Model filename to save/resume (asked if omitted)")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None, help="Training device, auto = cuda if available (asked if omitted)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: each method's own setup() default)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Final output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--ckpt-dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help=f"Intermediate weights directory (default: {DEFAULT_CHECKPOINT_DIR})")
    parser.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR, help=f"Session/iteration CSV log directory (default: {DEFAULT_LOG_DIR})")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help=f"Data directory path (default: {DEFAULT_DATA_DIR})")

    args = parser.parse_args()

    method, minutes, epochs, cache_mode, filename, device_choice = prompt_user_config(args)

    if method not in TRAINING_MODULES:
        raise ValueError(f"Unknown method '{method}'. Registered modules: {list(TRAINING_MODULES.keys())}")

    try:
        device = resolve_device(device_choice)
    except (RuntimeError, ValueError) as e:
        print(f"[DEVICE ERROR] {e}")
        sys.exit(1)
    print(f"[DEVICE] Training on {describe_device(device)}")

    output_dir = Path(args.output_dir)
    ckpt_dir = Path(args.ckpt_dir)
    data_dir = Path(args.data_dir)
    cache_dir = Path(DEFAULT_CACHE_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / filename
    checkpoint_path = ckpt_dir / f"part_{filename}"
    used_data_cache_file = cache_dir / DEFAULT_USED_DATA_FILE

    if method in DATASET_METHODS:
        dataset_files, new_dataset_files = select_dataset_files(data_dir, used_data_cache_file, cache_mode)
    else:
        dataset_files, new_dataset_files = [], []

    module_name = TRAINING_MODULES[method]
    module_path = Path(*module_name.split(".")).with_suffix(".py")

    # ==========================================================================
    # SELF-DIAGNOSTIC & CACHE CLEARING BLOCK
    # ==========================================================================
    if not module_path.exists():
        print(f"[ERROR] Could not find file '{module_path.resolve()}' on disk!")
        sys.exit(1)

    file_content = module_path.read_text(encoding="utf-8", errors="ignore")
    if "def setup" not in file_content:
        print(f"[ERROR] '{module_path.name}' exists on disk, but does NOT contain 'def setup' in its text content.")
        sys.exit(1)
    else:
        print(f"[DIAGNOSTIC] Verified 'def setup' exists in {module_path.resolve()} on disk.")

    # Clear bytecode caches
    for p in Path(".").glob("**/__pycache__"):
        shutil.rmtree(p, ignore_errors=True)

    # Clear sys.modules cache
    package_name = module_name.split(".")[0]
    for mod_key in list(sys.modules.keys()):
        if package_name in mod_key:
            del sys.modules[mod_key]

    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        print(f"[ERROR] Failed to import '{module_name}': {e}")
        sys.exit(1)

    # Setup training model and step function
    if hasattr(module, "setup"):
        lr_kwargs = {"lr": args.lr} if args.lr is not None else {}
        model, step_fn, method_name = module.setup(data_files=dataset_files, device=device, **lr_kwargs)

        results = run_session(model, step_fn, method_name, save_path, checkpoint_path,
                              epochs=epochs, time_limit_minutes=minutes, log_dir=args.log_dir,
                              device_name=describe_device(device))
        if results["iterations"] > 0:
            mark_files_used(used_data_cache_file, new_dataset_files)
    else:
        available_attrs = [attr for attr in dir(module) if not attr.startswith("_")]
        print(f"[ERROR] Module '{module_name}' loaded, but does not expose a `setup` function.")
        print(f"[DEBUG] Available attributes in {module_name}: {available_attrs}")
        raise AttributeError(f"Module '{method}' must define a `setup(lr, data_files)` function.")

    display_results(results)
