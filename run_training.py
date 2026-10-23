"""
Main training runner wrapper script with centralized lifecycle management,
live checkpointing, auto-resuming, health diagnostics, and dataset tracking.
"""

import argparse
import importlib
import sys
import shutil
import pickle
from pathlib import Path

import torch

# ==============================================================================
# DEFAULT CONFIGURATION
# ==============================================================================
DEFAULT_METHOD = "bc"
DEFAULT_EPOCHS = 30  # Each epoch is one full pass over the training data
DEFAULT_OUTPUT_DIR = "temp_models"
DEFAULT_CHECKPOINT_DIR = "temp_modelparts"
DEFAULT_DATA_DIR = "./server/data"
DEFAULT_CACHE_DIR = "./Train_BC"         # Moved cache directory inside ./Train_BC
DEFAULT_USED_DATA_FILE = "used_data.txt"

# Safety and diagnostic thresholds
STAGNATION_PCT_THRESHOLD = 20.0
STAGNATION_MIN_ITERATIONS = 10
MEAN_COLLAPSE_MIN_LOSS = 1.3
MEAN_COLLAPSE_MAX_LOSS = 1.5
EXPLODING_LOSS_THRESHOLD = 1e5
LOW_EPOCH_THRESHOLD = 50

# Maps training keys to their module import path (folder.file)
TRAINING_MODULES = {
    "bc": "Train_BC.Train_BC",
    "semisl": "Train_SemiSL.train_semisl",
    "rl": "Train_RL.train_rl",
}

# Methods that learn from the recorded dataset (RL makes its own data)
DATASET_METHODS = {"bc", "semisl"}


def prompt_user_config() -> tuple[str, int, str, str]:
    """Interactively prompts the user for training method, epochs, cache mode, and model filename."""
    print("=" * 60)
    print(" TRAINING RUNNER CONFIGURATION")
    print("=" * 60)

    method_input = input(f"1. Enter training method {list(TRAINING_MODULES)} (default: {DEFAULT_METHOD}): ").strip().lower()
    method = method_input if method_input else DEFAULT_METHOD

    epochs_input = input(f"2. Enter number of epochs (RL: iterations) (default: {DEFAULT_EPOCHS}): ").strip()
    epochs = int(epochs_input) if epochs_input else DEFAULT_EPOCHS

    if method not in DATASET_METHODS:
        cache_input = "i"
    else:
        print("3. Dataset selection:")
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

    default_filename = f"{method}_model.pt"
    filename_input = input(f"4. Enter model filename to save/resume (default: {default_filename}): ").strip()
    filename = filename_input if filename_input else default_filename
    if not filename.endswith(".pt"):
        filename += ".pt"

    print("=" * 60 + "\n")
    return method, epochs, cache_mode, filename


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

    if MEAN_COLLAPSE_MIN_LOSS < final_loss < MEAN_COLLAPSE_MAX_LOSS:
        warnings.append("[WARNING] Final loss is near ~1.41 (possible mean collapse).")

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
    print(f" Iterations      : {iterations}")
    print(f" Start Loss      : {start_loss:.6f}")
    print(f" Final Loss      : {final_loss:.6f}")
    print(f" Improvement     : {diff:.6f} ({pct:.1f}%)")
    print(f" Partial Weights : {stats.get('checkpoint_path', 'N/A')}")
    print(f" Final Model     : {stats.get('save_path', 'N/A')}")

    if warnings:
        print("\n" + "-" * 55)
        print(" SAFETY & DIAGNOSTIC ALERTS:")
        for warning in warnings:
            print(f"   * {warning}")
    print("=" * 55 + "\n")


def execute_training_loop(model, step_fn, epochs, checkpoint_path, save_path):
    resumed = False
    checkpoint_path = Path(checkpoint_path)
    save_path = Path(save_path)

    for path in [checkpoint_path, save_path]:
        if path.exists():
            print(f"[RESUME] Found existing weights at '{path}'. Loading via pickle...")
            try:
                # Bypass torch.load zip reader and load via pickle
                with open(path, "rb") as f:
                    state_dict = pickle.load(f)
                model.load_state_dict(state_dict)
                resumed = True
                break
            except Exception as e:
                # Don't silently train from scratch and overwrite the old file
                print(f"[RESUME ERROR] Failed to load '{path}': {e}")
                print("[RESUME ERROR] The weights are probably from an older architecture. "
                      "Choose a new model filename to train from scratch.")
                sys.exit(1)

    start_loss = None
    final_loss = 0.0
    completed_epochs = 0

    print(f"[TRAIN] Starting execution ({epochs} epochs)...")

    def atomic_save(state_dict, target_path):
        """Saves weights safely using Python pickle, bypassing PyTorch zip bugs entirely."""
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_suffix(".tmp")

        # Write using standard pickle to a temp file, then atomically replace
        with open(temp_path, "wb") as f:
            pickle.dump(state_dict, f)
        temp_path.replace(target_path)

    try:
        for epoch in range(1, epochs + 1):
            loss_val = step_fn(epoch)

            if start_loss is None:
                start_loss = loss_val
            final_loss = loss_val
            completed_epochs += 1

            # Save checkpoint after every epoch
            atomic_save(model.state_dict(), checkpoint_path)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] KeyboardInterrupt detected! Flushing current model weights safely...")
        atomic_save(model.state_dict(), checkpoint_path)
        print(f"[CHECKPOINT] Saved state safely to: {checkpoint_path}")

    atomic_save(model.state_dict(), save_path)

    return {
        "resumed": resumed,
        "completed_epochs": completed_epochs,
        "start_loss": start_loss if start_loss is not None else 0.0,
        "final_loss": final_loss,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training methods safely")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: each method's own setup() default)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Final output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--ckpt-dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help=f"Intermediate weights directory (default: {DEFAULT_CHECKPOINT_DIR})")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help=f"Data directory path (default: {DEFAULT_DATA_DIR})")

    args = parser.parse_args()

    method, epochs, cache_mode, filename = prompt_user_config()

    if method not in TRAINING_MODULES:
        raise ValueError(f"Unknown method '{method}'. Registered modules: {list(TRAINING_MODULES.keys())}")

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
        try:
            model, step_fn, method_name = module.setup(data_files=dataset_files, **lr_kwargs)
        except TypeError:
            model, step_fn, method_name = module.setup(**lr_kwargs)

        loop_stats = execute_training_loop(model, step_fn, epochs, checkpoint_path, save_path)
        if loop_stats["completed_epochs"] > 0:
            mark_files_used(used_data_cache_file, new_dataset_files)

        results = {
            "name": f"{method_name}{' [Resumed]' if loop_stats['resumed'] else ''}",
            "iterations": loop_stats["completed_epochs"],
            "start_loss": loop_stats["start_loss"],
            "final_loss": loop_stats["final_loss"],
            "save_path": str(save_path),
            "checkpoint_path": str(checkpoint_path),
        }
    else:
        available_attrs = [attr for attr in dir(module) if not attr.startswith("_")]
        print(f"[ERROR] Module '{module_name}' loaded, but does not expose a `setup` function.")
        print(f"[DEBUG] Available attributes in {module_name}: {available_attrs}")
        raise AttributeError(f"Module '{method}' must define a `setup(lr, data_files)` function.")

    display_results(results)
