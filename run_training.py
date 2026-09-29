"""
Main training runner wrapper script with centralized lifecycle management,
live checkpointing, auto-resuming, and health diagnostics.
"""

import sys
import argparse
import importlib
from pathlib import Path
import torch

# ==============================================================================
# DEFAULT RUNNER & HYPERPARAMETER CONFIGURATION
# ==============================================================================
DEFAULT_METHOD = "bc"
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-3
DEFAULT_OUTPUT_DIR = "temp_models"
DEFAULT_CHECKPOINT_DIR = "temp_modelparts"

# Safety and diagnostic thresholds
STAGNATION_PCT_THRESHOLD = 20.0
STAGNATION_MIN_ITERATIONS = 100
MEAN_COLLAPSE_MIN_LOSS = 1.3
MEAN_COLLAPSE_MAX_LOSS = 1.5
EXPLODING_LOSS_THRESHOLD = 1e5
LOW_EPOCH_THRESHOLD = 50

TRAINING_METHODS = {
    "bc": "Train_BC.train_bc",
    # Add future methods here:
    # "semisl": "Train_SemiSL.train_semisl",
    # "rl": "Train_RL.train_rl",
}


def prompt_model_filename(default_name: str) -> str:
    """Prompts the user for a model output filename in the terminal."""
    print("=" * 55)
    user_input = input(f"Enter model filename to save/resume (default: {default_name}): ").strip()
    filename = user_input if user_input else default_name
    if not filename.endswith(".pt"):
        filename += ".pt"
    print("=" * 55 + "\n")
    return filename


def analyze_training_health(stats: dict) -> list[str]:
    """Runs safety diagnostics on returned training metrics."""
    warnings = []

    start_loss = stats.get("start_loss", 0.0)
    final_loss = stats.get("final_loss", 0.0)
    iterations = stats.get("iterations", 1)

    diff = start_loss - final_loss
    pct = (diff / start_loss) * 100 if start_loss > 0 else 0

    # 1. Stagnation check
    if pct < STAGNATION_PCT_THRESHOLD and iterations >= STAGNATION_MIN_ITERATIONS:
        warnings.append(
            f"[CRITICAL] Training stalled on a plateau! Less than {STAGNATION_PCT_THRESHOLD}% loss reduction. "
            "Likely caused by dead ReLU neurons in deep layers (512 units x 5) or BatchNorm issues."
        )

    # 2. Mean collapse check
    if MEAN_COLLAPSE_MIN_LOSS < final_loss < MEAN_COLLAPSE_MAX_LOSS:
        warnings.append(
            "[WARNING] Final loss is near ~1.41. This indicates the network collapsed into predicting "
            "the mean zero output for all samples instead of learning gradients."
        )

    # 3. Exploding loss / NaN check
    if final_loss != final_loss or final_loss > EXPLODING_LOSS_THRESHOLD:
        warnings.append("[CRITICAL] Loss exploded or resulted in NaN! Lower your learning rate (--lr).")

    # 4. Low epoch count
    if final_loss > 0.1 and iterations < LOW_EPOCH_THRESHOLD:
        warnings.append(f"[NOTE] Epoch count is very low (< {LOW_EPOCH_THRESHOLD}). Consider increasing --epochs.")

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
    """Universal training execution loop with live disk saving and Ctrl+C trapping."""
    # 1. Auto-resume weights if file exists in temp_modelparts or temp_models
    resumed = False
    for path in [checkpoint_path, save_path]:
        if path.exists():
            print(f"[RESUME] Found existing weights at '{path}'. Loading into model...")
            model.load_state_dict(torch.load(path, weights_only=True))
            resumed = True
            break

    start_loss = None
    final_loss = 0.0
    completed_epochs = 0

    print(f"[TRAIN] Starting execution ({epochs} epochs)...")

    # 2. Main epoch loop with Ctrl+C interrupt safety
    try:
        for epoch in range(1, epochs + 1):
            loss_val = step_fn(epoch)

            if start_loss is None:
                start_loss = loss_val
            final_loss = loss_val
            completed_epochs += 1

            # Save state to disk on EVERY epoch
            torch.save(model.state_dict(), checkpoint_path)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] KeyboardInterrupt detected! Flushing current model weights to disk...")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[CHECKPOINT] Saved state safely to: {checkpoint_path}")

    # 3. Finalize saved model weights to temp_models
    torch.save(model.state_dict(), save_path)

    return {
        "resumed": resumed,
        "completed_epochs": completed_epochs,
        "start_loss": start_loss if start_loss is not None else 0.0,
        "final_loss": final_loss,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training methods safely")
    parser.add_argument("method", nargs="?", default=DEFAULT_METHOD, help=f"Training method key (default: {DEFAULT_METHOD})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Number of training iterations (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Final output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--ckpt-dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help=f"Intermediate weights directory (default: {DEFAULT_CHECKPOINT_DIR})")

    args = parser.parse_args()

    if args.method not in TRAINING_METHODS:
        raise ValueError(f"Unknown method '{args.method}'. Registered: {list(TRAINING_METHODS.keys())}")

    # Prompt user for model file name
    default_filename = f"{args.method}_model.pt"
    filename = prompt_model_filename(default_filename)

    # Ensure directories exist
    output_dir = Path(args.output_dir)
    ckpt_dir = Path(args.ckpt_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / filename
    checkpoint_path = ckpt_dir / f"part_{filename}"

    # Import selected module
    module = importlib.import_module(TRAINING_METHODS[args.method])

    # Check for setup function or legacy train function
    if hasattr(module, "setup"):
        model, step_fn, method_name = module.setup(lr=args.lr)
        loop_stats = execute_training_loop(model, step_fn, args.epochs, checkpoint_path, save_path)

        results = {
            "name": f"{method_name}{' [Resumed]' if loop_stats['resumed'] else ''}",
            "iterations": loop_stats["completed_epochs"],
            "start_loss": loop_stats["start_loss"],
            "final_loss": loop_stats["final_loss"],
            "save_path": str(save_path),
            "checkpoint_path": str(checkpoint_path),
        }
    else:
        # Fallback for legacy train() implementations
        results = module.train(epochs=args.epochs, lr=args.lr, save_path=str(save_path))
        results.setdefault("save_path", str(save_path))
        results.setdefault("checkpoint_path", str(checkpoint_path))

    # Run health checks and display summary
    display_results(results)
