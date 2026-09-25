"""
Main training runner wrapper script with safety and health checks
"""

import sys
import argparse
import importlib

TRAINING_METHODS = {
    "bc": "Train_BC.train_bc",
}


def analyze_training_health(stats: dict) -> list[str]:
    """Runs safety diagnostics on returned training metrics."""
    warnings = []

    start_loss = stats.get("start_loss", 0.0)
    final_loss = stats.get("final_loss", 0.0)
    iterations = stats.get("iterations", 1)

    # Calculate improvement metrics
    diff = start_loss - final_loss
    pct = (diff / start_loss) * 100 if start_loss > 0 else 0

    # 1. Check for complete stagnation / flat gradient plateau
    if pct < 20.0 and iterations >= 100:
        warnings.append(
            "[CRITICAL] Training stalled on a plateau! Less than 20% loss reduction. "
            "Likely caused by dead ReLU neurons in deep layers (512 units x 5) or BatchNorm issues."
        )

    # 2. Check for trivial output plateau (e.g., network outputting near 0 for all predictions)
    if 1.3 < final_loss < 1.5:
        warnings.append(
            "[WARNING] Final loss is near ~1.41. This indicates the network collapsed into predicting "
            "the mean zero output for all samples instead of learning gradients."
        )

    # 3. Check for exploding loss / NaN values
    if final_loss != final_loss or final_loss > 1e5:  # NaN check or explosion
        warnings.append("[CRITICAL] Loss exploded or resulted in NaN! Lower your learning rate (--lr).")

    # 4. Check for insufficient iterations
    if final_loss > 0.1 and iterations < 50:
        warnings.append("[NOTE] Epoch count is very low (< 50). Consider increasing --epochs.")

    return warnings


def display_results(stats: dict) -> None:
    start_loss = stats['start_loss']
    final_loss = stats['final_loss']
    iterations = stats['iterations']

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

    if warnings:
        print("\n" + "-" * 55)
        print(" SAFETY & DIAGNOSTIC ALERTS:")
        for warning in warnings:
            print(f"  * {warning}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training methods safely")
    parser.add_argument("method", nargs="?", default="bc", help="Training method key (default: bc)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()

    if args.method not in TRAINING_METHODS:
        raise ValueError(f"Unknown method '{args.method}'. Registered: {list(TRAINING_METHODS.keys())}")

    module = importlib.import_module(TRAINING_METHODS[args.method])

    # Execute training
    results = module.train(epochs=args.epochs, lr=args.lr)

    # Run safety checks and display
    display_results(results)
