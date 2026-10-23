"""
Behaviour Cloning (BC) Training Module with Comprehensive Diagnostics

The model sees state S (a camera frame), predicts action Â (left/right wheel),
and is trained with a supervised loss against the expert's action A.
"""

import sys
from pathlib import Path

import torch
from torch import nn, optim

sys.path.append(str(Path(__file__).resolve().parent.parent))
from arkitektur import DrivingNetwork
from training_common import (
    AugmentationSchedule,
    DrivingDataset,
    action_loss,
    augment_labeled_batch,
    batch_indices,
    evaluate_loss,
    print_batch_diagnostics,
    print_mean_baseline,
    split_files_by_recording,
    to_cpu,
    to_float_images,
)

BATCH_SIZE = 64


def setup(lr: float = 1e-3, data_files=None, device="cpu"):
    torch.manual_seed(42)
    device = torch.device(device)

    if not data_files:
        raise ValueError("[BC Training Error] No data files were passed! Check your ./data directory.")

    train_files, val_files = split_files_by_recording(data_files)
    print(f"[Split] {len(train_files)} recordings for training, {len(val_files)} held out for validation.")
    for path in val_files:
        print(f"  * validation: {Path(path).name}")

    train_set = DrivingDataset(train_files, filter_zeros=True, name="Train")
    if len(train_set) == 0:
        raise ValueError("[BC Training Error] Training set is empty after filtering zeros!")

    val_set = DrivingDataset(val_files, filter_zeros=True, name="Validation") if val_files else None
    if val_set is not None and len(val_set) == 0:
        val_set = None
    if val_set is None:
        print("[Split WARNING] No validation data. Overfitting can't be detected.")

    model = DrivingNetwork(in_channels=3)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)

    if val_set is not None:
        print_mean_baseline(criterion, train_set.actions, val_set.actions)

    # All frames live on the training device; batches are augmented there
    model.to(device)
    train_set.to(device)
    if val_set is not None:
        val_set.to(device)
    model.train()
    batch_size = min(BATCH_SIZE, len(train_set))
    batches_per_epoch = -(-len(train_set) // batch_size)

    # --- DIAGNOSTIC SANITY CHECK ---
    print("\n[DIAGNOSTIC] Running model forward-pass sanity check...")
    try:
        sample_images = to_float_images(train_set.images[:2])
        sample_actions = train_set.actions[:2]
        model.eval()
        with torch.no_grad():
            p_l, p_r = model(sample_images)
        model.train()
        for i in range(len(sample_actions)):
            print(f"  * Test Sample {i + 1} -> Pred: [{p_l[i].item():+.2f}, {p_r[i].item():+.2f}] | Target: [{sample_actions[i, 0].item():+.2f}, {sample_actions[i, 1].item():+.2f}]")
    except Exception as e:
        print(f"[DIAGNOSTIC WARNING] Sanity check encountered an issue: {e}")
    print("[DIAGNOSTIC] Sanity check complete.\n")

    augmentation = AugmentationSchedule()

    def train_step(epoch: int) -> dict:
        """Runs one full pass over the training data and returns its metrics."""
        model.train()
        total_loss = torch.zeros((), device=device)

        for batch_idx, idx in enumerate(batch_indices(len(train_set), batch_size, device), start=1):
            images, actions = augment_labeled_batch(
                to_float_images(train_set.images[idx]), train_set.actions[idx], augmentation.strength
            )

            optimizer.zero_grad()
            loss, pred_left, pred_right = action_loss(model, criterion, images, actions)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Summed on the device; calling .item() every batch would stall the GPU
            total_loss += loss.detach() * len(idx)

            # Print detailed diagnostics every 50 batches
            if batch_idx % 50 == 0:
                print_batch_diagnostics(batch_idx, batches_per_epoch, loss.item(), pred_left[0], pred_right[0], actions[0, 0], actions[0, 1])

        train_loss = total_loss.item() / len(train_set)
        message = f"[EPOCH {epoch}] Train loss: {train_loss:.5f}"
        strength = augmentation.strength  # Used this epoch, may go up below

        val_loss = None
        if val_set is not None:
            val_loss = evaluate_loss(model, criterion, val_set)
            message += f" | Val loss: {val_loss:.5f}"
            if augmentation.update(val_loss):
                message += "  (best so far)"
        message += f" | Augmentation x{strength:.2f}"

        print(message)
        return {"loss": train_loss, "val_loss": val_loss, "augment_strength": strength}

    # Saved/restored by run_training.py so resumed sessions continue seamlessly
    def get_state() -> dict:
        return {"optimizer": to_cpu(optimizer.state_dict()), **augmentation.state_dict()}

    def set_state(state: dict) -> None:
        optimizer.load_state_dict(state["optimizer"])
        augmentation.load_state_dict(state)

    train_step.get_state = get_state
    train_step.set_state = set_state

    return model, train_step, "Behaviour Cloning (BC)"
