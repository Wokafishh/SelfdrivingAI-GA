"""
Semi-Supervised Learning (SemiSL) with a "mean teacher".

Two models with the same architecture:
  - Student: trained normally with gradients and backpropagation.
  - Teacher: no gradients. After every step its weights W move slightly
    towards the Student's (EMA): W_teacher = d * W_teacher + (1 - d) * W_student

Two data streams per step (labeled : unlabeled = 1 : 8 frames):
  - Labeled (state S + expert action A): supervised loss, same as BC.
  - Unlabeled (state S only): the Student gets a noisy version of the frame,
    the Teacher gets the clean frame. Consistency loss = MSE between them.

The Teacher is the model that gets saved (it is usually the more stable one).
"""

import sys
import copy
import math
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from arkitektur import DrivingNetwork
from training_common import (
    DrivingDataset,
    UnlabeledDataset,
    action_loss,
    evaluate_loss,
    hash_bucket,
    print_batch_diagnostics,
    print_mean_baseline,
    split_files_by_recording,
)

# Share of training recordings whose actions are used as labels. The rest are
# used as unlabeled data (their actions are ignored). Lower this to test how well
# SemiSL does with little labeled data; set it to 100 to use every label.
LABELED_PERCENT = 25

# Extra recordings without any actions (state S only), e.g. just driving around filming
UNLABELED_DATA_DIR = ROOT_DIR / "server" / "unlabeled_data"

LABELED_BATCH_SIZE = 16
UNLABELED_BATCH_SIZE = 128  # 1:8 ratio between labeled and unlabeled frames
EMA_DECAY = 0.99
CONSISTENCY_WEIGHT = 5.0
CONSISTENCY_RAMPUP_EPOCHS = 5  # Consistency loss is faded in, the Teacher is random at first


def consistency_weight(epoch: int) -> float:
    """Sigmoid ramp-up from the original mean teacher paper (Tarvainen & Valpola, 2017)."""
    if CONSISTENCY_RAMPUP_EPOCHS <= 0:
        return CONSISTENCY_WEIGHT
    t = min(epoch / CONSISTENCY_RAMPUP_EPOCHS, 1.0)
    return CONSISTENCY_WEIGHT * math.exp(-5.0 * (1.0 - t) ** 2)


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    """Moves the Teacher's weights a tiny bit towards the Student's."""
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.mul_(decay).add_(s_param, alpha=1.0 - decay)
    # BatchNorm running statistics are copied directly
    for t_buf, s_buf in zip(teacher.buffers(), student.buffers()):
        t_buf.copy_(s_buf)


def cycle(loader):
    """Iterates a DataLoader forever, reshuffling each pass."""
    while True:
        for batch in loader:
            yield batch


def setup(lr: float = 1e-3, data_files=None):
    torch.manual_seed(42)

    if not data_files:
        raise ValueError("[SemiSL Training Error] No data files were passed! Check your ./data directory.")

    train_files, val_files = split_files_by_recording(data_files)
    labeled_files = [p for p in train_files if hash_bucket(p, salt="labeled") < LABELED_PERCENT]
    if not labeled_files and train_files:
        labeled_files = [min(train_files, key=lambda p: hash_bucket(p, salt="labeled"))]

    extra_unlabeled = sorted(str(p) for p in UNLABELED_DATA_DIR.glob("*.pkl")) if UNLABELED_DATA_DIR.exists() else []

    print(f"[Split] {len(train_files)} training recordings, {len(val_files)} held out for validation.")
    print(f"[Split] Labels used from {len(labeled_files)} recordings ({LABELED_PERCENT}%). "
          f"All {len(train_files)} training recordings + {len(extra_unlabeled)} extra unlabeled "
          f"recordings feed the unlabeled stream.")

    labeled_set = DrivingDataset(labeled_files, filter_zeros=True, augment=True, name="Labeled")
    if len(labeled_set) == 0:
        raise ValueError("[SemiSL Training Error] Labeled set is empty after filtering zeros!")
    unlabeled_set = UnlabeledDataset(train_files + extra_unlabeled, name="Unlabeled")

    val_set = DrivingDataset(val_files, filter_zeros=True, augment=False, name="Validation") if val_files else None
    if val_set is not None and len(val_set) == 0:
        val_set = None
    if val_set is None:
        print("[Split WARNING] No validation data. Overfitting can't be detected.")

    labeled_loader = DataLoader(labeled_set, batch_size=min(LABELED_BATCH_SIZE, len(labeled_set)), shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=min(UNLABELED_BATCH_SIZE, len(unlabeled_set)), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256) if val_set is not None else None

    labeled_iter = cycle(labeled_loader)
    unlabeled_iter = cycle(unlabeled_loader)
    steps_per_epoch = max(len(labeled_loader), len(unlabeled_loader))

    student = DrivingNetwork(in_channels=3)
    teacher = copy.deepcopy(student)
    for param in teacher.parameters():
        param.requires_grad_(False)

    optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=1e-4)
    supervised_criterion = nn.HuberLoss(delta=1.0)  # Same supervised loss as BC
    consistency_criterion = nn.MSELoss()

    if val_set is not None:
        print_mean_baseline(supervised_criterion, labeled_set.actions, val_set.actions)
    print(f"\n[SemiSL] {steps_per_epoch} steps per epoch | EMA decay {EMA_DECAY} | "
          f"consistency weight up to {CONSISTENCY_WEIGHT} over {CONSISTENCY_RAMPUP_EPOCHS} epochs\n")

    best_val_loss = float("inf")
    first_step = True

    def train_step(epoch: int) -> float:
        """One epoch of mixed labeled/unlabeled steps. Returns the mean supervised loss
        (comparable with BC; the total loss grows while the consistency weight ramps up)."""
        nonlocal best_val_loss, first_step

        if first_step:
            # run_training loads resumed weights into the Teacher (the saved model),
            # so start the Student from the same weights
            student.load_state_dict(teacher.state_dict())
            first_step = False

        student.train()
        teacher.eval()
        weight = consistency_weight(epoch - 1)
        totals = {"loss": 0.0, "supervised": 0.0, "consistency": 0.0}

        for step in range(1, steps_per_epoch + 1):
            images, expert_left, expert_right = next(labeled_iter)
            clean, noisy = next(unlabeled_iter)

            optimizer.zero_grad()

            # Labeled stream: same as behaviour cloning
            supervised, pred_left, pred_right = action_loss(student, supervised_criterion, images, expert_left, expert_right)

            # Unlabeled stream: Student on noisy frame vs Teacher on clean frame
            with torch.no_grad():
                teacher_left, teacher_right = teacher(clean)
            student_left, student_right = student(noisy)
            consistency = (
                consistency_criterion(student_left, teacher_left)
                + consistency_criterion(student_right, teacher_right)
            )

            loss = supervised + weight * consistency
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            ema_update(teacher, student, EMA_DECAY)

            totals["loss"] += loss.item()
            totals["supervised"] += supervised.item()
            totals["consistency"] += consistency.item()

            if step % 50 == 0:
                print_batch_diagnostics(step, steps_per_epoch, loss.item(), pred_left[0], pred_right[0], expert_left[0], expert_right[0])

        means = {k: v / steps_per_epoch for k, v in totals.items()}
        message = (
            f"[EPOCH {epoch}] Loss: {means['loss']:.5f} "
            f"(supervised {means['supervised']:.5f}, consistency {means['consistency']:.5f} x{weight:.2f})"
        )

        if val_loader is not None:
            teacher_val = evaluate_loss(teacher, supervised_criterion, val_loader)
            student_val = evaluate_loss(student, supervised_criterion, val_loader)
            message += f" | Val loss teacher: {teacher_val:.5f}, student: {student_val:.5f}"
            if teacher_val < best_val_loss:
                best_val_loss = teacher_val
                message += "  (best so far)"

        print(message)
        return means["supervised"]

    return teacher, train_step, "Semi-Supervised Learning (Mean Teacher)"
