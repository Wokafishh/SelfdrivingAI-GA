"""
Self-Supervised / Semi-Supervised Learning (SSL) — base script.

Workflow (as confirmed):
  1. Small labeled batch  -> student predicts -> compare to true label -> supervised loss (= BC).
  2. Large unlabeled batch (+noise) -> student predicts on noisy input;
     teacher (EMA copy of student, no grad) predicts on clean input ->
     compare -> consistency loss.
  3. loss = supervised + ramp * consistency -> backward -> STUDENT weights update only.
  4. teacher = decay * teacher + (1 - decay) * student   (no gradient, pure averaging)
"""
import torch
import torch.nn as nn
import torch.optim.swa_utils as swa_utils

# ============================== CONFIG ==============================
OBS_DIM = 8              # input (state) dimensionality
ACT_DIM = 2               # output (action) dimensionality
HIDDEN = 128               # hidden layer width

N_LABELED = 200            # size of the small labeled pool
N_UNLABELED = 2000         # size of the large unlabeled pool

BATCH_LABELED = 32         # labeled samples per step
BATCH_UNLABELED = 128      # unlabeled samples per step

NOISE_STD = 0.05           # std of noise added before feeding the student
EMA_DECAY = 0.99           # how slowly the teacher tracks the student
CONSISTENCY_WEIGHT = 1.0   # max weight of the consistency loss
RAMP_STEPS = 500           # steps to linearly ramp consistency weight 0 -> 1

TOTAL_STEPS = 2000
LR = 3e-4
GRAD_CLIP = 1.0
LOG_EVERY = 50

SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "ssl_checkpoint.pt"
# ======================================================================


def main():
    print(f"[setup] device={DEVICE}  seed={SEED}")
    torch.manual_seed(SEED)

    labeled_x, labeled_y, unlabeled_x = generate_data()
    labeled_x, labeled_y = labeled_x.to(DEVICE), labeled_y.to(DEVICE)
    unlabeled_x = unlabeled_x.to(DEVICE)
    print(f"[data] labeled={labeled_x.shape}  labeled_targets={labeled_y.shape}  unlabeled={unlabeled_x.shape}")

    student = build_model().to(DEVICE)

    # teacher = EMA copy of student. torch.optim.swa_utils.AveragedModel
    # handles the running-average bookkeeping natively.
    teacher = swa_utils.AveragedModel(student, avg_fn=ema_avg_fn).to(DEVICE)
    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    mse = nn.MSELoss()

    print(f"[model] student params={count_params(student):,}")
    print(f"[train] starting {TOTAL_STEPS} steps\n")

    for step in range(1, TOTAL_STEPS + 1):
        student.train()

        # ---- 1. supervised loss on labeled batch (this part = plain BC) ----
        lab_idx = torch.randint(0, N_LABELED, (BATCH_LABELED,), device=DEVICE)
        x_lab, y_lab = labeled_x[lab_idx], labeled_y[lab_idx]
        pred_lab = student(x_lab)
        sup_loss = mse(pred_lab, y_lab)

        # ---- 2. consistency loss on unlabeled (+labeled) batch ----
        unlab_idx = torch.randint(0, N_UNLABELED, (BATCH_UNLABELED,), device=DEVICE)
        x_unlab = unlabeled_x[unlab_idx]
        x_all = torch.cat([x_lab, x_unlab], dim=0)

        noise = torch.randn_like(x_all) * NOISE_STD
        student_pred = student(x_all + noise)
        with torch.no_grad():
            teacher_pred = teacher(x_all)
        consistency_loss = mse(student_pred, teacher_pred)

        # ---- 3. combine + backward (student only) ----
        ramp = min(1.0, step / RAMP_STEPS)
        loss = sup_loss + CONSISTENCY_WEIGHT * ramp * consistency_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        optimizer.step()

        # ---- 4. teacher drifts toward student (no gradient) ----
        teacher.update_parameters(student)

        if step % LOG_EVERY == 0 or step == 1:
            print(
                f"step {step:5d}/{TOTAL_STEPS}  "
                f"sup_loss={sup_loss.item():.4f}  "
                f"consistency_loss={consistency_loss.item():.4f}  "
                f"ramp={ramp:.2f}  "
                f"total_loss={loss.item():.4f}"
            )

    print("\n[train] done.")
    torch.save(
        {"student_state": student.state_dict(), "teacher_state": teacher.module.state_dict()},
        CKPT_PATH,
    )
    print(f"[save] checkpoint written to {CKPT_PATH}")


if __name__ == "__main__":
    main()


# ============================== HELPERS ==============================

def build_model() -> nn.Module:
    """Simple MLP: state -> action."""
    return nn.Sequential(
        nn.Linear(OBS_DIM, HIDDEN),
        nn.Tanh(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.Tanh(),
        nn.Linear(HIDDEN, ACT_DIM),
    )


def ema_avg_fn(averaged_param: torch.Tensor, model_param: torch.Tensor, num_averaged: torch.Tensor) -> torch.Tensor:
    """avg_fn for torch.optim.swa_utils.AveragedModel — implements
    teacher = EMA_DECAY * teacher + (1 - EMA_DECAY) * student.
    """
    return EMA_DECAY * averaged_param + (1.0 - EMA_DECAY) * model_param


def generate_data():
    """Synthetic stand-in dataset, built entirely with torch ops.

    A fixed random ground-truth function maps state -> action. The labeled
    set gets that true action (what a human/expert label would look like);
    the unlabeled set only gets states, no action — exactly the SSL setup.
    Swap this out for your real small-labeled / large-unlabeled data.
    """
    gen = torch.Generator().manual_seed(SEED)

    true_w1 = torch.randn(OBS_DIM, HIDDEN, generator=gen)
    true_w2 = torch.randn(HIDDEN, ACT_DIM, generator=gen)

    def true_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x @ true_w1) @ true_w2

    labeled_x = torch.randn(N_LABELED, OBS_DIM, generator=gen)
    labeled_y = true_fn(labeled_x) + 0.01 * torch.randn(N_LABELED, ACT_DIM, generator=gen)

    unlabeled_x = torch.randn(N_UNLABELED, OBS_DIM, generator=gen)

    return labeled_x, labeled_y, unlabeled_x


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
