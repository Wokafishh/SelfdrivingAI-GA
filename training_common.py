"""
Shared training code for the BC and SemiSL pipelines:
diagnostics, recording loading, train/validation split, augmentation and evaluation.
"""

import io
import zlib
import pickle
from pathlib import Path
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from arkitektur import IMAGE_SIZE, image_to_uint8_tensor

ImageFile.LOAD_TRUNCATED_IMAGES = True

VALIDATION_PERCENT = 15  # Share of recordings held out for validation
IDLE_THRESHOLD = 0.02    # |left| and |right| below this counts as standing still
NOISE_STD = 0.03         # Gaussian pixel noise added to "noisy" frames


# ==============================================================================
# [SHARED TRAINING DIAGNOSTICS] - Reusable across BC, RL, and SemiSL pipelines
# ==============================================================================
def safe_to_scalar(tensor_or_val) -> float:
    """Safely converts any tensor, slice, or scalar to a standard Python float."""
    if isinstance(tensor_or_val, torch.Tensor):
        return tensor_or_val.detach().float().mean().item()
    return float(tensor_or_val)


def print_batch_diagnostics(step, total_steps, loss, pred_l, pred_r, target_l, target_r):
    """Logs detailed step-by-step telemetry for monitoring training health."""
    pl_val = safe_to_scalar(pred_l)
    pr_val = safe_to_scalar(pred_r)
    tl_val = safe_to_scalar(target_l)
    tr_val = safe_to_scalar(target_r)

    print(
        f"[TRAIN DIAGNOSTIC | Step {step}/{total_steps}] "
        f"Loss: {loss:.5f} | "
        f"Pred L/R: [{pl_val:+.2f}, {pr_val:+.2f}] vs "
        f"Target L/R: [{tl_val:+.2f}, {tr_val:+.2f}]"
    )
# ==============================================================================


def hash_bucket(path, salt: str = "") -> int:
    """Stable 0-99 bucket for a file, based only on its name."""
    return zlib.crc32((salt + Path(path).name).encode("utf-8")) % 100


def split_files_by_recording(data_files) -> tuple[list[str], list[str]]:
    """Splits whole recordings into train/validation.

    Splitting per recording (not per frame) matters: neighbouring frames are
    nearly identical, so a per-frame split would leak training data into
    validation. The split is based on a hash of the filename, so a recording
    always lands on the same side, even when new recordings are added later.
    """
    pkl_files = sorted(str(p) for p in data_files if Path(p).suffix.lower() == ".pkl")
    if len(pkl_files) < 2:
        return pkl_files, []

    val_files = [p for p in pkl_files if hash_bucket(p) < VALIDATION_PERCENT]
    if not val_files:
        val_files = [min(pkl_files, key=hash_bucket)]
    train_files = [p for p in pkl_files if p not in val_files]
    return train_files, val_files


# --- Augmentation (operates on float tensors (3, 64, 64) in [0, 1]) ---
_random_crop = transforms.RandomResizedCrop(
    IMAGE_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1), antialias=True
)
_color_jitter = transforms.ColorJitter(
    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03
)


def augment_frame(image: torch.Tensor) -> torch.Tensor:
    """Random small crop/zoom plus brightness, contrast, saturation and hue changes."""
    return _color_jitter(_random_crop(image))


def noisy_frame(image: torch.Tensor) -> torch.Tensor:
    """A "noisy" version of a frame for consistency training.

    No mirroring here: a mirrored frame should get mirrored wheel values,
    which would break the "same state, same action" idea of consistency loss.
    """
    image = augment_frame(image)
    image = image + torch.randn_like(image) * NOISE_STD
    return image.clamp(0.0, 1.0)


def load_recording_file(path: Path, name: str):
    """Loads one recording (a pickled list of {'image', 'action'} dicts), or None on failure."""
    try:
        with open(path, "rb") as f:
            data_chunk = pickle.load(f)
    except Exception as e:
        print(f"[{name} Warning] Failed to load {path.name}: {e}")
        return None
    return data_chunk if isinstance(data_chunk, list) else None


def decode_frame(sample) -> torch.Tensor | None:
    try:
        return image_to_uint8_tensor(Image.open(io.BytesIO(sample["image"])))
    except Exception:
        return None


def stack_frames(images: list[torch.Tensor]) -> torch.Tensor:
    if images:
        return torch.stack(images)
    return torch.empty((0, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)


class DrivingDataset(Dataset):
    """Labeled frames (state S, expert action A) kept in memory as 64x64 uint8 tensors."""
    def __init__(self, file_paths, filter_zeros=True, augment=False, name="Dataset"):
        self.augment = augment

        print(f"[{name}] Loading {len(file_paths)} recording files...")
        images = []
        actions = []
        total_raw = 0
        filtered_count = 0
        unlabeled_count = 0
        decode_errors = 0

        for file_path in file_paths:
            path = Path(file_path)
            if path.suffix.lower() != ".pkl":
                continue
            data_chunk = load_recording_file(path, name)
            if data_chunk is None:
                continue

            for sample in data_chunk:
                total_raw += 1
                action = sample.get("action")
                if action is None:
                    unlabeled_count += 1
                    continue
                l_val, r_val = float(action[0]), float(action[1])

                if filter_zeros and abs(l_val) < IDLE_THRESHOLD and abs(r_val) < IDLE_THRESHOLD:
                    filtered_count += 1
                    continue

                image = decode_frame(sample)
                if image is None:
                    decode_errors += 1
                    continue
                images.append(image)
                actions.append((l_val, r_val))

        self.images = stack_frames(images)
        self.actions = torch.tensor(actions, dtype=torch.float32).reshape(-1, 2)

        print(f"[{name} Stats] Raw frames parsed: {total_raw}")
        print(f"[{name} Stats] Filtered out idle (0,0) frames: {filtered_count}")
        if unlabeled_count:
            print(f"[{name} Stats] Frames without an action skipped: {unlabeled_count}")
        print(f"[{name} Stats] Undecodable frames skipped: {decode_errors}")
        print(f"[{name} Stats] Active frames remaining: {len(self)}")

        if len(self) > 0:
            lefts, rights = self.actions[:, 0], self.actions[:, 1]
            print(f"[{name} Action Analysis]")
            print(f"  * Left Motor  -> Min: {lefts.min():.2f}, Max: {lefts.max():.2f}, Mean: {lefts.mean():.2f}")
            print(f"  * Right Motor -> Min: {rights.min():.2f}, Max: {rights.max():.2f}, Mean: {rights.mean():.2f}")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        image = self.images[idx].float() / 255.0
        left, right = self.actions[idx]

        if self.augment:
            # Mirror the frame: a left turn becomes a right turn, so the wheels swap
            if torch.rand(1).item() < 0.5:
                image = torch.flip(image, dims=[2])
                left, right = right, left
            image = augment_frame(image)

        return image, left.reshape(1), right.reshape(1)


class UnlabeledDataset(Dataset):
    """Frames used without their actions (state S only). Returns (clean, noisy) pairs."""
    def __init__(self, file_paths, name="Unlabeled"):
        print(f"[{name}] Loading {len(file_paths)} recording files...")
        images = []
        decode_errors = 0

        for file_path in file_paths:
            path = Path(file_path)
            if path.suffix.lower() != ".pkl":
                continue
            data_chunk = load_recording_file(path, name)
            if data_chunk is None:
                continue
            for sample in data_chunk:
                image = decode_frame(sample)
                if image is None:
                    decode_errors += 1
                    continue
                images.append(image)

        self.images = stack_frames(images)
        print(f"[{name} Stats] Frames loaded: {len(self)} (undecodable skipped: {decode_errors})")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        clean = self.images[idx].float() / 255.0
        return clean, noisy_frame(clean)


def action_loss(model, criterion, images, expert_left, expert_right):
    """Supervised loss between the model's actions Â and the expert's actions A."""
    pred_left, pred_right = model(images)
    loss = criterion(pred_left, expert_left) + criterion(pred_right, expert_right)
    return loss, pred_left, pred_right


@torch.no_grad()
def evaluate_loss(model, criterion, loader) -> float:
    """Mean supervised loss over a loader, with the model in eval mode."""
    was_training = model.training
    model.eval()
    total_loss, total_samples = 0.0, 0
    for images, expert_left, expert_right in loader:
        loss, _, _ = action_loss(model, criterion, images, expert_left, expert_right)
        total_loss += loss.item() * len(images)
        total_samples += len(images)
    model.train(was_training)
    return total_loss / total_samples


def print_mean_baseline(criterion, train_actions, val_actions) -> None:
    """Prints the val loss of always predicting the average action (the "ignore the image" score)."""
    mean_action = train_actions.mean(dim=0, keepdim=True).expand_as(val_actions)
    baseline = (
        criterion(mean_action[:, 0], val_actions[:, 0])
        + criterion(mean_action[:, 1], val_actions[:, 1])
    ).item()
    print(f"\n[BASELINE] Val loss when always predicting the average action: {baseline:.5f}")
    print("[BASELINE] Val loss must get clearly below this, otherwise the model ignores the image.")
