"""
Shared training code for the BC, SemiSL and RL pipelines:
device selection, diagnostics, recording loading, train/validation split,
batch augmentation and evaluation.

Training works on CPU (e.g. Linux laptop) and CUDA (e.g. Windows + RTX 3060).
All frames are kept in memory on the training device and augmented a whole
batch at a time there, so the GPU isn't waiting for the CPU.
"""

import io
import math
import zlib
import pickle
from pathlib import Path
from PIL import Image, ImageFile

import torch
import torch.nn.functional as F

from arkitektur import IMAGE_SIZE, image_to_uint8_tensor

ImageFile.LOAD_TRUNCATED_IMAGES = True

VALIDATION_PERCENT = 15  # Share of recordings held out for validation
IDLE_THRESHOLD = 0.02    # |left| and |right| below this counts as standing still
NOISE_STD = 0.03         # Gaussian pixel noise added to "noisy" frames

# Augmentation strength
CROP_SCALE = (0.85, 1.0)   # Share of the image area kept by the random crop
CROP_RATIO = (0.9, 1.1)    # Aspect ratio range of the crop
BRIGHTNESS = 0.3
CONTRAST = 0.3
SATURATION = 0.3
HUE = 0.03                 # Fraction of a full turn around the color wheel


# ==============================================================================
# DEVICE
# ==============================================================================
def resolve_device(name: str = "auto") -> torch.device:
    """"auto" = CUDA if available, else CPU. Asking for CUDA without a GPU is an error."""
    name = (name or "auto").lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this torch install can't see a GPU "
                f"(torch {torch.__version__}). Install the CUDA build: "
                "pip install -r requirements-cuda.txt"
            )
        torch.backends.cudnn.benchmark = True  # Input size never changes, let cuDNN pick fast kernels
    elif name != "cpu":
        raise ValueError(f"Unknown device '{name}', use cpu, cuda or auto.")
    return torch.device(name)


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return "cpu"


def to_cpu(obj):
    """Recursively moves tensors to the CPU, so saved files load on computers without CUDA."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return type(obj)((key, to_cpu(value)) for key, value in obj.items())
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(value) for value in obj)
    return obj


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


# ==============================================================================
# BATCH AUGMENTATION (float tensors (B, 3, 64, 64) in [0, 1], on any device)
# ==============================================================================
_RGB_TO_YIQ = torch.tensor([
    [0.299, 0.587, 0.114],
    [0.596, -0.274, -0.322],
    [0.211, -0.523, 0.312],
])
_YIQ_TO_RGB = torch.linalg.inv(_RGB_TO_YIQ)


def _uniform(batch: int, low: float, high: float, device) -> torch.Tensor:
    return torch.empty(batch, device=device).uniform_(low, high)


def _grayscale(images: torch.Tensor) -> torch.Tensor:
    return (0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3])


def augment_images(images: torch.Tensor) -> torch.Tensor:
    """Random small crop/zoom plus brightness, contrast, saturation and hue changes, per image."""
    batch, device = images.shape[0], images.device

    # Random resized crop, done as a per-image zoom + shift
    area = _uniform(batch, *CROP_SCALE, device)
    ratio = _uniform(batch, math.log(CROP_RATIO[0]), math.log(CROP_RATIO[1]), device).exp()
    width = (area * ratio).sqrt().clamp(max=1.0)
    height = (area / ratio).sqrt().clamp(max=1.0)
    theta = torch.zeros(batch, 2, 3, device=device)
    theta[:, 0, 0] = width
    theta[:, 1, 1] = height
    theta[:, 0, 2] = (torch.rand(batch, device=device) * 2 - 1) * (1 - width)
    theta[:, 1, 2] = (torch.rand(batch, device=device) * 2 - 1) * (1 - height)
    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    images = F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)

    # Brightness
    images = (images * _uniform(batch, 1 - BRIGHTNESS, 1 + BRIGHTNESS, device).view(-1, 1, 1, 1)).clamp(0, 1)

    # Contrast: blend towards the image's mean gray level
    mean_gray = _grayscale(images).mean(dim=(1, 2, 3), keepdim=True)
    factor = _uniform(batch, 1 - CONTRAST, 1 + CONTRAST, device).view(-1, 1, 1, 1)
    images = ((images - mean_gray) * factor + mean_gray).clamp(0, 1)

    # Saturation: blend towards the grayscale image
    gray = _grayscale(images)
    factor = _uniform(batch, 1 - SATURATION, 1 + SATURATION, device).view(-1, 1, 1, 1)
    images = ((images - gray) * factor + gray).clamp(0, 1)

    # Hue: rotate the color part (I, Q) in YIQ color space
    angle = _uniform(batch, -HUE, HUE, device) * 2 * math.pi
    rotation = torch.zeros(batch, 3, 3, device=device)
    rotation[:, 0, 0] = 1.0
    rotation[:, 1, 1] = angle.cos()
    rotation[:, 1, 2] = -angle.sin()
    rotation[:, 2, 1] = angle.sin()
    rotation[:, 2, 2] = angle.cos()
    matrix = _YIQ_TO_RGB.to(device) @ rotation @ _RGB_TO_YIQ.to(device)
    images = torch.einsum("bij,bjhw->bihw", matrix, images).clamp(0, 1)

    return images


def augment_labeled_batch(images: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Augmentation for (state S, action A) pairs. Mirrored frames get swapped wheel values."""
    # Mirror the frame: a left turn becomes a right turn, so the wheels swap
    flip = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(flip.view(-1, 1, 1, 1), images.flip(-1), images)
    actions = torch.where(flip.view(-1, 1), actions.flip(-1), actions)
    return augment_images(images), actions


def noisy_images(images: torch.Tensor) -> torch.Tensor:
    """A "noisy" version of frames for consistency training.

    No mirroring here: a mirrored frame should get mirrored wheel values,
    which would break the "same state, same action" idea of consistency loss.
    """
    images = augment_images(images)
    return (images + torch.randn_like(images) * NOISE_STD).clamp(0, 1)


def to_float_images(images_uint8: torch.Tensor) -> torch.Tensor:
    return images_uint8.float() / 255.0


def batch_indices(count: int, batch_size: int, device, shuffle: bool = True):
    """Yields index tensors for one pass over `count` samples."""
    order = torch.randperm(count, device=device) if shuffle else torch.arange(count, device=device)
    yield from order.split(batch_size)


def endless_batch_indices(count: int, batch_size: int, device):
    """Yields shuffled index batches forever, reshuffling every pass."""
    while True:
        yield from batch_indices(count, batch_size, device, shuffle=True)


# ==============================================================================
# LOADING RECORDINGS
# ==============================================================================
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


class DrivingDataset:
    """Labeled frames (state S, expert action A).

    images: uint8 (N, 3, 64, 64), actions: float (N, 2) as (left, right).
    Call .to(device) to move both to the training device.
    """
    def __init__(self, file_paths, filter_zeros=True, name="Dataset"):
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

    def to(self, device):
        self.images = self.images.to(device)
        self.actions = self.actions.to(device)
        return self


class UnlabeledDataset:
    """Frames used without their actions (state S only). images: uint8 (N, 3, 64, 64)."""
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

    def to(self, device):
        self.images = self.images.to(device)
        return self


# ==============================================================================
# LOSS & EVALUATION
# ==============================================================================
def action_loss(model, criterion, images, actions):
    """Supervised loss between the model's actions Â and the expert's actions A (N, 2)."""
    pred_left, pred_right = model(images)
    loss = criterion(pred_left, actions[:, 0:1]) + criterion(pred_right, actions[:, 1:2])
    return loss, pred_left, pred_right


@torch.no_grad()
def evaluate_loss(model, criterion, dataset: DrivingDataset, batch_size: int = 256) -> float:
    """Mean supervised loss over a dataset (no augmentation), with the model in eval mode."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    for idx in batch_indices(len(dataset), batch_size, dataset.images.device, shuffle=False):
        loss, _, _ = action_loss(model, criterion, to_float_images(dataset.images[idx]), dataset.actions[idx])
        total_loss += loss.item() * len(idx)
    model.train(was_training)
    return total_loss / len(dataset)


def print_mean_baseline(criterion, train_actions, val_actions) -> None:
    """Prints the val loss of always predicting the average action (the "ignore the image" score)."""
    mean_action = train_actions.mean(dim=0, keepdim=True).expand_as(val_actions)
    baseline = (
        criterion(mean_action[:, 0], val_actions[:, 0])
        + criterion(mean_action[:, 1], val_actions[:, 1])
    ).item()
    print(f"\n[BASELINE] Val loss when always predicting the average action: {baseline:.5f}")
    print("[BASELINE] Val loss must get clearly below this, otherwise the model ignores the image.")
