"""
live_distillation_mobilevit.py
-------------------------------
USAGE:

python3 Yolo_Seg/vit_training/dinov3_distill_training/live_distillation_mobilevit.py \
  --image-dir /path/to/images \
  --weights /path/to/dinov3_weights.pth \
  --checkpoint-dir checkpoints_split_norm_448 \
  --init-from /path/to/legacy_epoch0005.pth \
  --image-size 448

Knowledge distillation: DINOv3 ViT-Large/16 (teacher) → MobileViT-S (student).

Tensor contract (verified by shape trace)
------------------------------------------
  Teacher  x_norm_patchtokens : [B, G*G, 1024]  →  [B, G, G, 1024]  L2-normalised
  Student  forward_distill()  : [B, G, G, 1024]  L2-normalised
  Both share the same augmented image, then use branch-specific normalisation.

Loss: normalised-sum (sum over C=1024 feature dim, mean over B×G×G spatial tokens)
  See LOSS DESIGN NOTE below for the mathematical justification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
import csv
import json
import math
import numpy as np
import os
import re
import sys
import time

try:
    import yaml
except ImportError:
    yaml = None

try:
    import cv2
except ImportError:
    cv2 = None

VIT_SCRIPTS_DIR = Path(os.environ.get(
    "VIT_SCRIPTS_DIR",
    Path(__file__).resolve().parents[2] / "scripts",
))
if not VIT_SCRIPTS_DIR.is_dir():
    raise ImportError(f"Shared ViT architecture directory not found: {VIT_SCRIPTS_DIR}")
if str(VIT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(VIT_SCRIPTS_DIR))

from mobilevit_distill import mobilevit_s_distill, MobileViT, count_parameters
from sonar_feature_engineering import make_sonar_features
import torchvision.transforms.functional as TF


TEACHER_MEAN = (0.485, 0.456, 0.406)
TEACHER_STD = (0.229, 0.224, 0.225)
STUDENT_MEAN = (0.4743, 0.5000, 0.4715)
STUDENT_STD = (0.1319, 0.1335, 0.1367)
SONAR_FEATURE_MEAN = (0.091898, 0.132347, 0.114527)
SONAR_FEATURE_STD = (0.172297, 0.202109, 0.186716)
RGB_PREPROCESSING_CONTRACT = "dinov3_imagenet_teacher__beex_student_v1"
FLS_PREPROCESSING_CONTRACT = "dinov3_imagenet_teacher__fls_sonar_greyscale_centercrop_jpeg_v1"

DEFAULT_RUN_CONFIG = {
    "image_dir": "/workspace/Datasets_VIT",
    "weights": "/workspace/Pretrained_Dino_based_MVIT_Distillation/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "epochs": 50,
    "batch_size": 128,
    "lr": 1e-3,
    "checkpoint_every": 1,
    "checkpoint_dir": "checkpoints",
    "resume_from": "",
    "init_from": "",
    "dinov3_repo": "/workspace/dinov3",
    "image_size": 448,
    "input_mode": "rgb",
    "student_mean": None,
    "student_std": None,
    "sonar_middle_channel": "wavelet_low",
    "sonar_third_channel": "sobel_edge",
    "sonar_wavelet": "haar",
    "sonar_occupancy_threshold": 128,
    "sonar_local_contrast_blur": 31,
    "sonar_edge_blur": 3,
    "sonar_mask_mode": "support_hull",
    "sonar_mask_threshold": 3,
    "sonar_mask_close_kernel": 21,
    "sonar_mask_min_coverage": 0.05,
    "lambda_sigreg": 0.0,
    "sigreg_max_tokens": 2048,
    "sigreg_projections": 256,
    "sigreg_warmup_epochs": 0,
}

CONFIG_GROUP_KEYS = {
    "paths",
    "training",
    "data",
    "input",
    "sonar",
    "loss",
    "sigreg",
    "resume",
    "checkpoint",
    "checkpoints",
}


# ---------------------------------------------------------------------------
# LOSS DESIGN NOTE — "normalised sum" for dense feature distillation
#
# Both tensors entering the loss have shape [B, G, G, 1024].
#
#   reduction='mean'  → divides by B × G × G × 1024 elements.
#                       A large mismatch in one of the 1024 feature dims is
#                       diluted by the 1023 dims already well-matched.
#                       Fine-grained per-feature errors are suppressed.
#
#   reduction='sum'   → gradient magnitude ≈ 25 M× larger than 'mean'.
#                       Must re-tune LR every time batch size changes.
#
#   CHOSEN: sum over C=1024 feature dim, mean over B × G × G spatial tokens.
#
#   MSE formula:    loss = mean_{B,H,W} [ sum_C (s_c - t_c)^2 ]
#                        = ((s - t)**2).sum(dim=-1).mean()
#
#   Cosine formula: loss = mean_{B,H,W} [ 1 - cos_sim_C(s, t) ]
#                        = (1 - F.cosine_similarity(s, t, dim=-1)).mean()
#
#   Both are batch-size-independent; both preserve full per-feature supervision
#   pressure at every spatial token.
#
#   Both tensors are L2-normalised over dim=-1 before the loss, so:
#     • MSE on unit vectors is bounded in [0, 4] per element.
#     • Cosine similarity is the dot product; (1−cos) ∈ [0, 2].
#     • The two losses are complementary: MSE penalises magnitude + direction,
#       cosine penalises direction only. Together they constrain both aspects.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 1.  Dataset
# ---------------------------------------------------------------------------

def parse_triplet(value, name: str) -> tuple:
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        parts = [float(part) for part in value]
        if len(parts) != 3:
            raise ValueError(f"{name} must contain three values, got: {value}")
        return tuple(parts)
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"{name} must contain three comma-separated numbers, got: {value}")
    return tuple(parts)


def normalise_config_key(key: str) -> str:
    key = str(key).strip().replace("-", "_")
    aliases = {
        "weights_path": "weights",
        "dinov3_weights": "weights",
        "checkpoint_every_n_epochs": "checkpoint_every",
        "resume": "resume_from",
        "init": "init_from",
    }
    return aliases.get(key, key)


def flatten_run_config(data: dict) -> dict:
    flat = {}
    for key, value in data.items():
        normalised_key = normalise_config_key(key)
        if isinstance(value, dict) and normalised_key in CONFIG_GROUP_KEYS:
            flat.update(flatten_run_config(value))
        else:
            flat[normalised_key] = value
    return flat


def load_yaml_run_config(path: str) -> dict:
    if not path:
        return {}
    if yaml is None:
        raise ImportError("PyYAML is required for --config. Install it with: pip install pyyaml")
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return flatten_run_config(data)


def serialise_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [serialise_value(item) for item in value]
    if isinstance(value, list):
        return [serialise_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialise_value(item) for key, item in value.items()}
    return value


def write_run_config_files(
    checkpoint_dir: str,
    run_config: dict,
    config_path: str = "",
    runtime_info: dict = None,
) -> None:
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(config_path or ""),
        "run_config": serialise_value(run_config),
        "runtime": serialise_value(runtime_info or {}),
    }

    json_path = ckpt_dir / "training_config_resolved.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    if yaml is not None:
        yaml_path = ckpt_dir / "training_config_resolved.yaml"
        with yaml_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)

    if config_path:
        source_path = Path(config_path).expanduser()
        if source_path.is_file():
            source_copy_path = ckpt_dir / "training_config_source.yaml"
            source_copy_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")


def append_training_log(checkpoint_dir: str, record: dict) -> None:
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = ckpt_dir / "training_log.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(serialise_value(record), sort_keys=True) + "\n")

    csv_path = ckpt_dir / "training_log.csv"
    csv_columns = [
        "epoch",
        "avg_loss",
        "avg_mse",
        "avg_cosine",
        "avg_sigreg",
        "avg_valid_patch_fraction",
        "lambda_sigreg",
        "lr",
        "epoch_seconds",
    ]
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_columns = reader.fieldnames or []
            existing_rows = list(reader)
        if existing_columns != csv_columns:
            migration_path = csv_path.with_suffix(".csv.tmp")
            with migration_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=csv_columns)
                writer.writeheader()
                writer.writerows(
                    {column: row.get(column, "") for column in csv_columns}
                    for row in existing_rows
                )
            migration_path.replace(csv_path)

    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        if write_header:
            writer.writeheader()
        writer.writerow({column: record.get(column, "") for column in csv_columns})


def preprocessing_contract(
    input_mode: str,
    student_mean: tuple,
    student_std: tuple,
    sonar_middle_channel: str,
    sonar_third_channel: str,
    sonar_wavelet: str,
    sonar_occupancy_threshold: int,
    sonar_local_contrast_blur: int,
    sonar_edge_blur: int,
) -> str:
    if input_mode == "rgb":
        return RGB_PREPROCESSING_CONTRACT
    if input_mode == "fls_grayscale":
        return FLS_PREPROCESSING_CONTRACT
    mean_key = "_".join(f"{v:.6f}" for v in student_mean)
    std_key = "_".join(f"{v:.6f}" for v in student_std)
    feature_prefix = (
        "dinov3_imagenet_teacher__fls_sonar_features"
        if input_mode == "fls_features"
        else "dinov3_imagenet_teacher__sonar_student"
    )
    return (
        feature_prefix +
        f"__{sonar_middle_channel}"
        f"__third_{sonar_third_channel}"
        f"__wavelet_{sonar_wavelet}"
        f"__occ_{sonar_occupancy_threshold}"
        f"__localblur_{sonar_local_contrast_blur}"
        f"__edgeblur_{sonar_edge_blur}"
        f"__mean_{mean_key}"
        f"__std_{std_key}"
    )


def spatial_mask_contract(
    input_mode: str,
    mask_mode: str,
    threshold: int,
    close_kernel: int,
    min_coverage: float,
) -> str:
    if input_mode == "rgb" or mask_mode == "none":
        return "all_grid_cells_v1"
    return (
        f"sonar_support_mask_v1__mode_{mask_mode}"
        f"__threshold_{int(threshold)}"
        f"__close_{int(close_kernel)}"
        f"__coverage_{float(min_coverage):.4f}"
    )


def make_sonar_support_mask(gray: np.ndarray, threshold: int, close_kernel: int) -> np.ndarray:
    """Recover the sonar support while retaining dark shadows inside it."""
    seed = (gray > int(threshold)).astype(np.uint8)
    if seed.sum() < 16:
        return seed.astype(bool)

    if cv2 is not None:
        opened = cv2.morphologyEx(
            seed,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        kernel = max(int(close_kernel), 3)
        if kernel % 2 == 0:
            kernel += 1
        closed = cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)),
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        if count <= 1:
            return closed.astype(bool)
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        component = (labels == largest).astype(np.uint8)
        ys, xs = np.where(component > 0)
        if len(xs) < 3:
            return component.astype(bool)
        hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
        support = np.zeros_like(component)
        cv2.fillConvexPoly(support, hull, 1)
        return support.astype(bool)

    support = np.zeros_like(seed, dtype=bool)
    for row in range(seed.shape[0]):
        columns = np.flatnonzero(seed[row])
        if columns.size:
            support[row, columns[0]:columns[-1] + 1] = True
    return support


class LiveAugmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        image_size: int = 448,
        input_mode: str = "rgb",
        sonar_middle_channel: str = "wavelet_low",
        sonar_third_channel: str = "sobel_edge",
        sonar_wavelet: str = "haar",
        sonar_occupancy_threshold: int = 128,
        sonar_local_contrast_blur: int = 31,
        sonar_edge_blur: int = 3,
        sonar_mask_mode: str = "support_hull",
        sonar_mask_threshold: int = 3,
        sonar_mask_close_kernel: int = 21,
    ):
        self.directory = Path(image_dir)
        self.image_size = int(image_size)
        self.input_mode = input_mode
        self.sonar_middle_channel = sonar_middle_channel
        self.sonar_third_channel = sonar_third_channel
        self.sonar_wavelet = sonar_wavelet
        self.sonar_occupancy_threshold = int(sonar_occupancy_threshold)
        self.sonar_local_contrast_blur = int(sonar_local_contrast_blur)
        self.sonar_edge_blur = int(sonar_edge_blur)
        self.sonar_mask_mode = sonar_mask_mode
        self.sonar_mask_threshold = int(sonar_mask_threshold)
        self.sonar_mask_close_kernel = int(sonar_mask_close_kernel)
        self.image_paths = [
            p for p in self.directory.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        # Normalisation is branch-specific and applied inside the training loop.
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx])
        if self.input_mode == "rgb":
            return self.transform(image.convert("RGB"))
        if self.input_mode == "fls_grayscale":
            gray = self._center_square(image.convert("L"))
            gray = TF.resize(
                gray,
                size=(self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            teacher_image = TF.to_tensor(gray.convert("RGB"))
            student_image = TF.to_tensor(gray).mul(2.0).sub(1.0)
            support = self._support_tensor(np.asarray(gray, dtype=np.uint8))
            return teacher_image, student_image, support
        if self.input_mode == "fls_features":
            gray = self._center_square(image.convert("L"))
            gray = TF.resize(
                gray,
                size=(self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            gray_np = np.asarray(gray, dtype=np.uint8)
            teacher_image = TF.to_tensor(gray.convert("RGB"))
            student_features = make_sonar_features(
                gray_np,
                middle_channel=self.sonar_middle_channel,
                wavelet=self.sonar_wavelet,
                occupancy_threshold=self.sonar_occupancy_threshold,
                local_contrast_blur=self.sonar_local_contrast_blur,
                edge_blur=self.sonar_edge_blur,
                third_channel=self.sonar_third_channel,
            )
            student_image = torch.from_numpy(student_features).permute(2, 0, 1).float()
            support = self._support_tensor(gray_np)
            return teacher_image, student_image, support
        if self.input_mode != "sonar_features":
            raise ValueError(f"Unsupported input_mode: {self.input_mode}")

        gray = image.convert("L")
        crop_params = transforms.RandomResizedCrop.get_params(
            gray,
            scale=(0.8, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
        )
        gray = TF.resized_crop(
            gray,
            *crop_params,
            size=(self.image_size, self.image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        if torch.rand(1).item() < 0.5:
            gray = TF.hflip(gray)

        gray_np = np.asarray(gray, dtype=np.uint8)
        teacher_image = TF.to_tensor(gray.convert("RGB"))
        student_features = make_sonar_features(
            gray_np,
            middle_channel=self.sonar_middle_channel,
            wavelet=self.sonar_wavelet,
            occupancy_threshold=self.sonar_occupancy_threshold,
            local_contrast_blur=self.sonar_local_contrast_blur,
            edge_blur=self.sonar_edge_blur,
            third_channel=self.sonar_third_channel,
        )
        student_image = torch.from_numpy(student_features).permute(2, 0, 1).float()
        support = self._support_tensor(gray_np)
        return teacher_image, student_image, support

    def _support_tensor(self, gray: np.ndarray) -> torch.Tensor:
        if self.sonar_mask_mode == "none":
            support = np.ones(gray.shape, dtype=bool)
        elif self.sonar_mask_mode == "support_hull":
            support = make_sonar_support_mask(
                gray,
                threshold=self.sonar_mask_threshold,
                close_kernel=self.sonar_mask_close_kernel,
            )
        else:
            raise ValueError(f"Unsupported sonar_mask_mode: {self.sonar_mask_mode}")
        return torch.from_numpy(support)

    @staticmethod
    def _center_square(image: Image.Image) -> Image.Image:
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        return TF.crop(image, top, left, side, side)


class DistillExportWrapper(nn.Module):
    """TorchScript export wrapper whose forward path is forward_distill()."""
    def __init__(self, student: MobileViT):
        super().__init__()
        self.student = student

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.student.forward_distill(x)


# ---------------------------------------------------------------------------
# 2.  Normalised-sum distillation losses (unchanged from original design)
# ---------------------------------------------------------------------------

def masked_patch_mean(values: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    if valid_mask is None:
        return values.mean()
    if values.shape != valid_mask.shape:
        raise ValueError(
            f"Patch values shape {tuple(values.shape)} does not match mask {tuple(valid_mask.shape)}"
        )
    valid_mask = valid_mask.to(device=values.device, dtype=values.dtype)
    denominator = valid_mask.sum()
    if denominator.item() < 1:
        raise ValueError("Spatial mask removed every patch in the batch")
    return (values * valid_mask).sum() / denominator


def mse_loss_sum_features(
    student: torch.Tensor,
    teacher: torch.Tensor,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Sum squared error over the C=1024 feature dimension,
    then mean over batch × spatial positions (B × G × G).

    Input shapes: [B, G, G, 1024]   (both L2-normalised)
    Returns:      scalar

    Every patch token contributes its total feature reconstruction error —
    no dilution across 1024 dims — while the gradient stays batch-size stable.
    """
    per_patch = ((student - teacher) ** 2).sum(dim=-1)
    return masked_patch_mean(per_patch, valid_mask)


def cosine_loss_sum_features(
    student: torch.Tensor,
    teacher: torch.Tensor,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    (1 − cosine_similarity) over the C=1024 feature dimension,
    then mean over B × G × G.

    Input shapes: [B, G, G, 1024]   (both L2-normalised)
    Returns:      scalar

    On unit-norm vectors, cosine_similarity = dot product, so this is
    numerically identical to F.cosine_embedding_loss with target=+1 but
    without the awkward ones tensor.
    Penalises directional misalignment orthogonally to MSE's magnitude signal.
    """
    cos_sim = F.cosine_similarity(student, teacher, dim=-1)  # [B, G, G]
    return masked_patch_mean(1.0 - cos_sim, valid_mask)


def sampled_patch_sigreg_loss(
    student_grid: torch.Tensor,
    max_tokens: int = 2048,
    projections: int = 256,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Lightweight SIGReg-style uniformity term over sampled patch embeddings.

    forward_distill() returns unit vectors, so the reference is a uniform
    unit-sphere sample rather than an unconstrained Gaussian. The loss is a
    sliced-Wasserstein proxy: project both sets onto random directions, sort
    each projection, and match their 1D distributions.
    """
    if valid_mask is None:
        tokens = student_grid.reshape(-1, student_grid.shape[-1]).float()
    else:
        if student_grid.shape[:-1] != valid_mask.shape:
            raise ValueError(
                f"Student grid shape {tuple(student_grid.shape[:-1])} does not "
                f"match mask {tuple(valid_mask.shape)}"
            )
        tokens = student_grid[valid_mask.to(student_grid.device)].float()
    if tokens.shape[0] < 2:
        return tokens.new_zeros(())

    max_tokens = int(max_tokens)
    if max_tokens > 0 and tokens.shape[0] > max_tokens:
        idx = torch.randperm(tokens.shape[0], device=tokens.device)[:max_tokens]
        tokens = tokens.index_select(0, idx)

    tokens = F.normalize(tokens, p=2, dim=-1)
    feature_dim = tokens.shape[-1]
    projections = max(int(projections), 1)

    reference = torch.randn_like(tokens)
    reference = F.normalize(reference, p=2, dim=-1)

    basis = torch.randn(feature_dim, projections, device=tokens.device, dtype=tokens.dtype)
    basis = F.normalize(basis, p=2, dim=0)

    projected_tokens = torch.sort(tokens @ basis, dim=0).values
    projected_reference = torch.sort(reference @ basis, dim=0).values
    return (projected_tokens - projected_reference).square().mean() * feature_dim


# ---------------------------------------------------------------------------
# 3.  TorchScript checkpoint helper
# ---------------------------------------------------------------------------

def save_torchscript_checkpoint(
    student: MobileViT,
    epoch: int,
    avg_loss: float,
    checkpoint_dir: str = "checkpoints",
    image_size: int = 448,
    preprocessing_contract: str = RGB_PREPROCESSING_CONTRACT,
    student_mean: tuple = STUDENT_MEAN,
    student_std: tuple = STUDENT_STD,
    input_mode: str = "rgb",
    in_channels: int = 3,
    sonar_middle_channel: str = "wavelet_low",
    sonar_third_channel: str = "sobel_edge",
    sonar_wavelet: str = "haar",
    sonar_occupancy_threshold: int = 128,
    sonar_local_contrast_blur: int = 31,
    sonar_edge_blur: int = 3,
    sonar_mask_mode: str = "support_hull",
    sonar_mask_threshold: int = 3,
    sonar_mask_close_kernel: int = 21,
    sonar_mask_min_coverage: float = 0.05,
    avg_valid_patch_fraction: float = 1.0,
    lambda_sigreg: float = 0.0,
    sigreg_max_tokens: int = 2048,
    sigreg_projections: int = 256,
    sigreg_warmup_epochs: int = 0,
    optimizer=None,
    scheduler=None,
    scaler=None,
) -> None:
    """
    Saves two artefacts per checkpoint:

      1. TorchScript  (.pt)  — portable; loads on any device without the
         class definition.  Uses torch.jit.trace (MobileViT's CNN+einops ops
         are fully traceable); falls back to torch.jit.script if trace fails.

      2. State dict   (.pth) — always written; safe fallback for resuming or
         re-initialising.  Includes epoch, avg_loss, and model_state_dict.

    The student is moved to CPU before serialisation for device portability.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    student.eval()
    # Dummy input: CPU, matches training resolution exactly
    image_size = int(image_size)
    grid_size = image_size // 16
    in_channels = int(in_channels)
    dummy_input = torch.randn(1, in_channels, image_size, image_size)

    ts_path = ckpt_dir / f"student_mobilevit_2M_dinov3_epoch{epoch:04d}.pt"
    saved_ts = False

    export_model = DistillExportWrapper(student).eval()

    # MobileViT uses einops.rearrange. Tracing the wrapper preserves the
    # distillation forward path; scripting the raw student would export the
    # classification forward and break live-tracker inference.
    try:
        traced = torch.jit.trace(export_model, dummy_input, strict=False)
        traced.save(str(ts_path))
        print(f"  [Checkpoint] TorchScript (traced)   -> {ts_path}  (loss={avg_loss:.4f})")
        saved_ts = True
    except Exception as trace_err:
        print(
            f"  [Checkpoint] torch.jit.trace failed ({trace_err}). "
            "Skipping .pt — use the .pth state dict to resume."
        )

    # Full checkpoint always written regardless of TorchScript outcome.
    pth_path = ckpt_dir / f"student_mobilevit_2M_Dinov3_based_epoch{epoch:04d}.pth"
    if input_mode == "fls_grayscale":
        model_type = "mobilevit_distill_fls_sonar_greyscale"
        student_normalization = {
            "formula": "float = uint8 / 127.5 - 1.0",
            "mean": [0.0],
            "std": [1.0],
        }
    elif input_mode == "fls_features":
        model_type = "mobilevit_distill_fls_sonar_features"
        student_normalization = {"mean": student_mean, "std": student_std}
    else:
        model_type = "mobilevit_distill"
        student_normalization = {"mean": student_mean, "std": student_std}

    ckpt = {
        "epoch": epoch,
        "avg_loss": avg_loss,
        "model_state_dict": student.state_dict(),
        "torchscript_saved": saved_ts,
        "model_type": model_type,
        "feature_dim": 1024,
        "in_channels": in_channels,
        "train_img_size": image_size,
        "train_grid_size": grid_size,
        "output_stride": 16,
        "inference_note": "forward_distill supports any square input divisible by 16",
        "input_mode": input_mode,
        "preprocessing_contract": preprocessing_contract,
        "spatial_mask_contract": spatial_mask_contract(
            input_mode,
            sonar_mask_mode,
            sonar_mask_threshold,
            sonar_mask_close_kernel,
            sonar_mask_min_coverage,
        ),
        "spatial_mask_config": {
            "mode": sonar_mask_mode if input_mode != "rgb" else "none",
            "threshold": int(sonar_mask_threshold),
            "close_kernel": int(sonar_mask_close_kernel),
            "min_grid_coverage": float(sonar_mask_min_coverage),
            "avg_valid_patch_fraction": float(avg_valid_patch_fraction),
        },
        "teacher_normalization": {"mean": TEACHER_MEAN, "std": TEACHER_STD},
        "student_normalization": student_normalization,
        "sonar_feature_config": {
            "middle_channel": sonar_middle_channel,
            "third_channel": sonar_third_channel,
            "wavelet": sonar_wavelet,
            "occupancy_threshold": int(sonar_occupancy_threshold),
            "local_contrast_blur": int(sonar_local_contrast_blur),
            "edge_blur": int(sonar_edge_blur),
        },
        "auxiliary_loss_config": {
            "lambda_sigreg": float(lambda_sigreg),
            "sigreg_max_tokens": int(sigreg_max_tokens),
            "sigreg_projections": int(sigreg_projections),
            "sigreg_warmup_epochs": int(sigreg_warmup_epochs),
        },
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()
    torch.save(ckpt, str(pth_path))
    print(f"  [Checkpoint] State dict             -> {pth_path}")

    student.train()


def get_latest_checkpoint(checkpoint_dir: str) -> str:
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return ""

    candidates = []
    for path in ckpt_dir.glob("student_mobilevit_2M_Dinov3_based_epoch*.pth"):
        match = re.search(r"epoch(\d+)", path.name)
        if match:
            candidates.append((int(match.group(1)), path))

    if not candidates:
        return ""
    return str(max(candidates, key=lambda item: item[0])[1])


def load_distill_checkpoint(
    ckpt_path: str,
    student: MobileViT,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device="cpu",
    expected_image_size: int = None,
    expected_preprocessing_contract: str = RGB_PREPROCESSING_CONTRACT,
    expected_spatial_mask_contract: str = "all_grid_cells_v1",
) -> tuple:
    """
    Resume a tagged checkpoint created with the current preprocessing contract.

    Returns:
        start_epoch, avg_loss
    """
    if not ckpt_path:
        return 1, 0.0

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        saved_contract = ckpt.get("preprocessing_contract")
        if saved_contract != expected_preprocessing_contract:
            raise ValueError(
                "[Resume] Checkpoint preprocessing contract is missing or incompatible. "
                "Use --init-from to reuse its student weights with a fresh optimiser "
                "and scheduler."
            )
        saved_mask_contract = ckpt.get("spatial_mask_contract")
        legacy_unmasked = (
            expected_spatial_mask_contract == "all_grid_cells_v1"
            and saved_mask_contract in {None, "all_grid_cells_v1"}
        )
        if not legacy_unmasked and saved_mask_contract != expected_spatial_mask_contract:
            raise ValueError(
                "[Resume] Checkpoint spatial-mask objective is missing or incompatible. "
                f"Saved={saved_mask_contract!r}, expected={expected_spatial_mask_contract!r}. "
                "Use --init-from to warm-start the student with a fresh optimiser and scheduler."
            )
        student.load_state_dict(ckpt["model_state_dict"], strict=True)
        saved_img_size = ckpt.get("train_img_size")
        if expected_image_size is not None and saved_img_size is not None:
            if int(saved_img_size) != int(expected_image_size):
                old_grid = ckpt.get("train_grid_size", "?")
                new_grid = int(expected_image_size) // 16
                print(
                    f"[Resume] Checkpoint was trained at image_size={saved_img_size} "
                    f"(grid={old_grid}); continuing at image_size={expected_image_size} "
                    f"(grid={new_grid}). Model weights are compatible because the "
                    "MobileViT distill backbone is fully spatial."
                )
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        epoch = int(ckpt.get("epoch", 0))
        avg_loss = float(ckpt.get("avg_loss", 0.0))
        print(f"[Resume] Loaded full checkpoint: {ckpt_path}")
        return epoch + 1, avg_loss

    raise ValueError(
        "[Resume] Raw model-only checkpoints cannot be used for exact resume. "
        "Use --init-from to warm-start a fresh experiment."
    )


def load_initial_student_weights(
    ckpt_path: str,
    student: MobileViT,
    device="cpu",
) -> None:
    """Warm-start student weights without restoring training state."""
    if not ckpt_path:
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        source_epoch = ckpt.get("epoch", "?")
        source_contract = ckpt.get("preprocessing_contract", "legacy-or-unknown")
        source_mask_contract = ckpt.get("spatial_mask_contract", "legacy-unmasked")
    else:
        state_dict = ckpt
        source_epoch = "?"
        source_contract = "raw-model-state"
        source_mask_contract = "unknown"

    student.load_state_dict(state_dict, strict=True)
    print(f"[Init] Warm-started student weights: {ckpt_path}")
    print(f"       Source epoch: {source_epoch}")
    print(f"       Source preprocessing contract: {source_contract}")
    print(f"       Source spatial-mask contract: {source_mask_contract}")
    print("       Fresh optimiser, scheduler, scaler, and epoch counter will be used.")


# ---------------------------------------------------------------------------
# 4.  Main distillation loop
# ---------------------------------------------------------------------------

def live_distillation(
    image_dir: str,
    weights_path: str,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    checkpoint_every: int = 5,
    checkpoint_dir: str = "checkpoints",
    resume_from: str = "",
    init_from: str = "",
    dinov3_repo: str = "/workspace/dinov3",
    image_size: int = 448,
    input_mode: str = "rgb",
    student_mean: tuple = None,
    student_std: tuple = None,
    sonar_middle_channel: str = "wavelet_low",
    sonar_third_channel: str = "sobel_edge",
    sonar_wavelet: str = "haar",
    sonar_occupancy_threshold: int = 128,
    sonar_local_contrast_blur: int = 31,
    sonar_edge_blur: int = 3,
    sonar_mask_mode: str = "support_hull",
    sonar_mask_threshold: int = 3,
    sonar_mask_close_kernel: int = 21,
    sonar_mask_min_coverage: float = 0.05,
    lambda_sigreg: float = 0.0,
    sigreg_max_tokens: int = 2048,
    sigreg_projections: int = 256,
    sigreg_warmup_epochs: int = 0,
    config_path: str = "",
    run_config: dict = None,
) -> None:

    image_size = int(image_size)
    epochs = int(epochs)
    batch_size = int(batch_size)
    checkpoint_every = int(checkpoint_every)
    lambda_sigreg = float(lambda_sigreg)
    sigreg_max_tokens = int(sigreg_max_tokens)
    sigreg_projections = int(sigreg_projections)
    sigreg_warmup_epochs = int(sigreg_warmup_epochs)
    sonar_mask_threshold = int(sonar_mask_threshold)
    sonar_mask_close_kernel = int(sonar_mask_close_kernel)
    sonar_mask_min_coverage = float(sonar_mask_min_coverage)
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if checkpoint_every < 1:
        raise ValueError(f"checkpoint_every must be >= 1, got {checkpoint_every}")
    if lambda_sigreg < 0:
        raise ValueError(f"lambda_sigreg must be >= 0, got {lambda_sigreg}")
    if sigreg_max_tokens < 2:
        raise ValueError(f"sigreg_max_tokens must be >= 2, got {sigreg_max_tokens}")
    if sigreg_projections < 1:
        raise ValueError(f"sigreg_projections must be >= 1, got {sigreg_projections}")
    if sigreg_warmup_epochs < 0:
        raise ValueError(f"sigreg_warmup_epochs must be >= 0, got {sigreg_warmup_epochs}")
    if sonar_mask_mode not in {"none", "support_hull"}:
        raise ValueError("sonar_mask_mode must be 'none' or 'support_hull'")
    if sonar_mask_threshold < 0 or sonar_mask_threshold > 255:
        raise ValueError("sonar_mask_threshold must be in [0, 255]")
    if sonar_mask_close_kernel < 1:
        raise ValueError("sonar_mask_close_kernel must be >= 1")
    if not 0.0 < sonar_mask_min_coverage <= 1.0:
        raise ValueError("sonar_mask_min_coverage must be in (0, 1]")
    if image_size % 16 != 0:
        raise ValueError(f"image_size must be divisible by 16 for DINOv3 ViT-L/16, got {image_size}")
    if not Path(image_dir).is_dir():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")
    dinov3_repo = str(Path(dinov3_repo).expanduser().resolve())
    if not Path(dinov3_repo).is_dir():
        raise FileNotFoundError(f"DINOv3 repo not found: {dinov3_repo}")
    expected_grid = image_size // 16
    if input_mode not in {"rgb", "sonar_features", "fls_grayscale", "fls_features"}:
        raise ValueError("--input-mode must be 'rgb', 'sonar_features', 'fls_grayscale', or 'fls_features'")
    using_default_student_stats = student_mean is None or student_std is None
    if input_mode == "fls_grayscale":
        student_mean = (0.0,)
        student_std = (1.0,)
    if student_mean is None:
        student_mean = SONAR_FEATURE_MEAN if input_mode in {"sonar_features", "fls_features"} else STUDENT_MEAN
    if student_std is None:
        student_std = SONAR_FEATURE_STD if input_mode in {"sonar_features", "fls_features"} else STUDENT_STD
    run_contract = preprocessing_contract(
        input_mode=input_mode,
        student_mean=student_mean,
        student_std=student_std,
        sonar_middle_channel=sonar_middle_channel,
        sonar_third_channel=sonar_third_channel,
        sonar_wavelet=sonar_wavelet,
        sonar_occupancy_threshold=sonar_occupancy_threshold,
        sonar_local_contrast_blur=sonar_local_contrast_blur,
        sonar_edge_blur=sonar_edge_blur,
    )
    run_mask_contract = spatial_mask_contract(
        input_mode,
        sonar_mask_mode,
        sonar_mask_threshold,
        sonar_mask_close_kernel,
        sonar_mask_min_coverage,
    )
    resolved_run_config = dict(run_config or {})
    resolved_run_config.update({
        "image_dir": image_dir,
        "weights": weights_path,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": float(lr),
        "checkpoint_every": checkpoint_every,
        "checkpoint_dir": checkpoint_dir,
        "resume_from": resume_from,
        "init_from": init_from,
        "dinov3_repo": dinov3_repo,
        "image_size": image_size,
        "input_mode": input_mode,
        "student_mean": student_mean,
        "student_std": student_std,
        "sonar_middle_channel": sonar_middle_channel,
        "sonar_third_channel": sonar_third_channel,
        "sonar_wavelet": sonar_wavelet,
        "sonar_occupancy_threshold": int(sonar_occupancy_threshold),
        "sonar_local_contrast_blur": int(sonar_local_contrast_blur),
        "sonar_edge_blur": int(sonar_edge_blur),
        "sonar_mask_mode": sonar_mask_mode,
        "sonar_mask_threshold": sonar_mask_threshold,
        "sonar_mask_close_kernel": sonar_mask_close_kernel,
        "sonar_mask_min_coverage": sonar_mask_min_coverage,
        "lambda_sigreg": lambda_sigreg,
        "sigreg_max_tokens": sigreg_max_tokens,
        "sigreg_projections": sigreg_projections,
        "sigreg_warmup_epochs": sigreg_warmup_epochs,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    print(f"Initialising Live Distillation on: {device}")
    print(f"Image size: {image_size}x{image_size}  ->  target grid: {expected_grid}x{expected_grid}")
    print(f"Input mode: {input_mode}")
    if input_mode in {"sonar_features", "fls_features"}:
        print(
            f"  Student channels: raw_robust, {sonar_middle_channel}, {sonar_third_channel} "
            f"(wavelet={sonar_wavelet}, occ={sonar_occupancy_threshold}, "
            f"local_blur={sonar_local_contrast_blur}, edge_blur={sonar_edge_blur})"
        )
        if input_mode == "fls_features":
            print("  FLS geometry: center-crop square, resize, no random crop")
        if using_default_student_stats:
            print(
                "  [WARN] Using built-in sonar feature stats. Recompute and pass "
                "--student-mean/--student-std when changing feature channels or datasets."
            )
    elif input_mode == "fls_grayscale":
        print("  Student channels: grayscale")
        print("  Student norm: uint8/127.5 - 1.0")
    else:
        print(f"  Student norm mean={student_mean} std={student_std}")
    if lambda_sigreg > 0.0:
        print(
            f"  Aux SIGReg: ENABLED  "
            f"(lambda={lambda_sigreg:g}, max_tokens={sigreg_max_tokens}, "
            f"projections={sigreg_projections}, warmup_epochs={sigreg_warmup_epochs})"
        )
    if input_mode != "rgb":
        print(
            f"  Spatial mask: mode={sonar_mask_mode} threshold={sonar_mask_threshold} "
            f"close={sonar_mask_close_kernel} min_grid_coverage={sonar_mask_min_coverage:g}"
        )

    # ------------------------------------------------------------------ #
    # A.  Frozen Teacher — DINOv3 ViT-Large/16                            #
    #                                                                      #
    # Output used: features["x_norm_patchtokens"]                         #
    #   Shape: [B, G*G, 1024]  →  reshaped to [B, G, G, 1024]             #
    #   Derivation: image_size / patch_size 16 = G                         #
    #   Then L2-normalised over dim=-1 to form the distillation target.   #
    # ------------------------------------------------------------------ #
    print("Loading Teacher Model (Frozen) ...")
    teacher = torch.hub.load(
        dinov3_repo, "dinov3_vitl16", source="local", pretrained=False
    )
    teacher.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True), strict=True
    )
    teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False
    total_teacher = sum(p.numel() for p in teacher.parameters()) / 1e6
    print(f"  Teacher parameters: {total_teacher:.1f} M  (all frozen)")

    # ------------------------------------------------------------------ #
    # B.  Trainable Student — MobileViT-S (distillation variant)          #
    #                                                                      #
    # Architectural changes over original MobileViT-S:                    #
    #   1. mv2[6] stride = 1 (was 2): keeps stride product at 16.          #
    #      Derivation: image_size/16 = G, matching the teacher grid.       #
    #   2. conv2 output = 1024 (was 640): matches teacher feature dim.     #
    #      This is the existing 1×1 conv inside MobileViT — not a head.   #
    #   3. pool and fc removed from distillation forward path.             #
    #                                                                      #
    # forward_distill(x) output: [B, G, G, 1024]  L2-normalised           #
    # ------------------------------------------------------------------ #
    print("Loading Student Model (MobileViT-S, distillation variant) ...")
    student_in_channels = 1 if input_mode == "fls_grayscale" else 3
    student = mobilevit_s_distill(
        image_size=(image_size, image_size),
        in_channels=student_in_channels,
    ).to(device)
    print(f"  Student parameters: {count_parameters(student) / 1e6:.2f} M  (all trainable)")

    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    # ------------------------------------------------------------------ #
    # C.  Data and Mixed Precision Scaler                                  #
    # ------------------------------------------------------------------ #
    dataset    = LiveAugmentationDataset(
        image_dir,
        image_size=image_size,
        input_mode=input_mode,
        sonar_middle_channel=sonar_middle_channel,
        sonar_third_channel=sonar_third_channel,
        sonar_wavelet=sonar_wavelet,
        sonar_occupancy_threshold=sonar_occupancy_threshold,
        sonar_local_contrast_blur=sonar_local_contrast_blur,
        sonar_edge_blur=sonar_edge_blur,
        sonar_mask_mode=sonar_mask_mode,
        sonar_mask_threshold=sonar_mask_threshold,
        sonar_mask_close_kernel=sonar_mask_close_kernel,
    )
    if len(dataset) == 0:
        raise FileNotFoundError(f"No images found under {image_dir}")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    if resume_from and init_from:
        raise ValueError("Use only one of --resume-from or --init-from, not both.")

    if resume_from == "auto":
        resume_from = get_latest_checkpoint(checkpoint_dir)
        if resume_from:
            print(f"[Resume] Auto-detected checkpoint: {resume_from}")
        else:
            print("[Resume] No checkpoint found in checkpoint_dir; starting fresh.")

    if init_from:
        load_initial_student_weights(init_from, student, device=device)
        start_epoch, avg_loss = 1, 0.0
    else:
        start_epoch, avg_loss = load_distill_checkpoint(
            resume_from,
            student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            expected_image_size=image_size,
            expected_preprocessing_contract=run_contract,
            expected_spatial_mask_contract=run_mask_contract,
        )
    student.to(device)
    resolved_run_config["resume_from"] = resume_from

    write_run_config_files(
        checkpoint_dir,
        resolved_run_config,
        config_path=config_path,
        runtime_info={
            "device": str(device),
            "amp_enabled": amp_enabled,
            "dataset_images": len(dataset),
            "dataloader_batches": len(dataloader),
            "expected_grid": expected_grid,
            "preprocessing_contract": run_contract,
            "spatial_mask_contract": run_mask_contract,
            "start_epoch": start_epoch,
            "teacher": "dinov3_vitl16",
            "student": "mobilevit_s_distill",
        },
    )

    if start_epoch > epochs:
        print(f"[Resume] Checkpoint epoch is already >= target epochs ({epochs}). Nothing to train.")
        return

    print(f"\nStarting Training: {len(dataset)} images, epochs {start_epoch}..{epochs}.\n")

    # ------------------------------------------------------------------ #
    # D.  Training Loop                                                    #
    # ------------------------------------------------------------------ #
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        epoch_lr = optimizer.param_groups[0]["lr"]
        student.train()
        running_loss = 0.0
        running_mse = 0.0
        running_cosine = 0.0
        running_sigreg = 0.0
        running_valid_patch_fraction = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            if input_mode in {"sonar_features", "fls_grayscale", "fls_features"}:
                teacher_input, student_input, pixel_valid_mask = batch
                teacher_input = teacher_input.to(device)
                student_input = student_input.to(device)
                pixel_valid_mask = pixel_valid_mask.to(device)
            else:
                teacher_input = batch.to(device)
                student_input = teacher_input
                pixel_valid_mask = None

            teacher_images = TF.normalize(
                teacher_input,
                mean=TEACHER_MEAN,
                std=TEACHER_STD,
            )

            if input_mode == "fls_grayscale":
                student_images = student_input
            else:
                student_images = TF.normalize(
                    student_input,
                    mean=student_mean,
                    std=student_std,
                )

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type,
                                    dtype=torch.bfloat16,
                                    enabled=amp_enabled):

                # --- Teacher target (frozen, no gradients) ---
                #
                # DINOv3 ViT-L/16:
                #   x_norm_patchtokens: [B, G*G, 1024]
                #   reshaped to:        [B, G, G, 1024]
                #   L2-normalised over dim=-1  (each token is a unit vector)
                with torch.no_grad():
                    features = teacher.forward_features(teacher_images)
                    patch_tokens = features["x_norm_patchtokens"]          # [B, G*G, 1024]
                    grid_size = int(math.sqrt(patch_tokens.shape[1]))
                    if grid_size * grid_size != patch_tokens.shape[1]:
                        raise ValueError(
                            f"Teacher patch token count is not square: {patch_tokens.shape[1]}"
                        )
                    if grid_size != expected_grid:
                        raise ValueError(
                            f"Teacher grid is {grid_size}x{grid_size}, expected "
                            f"{expected_grid}x{expected_grid} for image_size={image_size}."
                        )
                    teacher_grid = patch_tokens.reshape(
                        teacher_input.shape[0], grid_size, grid_size, -1
                    )                                                      # [B, H, W, 1024]
                    teacher_grid = F.normalize(teacher_grid, p=2, dim=-1)

                # Convert the image-space sonar support mask into the exact
                # teacher/student patch grid. A low coverage threshold keeps
                # boundary patches while rejecting external black padding.
                if pixel_valid_mask is None:
                    grid_valid_mask = None
                    valid_patch_fraction = teacher_grid.new_ones(())
                else:
                    patch_coverage = F.interpolate(
                        pixel_valid_mask[:, None].float(),
                        size=teacher_grid.shape[1:3],
                        mode="area",
                    )[:, 0]
                    grid_valid_mask = patch_coverage >= sonar_mask_min_coverage
                    valid_per_sample = grid_valid_mask.flatten(1).sum(dim=1)
                    if torch.any(valid_per_sample == 0):
                        bad_indices = torch.nonzero(
                            valid_per_sample == 0, as_tuple=False
                        ).flatten().tolist()
                        raise ValueError(
                            "Sonar support mask removed every patch for batch "
                            f"sample(s) {bad_indices}. Lower sonar_mask_min_coverage "
                            "or inspect the source images."
                        )
                    valid_patch_fraction = grid_valid_mask.float().mean()

                # --- Student prediction (with gradients) ---
                #
                # MobileViT-S distillation variant:
                #   stride product = 16  →  image_size/16 = G spatial grid
                #   conv2 output dim = 1024  →  matches teacher feature dim
                #   forward_distill: [B, 1024, G, G] → permute → L2-norm
                #   output: [B, G, G, 1024]  (unit vectors, same as teacher)
                student_grid = student.forward_distill(student_images) # [B, G, G, 1024]
                if student_grid.shape[1:3] != teacher_grid.shape[1:3]:
                    raise ValueError(
                        f"Student grid {tuple(student_grid.shape[1:3])} does not "
                        f"match teacher grid {tuple(teacher_grid.shape[1:3])}."
                    )

                # --- Normalised-sum losses ---
                #
                # MSE:    sum_C (s−t)²  then mean over B×G×G
                #         → total reconstruction error per token, batch-stable
                #
                # Cosine: (1 − cos_sim_C(s, t))  then mean over B×G×G
                #         → directional alignment, complements MSE
                #
                # Both losses are bounded and batch-size-independent.
                mse_loss = mse_loss_sum_features(
                    student_grid, teacher_grid, grid_valid_mask
                )
                cosine_loss = cosine_loss_sum_features(
                    student_grid, teacher_grid, grid_valid_mask
                )
                distill_loss = mse_loss + cosine_loss

                sigreg_loss = student_grid.new_zeros(())
                effective_lambda_sigreg = 0.0
                if lambda_sigreg > 0.0 and epoch > sigreg_warmup_epochs:
                    with torch.amp.autocast(device_type=device.type, enabled=False):
                        sigreg_loss = sampled_patch_sigreg_loss(
                            student_grid,
                            max_tokens=sigreg_max_tokens,
                            projections=sigreg_projections,
                            valid_mask=grid_valid_mask,
                        )
                    effective_lambda_sigreg = lambda_sigreg

                total_loss = distill_loss + effective_lambda_sigreg * sigreg_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += total_loss.item()
            running_mse += mse_loss.item()
            running_cosine += cosine_loss.item()
            running_sigreg += sigreg_loss.item()
            running_valid_patch_fraction += valid_patch_fraction.item()
            postfix = {
                "loss": f"{total_loss.item():.4f}",
                "mse":  f"{mse_loss.item():.4f}",
                "cos":  f"{cosine_loss.item():.4f}",
            }
            if pixel_valid_mask is not None:
                postfix["valid"] = f"{valid_patch_fraction.item() * 100:.1f}%"
            if lambda_sigreg > 0.0:
                postfix["sig"] = f"{sigreg_loss.item():.4f}"
                postfix["sigλ"] = f"{effective_lambda_sigreg:g}"
            pbar.set_postfix(postfix)

        scheduler.step()
        avg_loss = running_loss / len(dataloader)
        avg_mse = running_mse / len(dataloader)
        avg_cosine = running_cosine / len(dataloader)
        avg_sigreg = running_sigreg / len(dataloader)
        avg_valid_patch_fraction = running_valid_patch_fraction / len(dataloader)
        epoch_seconds = time.time() - epoch_start_time
        print(f"Epoch {epoch} Complete — Avg Loss: {avg_loss:.4f}")
        if input_mode != "rgb":
            print(f"  Valid sonar patches: {avg_valid_patch_fraction * 100:.2f}%")
        if lambda_sigreg > 0.0:
            print(
                "  Components — "
                f"mse={avg_mse:.4f}  "
                f"cos={avg_cosine:.4f}  "
                f"sigreg={avg_sigreg:.4f}"
            )
        append_training_log(
            checkpoint_dir,
            {
                "epoch": epoch,
                "avg_loss": round(avg_loss, 8),
                "avg_mse": round(avg_mse, 8),
                "avg_cosine": round(avg_cosine, 8),
                "avg_sigreg": round(avg_sigreg, 8),
                "avg_valid_patch_fraction": round(avg_valid_patch_fraction, 8),
                "lambda_sigreg": lambda_sigreg,
                "lr": epoch_lr,
                "epoch_seconds": round(epoch_seconds, 3),
            },
        )

        if epoch % checkpoint_every == 0:
            print(f"  Saving checkpoint at epoch {epoch} ...")
            student.cpu()
            save_torchscript_checkpoint(
                student,
                epoch,
                avg_loss,
                checkpoint_dir,
                image_size=image_size,
                preprocessing_contract=run_contract,
                student_mean=student_mean,
                student_std=student_std,
                input_mode=input_mode,
                in_channels=student_in_channels,
                sonar_middle_channel=sonar_middle_channel,
                sonar_third_channel=sonar_third_channel,
                sonar_wavelet=sonar_wavelet,
                sonar_occupancy_threshold=sonar_occupancy_threshold,
                sonar_local_contrast_blur=sonar_local_contrast_blur,
                sonar_edge_blur=sonar_edge_blur,
                sonar_mask_mode=sonar_mask_mode,
                sonar_mask_threshold=sonar_mask_threshold,
                sonar_mask_close_kernel=sonar_mask_close_kernel,
                sonar_mask_min_coverage=sonar_mask_min_coverage,
                avg_valid_patch_fraction=avg_valid_patch_fraction,
                lambda_sigreg=lambda_sigreg,
                sigreg_max_tokens=sigreg_max_tokens,
                sigreg_projections=sigreg_projections,
                sigreg_warmup_epochs=sigreg_warmup_epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            student.to(device)

    # Final save (unconditional)
    print("\nSaving final model ...")
    student.cpu()
    save_torchscript_checkpoint(
        student,
        epochs,
        avg_loss,
        checkpoint_dir,
        image_size=image_size,
        preprocessing_contract=run_contract,
        student_mean=student_mean,
        student_std=student_std,
        input_mode=input_mode,
        in_channels=student_in_channels,
        sonar_middle_channel=sonar_middle_channel,
        sonar_third_channel=sonar_third_channel,
        sonar_wavelet=sonar_wavelet,
        sonar_occupancy_threshold=sonar_occupancy_threshold,
        sonar_local_contrast_blur=sonar_local_contrast_blur,
        sonar_edge_blur=sonar_edge_blur,
        sonar_mask_mode=sonar_mask_mode,
        sonar_mask_threshold=sonar_mask_threshold,
        sonar_mask_close_kernel=sonar_mask_close_kernel,
        sonar_mask_min_coverage=sonar_mask_min_coverage,
        avg_valid_patch_fraction=avg_valid_patch_fraction,
        lambda_sigreg=lambda_sigreg,
        sigreg_max_tokens=sigreg_max_tokens,
        sigreg_projections=sigreg_projections,
        sigreg_warmup_epochs=sigreg_warmup_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    final_path = Path(checkpoint_dir) / "distilled_student_mobilevit_1024d_2M_dinov3.pth"
    torch.save(student.state_dict(), str(final_path))
    print(f"Final raw state dict -> {final_path}")
    print("Training complete!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="",
                        help="YAML file containing all live distillation training parameters.")
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume-from", default=None,
                        help="'auto', a checkpoint path, or empty string for fresh training.")
    parser.add_argument("--init-from", default=None,
                        help="Warm-start student weights only; resets optimiser, scheduler, scaler, and epoch.")
    parser.add_argument("--dinov3-repo", default=None,
                        help="Local facebookresearch/dinov3 repository path used by torch.hub.load.")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Square crop size. Must be divisible by 16. Use 512 for 32x32 features.")
    parser.add_argument("--input-mode", choices=["rgb", "sonar_features", "fls_grayscale", "fls_features"], default=None,
                        help="rgb keeps the camera pipeline; sonar_features uses engineered bounded-mosaic channels; fls_grayscale uses one-channel FLS sonar; fls_features uses engineered FLS channels.")
    parser.add_argument("--student-mean", default=None,
                        help="Comma-separated student channel mean. Defaults to BeeX RGB or sonar-feature stats by input mode.")
    parser.add_argument("--student-std", default=None,
                        help="Comma-separated student channel std. Defaults to BeeX RGB or sonar-feature stats by input mode.")
    parser.add_argument("--sonar-middle-channel",
                        choices=["clahe", "occupancy", "inverse_occupancy", "wavelet_low", "wavelet_high", "local_contrast"],
                        default=None)
    parser.add_argument("--sonar-third-channel",
                        choices=["sobel_edge", "local_contrast", "raw_robust"],
                        default=None)
    parser.add_argument("--sonar-wavelet", default=None)
    parser.add_argument("--sonar-occupancy-threshold", type=int, default=None)
    parser.add_argument("--sonar-local-contrast-blur", type=int, default=None)
    parser.add_argument("--sonar-edge-blur", type=int, default=None)
    parser.add_argument("--sonar-mask-mode", choices=["none", "support_hull"], default=None,
                        help="Exclude external sonar padding from patch losses; use none for legacy behavior.")
    parser.add_argument("--sonar-mask-threshold", type=int, default=None,
                        help="Raw grayscale threshold used to seed the sonar support hull.")
    parser.add_argument("--sonar-mask-close-kernel", type=int, default=None,
                        help="Morphology kernel used before filling the sonar support hull.")
    parser.add_argument("--sonar-mask-min-coverage", type=float, default=None,
                        help="Minimum valid image-area fraction required to retain a grid patch.")
    parser.add_argument("--lambda-sigreg", type=float, default=None,
                        help="Weight for sampled patch SIGReg/uniformity auxiliary loss. 0 disables it.")
    parser.add_argument("--sigreg-max-tokens", type=int, default=None,
                        help="Maximum sampled student patch tokens per batch for SIGReg.")
    parser.add_argument("--sigreg-projections", type=int, default=None,
                        help="Number of random sliced projections used by SIGReg.")
    parser.add_argument("--sigreg-warmup-epochs", type=int, default=None,
                        help="Initial epochs trained with distillation only before enabling SIGReg.")
    args = parser.parse_args()

    yaml_config = load_yaml_run_config(args.config)
    metadata_keys = {"run_name", "description", "notes"}
    unknown_keys = sorted(
        key for key in yaml_config
        if key not in DEFAULT_RUN_CONFIG and key not in metadata_keys
    )
    if unknown_keys:
        raise ValueError(f"Unknown config key(s): {unknown_keys}")

    run_metadata = {
        key: yaml_config.pop(key)
        for key in list(yaml_config.keys())
        if key in metadata_keys
    }
    run_config = dict(DEFAULT_RUN_CONFIG)
    run_config.update(yaml_config)

    for key in DEFAULT_RUN_CONFIG:
        value = getattr(args, key, None)
        if value is not None:
            run_config[key] = value
    if run_metadata:
        run_config["_metadata"] = run_metadata

    student_mean = parse_triplet(run_config.get("student_mean"), "--student-mean")
    student_std = parse_triplet(run_config.get("student_std"), "--student-std")
    run_config["student_mean"] = student_mean
    run_config["student_std"] = student_std

    live_distillation(
        run_config["image_dir"],
        run_config["weights"],
        epochs=run_config["epochs"],
        batch_size=run_config["batch_size"],
        lr=run_config["lr"],
        checkpoint_every=run_config["checkpoint_every"],
        checkpoint_dir=run_config["checkpoint_dir"],
        resume_from=run_config["resume_from"],
        init_from=run_config["init_from"],
        dinov3_repo=run_config["dinov3_repo"],
        image_size=run_config["image_size"],
        input_mode=run_config["input_mode"],
        student_mean=student_mean,
        student_std=student_std,
        sonar_middle_channel=run_config["sonar_middle_channel"],
        sonar_third_channel=run_config["sonar_third_channel"],
        sonar_wavelet=run_config["sonar_wavelet"],
        sonar_occupancy_threshold=run_config["sonar_occupancy_threshold"],
        sonar_local_contrast_blur=run_config["sonar_local_contrast_blur"],
        sonar_edge_blur=run_config["sonar_edge_blur"],
        sonar_mask_mode=run_config["sonar_mask_mode"],
        sonar_mask_threshold=run_config["sonar_mask_threshold"],
        sonar_mask_close_kernel=run_config["sonar_mask_close_kernel"],
        sonar_mask_min_coverage=run_config["sonar_mask_min_coverage"],
        lambda_sigreg=run_config["lambda_sigreg"],
        sigreg_max_tokens=run_config["sigreg_max_tokens"],
        sigreg_projections=run_config["sigreg_projections"],
        sigreg_warmup_epochs=run_config["sigreg_warmup_epochs"],
        config_path=args.config,
        run_config=run_config,
    )
