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
import math
import numpy as np
import os
import re
import sys

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

def parse_triplet(value: str, name: str) -> tuple:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"{name} must contain three comma-separated numbers, got: {value}")
    return tuple(parts)


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
            return teacher_image, student_image
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
            return teacher_image, student_image
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
        return teacher_image, student_image

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

def mse_loss_sum_features(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """
    Sum squared error over the C=1024 feature dimension,
    then mean over batch × spatial positions (B × G × G).

    Input shapes: [B, G, G, 1024]   (both L2-normalised)
    Returns:      scalar

    Every patch token contributes its total feature reconstruction error —
    no dilution across 1024 dims — while the gradient stays batch-size stable.
    """
    return ((student - teacher) ** 2).sum(dim=-1).mean()


def cosine_loss_sum_features(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
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
    return (1.0 - cos_sim).mean()


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
    else:
        state_dict = ckpt
        source_epoch = "?"
        source_contract = "raw-model-state"

    student.load_state_dict(state_dict, strict=True)
    print(f"[Init] Warm-started student weights: {ckpt_path}")
    print(f"       Source epoch: {source_epoch}")
    print(f"       Source preprocessing contract: {source_contract}")
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
) -> None:

    image_size = int(image_size)
    epochs = int(epochs)
    batch_size = int(batch_size)
    checkpoint_every = int(checkpoint_every)
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if checkpoint_every < 1:
        raise ValueError(f"checkpoint_every must be >= 1, got {checkpoint_every}")
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
        )
    student.to(device)

    if start_epoch > epochs:
        print(f"[Resume] Checkpoint epoch is already >= target epochs ({epochs}). Nothing to train.")
        return

    print(f"\nStarting Training: {len(dataset)} images, epochs {start_epoch}..{epochs}.\n")

    # ------------------------------------------------------------------ #
    # D.  Training Loop                                                    #
    # ------------------------------------------------------------------ #
    for epoch in range(start_epoch, epochs + 1):
        student.train()
        running_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            if input_mode in {"sonar_features", "fls_grayscale", "fls_features"}:
                teacher_input, student_input = batch
                teacher_input = teacher_input.to(device)
                student_input = student_input.to(device)
            else:
                teacher_input = batch.to(device)
                student_input = teacher_input

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
                mse_loss    = mse_loss_sum_features(student_grid, teacher_grid)
                cosine_loss = cosine_loss_sum_features(student_grid, teacher_grid)
                total_loss  = mse_loss + cosine_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += total_loss.item()
            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "mse":  f"{mse_loss.item():.4f}",
                "cos":  f"{cosine_loss.item():.4f}",
            })

        scheduler.step()
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch} Complete — Avg Loss: {avg_loss:.4f}")

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
    parser.add_argument("--image-dir", default="/workspace/Datasets_VIT")
    parser.add_argument("--weights", default="/workspace/Pretrained_Dino_based_MVIT_Distillation/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume-from", default="",
                        help="'auto', a checkpoint path, or empty string for fresh training.")
    parser.add_argument("--init-from", default="",
                        help="Warm-start student weights only; resets optimiser, scheduler, scaler, and epoch.")
    parser.add_argument("--dinov3-repo", default="/workspace/dinov3",
                        help="Local facebookresearch/dinov3 repository path used by torch.hub.load.")
    parser.add_argument("--image-size", type=int, default=448,
                        help="Square crop size. Must be divisible by 16. Use 512 for 32x32 features.")
    parser.add_argument("--input-mode", choices=["rgb", "sonar_features", "fls_grayscale", "fls_features"], default="rgb",
                        help="rgb keeps the camera pipeline; sonar_features uses engineered bounded-mosaic channels; fls_grayscale uses one-channel FLS sonar; fls_features uses engineered FLS channels.")
    parser.add_argument("--student-mean", default="",
                        help="Comma-separated student channel mean. Defaults to BeeX RGB or sonar-feature stats by input mode.")
    parser.add_argument("--student-std", default="",
                        help="Comma-separated student channel std. Defaults to BeeX RGB or sonar-feature stats by input mode.")
    parser.add_argument("--sonar-middle-channel",
                        choices=["clahe", "occupancy", "inverse_occupancy", "wavelet_low", "wavelet_high", "local_contrast"],
                        default="wavelet_low")
    parser.add_argument("--sonar-third-channel",
                        choices=["sobel_edge", "local_contrast", "raw_robust"],
                        default="sobel_edge")
    parser.add_argument("--sonar-wavelet", default="haar")
    parser.add_argument("--sonar-occupancy-threshold", type=int, default=128)
    parser.add_argument("--sonar-local-contrast-blur", type=int, default=31)
    parser.add_argument("--sonar-edge-blur", type=int, default=3)
    args = parser.parse_args()
    student_mean = parse_triplet(args.student_mean, "--student-mean") if args.student_mean else None
    student_std = parse_triplet(args.student_std, "--student-std") if args.student_std else None

    live_distillation(
        args.image_dir,
        args.weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        init_from=args.init_from,
        dinov3_repo=args.dinov3_repo,
        image_size=args.image_size,
        input_mode=args.input_mode,
        student_mean=student_mean,
        student_std=student_std,
        sonar_middle_channel=args.sonar_middle_channel,
        sonar_third_channel=args.sonar_third_channel,
        sonar_wavelet=args.sonar_wavelet,
        sonar_occupancy_threshold=args.sonar_occupancy_threshold,
        sonar_local_contrast_blur=args.sonar_local_contrast_blur,
        sonar_edge_blur=args.sonar_edge_blur,
    )