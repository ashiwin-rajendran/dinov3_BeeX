#!/usr/bin/env python3
"""Visualise DINOv3 -> MobileViT distillation across saved checkpoints.

The probe image and geometric preprocessing remain fixed for every checkpoint.
This makes changes between epochs attributable to student weights, not random
augmentation. The training script is imported but never modified or executed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

try:
    import cv2
except ImportError:
    cv2 = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from live_distillation_mobilevit import (
    STUDENT_MEAN,
    STUDENT_STD,
    TEACHER_MEAN,
    TEACHER_STD,
    make_sonar_features,
    mobilevit_s_distill,
)


CHECKPOINT_PATTERN = "student_mobilevit_2M_Dinov3_based_epoch*.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise DINOv3 teacher-to-MobileViT distillation over epochs."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--teacher-weights", required=True)
    parser.add_argument("--dinov3-repo", required=True)
    parser.add_argument("--probe-image", required=True)
    parser.add_argument("--output-dir", default="", help="Default: CHECKPOINT_DIR/distillation_visuals")
    parser.add_argument("--device", default="", help="Default: cuda when available, otherwise cpu")
    parser.add_argument("--max-checkpoints", type=int, default=0, help="0 processes every checkpoint")
    parser.add_argument("--checkpoint", default="", help="Optional single .pth checkpoint")
    parser.add_argument("--query-x", type=float, default=0.5, help="Query x as fraction of image width")
    parser.add_argument("--query-y", type=float, default=0.5, help="Query y as fraction of image height")
    parser.add_argument("--metric-sample-tokens", type=int, default=256)
    parser.add_argument("--mask-threshold", type=int, default=3,
                        help="Pixel threshold used to recover the sonar support region")
    parser.add_argument("--mask-close-kernel", type=int, default=21)
    parser.add_argument("--mask-min-coverage", type=float, default=0.05,
                        help="Minimum sonar-support coverage required for a grid cell")
    parser.add_argument("--include-background", action="store_true",
                        help="Include external black padding in PCA and probe metrics")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true", help="Rebuild already processed epochs")
    parser.add_argument("--watch", action="store_true", help="Process new checkpoints as they appear")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--make-gif", action="store_true")
    parser.add_argument("--gif-duration-ms", type=int, default=700)
    return parser.parse_args()


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch(\d+)", path.name)
    return int(match.group(1)) if match else -1


def find_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoint:
        path = Path(args.checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return [path]

    root = Path(args.checkpoint_dir).expanduser().resolve()
    paths = sorted(root.glob(CHECKPOINT_PATTERN), key=checkpoint_epoch)
    if args.max_checkpoints > 0:
        paths = paths[-args.max_checkpoints:]
    return paths


def torch_load(path: Path, weights_only: bool = False):
    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_metadata(path: Path) -> dict:
    ckpt = torch_load(path)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return ckpt


def triplet(value, default) -> tuple[float, ...]:
    if value is None:
        return tuple(float(x) for x in default)
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None:
        return tuple(float(x) for x in default)
    return tuple(float(x) for x in value)


def center_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def fixed_square(image: Image.Image, size: int) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image)
    return ImageOps.fit(image, (size, size), method=resampling.BILINEAR, centering=(0.5, 0.5))


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[..., None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def normalize(tensor: torch.Tensor, mean, std) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1).clamp_min(1e-8)
    return (tensor - mean_t) / std_t


def to_u8(image01: np.ndarray) -> np.ndarray:
    return np.uint8(np.clip(image01 * 255.0, 0.0, 255.0))


def sonar_support_mask(gray: np.ndarray, threshold: int, close_kernel: int) -> np.ndarray:
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

    # Dependency-free fallback: fill the occupied span of every sonar row.
    support = np.zeros_like(seed, dtype=bool)
    for row in range(seed.shape[0]):
        columns = np.flatnonzero(seed[row])
        if columns.size:
            support[row, columns[0]:columns[-1] + 1] = True
    return support


def grid_support_mask(pixel_mask: np.ndarray, height: int, width: int, min_coverage: float) -> torch.Tensor:
    tensor = torch.from_numpy(pixel_mask.astype(np.float32))[None, None]
    coverage = F.interpolate(tensor, size=(height, width), mode="area")[0, 0]
    return coverage >= float(min_coverage)


def prepare_probe(image_path: Path, ckpt: dict, args: argparse.Namespace) -> dict:
    image_size = int(ckpt.get("train_img_size", 448))
    input_mode = str(ckpt.get("input_mode", "rgb"))
    sonar = dict(ckpt.get("sonar_feature_config") or {})
    student_norm = dict(ckpt.get("student_normalization") or {})
    student_mean = triplet(student_norm.get("mean"), STUDENT_MEAN)
    student_std = triplet(student_norm.get("std"), STUDENT_STD)

    source = Image.open(image_path)
    panels = []

    if input_mode == "rgb":
        geometry = fixed_square(source.convert("RGB"), image_size)
        teacher_raw = pil_to_tensor(geometry)
        student_raw = teacher_raw.clone()
        display = np.asarray(geometry)
        support = np.ones((image_size, image_size), dtype=bool)
        panels.append(("student input: RGB", display))
    elif input_mode == "fls_grayscale":
        geometry = center_square(source.convert("L"))
        geometry = fixed_square(geometry, image_size)
        teacher_raw = pil_to_tensor(geometry.convert("RGB"))
        student_raw = pil_to_tensor(geometry).mul(2.0).sub(1.0)
        display = np.asarray(geometry)
        support = sonar_support_mask(display, args.mask_threshold, args.mask_close_kernel)
        panels.append(("student input: grayscale [-1,1]", display))
    elif input_mode in {"fls_features", "sonar_features"}:
        gray = source.convert("L")
        if input_mode == "fls_features":
            gray = center_square(gray)
        gray = fixed_square(gray, image_size)
        gray_np = np.asarray(gray, dtype=np.uint8)
        support = sonar_support_mask(gray_np, args.mask_threshold, args.mask_close_kernel)
        teacher_raw = pil_to_tensor(gray.convert("RGB"))
        features = make_sonar_features(
            gray_np,
            middle_channel=sonar.get("middle_channel", "wavelet_low"),
            third_channel=sonar.get("third_channel", "sobel_edge"),
            wavelet=sonar.get("wavelet", "haar"),
            occupancy_threshold=int(sonar.get("occupancy_threshold", 128)),
            local_contrast_blur=int(sonar.get("local_contrast_blur", 31)),
            edge_blur=int(sonar.get("edge_blur", 3)),
        )
        student_raw = torch.from_numpy(features).permute(2, 0, 1).float()
        display = to_u8(features)
        panels.extend([
            ("raw robust", to_u8(features[..., 0])),
            (sonar.get("middle_channel", "middle"), to_u8(features[..., 1])),
            (sonar.get("third_channel", "third"), to_u8(features[..., 2])),
            ("student feature RGB", display),
        ])
    else:
        raise ValueError(f"Unsupported input_mode in checkpoint: {input_mode}")

    teacher = normalize(teacher_raw, TEACHER_MEAN, TEACHER_STD).unsqueeze(0)
    if input_mode == "fls_grayscale":
        student = student_raw.unsqueeze(0)
    else:
        student = normalize(student_raw, student_mean, student_std).unsqueeze(0)

    return {
        "teacher": teacher,
        "student": student,
        "display": display,
        "panels": panels,
        "image_size": image_size,
        "input_mode": input_mode,
        "in_channels": int(ckpt.get("in_channels", student.shape[1])),
        "contract": ckpt.get("preprocessing_contract", "unknown"),
        "student_mean": student_mean,
        "student_std": student_std,
        "pixel_support_mask": support,
    }


def load_teacher(repo: Path, weights: Path, device: torch.device):
    try:
        teacher = torch.hub.load(str(repo), "dinov3_vitl16", source="local", pretrained=False)
    except TypeError as exc:
        if sys.version_info < (3, 10) and "unsupported operand type" in str(exc):
            raise RuntimeError(
                "The local DINOv3 repository requires Python 3.10 or newer. "
                f"This interpreter is Python {sys.version_info.major}.{sys.version_info.minor}. "
                "Run this visualiser in the same Python 3.11/3.12 environment used for distillation."
            ) from exc
        raise
    state = torch_load(weights, weights_only=True)
    teacher.load_state_dict(state, strict=True)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def teacher_grid(teacher, tensor: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        output = teacher.forward_features(tensor)
        tokens = output["x_norm_patchtokens"]
        side = int(round(math.sqrt(tokens.shape[1])))
        if side * side != tokens.shape[1]:
            raise ValueError(f"Teacher token count is not square: {tokens.shape[1]}")
        grid = tokens.reshape(1, side, side, -1)
        return F.normalize(grid.float(), p=2, dim=-1)


def load_student(ckpt: dict, probe: dict, device: torch.device):
    student = mobilevit_s_distill(
        image_size=(probe["image_size"], probe["image_size"]),
        in_channels=probe["in_channels"],
    )
    student.load_state_dict(ckpt["model_state_dict"], strict=True)
    return student.to(device).eval()


def teacher_pca_basis(grid: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = grid[0][valid_mask.to(grid.device)].float()
    center = tokens.mean(dim=0, keepdim=True)
    _, _, basis = torch.pca_lowrank(tokens - center, q=3, center=False)
    return center, basis


def project_rgb(grid: torch.Tensor, center: torch.Tensor, basis: torch.Tensor, valid_mask: torch.Tensor, bounds=None):
    height, width = grid.shape[1:3]
    projected = (grid.reshape(-1, grid.shape[-1]).float() - center) @ basis
    array = projected.reshape(height, width, 3).detach().cpu().numpy()
    valid_numpy = valid_mask.cpu().numpy()
    if bounds is None:
        valid_values = array[valid_numpy]
        lo = np.percentile(valid_values, 2.0, axis=0)
        hi = np.percentile(valid_values, 98.0, axis=0)
    else:
        lo, hi = bounds
    scaled = np.clip((array - lo) / np.maximum(hi - lo, 1e-8), 0.0, 1.0)
    scaled[~valid_numpy] = 0.0
    return scaled, (lo, hi)


def query_similarity(grid: torch.Tensor, query_y: int, query_x: int, valid_mask: torch.Tensor) -> np.ndarray:
    features = F.normalize(grid[0].float(), p=2, dim=-1)
    reference = features[query_y, query_x]
    similarity = torch.einsum("hwc,c->hw", features, reference).detach().cpu().numpy()
    similarity[~valid_mask.cpu().numpy()] = np.nan
    return similarity


def nearest_valid_query(valid_mask: torch.Tensor, query_y: int, query_x: int) -> tuple[int, int]:
    if bool(valid_mask[query_y, query_x]):
        return query_y, query_x
    coords = torch.nonzero(valid_mask, as_tuple=False)
    if coords.numel() == 0:
        raise ValueError("Sonar support mask contains no valid grid cells")
    target = torch.tensor([query_y, query_x], dtype=coords.dtype)
    index = torch.argmin((coords - target).square().sum(dim=1))
    return int(coords[index, 0]), int(coords[index, 1])


def pairwise_uniformity(tokens: torch.Tensor, sample_count: int, seed: int) -> float:
    flat = F.normalize(tokens.reshape(-1, tokens.shape[-1]).float(), p=2, dim=-1)
    count = min(int(sample_count), flat.shape[0])
    generator = torch.Generator(device=flat.device).manual_seed(seed)
    indices = torch.randperm(flat.shape[0], generator=generator, device=flat.device)[:count]
    sample = flat.index_select(0, indices)
    similarity = sample @ sample.T
    mask = ~torch.eye(count, dtype=torch.bool, device=flat.device)
    return float(similarity[mask].mean().cpu())


def effective_rank(tokens: torch.Tensor, sample_count: int, seed: int) -> float:
    flat = tokens.reshape(-1, tokens.shape[-1]).float()
    count = min(int(sample_count), flat.shape[0])
    generator = torch.Generator(device=flat.device).manual_seed(seed)
    indices = torch.randperm(flat.shape[0], generator=generator, device=flat.device)[:count]
    sample = flat.index_select(0, indices)
    sample = sample - sample.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(sample)
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    return float(entropy.exp().cpu())


def spatial_smoothness(grid: torch.Tensor, valid_mask: torch.Tensor) -> float:
    grid = F.normalize(grid.float(), p=2, dim=-1)
    horizontal = (grid[:, :, 1:] * grid[:, :, :-1]).sum(dim=-1)[0]
    vertical = (grid[:, 1:] * grid[:, :-1]).sum(dim=-1)[0]
    horizontal_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
    vertical_valid = valid_mask[1:] & valid_mask[:-1]
    values = []
    if bool(horizontal_valid.any()):
        values.append(horizontal[horizontal_valid].mean())
    if bool(vertical_valid.any()):
        values.append(vertical[vertical_valid].mean())
    if not values:
        return float("nan")
    return float(torch.stack(values).mean().cpu())


def evaluate(teacher: torch.Tensor, student: torch.Tensor, valid_mask: torch.Tensor, args, epoch: int) -> tuple[dict, dict]:
    cosine = F.cosine_similarity(student, teacher, dim=-1)[0]
    mse_map = (student - teacher).square().sum(dim=-1)[0]
    valid = valid_mask.to(cosine.device)
    valid_cosine = cosine[valid]
    valid_mse = mse_map[valid]
    teacher_tokens = teacher[0][valid]
    student_tokens = student[0][valid]
    metrics = {
        "epoch": int(epoch),
        "probe_mse": float(valid_mse.mean().cpu()),
        "probe_cosine_loss": float((1.0 - valid_cosine).mean().cpu()),
        "probe_total_loss": float((valid_mse.mean() + (1.0 - valid_cosine).mean()).cpu()),
        "cosine_mean": float(valid_cosine.mean().cpu()),
        "cosine_p10": float(torch.quantile(valid_cosine, 0.10).cpu()),
        "cosine_p50": float(torch.quantile(valid_cosine, 0.50).cpu()),
        "cosine_p90": float(torch.quantile(valid_cosine, 0.90).cpu()),
        "mse_p90": float(torch.quantile(valid_mse, 0.90).cpu()),
        "student_pair_cosine": pairwise_uniformity(student_tokens, args.metric_sample_tokens, args.seed + epoch),
        "teacher_pair_cosine": pairwise_uniformity(teacher_tokens, args.metric_sample_tokens, args.seed),
        "student_effective_rank": effective_rank(student_tokens, args.metric_sample_tokens, args.seed + epoch),
        "teacher_effective_rank": effective_rank(teacher_tokens, args.metric_sample_tokens, args.seed),
        "student_spatial_smoothness": spatial_smoothness(student, valid_mask),
        "teacher_spatial_smoothness": spatial_smoothness(teacher, valid_mask),
        "valid_patch_fraction": float(valid_mask.float().mean()),
    }
    cosine_map = cosine.detach().cpu().numpy()
    mse_numpy = mse_map.detach().cpu().numpy()
    invalid = ~valid_mask.cpu().numpy()
    cosine_map[invalid] = np.nan
    mse_numpy[invalid] = np.nan
    maps = {
        "cosine": cosine_map,
        "mse": mse_numpy,
    }
    return metrics, maps


def read_training_log(checkpoint_dir: Path) -> list[dict]:
    path = checkpoint_dir / "training_log.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def numeric_series(rows: list[dict], key: str):
    xs, ys = [], []
    for row in rows:
        try:
            xs.append(int(float(row["epoch"])))
            ys.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            continue
    return xs, ys


def show_image(axis, image, title: str, cmap=None, vmin=None, vmax=None):
    if isinstance(cmap, str):
        cmap = copy.copy(plt.get_cmap(cmap))
        cmap.set_bad("black")
    artist = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=10)
    axis.axis("off")
    return artist


def plot_panel(
    output_path: Path,
    checkpoint_path: Path,
    ckpt: dict,
    probe: dict,
    teacher: torch.Tensor,
    student: torch.Tensor,
    valid_mask: torch.Tensor,
    metrics: dict,
    maps: dict,
    history: list[dict],
    training_log: list[dict],
    args: argparse.Namespace,
):
    grid_h, grid_w = teacher.shape[1:3]
    requested_x = int(np.clip(round(args.query_x * (grid_w - 1)), 0, grid_w - 1))
    requested_y = int(np.clip(round(args.query_y * (grid_h - 1)), 0, grid_h - 1))
    query_y, query_x = nearest_valid_query(valid_mask, requested_y, requested_x)
    center, basis = teacher_pca_basis(teacher, valid_mask)
    teacher_rgb, bounds = project_rgb(teacher, center, basis, valid_mask)
    student_rgb, _ = project_rgb(student, center, basis, valid_mask, bounds=bounds)
    teacher_sim = query_similarity(teacher, query_y, query_x, valid_mask)
    student_sim = query_similarity(student, query_y, query_x, valid_mask)
    sim_diff = np.abs(student_sim - teacher_sim)

    fig, axes = plt.subplots(4, 4, figsize=(19, 17))
    epoch = metrics["epoch"]
    fig.suptitle(
        f"DINOv3 ViT-L/16 -> MobileViT-S Distillation | epoch {epoch} | "
        f"{probe['input_mode']} | grid {grid_h}x{grid_w}",
        fontsize=15,
    )

    show_image(axes[0, 0], probe["display"], "Fixed probe / student-visible input")
    available = probe["panels"][:3]
    for index in range(3):
        if index < len(available):
            title, panel = available[index]
            show_image(axes[0, index + 1], panel, title, cmap="gray" if panel.ndim == 2 else None)
        else:
            axes[0, index + 1].axis("off")

    show_image(axes[1, 0], teacher_rgb, "Teacher patch embeddings: PCA RGB")
    show_image(axes[1, 1], student_rgb, "Student patch embeddings: teacher PCA basis")
    cos_artist = show_image(axes[1, 2], maps["cosine"], "Teacher-student cosine per patch", "viridis", -1, 1)
    fig.colorbar(cos_artist, ax=axes[1, 2], fraction=0.046)
    mse_artist = show_image(axes[1, 3], maps["mse"], "Squared error summed over 1024 dims", "magma")
    fig.colorbar(mse_artist, ax=axes[1, 3], fraction=0.046)

    for axis, data, title in [
        (axes[2, 0], teacher_sim, "Teacher query-to-all similarity"),
        (axes[2, 1], student_sim, "Student query-to-all similarity"),
        (axes[2, 2], sim_diff, "Absolute similarity-map difference"),
    ]:
        artist = show_image(axis, data, title, "inferno")
        axis.scatter([query_x], [query_y], marker="s", s=55, facecolors="none", edgecolors="cyan", linewidths=1.8)
        fig.colorbar(artist, ax=axis, fraction=0.046)

    finite_cosine = maps["cosine"][np.isfinite(maps["cosine"])]
    axes[2, 3].hist(finite_cosine, bins=40, range=(-1, 1), color="#277da1", alpha=0.85)
    axes[2, 3].axvline(metrics["cosine_mean"], color="#f94144", linewidth=2, label="mean")
    axes[2, 3].set_title("Patch cosine distribution")
    axes[2, 3].set_xlabel("cosine(student, teacher)")
    axes[2, 3].set_ylabel("patches")
    axes[2, 3].legend()

    train_epoch, train_total = numeric_series(training_log, "avg_loss")
    _, train_mse = numeric_series(training_log, "avg_mse")
    _, train_cos = numeric_series(training_log, "avg_cosine")
    if train_epoch:
        axes[3, 0].plot(train_epoch, train_total, label="train total", linewidth=2)
        if train_mse:
            axes[3, 0].plot(train_epoch[:len(train_mse)], train_mse, label="train mse")
        if train_cos:
            axes[3, 0].plot(train_epoch[:len(train_cos)], train_cos, label="train cosine")
    axes[3, 0].axvline(epoch, color="black", linestyle="--", alpha=0.5)
    axes[3, 0].set_title("Whole-dataset training losses")
    axes[3, 0].set_xlabel("epoch")
    if train_epoch:
        axes[3, 0].legend(fontsize=8)

    history_with_current = sorted(history + [metrics], key=lambda row: row["epoch"])
    hx = [row["epoch"] for row in history_with_current]
    axes[3, 1].plot(hx, [row["probe_total_loss"] for row in history_with_current], marker="o", label="probe total")
    axes[3, 1].plot(hx, [row["probe_mse"] for row in history_with_current], marker="o", label="probe mse")
    axes[3, 1].plot(hx, [row["probe_cosine_loss"] for row in history_with_current], marker="o", label="probe cosine")
    axes[3, 1].set_title("Fixed-probe losses")
    axes[3, 1].set_xlabel("epoch")
    axes[3, 1].legend(fontsize=8)

    axes[3, 2].plot(hx, [row["cosine_mean"] for row in history_with_current], marker="o", label="teacher alignment")
    axes[3, 2].plot(hx, [row["student_pair_cosine"] for row in history_with_current], marker="o", label="student pair cosine")
    axes[3, 2].axhline(metrics["teacher_pair_cosine"], color="black", linestyle="--", label="teacher pair cosine")
    axes[3, 2].set_title("Alignment and redundancy")
    axes[3, 2].set_xlabel("epoch")
    axes[3, 2].legend(fontsize=8)

    axes[3, 3].axis("off")
    notes = [
        f"checkpoint: {checkpoint_path.name}",
        f"checkpoint avg_loss: {float(ckpt.get('avg_loss', float('nan'))):.6f}",
        f"probe total: {metrics['probe_total_loss']:.6f}",
        f"probe MSE: {metrics['probe_mse']:.6f}",
        f"probe cosine loss: {metrics['probe_cosine_loss']:.6f}",
        f"cosine mean / p10: {metrics['cosine_mean']:.4f} / {metrics['cosine_p10']:.4f}",
        f"student eRank: {metrics['student_effective_rank']:.2f}",
        f"teacher eRank: {metrics['teacher_effective_rank']:.2f}",
        f"student pair cosine: {metrics['student_pair_cosine']:.4f}",
        f"teacher pair cosine: {metrics['teacher_pair_cosine']:.4f}",
        f"student smoothness: {metrics['student_spatial_smoothness']:.4f}",
        f"teacher smoothness: {metrics['teacher_spatial_smoothness']:.4f}",
        f"valid sonar patches: {metrics['valid_patch_fraction'] * 100:.1f}%",
        f"checkpoint mask objective: {ckpt.get('spatial_mask_contract', 'legacy-unmasked')}",
        "",
        "Reading guide:",
        "- PCA colours should increasingly resemble the teacher spatially.",
        "- Cosine should rise while MSE and similarity-map difference fall.",
        "- Very high pair cosine means redundant/collapsed patch features.",
        "- Effective rank should not collapse toward 1.",
        "- Black external padding is excluded from these diagnostic metrics.",
        "- Legacy checkpoints were trained on all cells; newly masked checkpoints exclude padding.",
    ]
    axes[3, 3].text(0.0, 1.0, "\n".join(notes), va="top", family="monospace", fontsize=9)

    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    fig.savefig(output_path, dpi=145)
    plt.close(fig)


def load_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else []


def save_history(path: Path, rows: list[dict]):
    rows = sorted(rows, key=lambda row: row["epoch"])
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if not rows:
        return
    csv_path = path.with_suffix(".csv")
    columns = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_best_summary(path: Path, rows: list[dict]):
    if not rows:
        return
    best_loss = min(rows, key=lambda row: row["probe_total_loss"])
    best_alignment = max(rows, key=lambda row: row["cosine_mean"])
    payload = {
        "selection_note": "Probe metrics are diagnostic; confirm the chosen checkpoint on a multi-image validation set.",
        "best_fixed_probe_loss": best_loss,
        "best_fixed_probe_alignment": best_alignment,
        "latest": max(rows, key=lambda row: row["epoch"]),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_gif(frame_paths: list[Path], output_path: Path, duration_ms: int):
    if not frame_paths:
        return
    frames = []
    resampling = getattr(Image, "Resampling", Image)
    for path in frame_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((1400, 1400), resampling.LANCZOS)
        frames.append(image.copy())
        image.close()
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(int(duration_ms), 50),
        loop=0,
        optimize=False,
    )
    for frame in frames:
        frame.close()


def save_pipeline_overview(output_path: Path):
    fig, axis = plt.subplots(figsize=(17, 6))
    axis.set_xlim(0, 17)
    axis.set_ylim(0, 7)
    axis.axis("off")
    axis.set_title("DINOv3 -> MobileViT Dense Distillation Pipeline", fontsize=17, pad=18)

    def box(x, y, width, height, title, detail, color):
        patch = plt.Rectangle((x, y), width, height, facecolor=color, edgecolor="#222222", linewidth=1.5)
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height * 0.66, title, ha="center", va="center", fontsize=11, weight="bold")
        axis.text(x + width / 2, y + height * 0.30, detail, ha="center", va="center", fontsize=8.5)

    def arrow(x0, y0, x1, y1, label=""):
        axis.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 1.8})
        if label:
            axis.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.18, label, ha="center", fontsize=8)

    box(0.3, 2.5, 2.0, 1.5, "Fixed probe image", "same pixels for every epoch", "#e9ecef")
    box(3.0, 4.7, 2.4, 1.5, "Teacher input", "RGB/grayscale + ImageNet norm", "#ffddd2")
    box(6.1, 4.7, 2.4, 1.5, "Frozen DINOv3 ViT-L/16", "x_norm_patchtokens", "#ffb4a2")
    box(9.2, 4.7, 2.3, 1.5, "Teacher target", "G x G x 1024 unit vectors", "#e5989b")
    box(3.0, 0.5, 2.4, 1.5, "Student input", "RGB, grayscale, or sonar features", "#caf0f8")
    box(6.1, 0.5, 2.4, 1.5, "Trainable MobileViT-S", "stride 16, dense output", "#90e0ef")
    box(9.2, 0.5, 2.3, 1.5, "Student prediction", "G x G x 1024 unit vectors", "#48cae4")
    box(12.3, 2.5, 2.0, 1.5, "Patch losses", "MSE + cosine + lambda*SIGReg", "#d8f3dc")
    box(15.0, 2.5, 1.7, 1.5, "Update", "student only", "#95d5b2")

    arrow(2.3, 3.5, 3.0, 5.25, "teacher branch")
    arrow(2.3, 3.0, 3.0, 1.25, "student branch")
    arrow(5.4, 5.45, 6.1, 5.45)
    arrow(8.5, 5.45, 9.2, 5.45)
    arrow(5.4, 1.25, 6.1, 1.25)
    arrow(8.5, 1.25, 9.2, 1.25)
    arrow(11.5, 5.25, 12.3, 3.55)
    arrow(11.5, 1.25, 12.3, 2.95)
    arrow(14.3, 3.25, 15.0, 3.25)
    axis.annotate("", xy=(7.3, 2.0), xytext=(15.8, 2.5), arrowprops={"arrowstyle": "->", "lw": 1.5, "linestyle": "--"})
    axis.text(11.6, 1.95, "back-propagation", ha="center", fontsize=8)
    axis.text(7.3, 4.35, "no gradients", ha="center", fontsize=9, color="#9d0208")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.metric_sample_tokens < 2:
        raise ValueError("--metric-sample-tokens must be >= 2")
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    probe_path = Path(args.probe_image).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else checkpoint_dir / "distillation_visuals" / probe_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_pipeline_overview(output_dir / "distillation_pipeline_overview.png")
    history_path = output_dir / "probe_metrics.json"
    history = load_history(history_path)
    processed = {int(row["epoch"]) for row in history}

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print("Loading frozen DINOv3 teacher once ...")
    teacher_model = load_teacher(
        Path(args.dinov3_repo).expanduser().resolve(),
        Path(args.teacher_weights).expanduser().resolve(),
        device,
    )

    teacher_cache = {}
    training_log = read_training_log(checkpoint_dir)
    force_pending = bool(args.force)

    while True:
        checkpoints = find_checkpoints(args)
        if not checkpoints and not args.watch:
            raise FileNotFoundError(
                f"No checkpoints matching {CHECKPOINT_PATTERN} in {checkpoint_dir}"
            )
        pending = [
            path for path in checkpoints
            if force_pending or checkpoint_epoch(path) not in processed
        ]
        for checkpoint_path in pending:
            epoch = checkpoint_epoch(checkpoint_path)
            print(f"[Epoch {epoch}] {checkpoint_path.name}")
            ckpt = checkpoint_metadata(checkpoint_path)
            probe = prepare_probe(probe_path, ckpt, args)
            cache_key = (
                probe["image_size"],
                probe["input_mode"],
                probe["contract"],
            )
            if cache_key not in teacher_cache:
                teacher_input = probe["teacher"].to(device)
                teacher_cache[cache_key] = teacher_grid(teacher_model, teacher_input).cpu()
            target_grid = teacher_cache[cache_key].to(device)
            if args.include_background or probe["input_mode"] == "rgb":
                valid_mask = torch.ones(target_grid.shape[1:3], dtype=torch.bool)
            else:
                valid_mask = grid_support_mask(
                    probe["pixel_support_mask"],
                    target_grid.shape[1],
                    target_grid.shape[2],
                    args.mask_min_coverage,
                )

            student_model = load_student(ckpt, probe, device)
            with torch.inference_mode():
                student_grid = student_model.forward_distill(probe["student"].to(device)).float()
            if student_grid.shape != target_grid.shape:
                raise ValueError(
                    f"Student grid {tuple(student_grid.shape)} != teacher grid {tuple(target_grid.shape)}"
                )

            metrics, maps = evaluate(target_grid, student_grid, valid_mask, args, epoch)
            metrics.update({
                "checkpoint": str(checkpoint_path),
                "checkpoint_avg_loss": float(ckpt.get("avg_loss", float("nan"))),
                "input_mode": probe["input_mode"],
                "image_size": int(probe["image_size"]),
                "grid_size": int(target_grid.shape[1]),
                "preprocessing_contract": probe["contract"],
                "probe_image": str(probe_path),
                "teacher_weights": str(Path(args.teacher_weights).expanduser().resolve()),
            })

            frame_path = output_dir / f"epoch_{epoch:04d}_distillation_probe.png"
            plot_panel(
                frame_path,
                checkpoint_path,
                ckpt,
                probe,
                target_grid,
                student_grid,
                valid_mask,
                metrics,
                maps,
                history,
                training_log,
                args,
            )
            history = [row for row in history if int(row["epoch"]) != epoch]
            history.append(metrics)
            save_history(history_path, history)
            save_best_summary(output_dir / "best_probe_checkpoint.json", history)
            processed.add(epoch)
            print(
                f"  saved={frame_path.name}  cosine={metrics['cosine_mean']:.4f}  "
                f"loss={metrics['probe_total_loss']:.4f}  erank={metrics['student_effective_rank']:.1f}"
            )
            del student_model, student_grid
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if args.make_gif:
            frames = sorted(output_dir.glob("epoch_*_distillation_probe.png"))
            make_gif(frames, output_dir / "distillation_progress.gif", args.gif_duration_ms)

        force_pending = False

        if not args.watch:
            break
        if not pending:
            print(f"No new checkpoints. Waiting {args.poll_seconds:g}s ...")
        time.sleep(max(args.poll_seconds, 1.0))

    print(f"Visualisations: {output_dir}")
    print(f"Probe metrics: {history_path}")


if __name__ == "__main__":
    main()
