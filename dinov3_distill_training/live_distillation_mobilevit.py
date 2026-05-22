"""
live_distillation_mobilevit.py
-------------------------------
USAGE:

python3 Yolo_Seg/vit_training/dinov3_distill_training/live_distillation_mobilevit.py \
  --image-dir /path/to/images \
  --weights /path/to/dinov3_weights.pth \
  --checkpoint-dir checkpoints \
  --resume-from auto \
  --image-size 512

Knowledge distillation: DINOv3 ViT-Large/16 (teacher) → MobileViT-S (student).

Tensor contract (verified by shape trace)
------------------------------------------
  Teacher  x_norm_patchtokens : [B, G*G, 1024]  →  [B, G, G, 1024]  L2-normalised
  Student  forward_distill()  : [B, G, G, 1024]  L2-normalised
  Both share the same [B, 3, image_size, image_size] input tensor.

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
import re

from mobilevit_distill import mobilevit_s_distill, MobileViT, count_parameters


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

class LiveAugmentationDataset(Dataset):
    def __init__(self, image_dir: str, image_size: int = 448):
        self.directory = Path(image_dir)
        self.image_size = int(image_size)
        self.image_paths = [
            p for p in self.directory.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        # Same normalisation as DINOv3 pretraining — critical so that teacher
        # and student receive identically pre-processed tensors.
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)


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
    dummy_input = torch.randn(1, 3, image_size, image_size)

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
    ckpt = {
        "epoch": epoch,
        "avg_loss": avg_loss,
        "model_state_dict": student.state_dict(),
        "torchscript_saved": saved_ts,
        "model_type": "mobilevit_distill",
        "feature_dim": 1024,
        "train_img_size": image_size,
        "train_grid_size": grid_size,
        "output_stride": 16,
        "inference_note": "forward_distill supports any square input divisible by 16",
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
) -> tuple:
    """
    Load a full or legacy student checkpoint.

    Returns:
        start_epoch, avg_loss
    """
    if not ckpt_path:
        return 1, 0.0

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
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

    student.load_state_dict(ckpt, strict=True)
    match = re.search(r"epoch[_-]?(\d+)", Path(ckpt_path).name)
    epoch = int(match.group(1)) if match else 0
    print(f"[Resume] Loaded legacy model-only checkpoint: {ckpt_path}")
    print("[Resume] Optimizer/scheduler/scaler state unavailable; they will restart.")
    return epoch + 1, 0.0


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
    image_size: int = 448,
) -> None:

    image_size = int(image_size)
    if image_size % 16 != 0:
        raise ValueError(f"image_size must be divisible by 16 for DINOv3 ViT-L/16, got {image_size}")
    expected_grid = image_size // 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    print(f"Initialising Live Distillation on: {device}")
    print(f"Image size: {image_size}x{image_size}  ->  target grid: {expected_grid}x{expected_grid}")

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
        "/workspace/dinov3", "dinov3_vitl16", source="local", pretrained=False
    )
    teacher.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True), strict=True
    )
    teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"  Teacher parameters: {count_parameters(teacher) / 1e6:.1f} M  (all frozen)")

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
    student = mobilevit_s_distill(image_size=(image_size, image_size)).to(device)
    print(f"  Student parameters: {count_parameters(student) / 1e6:.2f} M  (all trainable)")

    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    # ------------------------------------------------------------------ #
    # C.  Data and Mixed Precision Scaler                                  #
    # ------------------------------------------------------------------ #
    dataset    = LiveAugmentationDataset(image_dir, image_size=image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    if resume_from == "auto":
        resume_from = get_latest_checkpoint(checkpoint_dir)
        if resume_from:
            print(f"[Resume] Auto-detected checkpoint: {resume_from}")
        else:
            print("[Resume] No checkpoint found in checkpoint_dir; starting fresh.")

    start_epoch, avg_loss = load_distill_checkpoint(
        resume_from,
        student,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        expected_image_size=image_size,
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
        for images in pbar:
            images = images.to(device)
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
                    features     = teacher.forward_features(images)
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
                        images.shape[0], grid_size, grid_size, -1
                    )                                                      # [B, H, W, 1024]
                    teacher_grid = F.normalize(teacher_grid, p=2, dim=-1)

                # --- Student prediction (with gradients) ---
                #
                # MobileViT-S distillation variant:
                #   stride product = 16  →  image_size/16 = G spatial grid
                #   conv2 output dim = 1024  →  matches teacher feature dim
                #   forward_distill: [B, 1024, G, G] → permute → L2-norm
                #   output: [B, G, G, 1024]  (unit vectors, same as teacher)
                student_grid = student.forward_distill(images)             # [B, G, G, 1024]
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
    parser.add_argument("--image-size", type=int, default=448,
                        help="Square crop size. Must be divisible by 16. Use 512 for 32x32 features.")
    args = parser.parse_args()

    live_distillation(
        args.image_dir,
        args.weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        image_size=args.image_size,
    )
