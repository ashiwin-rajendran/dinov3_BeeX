import os
import cv2
import shutil
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_DIR  = '/workspace/datasets/AUV_Datasets/train'   # flat folder, no subdirs
TEST_DIR   = '/workspace/datasets/AUV_Datasets/test'    # flat folder, no subdirs

# Output — creates the subdir structure DINOv3 needs
TRAIN_OUT  = '/workspace/datasets/AUV_Datasets_Clean/train'
TEST_OUT   = '/workspace/datasets/AUV_Datasets_Clean/test'

BLUR_THRESHOLD    = 80.0   # Laplacian variance — below this = too blurry
SIMILARITY_STRIDE = 5      # Keep every Nth frame (removes temporal duplicates)
# ─────────────────────────────────────────────────────────────────────────────


def is_blurry(img_path, threshold=BLUR_THRESHOLD):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True   # unreadable = treat as corrupted
    score = cv2.Laplacian(img, cv2.CV_64F).var()
    return score < threshold


def is_corrupted(img_path):
    try:
        img = Image.open(img_path)
        img.verify()
        return False
    except Exception:
        return True


def process_flat_dir(src_dir, dst_dir, split_name):
    """
    Processes a flat directory (no subdirs) of images.
    Copies filtered frames into dst_dir (which acts as a single pseudo-class).
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Collect all jpg/png images directly in the folder
    frames = sorted(src_dir.glob('*.jpg')) + \
             sorted(src_dir.glob('*.JPG')) + \
             sorted(src_dir.glob('*.png')) + \
             sorted(src_dir.glob('*.PNG'))

    if not frames:
        print(f"[{split_name}] No images found in {src_dir}")
        return

    print(f"\n[{split_name}] Found {len(frames)} total frames in {src_dir}")

    kept = skipped_blur = skipped_corrupt = skipped_dup = 0

    for i, frame in enumerate(tqdm(frames, desc=split_name)):

        # # Skip near-duplicate sequential frames
        # if i % SIMILARITY_STRIDE != 0:
        #     skipped_dup += 1
        #     continue

        # Skip corrupted
        if is_corrupted(frame):
            skipped_corrupt += 1
            continue

        # Skip blurry
        if is_blurry(frame):
            skipped_blur += 1
            continue

        # Copy good frame to output
        shutil.copy2(frame, dst_dir / frame.name)
        kept += 1

    total_removed = skipped_dup + skipped_blur + skipped_corrupt
    print(f"\n{'='*45}")
    print(f"[{split_name}] Results:")
    print(f"  Input total:        {len(frames)}")
    print(f"  Kept:               {kept}")
    print(f"  Skipped (temporal): {skipped_dup}")
    print(f"  Skipped (blur):     {skipped_blur}")
    print(f"  Skipped (corrupt):  {skipped_corrupt}")
    print(f"  Total removed:      {total_removed}")
    print(f"  Output dir:         {dst_dir}")
    print(f"{'='*45}")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    process_flat_dir(TRAIN_DIR, TRAIN_OUT, split_name='TRAIN')
    process_flat_dir(TEST_DIR,  TEST_OUT,  split_name='TEST/VAL')

    print("\n✅ Done. Final structure:")
    print("  /workspace/auv_dataset_clean/")
    print("    train/auv/   ← SSL training frames")
    print("    val/auv/     ← validation frames")
