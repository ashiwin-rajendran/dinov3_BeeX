#!/usr/bin/env python3
"""
extract_beex_images.py

    python3 extract_beex_images.py \
        --input  /data/missions \
        --output /data/frames  \
        [--checkpoint /data/checkpoint.json] \
        [--topic  /ikan/front_cam/ml_clahe/compressed] \
        [--fps    1] \
        [--width  512] \
        [--height 512] \
        [--jpg-quality 95] \
        [--workers 1] \
        [--force]
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil

import cv2
import numpy as np

try:
    import rosbag
    from sensor_msgs.msg import CompressedImage
except ImportError as exc:
    sys.exit(
        f"[ERROR] Could not import ROS packages ({exc}).\n"
        "Please source your ROS environment first:\n"
        "  source /opt/ros/<distro>/setup.bash\n"
        "Then re-run this script."
    )

try:
    import torch
    import torch.nn.functional as F
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

TOPIC_CANDIDATES = [
    "/ikan/front_cam/ml_clahe/compressed",
    "/ikan/front_cam/image_color/clahe/compressed",
]

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

CHECKPOINT_VERSION = 1


def _bag_fingerprint(bag_path: Path) -> str:
    """Stable identifier: relative-path + file-size + mtime (no hashing cost)."""
    stat = bag_path.stat()
    raw = f"{bag_path}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.md5(raw.encode()).hexdigest()


def load_checkpoint(path: Path) -> Dict[str, dict]:
    """Return mapping  fingerprint -> metadata dict."""
    if not path.exists():
        return {}
    try:
        with path.open("r") as fh:
            data = json.load(fh)
        if data.get("version") != CHECKPOINT_VERSION:
            log.warning("Checkpoint version mismatch - starting fresh.")
            return {}
        return data.get("processed", {})
    except (json.JSONDecodeError, KeyError) as exc:
        log.warning("Could not read checkpoint (%s) - starting fresh.", exc)
        return {}


def save_checkpoint(path: Path, processed: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as fh:
        json.dump({"version": CHECKPOINT_VERSION, "processed": processed}, fh, indent=2)
    tmp.replace(path)  # atomic on POSIX


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


@dataclass
class ExtractionConfig:
    topic: str
    fps: float
    width: int
    height: int
    jpg_quality: int
    sim_threshold: float = 1.0


class FeatureExtractor:
    _instance = None

    def __init__(self):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch/Torchvision not available for feature extraction.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading ResNet18 on %s for feature extraction...", self.device)

        # Load pre-trained ResNet18 and remove the classification head
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        self.model = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.model.eval()
        self.model.to(self.device)

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Hardcoded Imagenet normalization values
        ])
        log.info("ResNet18 loaded on %s.", self.device)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @torch.no_grad()
    def extract(self, img_bgr: np.ndarray) -> torch.Tensor:
        # Convert BGR (cv2) to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess(img_rgb).unsqueeze(0).to(self.device)
        feature = self.model(tensor)
        # Flatten and normalize
        feature = feature.view(-1)
        return F.normalize(feature, p=2, dim=0)


def _decode_compressed_image(data: bytes) -> Optional[np.ndarray]:
    """Decode a CompressedImage payload (JPEG/PNG) to a BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img  # None if decode fails


def extract_frames_from_bag(bag_path, out_dir, cfg) -> Tuple[int, int, int]:
    staging_dir = out_dir.parent / f"_tmp_{out_dir.name}"

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
        log.warning("Removed stale staging dir: %s", staging_dir)

    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        interval_ns = int(1e9 / cfg.fps)
        last_saved_ns: Optional[int] = None
        frames_saved = 0
        messages_read = 0
        frames_skipped = 0

        saved_features = []
        extractor = None
        if cfg.sim_threshold < 1.0:
            if not TORCH_AVAILABLE:
                log.warning("PyTorch not available, skipping similarity filter.")
            else:
                extractor = FeatureExtractor.get_instance()

        with rosbag.Bag(str(bag_path), "r", allow_unindexed=True) as bag:
            topics_info = bag.get_type_and_topic_info().topics
            resolved_topic = cfg.topic if cfg.topic in topics_info else None
            if resolved_topic is None:
                for candidate in TOPIC_CANDIDATES:
                    if candidate in topics_info:
                        resolved_topic = candidate
                        log.info("Primary topic not found — using fallback '%s' in %s", resolved_topic, bag_path.name)
                        break

            if resolved_topic is None:
                log.warning(
                    "No recognised camera topic found in %s — available: %s", bag_path.name, list(topics_info.keys())
                )
                return 0, 0, 0

            for _topic, msg, t in bag.read_messages(topics=[resolved_topic]):
                messages_read += 1
                t_ns = t.to_nsec()

                if last_saved_ns is not None and (t_ns - last_saved_ns) < interval_ns:
                    continue

                img = _decode_compressed_image(bytes(msg.data))
                if img is None:
                    log.debug("Decode failed at t=%s in %s", t, bag_path.name)
                    continue

                if img.shape[:2] != (cfg.height, cfg.width):
                    img = cv2.resize(img, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)

                if extractor is not None:
                    feat = extractor.extract(img)
                    if saved_features:
                        saved_tensor = torch.stack(saved_features)
                        sims = torch.mv(saved_tensor, feat)
                        max_sim = sims.max().item()
                        if max_sim >= cfg.sim_threshold:
                            log.debug("Skipping frame at %s, similarity %.3f >= %.3f", t, max_sim, cfg.sim_threshold)
                            frames_skipped += 1
                            continue
                    saved_features.append(feat)

                fname = staging_dir / f"frame_{frames_saved:06d}.jpg"
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpg_quality])
                if not ok:
                    log.warning("JPEG encode failed for frame %d in %s", frames_saved, bag_path.name)
                    continue

                fname.write_bytes(buf.tobytes())
                last_saved_ns = t_ns
                frames_saved += 1

        if out_dir.exists():
            shutil.rmtree(out_dir)
        staging_dir.rename(out_dir)
        return frames_saved, messages_read, frames_skipped

    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# Worker (Multiprocessing)
# ---------------------------------------------------------------------------


def _worker(args: tuple) -> dict:
    (
        bag_path_str,
        out_dir_str,
        topic,
        fps,
        width,
        height,
        jpg_quality,
        sim_threshold,
    ) = args

    bag_path = Path(bag_path_str)
    out_dir = Path(out_dir_str)
    cfg = ExtractionConfig(
        topic=topic,
        fps=fps,
        width=width,
        height=height,
        jpg_quality=jpg_quality,
        sim_threshold=sim_threshold,
    )

    t0 = time.time()
    result = {
        "bag": bag_path_str,
        "out_dir": out_dir_str,
        "success": False,
        "frames_saved": 0,
        "messages_read": 0,
        "frames_skipped": 0,
        "elapsed_s": 0.0,
        "error": None,
    }

    try:
        frames_saved, messages_read, frames_skipped = extract_frames_from_bag(bag_path, out_dir, cfg)
        result.update(
            success=True,
            frames_saved=frames_saved,
            messages_read=messages_read,
            frames_skipped=frames_skipped,
        )
    except Exception as exc:  # pylint: disable=broad-except
        result["error"] = str(exc)
        log.error("FAILED %s: %s", bag_path.name, exc, exc_info=True)
    finally:
        result["elapsed_s"] = round(time.time() - t0, 2)

    return result


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_beex_files(root: Path) -> List[Path]:
    found = sorted(root.rglob("*.beex"))
    log.info("Discovered %d .beex file(s) under %s", len(found), root)
    return found


# ---------------------------------------------------------------------------
# Output path derivation
# ---------------------------------------------------------------------------


def derive_output_dir(bag_path: Path, input_root: Path, output_root: Path) -> Path:
    """
    Mirror the relative path from input_root into output_root
    """
    rel = bag_path.relative_to(input_root)
    stem_dir = rel.parent / rel.stem
    return output_root / stem_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract 1-FPS frames from .beex rosbag files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path, help="Root input directory.")
    p.add_argument("--output", required=True, type=Path, help="Root output directory.")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint JSON. Defaults to <output>/.checkpoint.json",
    )
    p.add_argument(
        "--topic",
        default="/ikan/front_cam/ml_clahe/compressed",
        help="ROS topic to extract.",
    )
    p.add_argument("--fps", type=float, default=1.0, help="Target extraction frame-rate.")
    p.add_argument("--width", type=int, default=512, help="Output frame width (px).")
    p.add_argument("--height", type=int, default=512, help="Output frame height (px).")
    p.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        metavar="Q",
        help="JPEG quality 0-100.",
    )
    p.add_argument(
        "--sim-threshold",
        type=float,
        default=1.0,
        help="Cosine similarity threshold for skipping duplicate images (e.g., 0.95). 1.0 disables filtering.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help=("Number of parallel worker processes. " "rosbag is I/O-bound so >2 rarely helps on spinning disks."),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore checkpoint and re-process all bags.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_root: Path = args.input.resolve()
    output_root: Path = args.output.resolve()

    if not input_root.is_dir():
        sys.exit(f"[ERROR] Input path does not exist or is not a directory: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path: Path = args.checkpoint or (output_root / ".checkpoint.json")

    # --- Load checkpoint --------------------------------------------------
    processed: Dict[str, dict] = {} if args.force else load_checkpoint(checkpoint_path)
    log.info(
        "Checkpoint: %d bag(s) already processed%s.",
        len(processed),
        " (--force: ignoring)" if args.force else "",
    )

    # --- Discover bags ----------------------------------------------------
    all_bags = discover_beex_files(input_root)
    if not all_bags:
        log.warning("No .beex files found. Exiting.")
        return

    # --- Filter out already-processed -------------------------------------
    pending: List[Tuple[Path, str]] = []
    for bag in all_bags:
        fp = _bag_fingerprint(bag)
        if fp in processed and processed[fp].get("success"):
            log.debug("Skipping (already done): %s", bag.name)
        else:
            pending.append((bag, fp))

    log.info(
        "%d bag(s) to process (%d skipped as already done).",
        len(pending),
        len(all_bags) - len(pending),
    )

    if not pending:
        log.info("Nothing to do. All bags are up-to-date.")
        return

    # --- Build worker arg tuples ------------------------------------------
    worker_args = []
    for bag_path, _fp in pending:
        out_dir = derive_output_dir(bag_path, input_root, output_root)
        worker_args.append(
            (
                str(bag_path),
                str(out_dir),
                args.topic,
                args.fps,
                args.width,
                args.height,
                args.jpg_quality,
                args.sim_threshold,
            )
        )

    # --- Process ----------------------------------------------------------
    total_frames = 0
    succeeded = 0
    failed = 0

    def _run_sequential():
        for w_args in worker_args:
            yield _worker(w_args)

    def _run_parallel():
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, w_args): w_args for w_args in worker_args}
            try:
                for fut in as_completed(futures):
                    yield fut.result()
            except KeyboardInterrupt:
                log.warning("Cancelling pending workers...")
                for f in futures:
                    f.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
                raise  # re-raise so the outer try/except in main catches it

    iterator = _run_sequential() if args.workers <= 1 else _run_parallel()

    # Map bag_path_str -> fingerprint for checkpoint updates
    fp_map = {str(bag): fp for bag, fp in pending}

    try:
        for i, result in enumerate(iterator, 1):
            bag_str = result["bag"]
            fp = fp_map[bag_str]
            bag_name = Path(bag_str).name

            if result["success"]:
                total_frames += result["frames_saved"]
                succeeded += 1
                skipped_msg = f" (skipped {result['frames_skipped']})" if result.get("frames_skipped") else ""
                log.info(
                    "[%d/%d] done %s  ->  %d frame(s)%s  (%.1fs)",
                    i,
                    len(pending),
                    bag_name,
                    result["frames_saved"],
                    skipped_msg,
                    result["elapsed_s"],
                )
                processed[fp] = {
                    "bag": bag_str,
                    "out_dir": result["out_dir"],
                    "success": True,
                    "frames_saved": result["frames_saved"],
                    "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            else:
                failed += 1
                log.error(
                    "[%d/%d] ✗ %s  (%.1fs)  error: %s", i, len(pending), bag_name, result["elapsed_s"], result["error"]
                )
                processed[fp] = {
                    "bag": bag_str,
                    "success": False,
                    "error": result["error"],
                    "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }

            save_checkpoint(checkpoint_path, processed)  # after every bag

    except KeyboardInterrupt:
        log.warning("\nCtrl+C received — saving checkpoint for %d completed bag(s) and exiting.", succeeded)
        save_checkpoint(checkpoint_path, processed)
        log.info("Checkpoint saved to %s  — re-run to continue.", checkpoint_path)
        sys.exit(130)  # standard SIGINT exit code

    # --- Summary ----------------------------------------------------------
    log.info(
        "Done. %d bag(s) processed: %d succeeded, %d failed. " "%d total frame(s) saved. Checkpoint: %s",
        len(pending),
        succeeded,
        failed,
        total_frames,
        checkpoint_path,
    )

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
