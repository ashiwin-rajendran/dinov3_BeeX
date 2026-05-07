import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Compose, ToImage, Resize, ToDtype
from pathlib import Path
from PIL import Image
from tqdm import tqdm


class FlatImageDataset(Dataset):
    """Loads all images from a flat directory (no subdirectory structure needed)."""

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    def __init__(self, folder, transform=None):
        self.folder    = Path(folder)
        self.transform = transform
        self.images    = [
            p for p in sorted(self.folder.iterdir())
            if p.suffix in self.EXTENSIONS
        ]
        if not self.images:
            raise RuntimeError(f"No images found in {folder}")
        print(f"Found {len(self.images)} images in {folder}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_DIR  = '/workspace/datasets/AUV_Datasets_Clean/train'
BATCH_SIZE = 4
NUM_WORKERS = 1
# ─────────────────────────────────────────────────────────────────────────────

transform = Compose([
    ToImage(),
    Resize((512, 512), antialias=True),
    ToDtype(torch.float32, scale=True),
])

dataset = FlatImageDataset(TRAIN_DIR, transform=transform)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                     num_workers=NUM_WORKERS, shuffle=False)

mean = torch.zeros(3)
std  = torch.zeros(3)
n    = 0

for imgs in tqdm(loader, desc="Computing stats"):
    b     = imgs.size(0)
    mean += imgs.mean(dim=[0, 2, 3]) * b
    std  += imgs.std(dim=[0, 2, 3])  * b
    n    += b

mean /= n
std  /= n

print(f"\n{'='*45}")
print(f"AUV Dataset Stats — save these!")
print(f"{'='*45}")
print(f"mean = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
print(f"std  = [{std[0]:.4f},  {std[1]:.4f},  {std[2]:.4f}]")
print(f"{'='*45}")
print(f"Total images processed: {n}")


# root@1bba5557444d:/workspace/scripts# python3 DatasetStats.py 
# Found 226799 images in /workspace/datasets/AUV_Datasets_Clean/train
# Computing stats: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 56700/56700 [11:57<00:00, 79.05it/s]

# =============================================
# AUV Dataset Stats — save these!
# =============================================
# mean = [0.5118, 0.5094, 0.5125]
# std  = [0.1240,  0.1278,  0.1188]
# =============================================
# Total images processed: 226799
# root@1bba5557444d:/workspace/scripts# 
