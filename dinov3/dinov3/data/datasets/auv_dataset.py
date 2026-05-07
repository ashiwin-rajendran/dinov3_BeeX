import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class AUVDataset(Dataset):

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    MEAN = [0.5118, 0.5094, 0.5125]
    STD  = [0.1240,  0.1278,  0.1188]

    def __init__(self, root, split='train', transform=None,
                 target_transform=None, transforms=None, **kwargs):

        self.root      = Path(root)
        self.split     = split
        self.transform = transform
        self.split_dir = self.root / split

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"[AUVDataset] Split directory not found: {self.split_dir}"
            )

        # Collect all images — flat or one level of subdirs
        self.images = []

        # First try flat (all images directly in split_dir)
        flat = [
            p for p in sorted(self.split_dir.iterdir())
            if p.is_file() and p.suffix in self.EXTENSIONS
        ]

        if flat:
            self.images = flat
        else:
            # Fall back: walk one level of subdirs
            for subdir in sorted(self.split_dir.iterdir()):
                if subdir.is_dir():
                    self.images += [
                        p for p in sorted(subdir.iterdir())
                        if p.is_file() and p.suffix in self.EXTENSIONS
                    ]

        if not self.images:
            raise RuntimeError(
                f"[AUVDataset] No images found in {self.split_dir}"
            )

        print(f"[AUVDataset] Split : {split}")
        print(f"[AUVDataset] Found : {len(self.images)} images")
        print(f"[AUVDataset] Root  : {self.split_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # Return dummy label 0 — SSL ignores labels
        return img, 0