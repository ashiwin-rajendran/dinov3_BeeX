import sys
sys.path.insert(0, '/workspace/dinov3')

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import (
    Compose, ToImage, Resize, ToDtype, Normalize, RandomHorizontalFlip
)
from dinov3.train.ssl_meta_arch import SSLMetaArch
from omegaconf import OmegaConf

print("=== Step 1: Dataset ===")
transform = Compose([
    ToImage(),
    Resize((224, 224), antialias=True),
    ToDtype(torch.float32, scale=True),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

dataset = ImageFolder(
    root='/workspace/others/tiny-imagenet-200/train',
    transform=transform
)
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
imgs, labels = next(iter(loader))
print(f"Batch shape: {imgs.shape}  Labels: {labels}")  # [2, 3, 224, 224]

print("\n=== Step 2: Model forward pass ===")
model = torch.hub.load(
    '/workspace/dinov3',
    'dinov3_vits16',
    source='local',
    pretrained=False
)
model.eval()
with torch.no_grad():
    out = model.forward_features(imgs)
print(f"CLS:   {out['x_norm_clstoken'].shape}")
print(f"Patch: {out['x_norm_patchtokens'].shape}")

print("\n=== All checks passed — pipeline is working ===")


# PYTHONPATH=${PWD} python /workspace/tests/TrainCheck.py 