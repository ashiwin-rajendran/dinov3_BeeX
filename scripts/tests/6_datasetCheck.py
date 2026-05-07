import sys
sys.path.insert(0, '/workspace/dinov3')

import torch
from torch.utils.data import DataLoader
from dinov3.data.datasets.auv_dataset import AUVDataset
from torchvision.transforms.v2 import (
    Compose, ToImage, Resize, ToDtype, Normalize, RandomHorizontalFlip
)

# With Computed Stats from the AUV datasets 
AUV_MEAN = [0.5118, 0.5094, 0.5125]
AUV_STD  = [0.1240,  0.1278,  0.1188] 

transform = Compose([
    ToImage(),
    Resize((512, 512), antialias=True),
    RandomHorizontalFlip(),
    ToDtype(torch.float32, scale=True),
    Normalize(mean=AUV_MEAN, std=AUV_STD),
])

dataset = AUVDataset(
    root='/workspace/datasets/AUV_Datasets_Clean',
    split='train',
    transform=transform
)

loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
imgs, labels = next(iter(loader))

print(f"Dataset size:  {len(dataset)}")
print(f"Batch shape:   {imgs.shape}")        # [4, 3, 224, 224]
print(f"Pixel range:   [{imgs.min():.2f}, {imgs.max():.2f}]")
print(f"Mean per channel: {imgs.mean(dim=[0,2,3])}")

# Verify the loaders.py integration
from dinov3.data.loaders import make_dataset
ds = make_dataset(
    dataset_str="AUVDataset:root=/workspace/datasets/AUV_Datasets_Clean:split=train"
)
print(f"\nloaders.py integration: {len(ds)} samples")


# cd /workspace/dinov3
# PYTHONPATH=${PWD} python /workspace/tests/datasetCheck.py