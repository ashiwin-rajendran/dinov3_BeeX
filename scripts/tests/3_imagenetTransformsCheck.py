from torchvision.transforms.v2 import (
    ToImage, Resize, ToDtype, Normalize, Compose, CenterCrop
)
import torch

transform = Compose([
    ToImage(),
    Resize(256, antialias=True),
    CenterCrop(224),
    ToDtype(torch.float32, scale=True),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),   # ImageNet stats
])