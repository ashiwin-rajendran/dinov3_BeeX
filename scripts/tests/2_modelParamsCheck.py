import torch
import sys
sys.path.insert(0, '/workspace/dinov3')  # root of the cloned repo

# Load architecture only — no weights needed yet
model = torch.hub.load(
    '/workspace/dinov3',   # local repo path
    'dinov3_vits16',     # start with ViT-S — smallest, fastest to test
    source='local',
    pretrained=False     # no weights, just architecture
)
model.eval()
print(model)
print(f"\nTotal params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")