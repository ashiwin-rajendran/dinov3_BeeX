import torch
import sys
sys.path.insert(0, '/path/to/dinov3')  # root of the cloned repo

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

model.eval()
dummy = torch.randn(1, 3, 224, 224)   # batch=1, RGB, 224×224

with torch.no_grad():
    out = model.forward_features(dummy)

# ViT outputs a dict with:
print("Output keys:", out.keys())
# Output keys: dict_keys(['x_norm_clstoken', 'x_storage_tokens', 'x_norm_patchtokens', 'x_prenorm', 'masks'])

cls   = out['x_norm_clstoken']      # shape: [1, 384]  for ViT-S
patch = out['x_norm_patchtokens']   # shape: [1, 196, 384]  (14×14 patches for 224px input)
reg   = out['x_storage_tokens']     # shape: [1, 4, 384]  (4 register tokens)

print(f"CLS token shape:   {cls.shape}")
print(f"Patch tokens shape: {patch.shape}")
print(f"Register tokens:   {reg.shape}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
dummy = dummy.to(device)

with torch.no_grad():
    out = model.forward_features(dummy)

print(f"Device: {device}")
print(f"CLS token device: {out['x_norm_clstoken'].device}")
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6:.1f} MB")