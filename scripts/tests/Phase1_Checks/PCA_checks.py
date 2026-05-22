import sys, os, random, torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace/dinov3")
from dinov3.models import vision_transformer as vits

CKPT = "/workspace/outputs/dinov3_vitl_dgx/phase1/teacher_64999_from_dcp.pth"
IMG_DIR = "/workspace/datasets/AUV_Datasets_Clean/train"
OUT = "/workspace/outputs/pca_check.png"
N_IMAGES = 200
IMG_SIZE = 512

# Load model
model = vits.__dict__["vit_large"](patch_size=16, ffn_layer="mlp", pos_embed_type="rope")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ckpt.get("model", ckpt)
sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
model.eval()
print("Model loaded")

# Preprocess
tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5118,0.5094,0.5125],[0.1240,0.1278,0.1188]),
])

# Sample images
all_imgs = list(Path(IMG_DIR).rglob("*.jpg"))[:5000]
selected = random.sample(all_imgs, min(N_IMAGES, len(all_imgs)))

# Extract CLS features
features = []
paths = []
print(f"Extracting features from {len(selected)} images...")
with torch.no_grad():
    for i, p in enumerate(selected):
        try:
            img = Image.open(p).convert("RGB")
            x = tfm(img).unsqueeze(0)
            out = model.forward_features(x)
            cls = out["x_norm_clstoken"].squeeze(0).numpy()
            features.append(cls)
            paths.append(p)
        except Exception as e:
            pass
        if i % 50 == 0:
            print(f"  {i}/{len(selected)}")

features = np.array(features)
print(f"Feature matrix: {features.shape}")

# PCA
from numpy.linalg import svd
F = features - features.mean(0)
_, _, Vt = svd(F, full_matrices=False)
pca = F @ Vt[:3].T  # first 3 PCs

# Variance explained
total_var = np.var(F, axis=0).sum()
pc_var = [np.var(pca[:,i]) for i in range(3)]
print(f"PC1 var explained: {pc_var[0]/total_var*100:.1f}%")
print(f"PC2 var explained: {pc_var[1]/total_var*100:.1f}%")
print(f"PC3 var explained: {pc_var[2]/total_var*100:.1f}%")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Phase 1 Feature PCA — iter ~64k\nPC1={pc_var[0]/total_var*100:.1f}%  PC2={pc_var[1]/total_var*100:.1f}%  PC3={pc_var[2]/total_var*100:.1f}%")

# PC1 vs PC2
scatter = axes[0].scatter(pca[:,0], pca[:,1], c=pca[:,2], 
                           cmap='viridis', alpha=0.6, s=15)
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
axes[0].set_title("PC1 vs PC2 (colored by PC3)")
plt.colorbar(scatter, ax=axes[0])

# PC1 vs PC3
scatter2 = axes[1].scatter(pca[:,0], pca[:,2], c=pca[:,1],
                            cmap='plasma', alpha=0.6, s=15)
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC3")
axes[1].set_title("PC1 vs PC3 (colored by PC2)")
plt.colorbar(scatter2, ax=axes[1])

plt.tight_layout()
plt.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"Saved: {OUT}")

# Summary verdict
spread = np.std(pca[:,0])
print(f"\nFeature spread (PC1 std): {spread:.3f}")
if pc_var[0]/total_var > 0.05:
    print("✅ PC1 explains >5% variance — features are structured")
else:
    print("⚠️  PC1 explains <5% variance — features may be collapsed")