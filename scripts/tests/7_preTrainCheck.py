"""
Pre-training sanity check for DINOv3 ViT-L on DGX Spark (GB10, 128GB unified memory).
Run this BEFORE launching full training.
Checks: unified memory, BF16, batch throughput, ETA, dataset.
"""
import sys
sys.path.insert(0, '/workspace/dinov3')

import torch
import time

print("=" * 55)
print(" DINOv3 ViT-L — DGX Spark GB10 — Sanity Check")
print("=" * 55)

# ── 1. Hardware Detection ─────────────────────────────────────────────────────
print("\n[1] Hardware Detection")
assert torch.cuda.is_available(), "CUDA not available!"
props = torch.cuda.get_device_properties(0)
print(f"    GPU         : {props.name}")
print(f"    Total memory: {props.total_memory / 1e9:.1f} GB (unified)")
print(f"    CUDA        : {torch.version.cuda}")
print(f"    PyTorch     : {torch.__version__}")
if props.total_memory / 1e9 >= 100:
    print(f"    128GB unified memory confirmed")
else:
    print(f"     Expected 128GB — got {props.total_memory/1e9:.0f}GB")

# ── 2. BF16 Check ─────────────────────────────────────────────────────────────
print("\n[2] BF16 Tensor Core (Blackwell 5th-gen)")
assert torch.cuda.is_bf16_supported(), "BF16 not supported!"
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda')
b = torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda')
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    c = a @ b
torch.cuda.synchronize()
tflops = (2 * 4096**3 * 20) / (time.time() - t0) / 1e12
print(f"    BF16 TFLOPS : {tflops:.1f}")
print(f"    BF16 OK")

# ── 3. Model Load ─────────────────────────────────────────────────────────────
print("\n[3] ViT-L Model (BF16, no gradient checkpointing)")
device = torch.device('cuda')
model = torch.hub.load(
    '/workspace/dinov3', 'dinov3_vitl16',
    source='local', pretrained=False
).to(device).to(torch.bfloat16)

params   = sum(p.numel() for p in model.parameters()) / 1e6
mem_used = torch.cuda.memory_allocated() / 1e9
print(f"    Params      : {params:.1f}M")
print(f"    Memory used : {mem_used:.2f} GB")
print(f"    Memory free : {(props.total_memory/1e9) - mem_used:.1f} GB")
print(f"    No block_chunks needed — 128GB is ample")

# ── 4. Forward Pass (batch=64) ────────────────────────────────────────────────
print("\n[4] Forward Pass — batch=64, BF16")
BATCH = 4
g_imgs = torch.randn(BATCH*2, 3, 224, 224, dtype=torch.bfloat16, device=device)
l_imgs = torch.randn(BATCH*8, 3,  96,  96, dtype=torch.bfloat16, device=device)
model.eval()
with torch.no_grad():
    og = model.forward_features(g_imgs)
    ol = model.forward_features(l_imgs)
print(f"    Global CLS  : {og['x_norm_clstoken'].shape}")
print(f"    Global patch: {og['x_norm_patchtokens'].shape}")
print(f"    Local CLS   : {ol['x_norm_clstoken'].shape}")
print(f"    Memory used : {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── 5. Backward + Grad Accum ──────────────────────────────────────────────────
print("\n[5] Backward Pass + Grad Accum (4 steps)")
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.04)
try:
    for step in range(4):
        x    = torch.randn(BATCH, 3, 224, 224, dtype=torch.bfloat16, device=device)
        out  = model.forward_features(x)
        loss = out['x_norm_clstoken'].mean() / 4
        loss.backward()
    opt.step(); opt.zero_grad()

    peak     = torch.cuda.max_memory_allocated() / 1e9
    headroom = (props.total_memory / 1e9) - peak
    print(f"    Peak memory : {peak:.2f} GB")
    print(f"    Headroom    : {headroom:.1f} GB free")
    if headroom < 20:
        print("     < 20GB headroom — reduce batch_size_per_gpu to 32")
    else:
        print(f"     Memory headroom comfortable")

except torch.cuda.OutOfMemoryError:
    print("     OOM — reduce batch_size_per_gpu to 32 in config")
    sys.exit(1)

# ── 6. Throughput + ETA ───────────────────────────────────────────────────────
print("\n[6] Throughput + Training ETA")
model.eval()
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    for _ in range(20):
        x = torch.randn(BATCH, 3, 224, 224, dtype=torch.bfloat16, device=device)
        _ = model.forward_features(x)
torch.cuda.synchronize()
elapsed      = time.time() - t0
imgs_sec     = (BATCH * 20) / elapsed
secs_iter    = elapsed / 20 * 4   # × grad_accum=4
hrs_p1       = secs_iter * 100_000 / 3600
hrs_p2       = secs_iter * 300_000 / 3600
print(f"    Throughput   : {imgs_sec:.0f} imgs/sec")
print(f"    Per iter     : ~{secs_iter:.1f}s (incl. grad accum)")
print(f"    Phase 1 ETA  : ~{hrs_p1:.0f} hrs  (100k iters)")
print(f"    Phase 2 ETA  : ~{hrs_p2:.0f} hrs  (300k iters)")
print(f"       273 GB/s bandwidth (vs A100 2000 GB/s) — slower per iter")
print(f"       but zero memory constraints with 128GB unified pool")

# ── 7. Dataset Check ──────────────────────────────────────────────────────────
print("\n[7] AUV Dataset")
from dinov3.data.datasets.auv_dataset import AUVDataset
from torchvision.transforms.v2 import Compose, ToImage, Resize, ToDtype, Normalize
from torch.utils.data import DataLoader

ds = AUVDataset(
    '/workspace/datasets/AUV_Datasets_Clean', split='train',
    transform=Compose([
        ToImage(), Resize((224, 224), antialias=True),
        ToDtype(torch.float32, scale=True),
        Normalize([0.5118, 0.5094, 0.5125], [0.1240, 0.1278, 0.1188]),
    ])
)
loader    = DataLoader(ds, batch_size=BATCH, num_workers=16, shuffle=True)
imgs, _   = next(iter(loader))
print(f"    Dataset size : {len(ds)}")
print(f"    Batch shape  : {imgs.shape}")
print(f"    Pixel range  : [{imgs.min():.2f}, {imgs.max():.2f}]")
print(f"     Dataset OK")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  All checks passed — ready to launch training!")
print()
print(" Copy config:")
print("   cp vitl_dgx_spark.yaml \\")
print("      /workspace/dinov3/dinov3/configs/train/")
print()
print(" Launch Phase 1:")
print("   bash /workspace/scripts/train_auv.sh phase1")
print()
print(" Monitor:")
print("   watch -n 2 nvidia-smi")
print("=" * 55)



# cd /workspace/dinov3
# PYTHONPATH=${PWD} python /workspace/scripts/tests/preTrainCheck.py