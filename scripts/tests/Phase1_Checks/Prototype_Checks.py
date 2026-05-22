# Check if DINO prototypes are being used uniformly
# Collapse = all images map to same few prototypes = training failed
import torch, os
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

ckpt = torch.load("/workspace/outputs/dinov3_vitl_dgx/phase1/full_64999.pth",
                  map_location="cpu", weights_only=False)
model_sd = ckpt["model"]

# Check DINO head last layer weight distribution
dino_keys = [k for k in model_sd if "dino_head" in k and "last_layer" in k]
ibot_keys = [k for k in model_sd if "ibot_head" in k and "last_layer" in k]

print("DINO head last layer:")
for k in dino_keys:
    w = model_sd[k]
    print(f"  {k}: shape={w.shape}  mean={w.mean():.4f}  std={w.std():.4f}  norm={w.norm():.2f}")

print("\niBOT head last layer:")    
for k in ibot_keys:
    w = model_sd[k]
    print(f"  {k}: shape={w.shape}  mean={w.mean():.4f}  std={w.std():.4f}  norm={w.norm():.2f}")

# Check teacher vs student weight similarity
teacher_keys = [k for k in model_sd if k.startswith("teacher.backbone.blocks.0")]
student_keys = [k for k in model_sd if k.startswith("student.backbone.blocks.0")]

if teacher_keys and student_keys:
    t_key = teacher_keys[0]
    s_key = student_keys[0]
    t_w = model_sd[t_key]
    s_w = model_sd[s_key]
    sim = torch.nn.functional.cosine_similarity(
        t_w.flatten().unsqueeze(0), s_w.flatten().unsqueeze(0)
    ).item()
    print(f"\nTeacher/Student block 0 similarity: {sim:.4f}")
    print("(~1.0 at start, should diverge slightly to 0.85-0.99 during training)")
    if sim > 0.8:
        print("✅ Teacher/student properly coupled via EMA")
    else:
        print("⚠️ Unusually low coupling")