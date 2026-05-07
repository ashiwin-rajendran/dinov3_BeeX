import argparse
import os
import random
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace/dinov3")
from dinov3.models import vision_transformer as vits

def load_model(checkpoint_path):
    model = vits.__dict__["vit_large"](
        patch_size=16,
        ffn_layer="mlp",
        pos_embed_type="rope",
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  Missing: {len(missing)}  Unexpected: {len(unexpected)}")
    model.eval()
    return model

def preprocess(image_path, img_size=224):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5118,0.5094,0.5125],[0.1240,0.1278,0.1188]), 
    ])
    img = Image.open(image_path).convert("RGB")
    return img.resize((img_size, img_size)), transform(img).unsqueeze(0)

def get_attention(model, tensor):
    saved = {}
    last_block = model.blocks[-1]
    orig_fwd = last_block.attn.forward

    def patched(x, rope=None):
        B, N, C = x.shape
        qkv = last_block.attn.qkv(x).reshape(B, N, 3,
            last_block.attn.num_heads, C // last_block.attn.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)

        # Apply RoPE if provided
        if rope is not None:
            q = last_block.attn.q_rope(q, rope) if hasattr(last_block.attn, 'q_rope') else q
            k = last_block.attn.k_rope(k, rope) if hasattr(last_block.attn, 'k_rope') else k

        attn = (q @ k.transpose(-2,-1)) * (q.shape[-1] ** -0.5)
        attn = attn.softmax(dim=-1)
        saved["attn"] = attn.detach()
        attn = last_block.attn.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = last_block.attn.proj(x)
        return last_block.attn.proj_drop(x)

    last_block.attn.forward = patched
    with torch.no_grad():
        model.forward_features(tensor)
    last_block.attn.forward = orig_fwd
    return saved["attn"]  # [1, H, N+1, N+1]

def visualize(image_path, model, output_dir, img_size=224):
    orig_img, tensor = preprocess(image_path, img_size)
    attn = get_attention(model, tensor)
    n_heads = attn.shape[1]
    nh = img_size // 16
    cls_attn = attn[0, :, 0, 1:].reshape(n_heads, nh, nh)

    ncols = min(n_heads, 8) + 2
    fig, axes = plt.subplots(1, ncols, figsize=(ncols*2.5, 3))
    fig.suptitle(f"DINOv3 Phase2 — {Path(image_path).name}", fontsize=10)

    axes[0].imshow(orig_img); axes[0].set_title("input", fontsize=8); axes[0].axis("off")

    mean_a = cls_attn.mean(0).numpy()
    mean_a = (mean_a - mean_a.min()) / (mean_a.max() - mean_a.min() + 1e-8)
    mean_up = np.array(Image.fromarray((mean_a*255).astype(np.uint8)).resize((img_size,img_size), Image.BILINEAR))/255.0
    axes[1].imshow(orig_img); axes[1].imshow(mean_up, alpha=0.6, cmap="inferno")
    axes[1].set_title("mean attn", fontsize=8); axes[1].axis("off")

    for h in range(min(n_heads, ncols-2)):
        a = cls_attn[h].numpy()
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        a_up = np.array(Image.fromarray((a*255).astype(np.uint8)).resize((img_size,img_size), Image.BILINEAR))/255.0
        axes[h+2].imshow(orig_img); axes[h+2].imshow(a_up, alpha=0.65, cmap="inferno")
        axes[h+2].set_title(f"h{h}", fontsize=8); axes[h+2].axis("off")

    plt.tight_layout()
    out = Path(output_dir) / f"attn_{Path(image_path).stem}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", default="/workspace/outputs/attention_maps")
    parser.add_argument("--n_images", type=int, default=6)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    model = load_model(args.checkpoint)

    exts = {".jpg",".jpeg",".png",".bmp"}
    all_imgs = [p for p in Path(args.image_dir).rglob("*") if p.suffix.lower() in exts]
    selected = random.sample(all_imgs, min(args.n_images, len(all_imgs)))
    print(f"Visualizing {len(selected)} images...")

    for i, img_path in enumerate(selected):
        print(f"[{i+1}/{len(selected)}] {img_path.name}")
        try:
            visualize(str(img_path), model, args.output_dir, args.img_size)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone! Results in: {args.output_dir}")

if __name__ == "__main__":
    main()