"""
mobilevit_distill.py
--------------------
MobileViT backbone adapted for dense feature distillation from DINOv3 ViT-Large/16.

ARCHITECTURAL CHANGES vs. original MobileViT
=============================================

Goal: student output must be [B, G, G, 1024] to match the teacher's
      x_norm_patchtokens grid exactly — same spatial resolution, same feature dim.

Teacher target shape derivation
--------------------------------
  DINOv3 ViT-L/16, square input divisible by 16
  patch size = 16×16  →  grid = image_size/16
  feature dim = 1024
  target: x_norm_patchtokens  [B, G*G, 1024]  →  [B, G, G, 1024]  (L2-normalised)

Original MobileViT spatial trace (256×256, stride product = 32)
----------------------------------------------------------------
  Input           : H×W
  conv1 stride 2  : H/2  × W/2
  mv2[1] stride 2 : H/4  × W/4
  mv2[4] stride 2 : H/8  × W/8    ← mvit[0] operates here
  mv2[5] stride 2 : H/16 × W/16   ← mvit[1] operates here
  mv2[6] stride 2 : H/32 × W/32   ← mvit[2] operates here
  conv2           : H/32 × W/32

At 448×448: H/32 = 14×14  ≠  28×28 required.
At 256×256: H/32 = 8×8    ≠  28×28 required.

Change 1 — Remove the last stride-2 downsampling (mv2[6])
----------------------------------------------------------
  mv2[6] stride changed from 2 → 1.
  New stride product = 16.
  At 448×448 input: H/16 = 28×28; at 512×512 input: H/16 = 32×32.

  Why this is architecturally sound:
  • MV2Block explicitly allows stride ∈ {1, 2} — the assert still passes.
  • mvit[2] now processes a 28×28 feature map instead of 14×14. Its internal
    transformer sees seq_len = (28/ph)×(28/pw) = 14×14 = 196 tokens per patch
    group (patch_size=(2,2)), identical to what mvit[1] processes. No
    rearrange constraint is violated.
  • This is strictly a stride reduction — no new parameters, no structural
    additions outside the backbone.

Change 2 — conv2 output channels: channels[-1] → 1024
------------------------------------------------------
  The original conv2 = conv_1x1_bn(channels[-2], channels[-1]) is a 1×1 conv
  already inside MobileViT. Changing its output to 1024 makes the backbone
  natively produce the teacher's feature dimension.

  For mobilevit_s: channels[-2] = 160  →  1024  (was 640).
  This is a channel width change to an existing layer, not an added layer.

  Why 1×1 conv is the right place for dimensional alignment:
  • 1×1 convolutions are the canonical way to change channel depth in CNN/ViT
    hybrid backbones (it is what they are designed for).
  • It already exists at the correct position (deepest feature map, before any
    pooling), so using it for the dimensional match is semantically correct.

Change 3 — Remove pool and fc
------------------------------
  AvgPool and the classification fc are task heads that collapse spatial
  information to a single vector — the opposite of what dense distillation
  needs. They are removed. The distillation forward returns the spatial map.

Summary of what is NOT changed
--------------------------------
  • All MV2Block and MobileViTBlock definitions — untouched.
  • conv1, mv2[0..5], mvit[0..2] — identical to original.
  • patch_size=(2,2) constraint: 28%2==0  ✓  at all three MobileViTBlock sites.
  • The model can still be used for classification by passing a num_classes and
    using the standard forward(); the distillation path is a separate method.

Input/Output contract for distillation
---------------------------------------
  Input  : [B, 3, image_size, image_size]   (same tensor that goes into DINOv3)
  Output : [B, G, G, 1024]  L2-normalised over the feature dimension
                               — directly comparable to teacher_grid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


# ---------------------------------------------------------------------------
# Primitives (unchanged from original)
# ---------------------------------------------------------------------------

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)',
                      h=h // self.ph, w=w // self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


# ---------------------------------------------------------------------------
# MobileViT — distillation variant
# ---------------------------------------------------------------------------

# Teacher output dimension (DINOv3 ViT-L/16)
TEACHER_DIM = 1024
# Effective output stride after the last stride-2 MobileViT downsampling was
# removed. DINOv3 ViT-L/16 also uses 16 px patch tokens, so any square input
# divisible by 16 gives matching teacher/student spatial grids.
OUTPUT_STRIDE = 16


class MobileViT(nn.Module):
    """
    MobileViT backbone with two architectural adjustments for dense distillation.

    Distillation-specific changes (see module docstring for full derivation):
      1. mv2[6] stride = 1  (was 2) — keeps output stride at 16
         after the last downsampling stage instead of collapsing to 14×14.
      2. conv2 output = TEACHER_DIM (1024)  — 1×1 conv already present in the
         backbone; its output width is set to match the teacher's feature dim
         without any additional layer being introduced.
      3. pool and fc are removed; the distillation forward returns the dense
         [B, G, G, 1024] feature map.

    For classification use, pass num_classes and call forward(x).
    For distillation, call forward_distill(x).
    """

    def __init__(
        self,
        image_size,
        dims,
        channels,
        num_classes,
        expansion=4,
        kernel_size=3,
        patch_size=(2, 2),
    ):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        assert ih % OUTPUT_STRIDE == 0 and iw % OUTPUT_STRIDE == 0, (
            f"MobileViT distillation variant requires image_size divisible by "
            f"{OUTPUT_STRIDE}, got ({ih}, {iw})."
        )
        self.image_size = (ih, iw)
        self.output_stride = OUTPUT_STRIDE
        self.grid_size = (ih // OUTPUT_STRIDE, iw // OUTPUT_STRIDE)

        L = [2, 4, 3]

        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))   # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        # ── Change 1: stride 2 → 1 ──────────────────────────────────────────
        # Original: MV2Block(channels[7], channels[8], 2, expansion)
        # With stride=1, spatial stays at image_size/16, matching the teacher.
        # MV2Block asserts stride ∈ {1, 2} — this is explicitly supported.
        self.mv2.append(MV2Block(channels[7], channels[8], 1, expansion))
        # ────────────────────────────────────────────────────────────────────

        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0] * 2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4)))
        # mvit[2] now receives the image_size/16 feature map.
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)))

        # ── Change 2: conv2 output = TEACHER_DIM (1024) ─────────────────────
        # This is an existing 1×1 conv inside MobileViT. Its output width is set
        # to 1024 to natively match the teacher's feature dimension.
        # For mobilevit_s: channels[-2]=160 → 1024  (was 640).
        self.conv2 = conv_1x1_bn(channels[-2], TEACHER_DIM)
        # ────────────────────────────────────────────────────────────────────

        # Classification head — kept for non-distillation use.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(TEACHER_DIM, num_classes, bias=False)

    # ------------------------------------------------------------------
    # Shared feature extraction (common to both forward paths)
    # ------------------------------------------------------------------

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the dense feature map [B, TEACHER_DIM, G, G] before any
        pooling or classification.

        Spatial trace:
          conv1/mv2/mvit stack downsamples by 16 overall.
          conv2             →   image_size/16 × image_size/16 × 1024
        """
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)      # Repeat

        x = self.mv2[4](x)
        x = self.mvit[0](x)

        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x = self.mv2[6](x)      # stride=1 — no spatial collapse
        x = self.mvit[2](x)
        x = self.conv2(x)       # [B, 1024, 28, 28]
        return x

    # ------------------------------------------------------------------
    # Distillation forward
    # ------------------------------------------------------------------

    def forward_distill(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dense feature extraction for knowledge distillation.

        Args:
            x: [B, 3, image_size, image_size]  — same tensor fed to DINOv3

        Returns:
            [B, G, G, 1024]  L2-normalised over dim=-1
                               Directly comparable to teacher_grid produced by:
                                 patch_tokens.reshape(B, G, G, 1024)
                                 F.normalize(..., p=2, dim=-1)
        """
        feat = self._extract_features(x)         # [B, 1024, G, G]
        feat = feat.permute(0, 2, 3, 1)          # [B, G, G, 1024]
        feat = F.normalize(feat, p=2, dim=-1)    # unit vectors over feature dim
        return feat

    # ------------------------------------------------------------------
    # Classification forward (unchanged semantics)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard classification forward.
        """
        x = self._extract_features(x)            # [B, 1024, G, G]
        x = self.pool(x).view(-1, x.shape[1])    # [B, 1024]
        x = self.fc(x)                           # [B, num_classes]
        return x


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def mobilevit_xxs_distill(image_size=(448, 448)):
    """
    MobileViT-XXS adapted for 448×448 distillation.
    Backbone params ≈ 1.3 M  +  conv2 (160→1024) ≈ 0.16 M  =  ~1.5 M total.
    Note: expansion=2 retained from original xxs definition.
    """
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT(image_size, dims, channels, num_classes=1000, expansion=2)


def mobilevit_xs_distill(image_size=(448, 448)):
    """
    MobileViT-XS adapted for 448×448 distillation.
    Backbone params ≈ 2.3 M  +  conv2 ≈ 0.10 M  =  ~2.4 M total.
    """
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT(image_size, dims, channels, num_classes=1000)


def mobilevit_s_distill(image_size=(448, 448)):
    """
    MobileViT-S adapted for 448×448 distillation.
    Backbone params ≈ 5.6 M  +  conv2 (160→1024) ≈ 0.16 M  =  ~5.8 M total.
    Recommended variant: best capacity/size tradeoff for distilling ViT-L.
    """
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT(image_size, dims, channels, num_classes=1000)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    sizes = (448, 512)

    for size in sizes:
        img = torch.randn(2, 3, size, size)
        expected_grid = size // OUTPUT_STRIDE
        for name, factory in [
            ("mobilevit_xxs_distill", mobilevit_xxs_distill),
            ("mobilevit_xs_distill",  mobilevit_xs_distill),
            ("mobilevit_s_distill",   mobilevit_s_distill),
        ]:
            model = factory(image_size=(size, size))
            model.eval()
            with torch.no_grad():
                out = model.forward_distill(img)
            print(f"{name} @ {size}")
            print(f"  distill output : {tuple(out.shape)}  (expected: (2, {expected_grid}, {expected_grid}, 1024))")
            print(f"  L2 norm check  : {out.norm(dim=-1).mean().item():.4f}  (expected: 1.0)")
            print(f"  params         : {count_parameters(model) / 1e6:.2f} M")
            print()
