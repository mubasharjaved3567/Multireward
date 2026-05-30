import os, math, urllib.request
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoModel


class AestheticMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128,   64), nn.Dropout(0.1),
            nn.Linear(64,    16),
            nn.Linear(16,     1),
        )
    def forward(self, x): return self.layers(x)


class RewardHeads:
    def __init__(self, device):
        self.device = device
        self._load_clip()
        self._load_aesthetic()
        self._load_pickscore()
        print("[RewardHeads] All 4 heads ready.")

    def _load_clip(self):
        import clip
        self.clip_model, self.clip_prep = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval().requires_grad_(False)
        print("  [OK] r_align   — CLIP ViT-B/32")

    def _load_aesthetic(self):
        import clip
        self.aes_clip, self.aes_prep = clip.load("ViT-L/14", device=self.device)
        self.aes_clip.eval().requires_grad_(False)
        w = "/tmp/aes.pth"
        if not os.path.exists(w):
            print("  Downloading aesthetic weights...")
            urllib.request.urlretrieve(
                "https://github.com/christophschuhmann/improved-aesthetic-predictor"
                "/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth", w)
        self.aes_mlp = AestheticMLP()
        self.aes_mlp.load_state_dict(torch.load(w, map_location="cpu"))
        self.aes_mlp.to(self.device).eval().requires_grad_(False)
        print("  [OK] r_aesthetic — LAION Aesthetic")

    def _load_pickscore(self):
        # FIX: PickScore on CPU — saves 3.8 GB VRAM
        self.pick_proc  = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.pick_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").to("cpu").eval()
        self.pick_model.requires_grad_(False)
        print("  [OK] r_preference — PickScore ViT-H (CPU)")

    @torch.no_grad()
    def score(self, prompt, image):
        import clip as C

        # r_align: CLIP cosine similarity (GPU)
        i   = self.clip_prep(image).unsqueeze(0).to(self.device)
        t   = C.tokenize([prompt], truncate=True).to(self.device)
        if_ = self.clip_model.encode_image(i);  if_ = if_ / if_.norm(dim=-1, keepdim=True)
        tf  = self.clip_model.encode_text(t);   tf  = tf  / tf.norm(dim=-1, keepdim=True)
        r_align = (if_ * tf).sum().item()

        # r_aesthetic: LAION MLP (GPU)
        a   = self.aes_prep(image).unsqueeze(0).to(self.device)
        f   = self.aes_clip.encode_image(a).float()
        f   = f / f.norm(dim=-1, keepdim=True)
        r_aesthetic = self.aes_mlp(f).item()

        # r_preference: PickScore (CPU — saves 3.8 GB VRAM)
        pi  = self.pick_proc(images=[image], return_tensors="pt", padding=True)
        pt  = self.pick_proc(text=[prompt], return_tensors="pt", padding=True, truncation=True)
        ie  = self.pick_model.get_image_features(**pi)
        ie  = ie.pooler_output if hasattr(ie, "pooler_output") else ie
        ie  = ie / ie.norm(dim=-1, keepdim=True)
        te  = self.pick_model.get_text_features(**pt)
        te  = te.pooler_output if hasattr(te, "pooler_output") else te
        te  = te / te.norm(dim=-1, keepdim=True)
        r_preference = (ie * te).sum().item()

        # r_quality: Laplacian variance (CPU numpy)
        g       = np.array(image.convert("L")).astype(np.float32)
        gy, gx  = np.gradient(g)
        lap     = np.gradient(gx)[1] + np.gradient(gy)[0]
        r_quality = float(np.var(lap))

        return {
            "r_align":      r_align,
            "r_aesthetic":  r_aesthetic,
            "r_preference": r_preference,
            "r_quality":    r_quality,
        }


def norm_scores(s):
    return {
        "r_align":      s["r_align"]      / 30.0,
        "r_aesthetic":  s["r_aesthetic"]  / 8.0,
        "r_preference": max(s["r_preference"] - 0.15, 0.0) / 0.2,
        "r_quality":    min(s["r_quality"] / 5000.0, 1.0),
    }


def smooth_weights(step_idx, total_steps):
    """
    Trajectory-aware weights that shift focus across denoising timesteps.
    Order: [r_align, r_aesthetic, r_preference, r_quality]

    Early steps (high noise):   r_align dominates   — establish text binding
    Mid steps:                  r_aesthetic peaks    — refine style/composition
                                r_preference peaks   — human preference signal
    Late steps (low noise):     r_quality rises      — sharpen fine details

    Returns normalised tensor summing to 1.
    """
    s = step_idx / max(total_steps - 1, 1)   # 0.0 → 1.0
    w = torch.tensor([
        1.0 - 0.5 * s,                              # r_align:      1.0 → 0.5
        math.sin(math.pi * s),                       # r_aesthetic:  0 → peak → 0
        math.sin(math.pi * max(s - 0.1, 0.0)),      # r_preference: delayed sin
        0.2 + 0.8 * s,                               # r_quality:    0.2 → 1.0
    ], dtype=torch.float32)
    return w / (w.sum() + 1e-8)