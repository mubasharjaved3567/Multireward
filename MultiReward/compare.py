# fid_compare.py
# ─────────────────────────────────────────────────────────────────────────────
# Run from inside MultiReward/ folder:
#   python fid_compare.py
#
# Folder structure expected:
#   MultiReward/
#   ├── images/
#   │   ├── coco/val2017/     ← COCO images downloaded here
#   │   ├── refl/             ← paste your test images here manually
#   │   ├── strw/             ← script generates and saves here
#   │   └── grids/            ← comparison grids saved here
#   ├── results/              ← fid_scores.json saved here
#   ├── data/
#   │   ├── test.json
#   │   └── refl_data.json
#   └── fid_compare.py        ← this file
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import shutil
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchmetrics.image.fid import FrechetInceptionDistance

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← only change hf_token
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    # data
    "test_data":     "data/test.json",
    "subset_size":   200,

    # COCO — you downloaded here
    "coco_source":   "images/coco/val2017",

    # ReFL — you pasted manually here
    "refl_dir":      "images/refl",

    # STRW — script generates and saves here
    "strw_dir":      "images/strw",

    # grids saved here
    "grids_dir":     "images/grids",

    # COCO subsets saved here
    "coco_sub1":     "images/coco/subset1",
    "coco_sub2":     "images/coco/subset2",
    "coco_sub3":     "images/coco/subset3",

    # results
    "results_dir":   "results",

    # HuggingFace
    "hf_repo":       "mubasharjaved/strw-ddpo-20steps",
    "hf_checkpoint": "checkpoint-400/model.safetensors",

    # inference
    "model_id":      "CompVis/stable-diffusion-v1-4",
    "infer_steps":   50,
    "guidance":      7.5,

    # grid display
    "grid_rows":     8,
}

# ─────────────────────────────────────────────────────────────────────────────
# CREATE ALL FOLDERS
# ─────────────────────────────────────────────────────────────────────────────

for d in [CFG["refl_dir"], CFG["strw_dir"], CFG["grids_dir"],
          CFG["coco_sub1"], CFG["coco_sub2"], CFG["coco_sub3"],
          CFG["results_dir"]]:
    os.makedirs(d, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")
print(f"Running from: {os.getcwd()}\n")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def pil_to_tensor(images, size=299):
    tensors = []
    for img in images:
        arr = np.array(img.resize((size, size))).astype(np.uint8)
        tensors.append(torch.tensor(arr).permute(2, 0, 1))
    return torch.stack(tensors)


def compute_fid(real_images, gen_images):
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    fid.update(pil_to_tensor(real_images).to(DEVICE), real=True)
    fid.update(pil_to_tensor(gen_images).to(DEVICE),  real=False)
    score = float(fid.compute().item())
    del fid
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return score


def load_images_from_folder(folder, n):
    """Load up to n images from a folder (flat or subfolders)."""
    images = []
    exts   = ('.jpg', '.jpeg', '.png', '.webp')

    # flat folder
    files = sorted([f for f in os.listdir(folder)
                    if f.lower().endswith(exts)])
    if files:
        for fname in files[:n]:
            try:
                images.append(Image.open(os.path.join(folder, fname)).convert("RGB"))
            except:
                pass
        return images

    # subfolders (one image per subfolder)
    subdirs = sorted([d for d in os.listdir(folder)
                      if os.path.isdir(os.path.join(folder, d))])
    for d in subdirs[:n]:
        sub   = os.path.join(folder, d)
        files = sorted([f for f in os.listdir(sub) if f.lower().endswith(exts)])
        if files:
            try:
                images.append(Image.open(os.path.join(sub, files[0])).convert("RGB"))
            except:
                pass
    return images


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load prompts
# ─────────────────────────────────────────────────────────────────────────────

print("[1/6] Loading prompts...")
with open(CFG["test_data"]) as f:
    test_data = json.load(f)
N       = CFG["subset_size"]
prompts = [item["prompt"] for item in test_data[:N]]
print(f"  {len(prompts)} prompts loaded\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Split COCO into 3 subsets → images/coco/subset1,2,3
# ─────────────────────────────────────────────────────────────────────────────

print("[2/6] Splitting COCO into 3 subsets...")

if not os.path.exists(CFG["coco_source"]):
    print(f"  ERROR: COCO not found at {CFG['coco_source']}")
    print(f"  Run this first:")
    print(f"    mkdir -p images/coco")
    print(f"    curl -L http://images.cocodataset.org/zips/val2017.zip -o images/coco/val2017.zip")
    print(f"    cd images/coco && unzip val2017.zip && rm val2017.zip && cd ../..")
    exit(1)

all_coco = sorted([f for f in os.listdir(CFG["coco_source"])
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
print(f"  Found {len(all_coco)} COCO images")

if len(all_coco) < 3 * N:
    N = len(all_coco) // 3
    print(f"  Adjusted subset_size to {N}")

sub_dirs   = [CFG["coco_sub1"], CFG["coco_sub2"], CFG["coco_sub3"]]
coco_subsets = []

for i in range(3):
    batch = all_coco[i * N : (i + 1) * N]
    imgs  = []
    for fname in batch:
        src = os.path.join(CFG["coco_source"], fname)
        dst = os.path.join(sub_dirs[i], fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        try:
            imgs.append(Image.open(dst).convert("RGB"))
        except:
            pass
    coco_subsets.append(imgs)
    print(f"  Subset {i+1}: {len(imgs)} images → images/coco/subset{i+1}/")

print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Load ReFL images from images/refl/
# ─────────────────────────────────────────────────────────────────────────────

print(f"[3/6] Loading ReFL images from images/refl/ ...")
refl_images = load_images_from_folder(CFG["refl_dir"], N)
print(f"  Loaded {len(refl_images)} ReFL images")
if len(refl_images) == 0:
    print(f"  WARNING: no images found in {CFG['refl_dir']}")
    print(f"  Paste your test images into images/refl/ and re-run.")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Generate STRW images → images/strw/
# ─────────────────────────────────────────────────────────────────────────────

print(f"[4/6] STRW images → images/strw/ ...")

existing = sorted([f for f in os.listdir(CFG["strw_dir"])
                   if f.lower().endswith('.png')])

if len(existing) >= N:
    print(f"  Already have {len(existing)} images — loading from images/strw/")
    strw_images = []
    for fname in existing[:N]:
        try:
            strw_images.append(
                Image.open(os.path.join(CFG["strw_dir"], fname)).convert("RGB")
            )
        except:
            pass
    print(f"  Loaded {len(strw_images)} STRW images")

else:
    print(f"  Generating {N} images from {CFG['hf_repo']} ...")
    print(f"  (saved to images/strw/ — next run loads from disk instantly)")

    from diffusers import StableDiffusionPipeline
    from huggingface_hub import hf_hub_download
    import safetensors.torch

    hf_token = None

    pipe = StableDiffusionPipeline.from_pretrained(
        CFG["model_id"],
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None,
    ).to(DEVICE)

    print(f"  Downloading weights: {CFG['hf_checkpoint']}")
    ckpt = hf_hub_download(
        repo_id=CFG["hf_repo"],
        filename=CFG["hf_checkpoint"],
        token=hf_token,
    )
    weights = safetensors.torch.load_file(ckpt)
    pipe.unet.load_state_dict(weights, strict=False)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    pipe.unet = pipe.unet.to(DEVICE, dtype=dtype)
    print("  Weights loaded. Generating...")

    strw_images = []
    for i, prompt in enumerate(prompts):
        with torch.no_grad():
            img = pipe(
                prompt,
                num_images_per_prompt=1,
                num_inference_steps=CFG["infer_steps"],
                guidance_scale=CFG["guidance"],
            ).images[0]
        strw_images.append(img)
        img.save(os.path.join(CFG["strw_dir"], f"{i:04d}.png"))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{N} generated")

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  {len(strw_images)} images saved to images/strw/")

print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Compute FID across 3 subsets
# ─────────────────────────────────────────────────────────────────────────────

print("[5/6] Computing FID scores (3 subsets × 2 models)...")

refl_fids, strw_fids = [], []

for i, coco_subset in enumerate(coco_subsets):
    print(f"\n  ── Subset {i+1}/3  ({len(coco_subset)} real COCO images) ──")

    if len(refl_images) >= 10:
        r = compute_fid(coco_subset, refl_images[:N])
        refl_fids.append(r)
        print(f"     ReFL FID : {r:.2f}")
    else:
        refl_fids.append(None)
        print(f"     ReFL FID : N/A (not enough images)")

    if len(strw_images) >= 10:
        s = compute_fid(coco_subset, strw_images[:N])
        strw_fids.append(s)
        print(f"     STRW FID : {s:.2f}")
    else:
        strw_fids.append(None)
        print(f"     STRW FID : N/A (not enough images)")

print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Save comparison grids → images/grids/
# ─────────────────────────────────────────────────────────────────────────────

print(f"[6/6] Saving comparison grids → images/grids/ ...")

ROWS     = min(CFG["grid_rows"], len(prompts),
               len(refl_images) if refl_images else 0,
               len(strw_images) if strw_images else 0)
IMG_W    = 256
IMG_H    = 256
PAD      = 6
LABEL_H  = 36
HEADER_H = 44
COLS     = 3

grid_w = PAD + COLS * (IMG_W + PAD)
grid_h = HEADER_H + ROWS * (IMG_H + LABEL_H + PAD) + PAD

try:
    font_hdr  = ImageFont.truetype(
        "/System/Library/Fonts/Helvetica.ttc", 14)
    font_body = ImageFont.truetype(
        "/System/Library/Fonts/Helvetica.ttc", 10)
except:
    font_hdr  = ImageFont.load_default()
    font_body = ImageFont.load_default()

col_labels = ["ReFL  (SD v1.4)", "STRW — Ours", "COCO Real"]
col_colors = [(160, 80, 80), (60, 110, 190), (60, 150, 90)]


def make_grid(refl_imgs, strw_imgs, coco_imgs, prompts_list, rows):
    g    = Image.new("RGB", (grid_w, grid_h), (240, 240, 240))
    draw = ImageDraw.Draw(g)

    # headers
    for c, (label, color) in enumerate(zip(col_labels, col_colors)):
        x0 = PAD + c * (IMG_W + PAD)
        draw.rectangle([x0, 4, x0 + IMG_W, HEADER_H - 4], fill=color)
        draw.text((x0 + IMG_W // 2, HEADER_H // 2),
                  label, fill="white", font=font_hdr, anchor="mm")

    # rows
    for row in range(rows):
        y = HEADER_H + row * (IMG_H + LABEL_H + PAD)
        for c, src in enumerate([refl_imgs, strw_imgs, coco_imgs]):
            x0 = PAD + c * (IMG_W + PAD)
            if src and row < len(src):
                g.paste(src[row].resize((IMG_W, IMG_H)), (x0, y))
            else:
                draw.rectangle([x0, y, x0+IMG_W, y+IMG_H], fill=(200, 200, 200))
                draw.text((x0+IMG_W//2, y+IMG_H//2), "N/A",
                          fill=(100,100,100), font=font_hdr, anchor="mm")

        # prompt label
        p = prompts_list[row]
        p = p[:75] + "..." if len(p) > 75 else p
        draw.text((grid_w // 2, y + IMG_H + 4), p,
                  fill=(50, 50, 50), font=font_body, anchor="mt")
    return g


# one grid per subset (COCO column changes)
for i, coco_sub in enumerate(coco_subsets):
    g    = make_grid(refl_images, strw_images, coco_sub, prompts, ROWS)
    path = os.path.join(CFG["grids_dir"], f"comparison_subset{i+1}.png")
    g.save(path)
    print(f"  Saved: {path}")

# one combined grid using subset1 as reference
g_main = make_grid(refl_images, strw_images, coco_subsets[0], prompts, ROWS)
main_path = os.path.join(CFG["grids_dir"], "comparison_main.png")
g_main.save(main_path)
print(f"  Saved: {main_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL TABLE
# ─────────────────────────────────────────────────────────────────────────────

valid_refl = [x for x in refl_fids if x is not None]
valid_strw = [x for x in strw_fids if x is not None]
refl_mean  = np.mean(valid_refl) if valid_refl else None
refl_std   = np.std(valid_refl)  if valid_refl else None
strw_mean  = np.mean(valid_strw) if valid_strw else None
strw_std   = np.std(valid_strw)  if valid_strw else None

def fmt3(vals):
    return "  ".join(f"{v:6.1f}" if v is not None else "   N/A" for v in vals)

print(f"\n{'='*64}")
print(f"FID Results  —  3 non-overlapping COCO subsets × {N} images")
print(f"Lower FID = closer to real photos = better")
print(f"{'='*64}")
print(f"{'Method':<30} {'S1':>7} {'S2':>7} {'S3':>7}  {'Mean±Std':>12}")
print(f"{'-'*64}")

if valid_refl:
    print(f"{'ReFL (SD v1.4 full UNet)':<30} {fmt3(refl_fids)}  {refl_mean:6.1f}±{refl_std:.1f}")
else:
    print(f"{'ReFL (SD v1.4 full UNet)':<30}  N/A")

if valid_strw:
    print(f"{'STRW ours (4-head reward)':<30} {fmt3(strw_fids)}  {strw_mean:6.1f}±{strw_std:.1f}")
else:
    print(f"{'STRW ours (4-head reward)':<30}  N/A")

print(f"{'='*64}")

if refl_mean and strw_mean:
    diff = refl_mean - strw_mean
    if strw_mean < refl_mean:
        consistent = all(s < r for s, r in zip(strw_fids, refl_fids))
        print(f"✓  STRW lower FID by {diff:.1f} pts | "
              f"Consistent across all 3: {'YES' if consistent else 'PARTIAL'}")
    else:
        print(f"✗  ReFL lower FID by {abs(diff):.1f} pts")
print(f"{'='*64}")

print(f"""
Folder summary:
  images/
  ├── coco/
  │   ├── subset1/  ({len(coco_subsets[0])} images)
  │   ├── subset2/  ({len(coco_subsets[1])} images)
  │   └── subset3/  ({len(coco_subsets[2])} images)
  ├── refl/         ({len(refl_images)} images)
  ├── strw/         ({len(strw_images)} images)
  └── grids/
      ├── comparison_main.png
      ├── comparison_subset1.png
      ├── comparison_subset2.png
      └── comparison_subset3.png
""")

# save JSON
results = {
    "subset_size": N,
    "refl_fids":   refl_fids,  "strw_fids":  strw_fids,
    "refl_mean":   refl_mean,  "refl_std":   refl_std,
    "strw_mean":   strw_mean,  "strw_std":   strw_std,
    "hf_repo":     CFG["hf_repo"],
    "hf_checkpoint": CFG["hf_checkpoint"],
}
out = os.path.join(CFG["results_dir"], "fid_scores.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"Scores saved: {out}")