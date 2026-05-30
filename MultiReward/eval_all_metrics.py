"""
Full Evaluation Script — Kaggle
Computes:
  1. FID (vs COCO) for SD v1.4 baseline + all STRW checkpoints
  2. All 4 reward scores (CLIP, Aesthetic, PickScore, Quality)
  
Run:
  python /kaggle/working/eval_all_metrics.py
"""

import os, json, torch, gc
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from diffusers import StableDiffusionPipeline, DDIMScheduler
from huggingface_hub import snapshot_download, login
from kaggle_secrets import UserSecretsClient
from torchmetrics.image.fid import FrechetInceptionDistance
import subprocess, shutil

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    "hf_repo":    "mubasharjaved/strw-4head-layer-routing",
    "test_json":  "/kaggle/input/datasets/mubasharalidataai/multirewards-final/test.json",
    "n":          100,
    "steps":      50,
    "seed":       42,
    "checkpoints": ["v4_step_100", "v4_step_200", "v4_step_300", "v4_step_400", "v4_step_500"],
    "out_dir":    "/kaggle/working/eval",
    "coco_dir":   "/kaggle/working/coco/val2017",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Login ─────────────────────────────────────────────────────────────────────
login(token=UserSecretsClient().get_secret("Cutie"))

# ── Setup dirs ────────────────────────────────────────────────────────────────
os.makedirs(CFG["out_dir"], exist_ok=True)
os.makedirs(CFG["coco_dir"], exist_ok=True)

# ── Load prompts ──────────────────────────────────────────────────────────────
with open(CFG["test_json"]) as f:
    data = json.load(f)
field   = next(k for k in ["prompt", "text", "caption"] if k in data[0])
prompts = [x[field] for x in data[:CFG["n"]]]
print(f"Loaded {len(prompts)} prompts")

# ── Download COCO ─────────────────────────────────────────────────────────────
if len(os.listdir(CFG["coco_dir"])) < 100:
    print("Downloading COCO val2017...")
    os.system("curl -sL http://images.cocodataset.org/zips/val2017.zip -o /kaggle/working/coco/val2017.zip")
    os.system("cd /kaggle/working/coco && unzip -q val2017.zip && rm val2017.zip")
print(f"COCO: {len(os.listdir(CFG['coco_dir']))} images")

# ── Helper: generate images ───────────────────────────────────────────────────
def generate_images(model_path, out_dir, prompts, steps=50):
    os.makedirs(out_dir, exist_ok=True)
    existing = [f for f in os.listdir(out_dir) if f.endswith('.png')]
    if len(existing) >= len(prompts):
        print(f"  Using {len(existing)} cached images")
        return

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, safety_checker=None,
    ).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(steps)
    pipe.set_progress_bar_config(disable=True)

    for i, prompt in enumerate(tqdm(prompts)):
        out = f"{out_dir}/{i:04d}.png"
        if os.path.exists(out): continue
        with torch.no_grad():
            img = pipe(prompt, num_inference_steps=steps,
                       generator=torch.Generator().manual_seed(CFG["seed"])).images[0]
        img.save(out)

    del pipe; gc.collect(); torch.cuda.empty_cache()
    print(f"  Generated {len(os.listdir(out_dir))} images")

# ── Helper: load images ───────────────────────────────────────────────────────
def load_folder(folder, n):
    exts = ('.jpg','.jpeg','.png','.webp')
    images = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(exts):
            try: images.append(Image.open(os.path.join(folder, fname)).convert("RGB"))
            except: pass
        if len(images) >= n: break
    return images

# ── Helper: compute FID ───────────────────────────────────────────────────────
def compute_fid(real_images, gen_images):
    def to_tensor(imgs, size=299):
        t = []
        for img in imgs:
            arr = np.array(img.resize((size,size))).astype(np.uint8)
            t.append(torch.tensor(arr).permute(2,0,1))
        return torch.stack(t)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    fid.update(to_tensor(real_images).to(DEVICE), real=True)
    fid.update(to_tensor(gen_images).to(DEVICE),  real=False)
    score = float(fid.compute().item())
    del fid; gc.collect(); torch.cuda.empty_cache()
    return score

# ── Helper: score images ──────────────────────────────────────────────────────
def score_images(img_dir, prompts):
    import clip as C
    scores = {k: [] for k in ["r_align", "r_aesthetic", "r_preference", "r_quality"]}

    # CLIP
    clip_model, clip_prep = C.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    # Aesthetic
    import urllib.request
    class AestheticMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(768,1024), torch.nn.Dropout(0.2),
                torch.nn.Linear(1024,128), torch.nn.Dropout(0.2),
                torch.nn.Linear(128,64),   torch.nn.Dropout(0.1),
                torch.nn.Linear(64,16), torch.nn.Linear(16,1),
            )
        def forward(self, x): return self.layers(x)

    aes_clip, aes_prep = C.load("ViT-L/14", device=DEVICE)
    aes_clip.eval()
    w = "/tmp/aes.pth"
    if not os.path.exists(w):
        urllib.request.urlretrieve(
            "https://github.com/christophschuhmann/improved-aesthetic-predictor"
            "/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth", w)
    aes_mlp = AestheticMLP()
    aes_mlp.load_state_dict(torch.load(w, map_location="cpu"))
    aes_mlp.to(DEVICE).eval()

    # PickScore
    from transformers import AutoProcessor, AutoModel
    pick_proc  = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    pick_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").to("cpu").eval()

    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    for i, (fname, prompt) in enumerate(tqdm(zip(files, prompts), total=len(prompts))):
        img = Image.open(os.path.join(img_dir, fname)).convert("RGB")

        # r_align
        with torch.no_grad():
            i_  = clip_prep(img).unsqueeze(0).to(DEVICE)
            t_  = C.tokenize([prompt], truncate=True).to(DEVICE)
            if_ = clip_model.encode_image(i_); if_ = if_/if_.norm(dim=-1,keepdim=True)
            tf_ = clip_model.encode_text(t_);  tf_ = tf_/tf_.norm(dim=-1,keepdim=True)
            scores["r_align"].append((if_*tf_).sum().item())

        # r_aesthetic
        with torch.no_grad():
            a = aes_prep(img).unsqueeze(0).to(DEVICE)
            f = aes_clip.encode_image(a).float()
            f = f/f.norm(dim=-1,keepdim=True)
            scores["r_aesthetic"].append(aes_mlp(f).item())

        # r_preference
        with torch.no_grad():
            pi = pick_proc(images=[img], return_tensors="pt", padding=True)
            pt = pick_proc(text=[prompt], return_tensors="pt", padding=True, truncation=True)
            ie = pick_model.get_image_features(**pi)
            if hasattr(ie,'pooler_output'): ie = ie.pooler_output
            ie = ie/ie.norm(dim=-1,keepdim=True)
            te = pick_model.get_text_features(**pt)
            if hasattr(te,'pooler_output'): te = te.pooler_output
            te = te/te.norm(dim=-1,keepdim=True)
            scores["r_preference"].append((ie*te).sum().item())

        # r_quality
        g = np.array(img.convert("L")).astype(np.float32)
        gy,gx = np.gradient(g)
        lap = np.gradient(gx)[1] + np.gradient(gy)[0]
        scores["r_quality"].append(float(np.var(lap)))

    del clip_model, aes_clip, aes_mlp, pick_model
    gc.collect(); torch.cuda.empty_cache()

    return {k: round(float(np.mean(v)), 4) for k,v in scores.items()}

# ── Step 1: Generate baseline SD v1.4 images ─────────────────────────────────
print("\n=== Generating Baseline SD v1.4 images ===")
baseline_dir = f"{CFG['out_dir']}/baseline"
generate_images("CompVis/stable-diffusion-v1-4", baseline_dir, prompts, CFG["steps"])

# ── Step 2: Generate STRW images for each checkpoint ─────────────────────────
print("\n=== Downloading + Generating STRW checkpoints ===")
ckpt_dirs = {}
for ckpt in CFG["checkpoints"]:
    print(f"\nCheckpoint: {ckpt}")
    out_dir = f"{CFG['out_dir']}/{ckpt}"

    # Check if checkpoint exists on HF
    try:
        local = snapshot_download(
            repo_id=CFG["hf_repo"],
            allow_patterns=[f"{ckpt}/*"],
            local_dir=f"/kaggle/working/ckpts/{ckpt}"
        )
        ckpt_path = f"/kaggle/working/ckpts/{ckpt}/{ckpt}"
        generate_images(ckpt_path, out_dir, prompts, CFG["steps"])
        ckpt_dirs[ckpt] = out_dir
    except Exception as e:
        print(f"  Skipping {ckpt}: {e}")

# ── Step 3: Load COCO reference images ───────────────────────────────────────
print("\n=== Loading COCO reference images ===")
coco_images = load_folder(CFG["coco_dir"], CFG["n"])
print(f"COCO: {len(coco_images)} images")

# ── Step 4: Compute FID ───────────────────────────────────────────────────────
print("\n=== Computing FID scores ===")
results = {}

# Baseline FID
baseline_imgs = load_folder(baseline_dir, CFG["n"])
baseline_fid  = compute_fid(coco_images, baseline_imgs)
results["baseline"] = {"fid": baseline_fid}
print(f"Baseline SD v1.4 FID: {baseline_fid:.2f}")

# STRW checkpoints FID
for ckpt, out_dir in ckpt_dirs.items():
    strw_imgs = load_folder(out_dir, CFG["n"])
    fid = compute_fid(coco_images, strw_imgs)
    results[ckpt] = {"fid": fid}
    print(f"{ckpt} FID: {fid:.2f}")

# ── Step 5: Score all images with 4 rewards ───────────────────────────────────
print("\n=== Scoring images with 4 rewards ===")

print("Scoring baseline...")
baseline_scores = score_images(baseline_dir, prompts)
results["baseline"]["scores"] = baseline_scores
print(f"Baseline scores: {baseline_scores}")

for ckpt, out_dir in ckpt_dirs.items():
    print(f"Scoring {ckpt}...")
    scores = score_images(out_dir, prompts)
    results[ckpt]["scores"] = scores
    print(f"{ckpt} scores: {scores}")

# ── Step 6: Results table ─────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"FULL EVALUATION RESULTS — {CFG['n']} images")
print(f"{'='*80}")
print(f"{'Model':<25} {'FID':>8} {'r_align':>10} {'r_aes':>8} {'r_pref':>8} {'r_qual':>10}")
print(f"{'-'*80}")

all_models = ["baseline"] + list(ckpt_dirs.keys())
for model in all_models:
    if model not in results: continue
    fid = results[model].get("fid", 0)
    sc  = results[model].get("scores", {})
    print(f"  {model:<23} {fid:>8.2f} "
          f"{sc.get('r_align',0):>10.4f} "
          f"{sc.get('r_aesthetic',0):>8.4f} "
          f"{sc.get('r_preference',0):>8.4f} "
          f"{sc.get('r_quality',0):>10.2f}")

print(f"{'='*80}")
print("Lower FID = better | Higher reward scores = better")

# Save
with open(f"{CFG['out_dir']}/full_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {CFG['out_dir']}/full_results.json")

# Upload to HF
from huggingface_hub import HfApi
api = HfApi()
try:
    api.upload_file(
        path_or_fileobj=f"{CFG['out_dir']}/full_results.json",
        path_in_repo="eval/full_results_v4.json",
        repo_id=CFG["hf_repo"]
    )
    print("Uploaded to HF ✅")
except Exception as e:
    print(f"HF upload failed: {e}")
