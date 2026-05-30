# compare_local.py
import json, os, torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
import matplotlib
import matplotlib.pyplot as plt

DEVICE    = "cpu"   # change to "cuda" if you have GPU
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_DIR  = os.path.join(SCRIPT_DIR, "checkpoint_step900")   # download from HF once
TEST_JSON = os.path.join(SCRIPT_DIR, "test.json")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Download checkpoint once if missing ───────────────────────────────────────
if not os.path.exists(CKPT_DIR):
    print("Downloading checkpoint_step900 from HuggingFace (one time)...")
    from huggingface_hub import snapshot_download
    local = snapshot_download(
        repo_id="mubasharjaved/strw-4head-layer-routing",
        allow_patterns=["checkpoint_step900/*"]
    )
    import shutil
    shutil.copytree(os.path.join(local, "checkpoint_step900"), CKPT_DIR)
    print(f"✓ Saved to {CKPT_DIR}")

# ── Load prompts ──────────────────────────────────────────────────────────────
with open(TEST_JSON) as f:
    test_data = json.load(f)
field   = 'prompt' if 'prompt' in test_data[0] else 'text'
prompts = [x[field] for x in test_data[:10]]
print(f"Loaded {len(prompts)} prompts")

# ── Load STRW pipeline ────────────────────────────────────────────────────────
print("Loading STRW pipeline...")
strw_pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32,
    safety_checker=None
)
strw_pipe.unet = UNet2DConditionModel.from_pretrained(
    os.path.join(CKPT_DIR, "unet"),
    torch_dtype=torch.float32
)
strw_pipe.scheduler = DDIMScheduler.from_config(strw_pipe.scheduler.config)
strw_pipe = strw_pipe.to(DEVICE)
strw_pipe.set_progress_bar_config(disable=True)

# ── Load ReFL pipeline ────────────────────────────────────────────────────────
print("Loading ReFL pipeline...")
refl_pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32,
    safety_checker=None
)
refl_pipe.scheduler = DDIMScheduler.from_config(refl_pipe.scheduler.config)
refl_pipe = refl_pipe.to(DEVICE)
refl_pipe.set_progress_bar_config(disable=True)

# ── Generate images ───────────────────────────────────────────────────────────
print("Generating 10 pairs (this may take a while on CPU)...")
refl_imgs = []
strw_imgs = []

for i, prompt in enumerate(prompts):
    print(f"  [{i+1}/10] {prompt[:60]}...")
    with torch.no_grad():
        r = refl_pipe([prompt], num_inference_steps=20, guidance_scale=7.5,
                      height=512, width=512,
                      generator=torch.Generator(DEVICE).manual_seed(42)).images[0]
        s = strw_pipe([prompt], num_inference_steps=20, guidance_scale=7.5,
                      height=512, width=512,
                      generator=torch.Generator(DEVICE).manual_seed(42)).images[0]
    refl_imgs.append(r)
    strw_imgs.append(s)

# ── Plot ──────────────────────────────────────────────────────────────────────
print("Creating plot...")
fig, axes = plt.subplots(10, 3, figsize=(15, 50),
                          gridspec_kw={'width_ratios': [2, 3, 3]})
fig.suptitle('ReFL vs STRW — Same Prompt, Same Seed (42)',
             fontsize=16, fontweight='bold', y=1.001)

for i in range(10):
    p      = prompts[i]
    p_wrap = '\n'.join([p[j:j+40] for j in range(0, min(len(p), 120), 40)])

    axes[i, 0].axis('off')
    axes[i, 0].text(0.5, 0.5, f"Prompt {i+1}:\n\n{p_wrap}",
                    ha='center', va='center', fontsize=8,
                    transform=axes[i, 0].transAxes,
                    bbox=dict(boxstyle='round', facecolor='#f0f4ff', alpha=0.8))

    axes[i, 1].imshow(refl_imgs[i])
    axes[i, 1].axis('off')
    axes[i, 1].set_title('ReFL (baseline)', color='#4C72B0',
                          fontsize=10, fontweight='bold')

    axes[i, 2].imshow(strw_imgs[i])
    axes[i, 2].axis('off')
    axes[i, 2].set_title('STRW (ours)', color='#DD8452',
                          fontsize=10, fontweight='bold')

plt.tight_layout()
out = os.path.join(PLOTS_DIR, "prompts_compare.png")
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.show()
print(f"✓ Saved: {out}")