# ══════════════════════════════════════════════════════════════════════════════
# fid_eval.py  —  FID Evaluation (Kaggle + Local)
# ══════════════════════════════════════════════════════════════════════════════
#
# KAGGLE:  set PLOT_ONLY = False  → computes FID, saves JSON + plot
# LOCAL:   set PLOT_ONLY = True   → reads existing JSON, saves plot
#
# ══════════════════════════════════════════════════════════════════════════════

import os, json, sys, subprocess

# ══════════════════════════════════════════════════════════════════════════════
# ── CONFIG ────────────────────────────────────────────────────────────────────
PLOT_ONLY = True    # True = local plot only | False = full FID compute
N         = 100     # number of images
# ══════════════════════════════════════════════════════════════════════════════

# ── Environment ───────────────────────────────────────────────────────────────
ON_KAGGLE  = os.path.exists('/kaggle/working')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if ON_KAGGLE:
    REFL_DIR    = '/kaggle/input/datasets/mubasharalidataai/final-training-v1/test_images/test_images'
    STRW_DIR    = '/kaggle/working/strw_images'
    COCO_DIR    = '/kaggle/working/coco/subset1'
    DIFFDB_DIR  = '/kaggle/working/diffusiondb_ref'
    RESULTS_DIR = '/kaggle/working/results'
    PLOTS_DIR   = '/kaggle/working/plots'
else:
    REFL_DIR    = os.path.join(SCRIPT_DIR, 'images', 'refl')
    STRW_DIR    = os.path.join(SCRIPT_DIR, 'images', 'strw')
    COCO_DIR    = os.path.join(SCRIPT_DIR, 'images', 'coco')
    DIFFDB_DIR  = os.path.join(SCRIPT_DIR, 'images', 'diffusiondb_ref')
    RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
    PLOTS_DIR   = os.path.join(SCRIPT_DIR, 'plots')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

RESULTS_JSON = os.path.join(RESULTS_DIR, 'fid_results.json')

print(f"{'='*55}")
print(f"  FID Eval | {'Kaggle' if ON_KAGGLE else 'Local'} | plot-only={PLOT_ONLY}")
print(f"  results  : {RESULTS_JSON}")
print(f"  plots    : {PLOTS_DIR}")
print(f"{'='*55}\n")

# ══════════════════════════════════════════════════════════════════════════════
# SHARED: plot function — works from any results dict
# ══════════════════════════════════════════════════════════════════════════════
def make_plot(results, plot_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # Only plot keys that exist in results
    all_pairs = [
        ('COCO val2017', 'coco_refl',   'coco_strw'),
        ('DiffusionDB',  'diffdb_refl', 'diffdb_strw'),
        ('Aesthetic',    'aes_refl',    'aes_strw'),
    ]
    pairs = [(n, rk, sk) for n, rk, sk in all_pairs
             if results.get(rk) is not None and results.get(sk) is not None]

    if not pairs:
        print("[ERROR] No valid result pairs to plot.")
        return

    refs        = [n  for n, _, _ in pairs]
    refl_scores = [results[rk] for _, rk, _ in pairs]
    strw_scores = [results[sk] for _, _, sk in pairs]

    x     = np.arange(len(refs))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('FID Evaluation — ReFL vs STRW (MultiReward)',
                 fontsize=14, fontweight='bold')

    # Bar chart
    ax = axes[0]
    b1 = ax.bar(x - width/2, refl_scores, width,
                label='ReFL (baseline)', color='#4C72B0', alpha=0.85)
    b2 = ax.bar(x + width/2, strw_scores, width,
                label='STRW (ours)',     color='#DD8452', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(refs)
    ax.set_ylabel('FID Score (↓ lower is better)')
    ax.set_title('FID by Reference Dataset')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.bar_label(b1, fmt='%.1f', padding=3, fontsize=9)
    ax.bar_label(b2, fmt='%.1f', padding=3, fontsize=9)
    ax.set_ylim(0, max(refl_scores + strw_scores) * 1.2)

    # Delta chart
    ax2    = axes[1]
    deltas = [s - r for r, s in zip(refl_scores, strw_scores)]
    colors = ['#2ca02c' if d < 0 else '#d62728' for d in deltas]
    b3     = ax2.bar(refs, deltas, color=colors, alpha=0.85)
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_ylabel('Δ FID  (STRW − ReFL)  ↓ negative = STRW better')
    ax2.set_title('STRW Improvement over ReFL')
    ax2.bar_label(b3, fmt='%+.1f', padding=3, fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(handles=[
        mpatches.Patch(color='#2ca02c', label='STRW better'),
        mpatches.Patch(color='#d62728', label='ReFL better'),
    ], fontsize=8)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Plot saved: {plot_path}")

    # Summary table
    print(f"\n{'='*65}")
    print(f"{'Reference':<22} {'ReFL':>10} {'STRW':>10} {'Δ':>10} {'Winner':>10}")
    print(f"{'-'*65}")
    for name, rk, sk in pairs:
        r, s = results[rk], results[sk]
        d    = s - r
        w    = 'STRW ✓' if s < r else 'ReFL'
        print(f"  {name:<20} {r:>10.2f} {s:>10.2f} {d:>+10.2f} {w:>10}")
    wins = sum(1 for _, rk, sk in pairs if results[sk] < results[rk])
    print(f"{'='*65}")
    print(f"STRW wins on {wins}/{len(pairs)} reference datasets\n")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT-ONLY MODE
# ══════════════════════════════════════════════════════════════════════════════
if PLOT_ONLY:
    if not os.path.exists(RESULTS_JSON):
        print(f"[ERROR] Not found: {RESULTS_JSON}")
        print(f"  Download fid_results.json from Kaggle and place it in:")
        print(f"  {RESULTS_DIR}/")
        sys.exit(1)

    with open(RESULTS_JSON) as f:
        results = json.load(f)
    print(f"[✓] Loaded results: {list(results.keys())}")

    plot_path = os.path.join(PLOTS_DIR, 'fid_plot.png')
    make_plot(results, plot_path)
    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
# FULL COMPUTE MODE (Kaggle)
# ══════════════════════════════════════════════════════════════════════════════
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'torch-fidelity', 'torchmetrics', 'datasets'], check=True)

import importlib, torchmetrics.image.fid
importlib.reload(torchmetrics.image.fid)
from torchmetrics.image.fid import FrechetInceptionDistance
import torch, numpy as np
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}\n")

def pil_to_tensor(images, size=299):
    tensors = []
    for img in images:
        arr = np.array(img.resize((size, size))).astype(np.uint8)
        tensors.append(torch.tensor(arr).permute(2, 0, 1))
    return torch.stack(tensors)

def compute_fid(real_imgs, gen_imgs, label=''):
    if len(real_imgs) < 2 or len(gen_imgs) < 2:
        print(f"  [SKIP] {label} — not enough images")
        return None
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    fid.update(pil_to_tensor(real_imgs).to(DEVICE), real=True)
    fid.update(pil_to_tensor(gen_imgs).to(DEVICE),  real=False)
    score = float(fid.compute().item())
    del fid; torch.cuda.empty_cache()
    return score

def load_folder(folder, n):
    exts = ('.jpg', '.jpeg', '.png', '.webp')
    imgs = []
    if not os.path.isdir(folder):
        print(f"  [WARN] Not found: {folder}"); return imgs
    for fname in sorted(os.listdir(folder))[:n*2]:
        if fname.lower().endswith(exts):
            try: imgs.append(Image.open(os.path.join(folder, fname)).convert('RGB'))
            except: pass
        if len(imgs) >= n: break
    return imgs

def load_nested(folder, n):
    exts = ('.jpg', '.jpeg', '.png', '.webp')
    imgs = []
    if not os.path.isdir(folder):
        print(f"  [WARN] Not found: {folder}"); return imgs
    for root, _, files in os.walk(folder):
        for fname in sorted(files):
            if fname.lower().endswith(exts):
                try: imgs.append(Image.open(os.path.join(root, fname)).convert('RGB'))
                except: pass
            if len(imgs) >= n: return imgs
    return imgs

print("[1/3] Loading images...")
refl_imgs  = load_nested(REFL_DIR,  N)
strw_imgs  = load_folder(STRW_DIR,  N)
coco_imgs  = load_folder(COCO_DIR,  N)
diffdb_imgs = load_folder(DIFFDB_DIR, N)
print(f"  ReFL={len(refl_imgs)}  STRW={len(strw_imgs)}  COCO={len(coco_imgs)}  DiffDB={len(diffdb_imgs)}")

print("\n[2/3] Computing FID...")
results = {}
for key, ref, gen, label in [
    ('coco_refl',   coco_imgs,   refl_imgs,  'COCO   vs ReFL'),
    ('coco_strw',   coco_imgs,   strw_imgs,  'COCO   vs STRW'),
    ('diffdb_refl', diffdb_imgs, refl_imgs,  'DiffDB vs ReFL'),
    ('diffdb_strw', diffdb_imgs, strw_imgs,  'DiffDB vs STRW'),
]:
    print(f"  {label}...")
    results[key] = compute_fid(ref, gen[:N], label)
    print(f"    → {results[key]:.2f}" if results[key] else "    → skipped")

with open(RESULTS_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n[✓] Saved: {RESULTS_JSON}")

print("\n[3/3] Plotting...")
make_plot(results, os.path.join(PLOTS_DIR, 'fid_plot.png'))

print(f"\n─── Download from Kaggle ──────────────────────────")
print(f"  {RESULTS_JSON}")
print(f"  {os.path.join(PLOTS_DIR, 'fid_plot.png')}")
print(f"───────────────────────────────────────────────────")