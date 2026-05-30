# ══════════════════════════════════════════════════════════════════════════════
# reward_eval.py  —  Reward Score Evaluation (Kaggle + Local)
# ══════════════════════════════════════════════════════════════════════════════
#
# KAGGLE:  set PLOT_ONLY = False  → scores images, saves JSON + plot
# LOCAL:   set PLOT_ONLY = True   → reads existing JSON, saves plot
#
# ══════════════════════════════════════════════════════════════════════════════

import os, json, sys
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# ── CONFIG ────────────────────────────────────────────────────────────────────
PLOT_ONLY = False   # True = local plot only | False = full scoring on Kaggle
N         = 100     # number of images to score
# ══════════════════════════════════════════════════════════════════════════════

# ── Environment ───────────────────────────────────────────────────────────────
ON_KAGGLE  = os.path.exists('/kaggle/working')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if ON_KAGGLE:
    WORKING_DIR = '/kaggle/working'
    REFL_DIR    = '/kaggle/input/datasets/mubasharalidataai/final-training-v1/test_images/test_images'
    STRW_DIR    = '/kaggle/working/strw_images'
    TEST_JSON   = '/kaggle/input/datasets/mubasharalidataai/final-training-v1/test.json'
    RESULTS_DIR = '/kaggle/working/results'
    PLOTS_DIR   = '/kaggle/working/plots'
else:
    WORKING_DIR = SCRIPT_DIR
    REFL_DIR    = os.path.join(SCRIPT_DIR, 'images', 'refl')
    STRW_DIR    = os.path.join(SCRIPT_DIR, 'images', 'strw')
    TEST_JSON   = os.path.join(SCRIPT_DIR, 'test.json')
    RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
    PLOTS_DIR   = os.path.join(SCRIPT_DIR, 'plots')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

RESULTS_JSON = os.path.join(RESULTS_DIR, 'reward_scores.json')
METRICS      = ['r_align', 'r_aesthetic', 'r_preference', 'r_quality']

print(f"{'='*55}")
print(f"  Reward Eval | {'Kaggle' if ON_KAGGLE else 'Local'} | plot-only={PLOT_ONLY}")
print(f"  results     : {RESULTS_JSON}")
print(f"{'='*55}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SHARED: plot + summary function
# ══════════════════════════════════════════════════════════════════════════════
def make_plot(refl_scores, strw_scores, plot_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    refl_means = {m: np.mean([s[m] for s in refl_scores]) for m in METRICS}
    strw_means = {m: np.mean([s[m] for s in strw_scores]) for m in METRICS}
    win_rates  = {m: sum(1 for i in range(len(strw_scores))
                         if strw_scores[i][m] > refl_scores[i][m])
                  for m in METRICS}

    x     = np.arange(len(METRICS))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Reward Score Comparison — ReFL vs STRW (MultiReward)',
                 fontsize=14, fontweight='bold')

    # Mean scores bar chart
    ax = axes[0]
    r_vals = [refl_means[m] for m in METRICS]
    s_vals = [strw_means[m] for m in METRICS]
    b1 = ax.bar(x - width/2, r_vals, width, label='ReFL (baseline)', color='#4C72B0', alpha=0.85)
    b2 = ax.bar(x + width/2, s_vals, width, label='STRW (ours)',     color='#DD8452', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(METRICS, rotation=10)
    ax.set_ylabel('Mean Reward Score (↑ higher is better)')
    ax.set_title('Mean Reward by Metric')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.bar_label(b1, fmt='%.3f', padding=3, fontsize=8)
    ax.bar_label(b2, fmt='%.3f', padding=3, fontsize=8)
    ax.set_ylim(0, max(r_vals + s_vals) * 1.25)

    # Win rate bar chart
    ax2    = axes[1]
    w_vals = [win_rates[m] / len(strw_scores) * 100 for m in METRICS]
    colors = ['#2ca02c' if w > 50 else '#d62728' for w in w_vals]
    b3 = ax2.bar(METRICS, w_vals, color=colors, alpha=0.85)
    ax2.axhline(50, color='black', linewidth=0.8, linestyle='--', label='50% baseline')
    ax2.set_ylabel('STRW Win Rate (%)')
    ax2.set_title('Per-Prompt Win Rate (STRW > ReFL)')
    ax2.set_ylim(0, 110)
    ax2.bar_label(b3, fmt='%.0f%%', padding=3, fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(handles=[
        mpatches.Patch(color='#2ca02c', label='STRW wins (>50%)'),
        mpatches.Patch(color='#d62728', label='ReFL wins (<50%)'),
    ], fontsize=8)
    ax2.set_xticklabels(METRICS, rotation=10)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Plot saved: {plot_path}")

    # Print summary table
    n = len(strw_scores)
    print(f"\n{'='*65}")
    print(f"Reward Score Comparison ({n} images) — Higher = Better")
    print(f"{'='*65}")
    print(f"{'Metric':<20} {'ReFL':>10} {'STRW':>10} {'Winner':>10}")
    print(f"{'-'*65}")
    for m in METRICS:
        r = refl_means[m]; s = strw_means[m]
        w = 'STRW ✓' if s > r else 'ReFL'
        print(f"  {m:<18} {r:>10.4f} {s:>10.4f} {w:>10}")
    print(f"{'='*65}")
    print(f"\nPer-prompt win rates:")
    for m in METRICS:
        wins = win_rates[m]
        print(f"  {m:<20} STRW wins {wins}/{n} ({wins/n*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT-ONLY MODE (local)
# ══════════════════════════════════════════════════════════════════════════════
if PLOT_ONLY:
    if not os.path.exists(RESULTS_JSON):
        print(f"[ERROR] Not found: {RESULTS_JSON}")
        print(f"  Download reward_scores.json from Kaggle → {RESULTS_DIR}/")
        sys.exit(1)

    with open(RESULTS_JSON) as f:
        data = json.load(f)

    refl_scores = data['refl_scores']
    strw_scores = data['strw_scores']
    print(f"[✓] Loaded: {len(refl_scores)} ReFL + {len(strw_scores)} STRW scores")

    make_plot(refl_scores, strw_scores,
              os.path.join(PLOTS_DIR, 'reward_plot.png'))
    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
# FULL COMPUTE MODE (Kaggle)
# ══════════════════════════════════════════════════════════════════════════════
from PIL import Image

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

# ── Load prompts ──────────────────────────────────────────────────────────────
print("[1/4] Loading prompts...")

# Try both test.json locations
test_json_candidates = [
    '/kaggle/input/datasets/mubasharalidataai/final-training-v1/test.json',
    os.path.join(WORKING_DIR, 'data', 'test.json'),
    TEST_JSON,
]
test_json_path = next((p for p in test_json_candidates if os.path.exists(p)), None)
if test_json_path is None:
    print("  [WARN] test.json not found — using generic prompts")
    prompts = [f"a high quality photo of a beautiful scene {i}" for i in range(N)]
else:
    with open(test_json_path) as f:
        test_data = json.load(f)
    field   = 'prompt' if 'prompt' in test_data[0] else 'text'
    prompts = [x[field] for x in test_data[:N]]
    print(f"  Loaded {len(prompts)} prompts from {test_json_path}")

# ── Load images ───────────────────────────────────────────────────────────────
print("\n[2/4] Loading images...")
refl_images = load_nested(REFL_DIR, N)
strw_images = load_folder(STRW_DIR, N)
print(f"  ReFL : {len(refl_images)}")
print(f"  STRW : {len(strw_images)}")

if len(refl_images) == 0 or len(strw_images) == 0:
    print("[ERROR] Missing images — check REFL_DIR / STRW_DIR")
    sys.exit(1)

n = min(N, len(refl_images), len(strw_images), len(prompts))
refl_images = refl_images[:n]
strw_images = strw_images[:n]
prompts     = prompts[:n]
print(f"  Scoring {n} images each")

# ── Load reward heads ─────────────────────────────────────────────────────────
print("\n[3/4] Loading reward heads...")
sys.path.append(WORKING_DIR)
from MultiReward.trainings.train_ddpo_v0.reward_heads import RewardHeads, norm_scores
device = "cuda"
heads  = RewardHeads(device)

# ── Score images ──────────────────────────────────────────────────────────────
print("\n[4/4] Scoring images...")

print("Scoring STRW images...")
strw_scores = []
for i in range(n):
    sc = heads.score(prompts[i], strw_images[i])
    strw_scores.append(norm_scores(sc))
    if (i + 1) % 10 == 0:
        print(f"  STRW: {i+1}/{n}")

print("\nScoring ReFL images...")
refl_scores = []
for i in range(n):
    sc = heads.score(prompts[i], refl_images[i])
    refl_scores.append(norm_scores(sc))
    if (i + 1) % 10 == 0:
        print(f"  ReFL: {i+1}/{n}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
results = {"refl_scores": refl_scores, "strw_scores": strw_scores}
with open(RESULTS_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n[✓] Saved: {RESULTS_JSON}")

# ── Plot + summary ────────────────────────────────────────────────────────────
make_plot(refl_scores, strw_scores,
          os.path.join(PLOTS_DIR, 'reward_plot.png'))

print(f"""
─── Download from Kaggle ──────────────────────────
  {RESULTS_JSON}
  {os.path.join(PLOTS_DIR, 'reward_plot.png')}
───────────────────────────────────────────────────
""")