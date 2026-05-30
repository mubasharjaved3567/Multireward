"""
plot_logs.py
Run locally from MultiReward/ folder:
    python plot_logs.py
    python plot_logs.py --log train_log.jsonl --out plots/
"""
import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

def rolling(vals, w=20):
    return np.convolve(vals, np.ones(w)/w, mode='valid')

def load_log(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} steps from {path}")
    return records

def plot_reward_weights(records, steps, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Reward Weight Activity per Training Step\n(smooth_weights driving the loss)',
                 fontsize=13, fontweight='bold')
    metrics = [
        ('sw_align',      'r_align → cross_attn',            '#3b82f6', axes[0,0]),
        ('sw_aesthetic',  'r_aesthetic → self_attn + conv',   '#f59e0b', axes[0,1]),
        ('sw_preference', 'r_preference → self + cross_attn', '#10b981', axes[1,0]),
        ('sw_quality',    'r_quality → resnets',              '#8b5cf6', axes[1,1]),
    ]
    for key, label, color, ax in metrics:
        vals = [r[key] for r in records]
        ax.plot(steps, vals, color=color, lw=0.8, alpha=0.4)
        if len(vals) > 20:
            ax.plot(steps[19:], rolling(vals), color=color, lw=2.5, label='20-step avg')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Weight')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, 'reward_weights.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

def plot_grad_norms(records, steps, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Gradient Norms per Layer Group\n(layer routing effect)',
                 fontsize=13, fontweight='bold')
    gnorms = [
        ('gnorm_cross_attn', 'cross_attn (r_align target)',   '#3b82f6', axes[0,0]),
        ('gnorm_self_attn',  'self_attn (r_aesthetic target)', '#f59e0b', axes[0,1]),
        ('gnorm_conv',       'conv (secondary target)',        '#10b981', axes[1,0]),
        ('gnorm_resnets',    'resnets (r_quality target)',     '#8b5cf6', axes[1,1]),
    ]
    for key, label, color, ax in gnorms:
        vals = [r[key] for r in records]
        ax.plot(steps, vals, color=color, lw=0.8, alpha=0.4)
        if len(vals) > 20:
            ax.plot(steps[19:], rolling(vals), color=color, lw=2.5, label='20-step avg')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Grad norm')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, 'grad_norms.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

def plot_training_progress(records, steps, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle('Training Progress', fontsize=13, fontweight='bold')

    R  = [r['composite_R'] for r in records]
    kl = [r['kl']          for r in records]

    ax1.plot(steps, R,  color='#3b82f6', lw=0.8, alpha=0.4)
    if len(R) > 20:
        ax1.plot(steps[19:], rolling(R), color='#3b82f6', lw=2.5, label='20-step avg')
    ax1.set_title('Composite Reward R')
    ax1.set_xlabel('Step')
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(steps, kl, color='#ef4444', lw=0.8, alpha=0.4)
    if len(kl) > 20:
        ax2.plot(steps[19:], rolling(kl), color='#ef4444', lw=2.5, label='20-step avg')
    ax2.set_title('KL Penalty')
    ax2.set_xlabel('Step')
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, 'training_progress.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

def plot_reward_by_denoising(records, out_dir):
    by_idx = defaultdict(list)
    for r in records:
        by_idx[r['reward_idx']].append(r)

    denoising_steps = sorted(by_idx.keys())
    dominant_by_step = []
    for idx in denoising_steps:
        block = by_idx[idx]
        dominant_by_step.append({
            'r_align':      np.mean([r['sw_align']      for r in block]),
            'r_aesthetic':  np.mean([r['sw_aesthetic']  for r in block]),
            'r_preference': np.mean([r['sw_preference'] for r in block]),
            'r_quality':    np.mean([r['sw_quality']    for r in block]),
        })

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#3b82f6', '#f59e0b', '#10b981', '#8b5cf6']
    labels = ['r_align (cross_attn)', 'r_aesthetic (self_attn)',
              'r_preference (self+cross)', 'r_quality (resnets)']
    keys   = ['r_align', 'r_aesthetic', 'r_preference', 'r_quality']

    for key, label, color in zip(keys, labels, colors):
        vals = [d[key] for d in dominant_by_step]
        ax.plot(denoising_steps, vals, color=color, lw=2.5,
                label=label, marker='o', ms=4)

    ax.set_xlabel(f'Denoising step index (0=early/noisy, {max(denoising_steps)}=late/clean)')
    ax.set_ylabel('Average smooth weight')
    ax.set_title('Reward Activity by Denoising Position\n'
                 'r_align high early → r_quality high late (as designed)')
    ax.legend(loc='center right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'reward_by_denoising_step.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

def print_summary(records):
    print(f"\n{'='*70}")
    print(f"Dominant Reward per Training Block")
    print(f"{'='*70}")
    print(f"{'Steps':<12} {'r_align':>10} {'r_aesthetic':>12} {'r_preference':>14} {'r_quality':>12}")
    print(f"{'-'*70}")
    chunk = max(1, len(records) // 5)
    for c in range(5):
        block  = records[c*chunk:(c+1)*chunk]
        if not block: continue
        counts = Counter()
        for r in block:
            sw = {'r_align': r['sw_align'], 'r_aesthetic': r['sw_aesthetic'],
                  'r_preference': r['sw_preference'], 'r_quality': r['sw_quality']}
            counts[max(sw, key=sw.get)] += 1
        s, e = c*chunk+1, (c+1)*chunk
        print(f"  {s}-{e:<8} "
              f"{counts['r_align']:>10} "
              f"{counts['r_aesthetic']:>12} "
              f"{counts['r_preference']:>14} "
              f"{counts['r_quality']:>12}")
    print(f"{'='*70}")

    print(f"\nReward score trends (first 100 vs last 100 steps):")
    print(f"{'Metric':<22} {'First 100':>12} {'Last 100':>12} {'Change':>10}")
    print(f"{'-'*58}")
    for key in ['r_align_norm', 'r_aesthetic_norm', 'r_preference_norm', 'r_quality_norm']:
        first = np.mean([r[key] for r in records[:100]])
        last  = np.mean([r[key] for r in records[-100:]])
        delta = last - first
        flag  = '↑' if delta > 0 else '↓'
        print(f"  {key:<20} {first:>12.4f} {last:>12.4f} {delta:>+9.4f} {flag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='train_log.jsonl', help='Path to train_log.jsonl')
    parser.add_argument('--out', default='plots',           help='Output directory for plots')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    records = load_log(args.log)
    steps   = [r['global_step'] for r in records]

    plot_reward_weights(records, steps, args.out)
    plot_grad_norms(records, steps, args.out)
    plot_training_progress(records, steps, args.out)
    plot_reward_by_denoising(records, args.out)
    print_summary(records)

    print(f"\nAll plots saved to {args.out}/")


if __name__ == '__main__':
    main()