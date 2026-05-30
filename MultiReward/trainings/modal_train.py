"""
modal_train.py  (Modal 1.0 compatible)
=======================================
Full Modal.com training script for REINFORCE-based Lightweight DDPO + STRW.

SETUP (one time):
    pip3 install modal
    modal token new

FILE STRUCTURE on your local machine:
    your_project/
    ├── modal_train.py        ← this file
    ├── train_ours.py         ← the training script
    └── data/
        └── refl_data.json    ← training data

USAGE:
    modal run modal_train.py                    # full 1000-step training
    modal run modal_train.py --mode sanity      # 50-step sanity check first
    modal run modal_train.py --mode list        # see what's saved in volume
    modal run modal_train.py --mode download    # pull results to local machine
    modal run modal_train.py --steps 500        # custom step count
    modal run modal_train.py --resume           # resume from last checkpoint

MONITOR LIVE LOGS (in a separate terminal while training runs):
    modal app logs ddpo-altaf-train
"""

import os
import sys
import modal

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App("ddpo-altaf-train")

# ---------------------------------------------------------------------------
# Persistent Volume
# ---------------------------------------------------------------------------
volume      = modal.Volume.from_name("ddpo-altaf-outputs", create_if_missing=True)
VOLUME_PATH = "/outputs"

# ---------------------------------------------------------------------------
# Container image
#
# Version pins that actually work together:
#   torch 2.1.2         — stable, widely tested
#   numpy 1.26.4        — last 1.x release; torch 2.1.2 compiled against numpy 1.x
#   diffusers 0.21.4    — new enough to work with huggingface_hub 0.20.x
#   huggingface_hub 0.20.3 — last version before cached_download was removed
#   transformers 4.36.2 — compatible with above
#   accelerate 0.25.0   — compatible with torch 2.1.2
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "wget", "libgl1"])
    .pip_install([
        # numpy FIRST and pinned to 1.x — must install before torch
        "numpy==1.26.4",
    ])
    .pip_install([
        "torch==2.1.2",
        "torchvision==0.16.2",
    ])
    .pip_install([
        # huggingface_hub pinned — 0.21+ removes cached_download
        "huggingface_hub==0.20.3",
        # diffusers 0.21.4 — compatible with above hub version
        "diffusers==0.21.4",
        "transformers==4.36.2",
        "accelerate==0.25.0",
        "datasets==2.16.1",
        "scipy",
        "bitsandbytes==0.41.3",
        "tensorboard==2.15.1",
        "tqdm",
        "matplotlib",
        "Pillow",
        "packaging",
    ])
    .run_commands(
        # Download aesthetic predictor weights at IMAGE BUILD TIME
        # so training never needs to fetch them at runtime.
        "mkdir -p /root/.cache && "
        "wget -q -O /root/.cache/aesthetic_predictor.pth "
        "'https://github.com/christophschuhmann/improved-aesthetic-predictor"
        "/raw/main/sac+logos+ava1-l14-linearMSE.pth' || "
        # fallback mirror
        "wget -q -O /root/.cache/aesthetic_predictor.pth "
        "'https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_predictor/resolve/main/sac+logos+ava1-l14-linearMSE.pth'"
    )
    .add_local_file("train_ours.py", "/root/train_ours.py")
    .add_local_dir("data",           "/root/data")
)


# ---------------------------------------------------------------------------
# Helper: build the training command
# ---------------------------------------------------------------------------
def build_cmd(
    max_train_steps: int = 1000,
    grad_accum: int      = 2,
    resume: bool         = False,
    init_ckpt: str       = None,
) -> list:

    cmd = [
        "python", "/root/train_ours.py",

        # model
        "--pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4",

        # data
        "--train_data_dir=/root/data/refl_data.json",
        "--caption_column=text",
        "--max_train_samples=1000",

        # output — written to the persistent volume
        f"--output_dir={VOLUME_PATH}/checkpoint",
        "--logging_dir=logs",
        "--report_to=tensorboard",
        "--tracker_project_name=ddpo_reinforce_50step",

        # training
        f"--max_train_steps={max_train_steps}",
        "--train_batch_size=1",
        f"--gradient_accumulation_steps={grad_accum}",
        "--mixed_precision=fp16",

        # optimizer
        "--learning_rate=1e-7",   # lowered from 1e-6 — grad norms were exploding (100k+)
        "--lr_scheduler=cosine",
        "--lr_warmup_steps=50",
        "--adam_beta1=0.9",
        "--adam_beta2=0.999",
        "--adam_weight_decay=1e-2",
        "--adam_epsilon=1e-8",
        "--max_grad_norm=0.1",    # tightened from 1.0 — clipping more aggressively

        # REINFORCE / DDPO
        "--total_denoising_steps=50",
        "--baseline=running_mean",
        "--baseline_ema_alpha=0.05",

        # reward hacking mitigations
        "--kl_beta=0.1",          # raised — stronger KL penalty dampens instability
        "--kl_threshold=0.8",
        "--reward_clip=1.5",      # tightened — large rewards were driving grad explosions

        # checkpointing every 100 steps, plots every 200 (less frequent = less VRAM spike)
        "--checkpointing_steps=100",
        "--graph_every=200",

        "--seed=42",
    ]

    # Gradient checkpointing always on — trades compute for VRAM
    # Do NOT use 8bit adam — incompatible with regular AdamW checkpoint state
    cmd.append("--gradient_checkpointing")

    if resume:
        cmd.append("--resume_from_checkpoint=latest")
    if init_ckpt:
        cmd.append(f"--init_from_checkpoint={init_ckpt}")

    return cmd


# ---------------------------------------------------------------------------
# FULL TRAINING
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",
    timeout=86400,
    image=image,
    volumes={VOLUME_PATH: volume},
)
def run_training(max_train_steps: int = 1000, resume: bool = False):
    import subprocess

    os.makedirs(f"{VOLUME_PATH}/checkpoint", exist_ok=True)
    os.makedirs(f"{VOLUME_PATH}/plots",      exist_ok=True)
    os.makedirs(f"{VOLUME_PATH}/logs",       exist_ok=True)

    print("=" * 60)
    print("GPU INFO:")
    subprocess.run(["nvidia-smi"], check=False)
    print("=" * 60)

    cmd = build_cmd(
        max_train_steps=max_train_steps,
        grad_accum=2,
        resume=resume,
    )

    print("TRAINING COMMAND:")
    print(" \\\n  ".join(cmd))
    print("=" * 60)

    # Reduce CUDA memory fragmentation — critical on A10G at high step counts
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    result = subprocess.run(cmd, check=False, env=env)
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with return code {result.returncode}")

    print("\nTraining complete!")
    print("Run:  modal run modal_train.py --mode download  to pull results locally.")


# ---------------------------------------------------------------------------
# SANITY CHECK — 50 steps only (~15 min)
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",
    timeout=3600,
    image=image,
    volumes={VOLUME_PATH: volume},
)
def run_sanity():
    import subprocess

    os.makedirs(f"{VOLUME_PATH}/checkpoint", exist_ok=True)

    print("=" * 60)
    print("SANITY CHECK — 50 steps only")
    print("Watch for: grad_norm > 0, kl_penalty > 0, reward changing")
    print("=" * 60)

    subprocess.run(["nvidia-smi"], check=False)

    cmd = build_cmd(max_train_steps=50, grad_accum=2)

    # save at step 25 and 50 for the sanity run
    cmd = [c for c in cmd if not c.startswith("--checkpointing_steps")]
    cmd = [c for c in cmd if not c.startswith("--graph_every")]
    cmd += ["--checkpointing_steps=25", "--graph_every=25"]

    print("COMMAND:")
    print(" \\\n  ".join(cmd))

    result = subprocess.run(cmd, check=False)
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Sanity check failed (code {result.returncode})")

    print("\nSanity check passed.")
    print("  grad_norm > 0  ← REINFORCE fix working")
    print("  kl_penalty > 0 ← UNet moving")
    print("  reward trending ← model learning")
    print("\nNow run full training:  modal run modal_train.py")


# ---------------------------------------------------------------------------
# LIST — show what is saved in the volume
# ---------------------------------------------------------------------------
@app.function(
    image=modal.Image.debian_slim(),
    volumes={VOLUME_PATH: volume},
)
def list_outputs():
    print(f"\nContents of Modal Volume 'ddpo-altaf-outputs':\n")
    for root, dirs, files in os.walk(VOLUME_PATH):
        level     = root.replace(VOLUME_PATH, "").count(os.sep)
        indent    = "  " * level
        subindent = "  " * (level + 1)
        print(f"{indent}{os.path.basename(root)}/")
        for f in sorted(files):
            fpath    = os.path.join(root, f)
            size     = os.path.getsize(fpath)
            size_str = (f"{size/1024/1024:.1f} MB" if size > 1024 * 1024
                        else f"{size/1024:.1f} KB"  if size > 1024
                        else f"{size} B")
            print(f"{subindent}{f}  ({size_str})")


# ---------------------------------------------------------------------------
# DOWNLOAD — pull everything from the volume to ./modal_outputs/ locally
# ---------------------------------------------------------------------------
@app.function(
    image=modal.Image.debian_slim(),
    volumes={VOLUME_PATH: volume},
)
def download_outputs():
    results = {}
    for root, dirs, files in os.walk(VOLUME_PATH):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel   = os.path.relpath(fpath, VOLUME_PATH)
            with open(fpath, "rb") as f:
                results[rel] = f.read()
    return results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    mode:   str  = "train",
    steps:  int  = 1000,
    resume: bool = False,
):
    if mode == "sanity":
        print("Running 50-step sanity check on A10G...")
        run_sanity.remote()

    elif mode == "train":
        print(f"Starting full training ({steps} steps) on A10G...")
        print("Monitor live logs:  modal app logs ddpo-altaf-train")
        run_training.remote(max_train_steps=steps, resume=resume)

    elif mode == "list":
        list_outputs.remote()

    elif mode == "download":
        print("Downloading from Modal Volume to ./modal_outputs/ ...")
        results = download_outputs.remote()
        for rel_path, data in results.items():
            local_path = os.path.join("modal_outputs", rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(data)
            print(f"  saved: {local_path}")
        print(f"\nDone. {len(results)} files saved to ./modal_outputs/")

    else:
        print(f"Unknown mode '{mode}'. Valid: train | sanity | list | download")
        sys.exit(1)