'''
@File        : train_ours.py
@Description : Lightweight DDPO with REINFORCE policy gradient, STRW reward weighting,
               layer routing via backward hooks, full UNet fine-tuning (50 steps).

ROOT CAUSE FIX (why previous run had grad_norm = 0 throughout):
  The old script computed loss = -R + kl, where R was computed on image_tensor.detach().
  The reward heads (CLIP, PickScore, Aesthetic) all operate on CPU-detached PIL images —
  they produce Python floats with NO computational graph connection to the UNet.
  Result: loss.backward() propagates zero gradient to every UNet parameter.

REINFORCE FIX:
  Instead of treating R as a differentiable function of UNet params (it is NOT —
  reward models are frozen and detached), we use the REINFORCE / policy gradient trick:

    ∇_θ J(θ) = E_{x ~ π_θ}[R(x) · ∇_θ log π_θ(x | c)]

  In denoising diffusion:
    log π_θ(x_t | x_{t+1}, c) = -||ε_θ(x_t, t, c) - ε̂||² / (2σ_t²)

  So the surrogate loss for REINFORCE is:
    L_REINFORCE(θ) = -R_detached · log π_θ(x_mid | x_{mid+1}, c)
                   = R_detached · ||ε_θ(x_mid, t_mid, c) - ε_target||²

  where ε_target is the DDPM noise target at that step (what the scheduler expected).
  This gives real gradients through ε_θ, which is the UNet output.

  KL penalty is added on top:
    L_total = L_REINFORCE + β · KL(π_θ || π_ref)

LAYER ROUTING FIX:
  The old hooks multiplied parameter gradients AFTER backward. Because reward heads
  produced no gradient, there was nothing to scale — hooks had no effect.
  With REINFORCE, real gradients flow. Hooks now correctly scale them per block.

  Additionally: hooks must be re-registered AFTER accelerator.prepare() because
  prepare() wraps self.unet in an AcceleratedModel object. The old code registered
  hooks BEFORE prepare(), on the unwrapped module, then prepare() created a new
  wrapper — the hooks were registered on a different object than the one being
  backward'd through.

FIXED WEIGHTS INITIALIZATION:
  On --resume_from_checkpoint or --init_from_checkpoint, we load UNet weights
  from a prior checkpoint and keep them as the STARTING point (not the ref).
  The frozen ref_unet is ALWAYS loaded from the original pretrained model.

50-STEP DENOISING:
  --total_denoising_steps=50 (was 20 or 30 in previous runs).
  Mid-timestep sampled uniformly from [0, 49].
'''

import argparse
import logging
import math
import os
import random
from pathlib import Path

import accelerate
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor

from PIL import Image

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available

if is_wandb_available():
    import wandb

check_min_version("0.16.0.dev0")
logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Reward head 1: CLIP score (ViT-B/32)
# ---------------------------------------------------------------------------
class CLIPScorer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def forward(self, images_pil, prompts):
        inputs = self.processor(
            text=prompts, images=images_pil,
            return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        outputs = self.model(**inputs)
        return outputs.logits_per_image.diagonal()


# ---------------------------------------------------------------------------
# Reward head 2: LAION Aesthetic Predictor (MLP on CLIP ViT-L/14)
# ---------------------------------------------------------------------------
class AestheticMLP(nn.Module):
    def __init__(self, input_size=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128),       nn.Dropout(0.2),
            nn.Linear(128, 64),         nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


class AestheticScorer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.clip      = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip.requires_grad_(False)
        self.mlp       = AestheticMLP(768).to(device)
        self.mlp.requires_grad_(False)
        self.device    = device
        self._load_weights()

    def _load_weights(self):
        import urllib.request
        # Priority 1: baked into Modal image at build time (no network needed)
        # Priority 2: local user cache (for running on your own machine)
        # Priority 3: download as last resort
        baked_path = "/root/.cache/aesthetic_predictor.pth"
        user_path  = os.path.expanduser("~/.cache/aesthetic_predictor.pth")
        if os.path.exists(baked_path):
            path = baked_path
            logger.info("Aesthetic predictor: using pre-baked weights.")
        elif os.path.exists(user_path):
            path = user_path
            logger.info("Aesthetic predictor: using cached weights.")
        else:
            path = user_path
            os.makedirs(os.path.dirname(path), exist_ok=True)
            url = ("https://github.com/christophschuhmann/"
                   "improved-aesthetic-predictor/raw/main/"
                   "sac+logos+ava1-l14-linearMSE.pth")
            logger.info("Aesthetic predictor: downloading weights...")
            urllib.request.urlretrieve(url, path)
        state_dict = torch.load(path, map_location=self.device)
        remapped   = {k.replace("layers.", ""): v for k, v in state_dict.items()}
        self.mlp.load_state_dict(remapped, strict=False)

    @torch.no_grad()
    def forward(self, images_pil):
        inputs = self.processor(images=images_pil, return_tensors="pt").to(self.device)
        feats  = self.clip.get_image_features(**inputs)
        feats  = feats / feats.norm(dim=-1, keepdim=True)
        return self.mlp(feats.float()).squeeze(-1)


# ---------------------------------------------------------------------------
# Reward head 3: PickScore (ViT-H/14)
# Loaded on-demand to save VRAM — unloaded immediately after scoring.
# ---------------------------------------------------------------------------
class PickScorer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    @torch.no_grad()
    def forward(self, images_pil, prompts):
        from transformers import AutoProcessor, AutoModel
        import gc
        processor  = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        model      = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").to(self.device)
        model.requires_grad_(False)
        img_inputs  = processor(images=images_pil, return_tensors="pt", padding=True).to(self.device)
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        img_embs    = model.get_image_features(**img_inputs)
        img_embs    = img_embs / img_embs.norm(dim=-1, keepdim=True)
        text_embs   = model.get_text_features(**text_inputs)
        text_embs   = text_embs / text_embs.norm(dim=-1, keepdim=True)
        scores      = (img_embs * text_embs).sum(dim=-1)
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
        return scores.cpu()


# ---------------------------------------------------------------------------
# Reward head 4: Laplacian variance (sharpness)
# ---------------------------------------------------------------------------
def laplacian_variance(image_tensor):
    '''image_tensor: [B,3,H,W] float in [0,1]. Returns [B] variance.'''
    gray   = (0.299 * image_tensor[:, 0]
            + 0.587 * image_tensor[:, 1]
            + 0.114 * image_tensor[:, 2]).unsqueeze(1)
    kernel = torch.tensor(
        [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
        dtype=image_tensor.dtype, device=image_tensor.device
    ).unsqueeze(0).unsqueeze(0)
    lap = F.conv2d(gray, kernel, padding=1)
    return lap.var(dim=[1, 2, 3])


# ---------------------------------------------------------------------------
# STRW weight schedule (equations 1–5)
# s = mid_timestep / total_steps  (0 = noise, 1 = clean)
# ---------------------------------------------------------------------------
def strw_weights(mid_timestep, total_steps=50):
    s         = mid_timestep / max(total_steps - 1, 1)
    w_clip    = s
    w_aes     = math.sin(math.pi * s)
    w_pick    = math.sin(math.pi * max(s - 0.1, 0.0))
    w_quality = 1.0 - s
    total     = w_clip + w_aes + w_pick + w_quality + 1e-8
    return w_clip/total, w_aes/total, w_pick/total, w_quality/total


# ---------------------------------------------------------------------------
# Running normalizer — Welford's online algorithm
# Keeps each reward head on zero-mean unit-variance scale.
# ---------------------------------------------------------------------------
class RunningNormalizer:
    def __init__(self):
        self.n    = 0
        self.mean = 0.0
        self.M2   = 0.0

    def update(self, values):
        for v in values:
            self.n   += 1
            delta     = float(v) - self.mean
            self.mean += delta / self.n
            self.M2  += delta * (float(v) - self.mean)

    @property
    def std(self):
        if self.n < 2:
            return 1.0
        return math.sqrt(self.M2 / (self.n - 1)) + 1e-8

    def normalize(self, t):
        return (t - self.mean) / self.std


# ---------------------------------------------------------------------------
# Layer Router
#
# FIX vs old code: hooks are registered AFTER accelerator.prepare() so they
# attach to the wrapped module that actually receives gradients.
# The register() method unwraps via .module if wrapped.
#
# Hooks scale existing gradients: grad_new = grad * weight[block]
#   down_blocks  <-- w_clip       (semantic, early denoising)
#   mid_block    <-- w_aes+w_pick (style + preference, mid)
#   up_blocks    <-- w_quality    (sharpness, late denoising)
# ---------------------------------------------------------------------------
class LayerRouter:
    def __init__(self):
        self.hooks = []
        self._w    = {'down': 1.0, 'mid': 1.0, 'up': 1.0}
        self._unet = None

    def attach(self, unet):
        '''Call AFTER accelerator.prepare(unet). Stores reference and registers hooks.'''
        self._unet = unet
        self.register()

    def update_weights(self, w_clip, w_aes, w_pick, w_quality):
        self._w['down'] = float(w_clip)
        self._w['mid']  = float(w_aes + w_pick)
        self._w['up']   = float(w_quality)

    def register(self):
        self.remove()
        if self._unet is None:
            raise RuntimeError("LayerRouter.attach(unet) must be called before register().")

        # Unwrap DDP / Accelerate wrapper
        raw = self._unet.module if hasattr(self._unet, 'module') else self._unet

        def make_hook(key):
            def hook(grad):
                if grad is None:
                    return grad
                w = self._w[key]
                if w == 0.0:
                    # Return zero tensor instead of None to avoid autograd confusion
                    return torch.zeros_like(grad)
                return grad * w
            return hook

        n_hooks = 0
        for block in raw.down_blocks:
            for p in block.parameters():
                if p.requires_grad:
                    self.hooks.append(p.register_hook(make_hook('down')))
                    n_hooks += 1

        if raw.mid_block is not None:
            for p in raw.mid_block.parameters():
                if p.requires_grad:
                    self.hooks.append(p.register_hook(make_hook('mid')))
                    n_hooks += 1

        for block in raw.up_blocks:
            for p in block.parameters():
                if p.requires_grad:
                    self.hooks.append(p.register_hook(make_hook('up')))
                    n_hooks += 1

        logger.info(f"LayerRouter: registered {n_hooks} gradient hooks on UNet.")

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ---------------------------------------------------------------------------
# Graph logger
# ---------------------------------------------------------------------------
class GraphLogger:
    def __init__(self, output_dir):
        self.plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        self.history = {
            'step': [], 'train_loss': [], 'reward': [], 'kl_penalty': [],
            'log_prob': [], 'reinforce_loss': [],
            'w_clip': [], 'w_aes': [], 'w_pick': [], 'w_quality': [],
            'mid_timestep': [],
            'r_clip': [], 'r_aesthetic': [], 'r_pick': [], 'r_laplacian': [],
            'grad_norm': [],
        }

    def update(self, step, d):
        self.history['step'].append(step)
        for k in self.history:
            if k != 'step' and k in d:
                self.history[k].append(d[k])

    def save_all(self, step, generated_images=None, prompts=None):
        self._plot_training_curves(step)
        self._plot_reward_heads(step)
        self._plot_strw_weights(step)
        self._plot_timestep_dist(step)
        self._plot_weight_trajectories(step)
        if generated_images:
            self._plot_generated_images(step, generated_images, prompts)

    def _plot_training_curves(self, step):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(f'Training curves (step {step})', fontsize=13)
        s = self.history['step']

        axes[0,0].plot(s, self.history['train_loss'], color='#E24B4A', lw=1.2)
        axes[0,0].set_title('Total loss')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].grid(True, alpha=0.3)

        axes[0,1].plot(s, self.history['reward'], color='#378ADD', lw=1.2)
        axes[0,1].set_title('Composite reward R (should trend UP)')
        axes[0,1].set_ylabel('R')
        axes[0,1].grid(True, alpha=0.3)

        axes[1,0].plot(s, self.history['kl_penalty'], color='#9FE1CB', lw=1.2)
        axes[1,0].set_title('KL penalty (should be small & stable)')
        axes[1,0].set_ylabel('KL')
        axes[1,0].grid(True, alpha=0.3)

        axes[1,1].plot(s, self.history['grad_norm'], color='#BA7517', lw=1.2)
        axes[1,1].set_title('Grad norm (MUST be > 0 — was 0 before fix)')
        axes[1,1].set_ylabel('||∇||')
        axes[1,1].grid(True, alpha=0.3)

        for ax in axes.flat:
            ax.set_xlabel('Step')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'training_curves_step{step}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    def _plot_reward_heads(self, step):
        s   = self.history['step']
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(f'Reward head scores (step {step})', fontsize=13)
        pairs = [('r_clip','#378ADD','CLIP'), ('r_aesthetic','#9FE1CB','Aesthetic'),
                 ('r_pick','#1D9E75','PickScore'), ('r_laplacian','#BA7517','Laplacian')]
        for ax, (k, c, lbl) in zip(axes.flat, pairs):
            ax.plot(s, self.history[k], color=c, lw=1.2)
            ax.set_title(lbl)
            ax.set_xlabel('Step')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'reward_heads_step{step}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    def _plot_strw_weights(self, step):
        s = self.history['mid_timestep']
        if not s:
            return
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f'STRW weights vs sampled timestep (step {step})', fontsize=13)
        for ax, (k, c, lbl) in zip(
            [axes[0], axes[0], axes[0], axes[0]],
            [('w_clip','#378ADD','CLIP'),('w_aes','#9FE1CB','Aesthetic'),
             ('w_pick','#1D9E75','PickScore'),('w_quality','#BA7517','Laplacian')]
        ):
            pass  # matplotlib can't easily do this in a loop on same ax
        axes[0].scatter(s, self.history['w_clip'],    color='#378ADD', alpha=0.4, s=8, label='CLIP')
        axes[0].scatter(s, self.history['w_aes'],     color='#9FE1CB', alpha=0.4, s=8, label='Aesthetic')
        axes[0].scatter(s, self.history['w_pick'],    color='#1D9E75', alpha=0.4, s=8, label='PickScore')
        axes[0].scatter(s, self.history['w_quality'], color='#BA7517', alpha=0.4, s=8, label='Laplacian')
        axes[0].set_xlabel('Sampled mid_timestep')
        axes[0].set_ylabel('Weight')
        axes[0].set_title('Actual weights at each training step')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(self.history['step'], self.history['log_prob'], color='#E24B4A', lw=1.2)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('log π_θ(x_mid)')
        axes[1].set_title('Log-prob of sampled step (REINFORCE surrogate)')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'strw_weights_step{step}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    def _plot_timestep_dist(self, step):
        ts = self.history['mid_timestep']
        if not ts:
            return
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(ts, bins=50, color='#7F77DD', edgecolor='white', linewidth=0.5)
        ax.axhline(len(ts)/50, color='#E24B4A', ls='--', lw=1.2, label='Expected uniform')
        ax.set_xlabel('Sampled mid_timestep')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Timestep distribution (step {step}) — should be uniform')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'timestep_dist_step{step}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    def _plot_weight_trajectories(self, step):
        sv      = np.linspace(0, 1, 200)
        w_c     = sv
        w_a     = np.sin(np.pi * sv)
        w_p     = np.sin(np.pi * np.maximum(sv - 0.1, 0))
        w_q     = 1 - sv
        total   = w_c + w_a + w_p + w_q + 1e-8
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle('STRW weight schedule — theoretical curves', fontsize=13)
        for ax, (wc, wa, wp, wq, title) in zip(axes, [
            (w_c, w_a, w_p, w_q, 'Raw weights'),
            (w_c/total, w_a/total, w_p/total, w_q/total, 'Normalized weights')
        ]):
            ax.plot(sv, wc, color='#378ADD', lw=2, label='w_align (CLIP)')
            ax.plot(sv, wa, color='#9FE1CB', lw=2, label='w_aesthetic')
            ax.plot(sv, wp, color='#1D9E75', lw=2, label='w_preference')
            ax.plot(sv, wq, color='#BA7517', lw=2, label='w_quality')
            ax.set_xlabel('s = t/T')
            ax.set_title(title)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'weight_trajectories_step{step}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    def _plot_generated_images(self, step, images, prompts=None):
        n    = len(images)
        cols = min(n, 4)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4+0.8))
        fig.suptitle(f'Generated images (step {step})', fontsize=13)
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]
        for idx, img in enumerate(images):
            r, c = divmod(idx, cols)
            axes[r][c].imshow(np.array(img))
            axes[r][c].axis('off')
            if prompts and idx < len(prompts):
                p = prompts[idx][:60] + '...' if len(prompts[idx]) > 60 else prompts[idx]
                axes[r][c].set_title(p, fontsize=7)
        for idx in range(len(images), rows*cols):
            r, c = divmod(idx, cols)
            axes[r][c].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'generated_images_step{step}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="REINFORCE-based Lightweight DDPO + STRW + Layer Routing (50 steps)"
    )
    # model
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--init_from_checkpoint", type=str, default=None,
                        help="Path to a prior UNet checkpoint to use as starting weights. "
                             "The frozen ref_unet is ALWAYS the original pretrained model, "
                             "not this checkpoint.")
    parser.add_argument("--revision", type=str, default=None)

    # data
    parser.add_argument("--train_data_dir",          type=str, default="data/refl_data.json")
    parser.add_argument("--caption_column",          type=str, default="text")
    parser.add_argument("--max_train_samples",       type=int, default=1000)
    parser.add_argument("--dataloader_num_workers",  type=int, default=0)

    # output
    parser.add_argument("--output_dir",              type=str, default="checkpoint/train_ours")
    parser.add_argument("--logging_dir",             type=str, default="logs")
    parser.add_argument("--report_to",               type=str, default="tensorboard")
    parser.add_argument("--tracker_project_name",    type=str, default="train_ours_reinforce")

    # training
    parser.add_argument("--seed",                        type=int,   default=42)
    parser.add_argument("--max_train_steps",             type=int,   default=1000)
    parser.add_argument("--num_train_epochs",            type=int,   default=200)
    parser.add_argument("--train_batch_size",            type=int,   default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=4)
    parser.add_argument("--gradient_checkpointing",      action="store_true")
    parser.add_argument("--mixed_precision",             type=str,   default="fp16",
                        choices=["no", "fp16", "bf16"])

    # optimizer
    parser.add_argument("--learning_rate",    type=float, default=1e-6)
    parser.add_argument("--lr_scheduler",     type=str,   default="cosine")
    parser.add_argument("--lr_warmup_steps",  type=int,   default=50)
    parser.add_argument("--adam_beta1",       type=float, default=0.9)
    parser.add_argument("--adam_beta2",       type=float, default=0.999)
    parser.add_argument("--adam_weight_decay",type=float, default=1e-2)
    parser.add_argument("--adam_epsilon",     type=float, default=1e-8)
    parser.add_argument("--max_grad_norm",    type=float, default=1.0)
    parser.add_argument("--use_8bit_adam",    action="store_true")

    # STRW / DDPO
    parser.add_argument("--total_denoising_steps", type=int,   default=50)

    # REINFORCE baseline (reduces variance)
    # "running_mean" = use exponential moving average of past rewards as baseline
    # "none"         = no baseline (higher variance, simpler)
    parser.add_argument("--baseline",              type=str,   default="running_mean",
                        choices=["running_mean", "none"])
    parser.add_argument("--baseline_ema_alpha",    type=float, default=0.05,
                        help="EMA decay for running mean baseline (lower = more stable).")

    # Reward hacking mitigations
    parser.add_argument("--kl_beta",      type=float, default=0.05)
    parser.add_argument("--kl_threshold", type=float, default=0.5)
    parser.add_argument("--reward_clip",  type=float, default=3.0)

    # checkpointing
    parser.add_argument("--checkpointing_steps",    type=int, default=100)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--local_rank",             type=int, default=-1)
    parser.add_argument("--graph_every",            type=int, default=100)

    args = parse_known_and_env(parser)
    return args


def parse_known_and_env(parser):
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:

    def __init__(self, args):

        logging_dir = os.path.join(args.output_dir, args.logging_dir)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=ProjectConfiguration(
                project_dir=args.output_dir,
                logging_dir=logging_dir,
            ),
        )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)

        if args.seed is not None:
            set_seed(args.seed)

        if self.accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)

        device = self.accelerator.device

        # ----------------------------------------------------------------
        # Base models
        # ----------------------------------------------------------------
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer",
            revision=args.revision
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder",
            revision=args.revision
        )
        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae",
            revision=args.revision
        )

        # Trainable UNet
        if args.init_from_checkpoint:
            logger.info(f"Loading UNet weights from checkpoint: {args.init_from_checkpoint}")
            self.unet = UNet2DConditionModel.from_pretrained(
                args.init_from_checkpoint, subfolder="unet"
            )
            logger.info("UNet initialised from checkpoint (fixed weights as starting point).")
        else:
            self.unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="unet",
                revision=args.revision
            )
            logger.info("UNet initialised from pretrained SD v1.4.")

        # Frozen reference UNet — ALWAYS original SD v1.4, never the checkpoint
        # This is the anchor for the KL penalty.
        self.ref_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet",
            revision=args.revision
        )
        self.ref_unet.requires_grad_(False)
        self.ref_unet.eval()

        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        if args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        # ----------------------------------------------------------------
        # Reward heads — on CPU to save VRAM
        # ----------------------------------------------------------------
        self.reward_device   = torch.device("cpu")
        self.clip_scorer     = CLIPScorer(self.reward_device)
        self.aesthetic_scorer= AestheticScorer(self.reward_device)
        self.pick_scorer     = PickScorer(self.reward_device)

        self.norm = {
            'clip':      RunningNormalizer(),
            'aesthetic': RunningNormalizer(),
            'pick':      RunningNormalizer(),
            'laplacian': RunningNormalizer(),
        }

        # ----------------------------------------------------------------
        # REINFORCE baseline (exponential moving average of reward)
        # Reduces gradient variance: advantage = R - baseline
        # ----------------------------------------------------------------
        self.reward_baseline = 0.0
        self.baseline_alpha  = args.baseline_ema_alpha

        # ----------------------------------------------------------------
        # Optimizer
        # ----------------------------------------------------------------
        if args.use_8bit_adam:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        self.optimizer = optimizer_cls(
            self.unet.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        # ----------------------------------------------------------------
        # Dataset
        # ----------------------------------------------------------------
        dataset = load_dataset(
            "json",
            data_files={"train": args.train_data_dir},
            cache_dir=None
        )

        def tokenize_captions(examples):
            captions = []
            for cap in examples[args.caption_column]:
                if isinstance(cap, str):
                    captions.append(cap)
                elif isinstance(cap, (list, np.ndarray)):
                    captions.append(random.choice(cap))
                else:
                    raise ValueError("Caption must be str or list.")
            return self.tokenizer(
                captions,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids

        def preprocess_train(examples):
            examples["input_ids"] = tokenize_captions(examples)
            examples["prompts"]   = examples[args.caption_column]
            return examples

        with self.accelerator.main_process_first():
            if args.max_train_samples is not None:
                dataset["train"] = (
                    dataset["train"]
                    .shuffle(seed=args.seed)
                    .select(range(args.max_train_samples))
                )
            self.train_dataset = dataset["train"].with_transform(preprocess_train)

        def collate_fn(examples):
            input_ids = torch.stack([ex["input_ids"] for ex in examples])
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            prompts   = [ex["prompts"] for ex in examples]
            return {"input_ids": input_ids, "prompts": prompts}

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )

        # ----------------------------------------------------------------
        # LR scheduler
        # ----------------------------------------------------------------
        overrode_max_train_steps = False
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / args.gradient_accumulation_steps
        )
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * self.num_update_steps_per_epoch
            overrode_max_train_steps = True

        self.lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

        # ----------------------------------------------------------------
        # Accelerator prepare — MUST happen before LayerRouter.attach()
        # ----------------------------------------------------------------
        (
            self.unet,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        self.text_encoder.to(device, dtype=self.weight_dtype)
        self.vae.to(device,          dtype=self.weight_dtype)
        self.ref_unet.to(device,     dtype=torch.float32)
        self.ref_unet.requires_grad_(False)
        self.ref_unet.eval()

        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / args.gradient_accumulation_steps
        )
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * self.num_update_steps_per_epoch
        args.num_train_epochs = math.ceil(
            args.max_train_steps / self.num_update_steps_per_epoch
        )

        # ----------------------------------------------------------------
        # Layer router — attach AFTER prepare()
        # ----------------------------------------------------------------
        self.layer_router = LayerRouter()
        self.layer_router.attach(self.unet)

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(args.tracker_project_name, vars(args))

        self.graph_logger = None
        if self.accelerator.is_main_process:
            self.graph_logger = GraphLogger(args.output_dir)
            self.graph_logger._plot_weight_trajectories(step=0)
            logger.info(f"Weight trajectory plot saved to {self.graph_logger.plot_dir}")

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _to_pil(self, image_tensor):
        imgs = []
        for i in range(image_tensor.shape[0]):
            arr = (image_tensor[i].float().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
            imgs.append(Image.fromarray(arr))
        return imgs

    def compute_composite_reward(self, image_tensor, prompts, mid_timestep, total_steps):
        '''
        Compute detached STRW composite reward.
        Returns scalar R (Python float) — no grad, not connected to UNet.
        Also returns STRW weights for layer routing.
        '''
        images_pil  = self._to_pil(image_tensor.detach())
        images_cpu  = image_tensor.float().detach().cpu()

        r_clip      = self.clip_scorer(images_pil, prompts).float()
        r_aesthetic = self.aesthetic_scorer(images_pil).float()
        r_pick      = self.pick_scorer(images_pil, prompts).float()
        r_laplacian = laplacian_variance(images_cpu).float()

        self.norm['clip'].update(r_clip.tolist())
        self.norm['aesthetic'].update(r_aesthetic.tolist())
        self.norm['pick'].update(r_pick.tolist())
        self.norm['laplacian'].update(r_laplacian.tolist())

        r_clip_n = torch.tensor(self.norm['clip'].normalize(r_clip).tolist())
        r_aes_n  = torch.tensor(self.norm['aesthetic'].normalize(r_aesthetic).tolist())
        r_pick_n = torch.tensor(self.norm['pick'].normalize(r_pick).tolist())
        r_lap_n  = torch.tensor(self.norm['laplacian'].normalize(r_laplacian).tolist())

        w_clip, w_aes, w_pick, w_quality = strw_weights(mid_timestep, total_steps)

        R_scalar = (
            w_clip    * r_clip_n.mean().item()
          + w_aes     * r_aes_n.mean().item()
          + w_pick    * r_pick_n.mean().item()
          + w_quality * r_lap_n.mean().item()
        )

        head_scores = {
            'r_clip':      r_clip.mean().item(),
            'r_aesthetic': r_aesthetic.mean().item(),
            'r_pick':      r_pick.mean().item(),
            'r_laplacian': r_laplacian.mean().item(),
        }

        return R_scalar, w_clip, w_aes, w_pick, w_quality, head_scores

    def compute_log_prob(self, noise_pred, noise_target, sigma_t):
        '''
        Compute log π_θ(x_mid | x_{mid+1}, c) under Gaussian noise model.

        In DDPM, the reverse step is:
          x_{t-1} = (x_t - σ_t · ε_θ) / scale  + noise

        The log-prob of the sampled x_{mid-1} under π_θ is:
          log π_θ = -||ε_θ - ε_target||² / (2 σ_t²)  + const

        For REINFORCE we just need the part that has grad:
          log_prob = -0.5 * ||ε_θ - ε_target||²

        ε_target: what noise the scheduler "expected" — approximated by
        the noise prediction of the frozen reference UNet (or the DDPM
        noise target derived from the latents).

        noise_pred:   [B,4,64,64] — trainable UNet output (has grad)
        noise_target: [B,4,64,64] — reference / target (detached, no grad)
        sigma_t:      scalar sigma at timestep t (from scheduler)
        '''
        sq_err   = ((noise_pred.float() - noise_target.float()) ** 2).mean()
        log_prob = -0.5 * sq_err / (sigma_t ** 2 + 1e-8)
        return log_prob

    def get_sigma(self, t):
        '''
        Get σ_t from the DDPM scheduler.
        alphas_cumprod[t] = ᾱ_t.  σ_t = sqrt(1 - ᾱ_t).
        '''
        ac = self.noise_scheduler.alphas_cumprod
        # t is a timestep value (e.g., 981), not an index.
        # Convert to scheduler index.
        idx = (self.noise_scheduler.timesteps == t).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            return 1.0
        ᾱ_t = ac[idx[0].item()].item()
        return math.sqrt(max(1.0 - ᾱ_t, 1e-8))

    def _generate_sample_images(self, prompts, args):
        self.unet.eval()
        images = []
        raw_unet = self.unet.module if hasattr(self.unet, 'module') else self.unet
        with torch.no_grad():
            for prompt in prompts:
                tokens = self.tokenizer(
                    [prompt],
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(self.accelerator.device)
                enc = self.text_encoder(tokens)[0]
                lat = torch.randn((1,4,64,64), device=self.accelerator.device)
                self.noise_scheduler.set_timesteps(
                    args.total_denoising_steps, device=self.accelerator.device
                )
                for t in self.noise_scheduler.timesteps:
                    lmi  = self.noise_scheduler.scale_model_input(lat, t)
                    pred = raw_unet(lmi, t, encoder_hidden_states=enc).sample
                    lat  = self.noise_scheduler.step(pred, t, lat).prev_sample
                lat = (1 / self.vae.config.scaling_factor) * lat
                img = self.vae.decode(lat.to(self.weight_dtype)).sample
                img = (img / 2 + 0.5).clamp(0, 1)
                arr = (img[0].float().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
                images.append(Image.fromarray(arr))
        self.unet.train()
        return images

    # ----------------------------------------------------------------
    # Main training loop
    # ----------------------------------------------------------------
    def train(self, args):

        total_batch_size = (
            args.train_batch_size
            * self.accelerator.num_processes
            * args.gradient_accumulation_steps
        )

        logger.info("***** REINFORCE-based Lightweight DDPO + STRW (50 steps) *****")
        logger.info(f"  Total examples           = {len(self.train_dataset)}")
        logger.info(f"  Batch size per device    = {args.train_batch_size}")
        logger.info(f"  Total batch size         = {total_batch_size}")
        logger.info(f"  Grad accum steps         = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimizer steps    = {args.max_train_steps}")
        logger.info(f"  Denoising steps          = {args.total_denoising_steps}")
        logger.info(f"  KL beta                  = {args.kl_beta}")
        logger.info(f"  KL threshold             = {args.kl_threshold}")
        logger.info(f"  Reward clip              = ±{args.reward_clip}")
        logger.info(f"  Learning rate            = {args.learning_rate}")
        logger.info(f"  REINFORCE baseline       = {args.baseline}")
        logger.info(f"")
        logger.info("  KEY FIX: Using REINFORCE surrogate loss.")
        logger.info("  L = -advantage · log_prob  (real gradients through ε_θ)")
        logger.info("  L_total = L_reinforce + β · KL(π_θ || π_ref)")

        global_step   = 0
        first_epoch   = 0
        skipped_steps = 0

        # Resume from checkpoint
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint == "latest":
                dirs = sorted(
                    [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                    key=lambda x: int(x.split("-")[1])
                )
                path = dirs[-1] if dirs else None
            else:
                path = os.path.basename(args.resume_from_checkpoint)
            if path:
                self.accelerator.print(f"Resuming from {path}")
                self.accelerator.load_state(os.path.join(args.output_dir, path))
                global_step = int(path.split("-")[1])
                first_epoch = global_step // self.num_update_steps_per_epoch

        progress_bar = tqdm(
            range(global_step, args.max_train_steps),
            disable=not self.accelerator.is_local_main_process
        )
        progress_bar.set_description("Steps")

        for epoch in range(first_epoch, args.num_train_epochs):
            self.unet.train()
            train_loss_accum = 0.0

            for step, batch in enumerate(self.train_dataloader):

                with self.accelerator.accumulate(self.unet):

                    # ------------------------------------------------
                    # Text encoding
                    # ------------------------------------------------
                    with torch.no_grad():
                        encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                    # ------------------------------------------------
                    # Sample starting noise
                    # ------------------------------------------------
                    latents = torch.randn(
                        (args.train_batch_size, 4, 64, 64),
                        device=self.accelerator.device
                    )

                    # 50-step scheduler
                    self.noise_scheduler.set_timesteps(
                        args.total_denoising_steps,
                        device=self.accelerator.device
                    )
                    timesteps = self.noise_scheduler.timesteps

                    # Sample mid_timestep uniformly from full 50-step trajectory
                    mid_timestep = random.randint(0, args.total_denoising_steps - 1)
                    t_mid        = timesteps[mid_timestep]

                    # ------------------------------------------------
                    # Phase 1: no_grad denoising up to mid_timestep - 1
                    # ------------------------------------------------
                    with torch.no_grad():
                        for t in timesteps[:mid_timestep]:
                            lmi        = self.noise_scheduler.scale_model_input(latents, t)
                            noise_pred = self.unet(
                                lmi, t,
                                encoder_hidden_states=encoder_hidden_states
                            ).sample
                            latents    = self.noise_scheduler.step(
                                noise_pred, t, latents
                            ).prev_sample

                    # ------------------------------------------------
                    # Phase 2: the ONE step with grad (REINFORCE step)
                    # ε_θ(x_mid, t_mid, c) — this has grad
                    # ------------------------------------------------
                    lmi_mid    = self.noise_scheduler.scale_model_input(latents, t_mid)

                    # Trainable UNet — GRAD ON
                    noise_pred = self.unet(
                        lmi_mid, t_mid,
                        encoder_hidden_states=encoder_hidden_states
                    ).sample   # [B, 4, 64, 64], has grad

                    # Reference UNet — GRAD OFF (for KL penalty AND log_prob target)
                    with torch.no_grad():
                        noise_pred_ref = self.ref_unet(
                            lmi_mid.float(),
                            t_mid,
                            encoder_hidden_states=encoder_hidden_states.float()
                        ).sample  # [B, 4, 64, 64], no grad

                    # Step forward from mid_timestep (for reward evaluation)
                    latents_mid = self.noise_scheduler.step(
                        noise_pred, t_mid, latents
                    ).prev_sample

                    # ------------------------------------------------
                    # Phase 3: no_grad — complete remaining steps to get x_0
                    # Use detached latents_mid so grad doesn't flow here
                    # ------------------------------------------------
                    latents_final = latents_mid.detach()
                    with torch.no_grad():
                        for t in timesteps[mid_timestep + 1:]:
                            lmi_r         = self.noise_scheduler.scale_model_input(latents_final, t)
                            noise_pred_r  = self.unet(
                                lmi_r, t,
                                encoder_hidden_states=encoder_hidden_states
                            ).sample
                            latents_final = self.noise_scheduler.step(
                                noise_pred_r, t, latents_final
                            ).prev_sample

                    # ------------------------------------------------
                    # Decode x_0 for reward computation
                    # ------------------------------------------------
                    with torch.no_grad():
                        latents_decode = (1 / self.vae.config.scaling_factor) * latents_final
                        image_decoded  = self.vae.decode(
                            latents_decode.to(self.weight_dtype)
                        ).sample
                        image_decoded  = (image_decoded / 2 + 0.5).clamp(0, 1)  # [B,3,512,512]

                    # ------------------------------------------------
                    # KL penalty — computed BEFORE reward (allows early skip)
                    # KL ≈ 0.5 ||ε_θ - ε_ref||²  (equation 8)
                    # noise_pred has grad; noise_pred_ref is detached → real KL gradient
                    # ------------------------------------------------
                    kl_penalty = 0.5 * (
                        (noise_pred.float() - noise_pred_ref.float()) ** 2
                    ).mean()

                    if kl_penalty.item() > args.kl_threshold:
                        skipped_steps += 1
                        logger.info(
                            f"Step {global_step}: KL={kl_penalty.item():.4f} "
                            f"> threshold {args.kl_threshold} — skipping"
                        )
                        self.optimizer.zero_grad()
                        self.accelerator.log(
                            {"skipped_steps": skipped_steps, "kl_penalty": kl_penalty.item()},
                            step=global_step
                        )
                        continue

                    # ------------------------------------------------
                    # Compute STRW composite reward (detached float)
                    # R has NO grad — this is correct for REINFORCE.
                    # ------------------------------------------------
                    R_scalar, w_clip, w_aes, w_pick, w_quality, head_scores = (
                        self.compute_composite_reward(
                            image_decoded, batch["prompts"],
                            mid_timestep, args.total_denoising_steps
                        )
                    )

                    # Clip reward
                    R_scalar = max(min(R_scalar, args.reward_clip), -args.reward_clip)

                    # ------------------------------------------------
                    # REINFORCE baseline (variance reduction)
                    # advantage = R - baseline
                    # ------------------------------------------------
                    if args.baseline == "running_mean":
                        advantage = R_scalar - self.reward_baseline
                        # Update EMA baseline
                        self.reward_baseline = (
                            (1 - self.baseline_alpha) * self.reward_baseline
                            + self.baseline_alpha     * R_scalar
                        )
                    else:
                        advantage = R_scalar

                    # ------------------------------------------------
                    # REINFORCE surrogate loss:
                    #   L_REINFORCE = -advantage · log π_θ(x_mid | x_{mid+1}, c)
                    #
                    # log π_θ = -0.5 ||ε_θ - ε_ref||² / σ_t²
                    # (Gaussian reverse process; ε_ref ≈ DDPM noise target)
                    #
                    # Because noise_pred requires grad and ε_ref does NOT,
                    # d(log_prob)/d(θ) = -(ε_θ - ε_ref) / σ_t²
                    # which is a real non-zero gradient into the UNet.
                    #
                    # Intuition: if advantage > 0, we INCREASE log_prob
                    # (make this noise prediction more likely).
                    #           if advantage < 0, we DECREASE log_prob.
                    # ------------------------------------------------
                    sigma_t  = self.get_sigma(t_mid)
                    log_prob = self.compute_log_prob(noise_pred, noise_pred_ref, sigma_t)
                    # log_prob has grad (through noise_pred)

                    L_reinforce = -float(advantage) * log_prob

                    # ------------------------------------------------
                    # Layer routing: update hook weights BEFORE backward
                    # ------------------------------------------------
                    self.layer_router.update_weights(w_clip, w_aes, w_pick, w_quality)

                    # ------------------------------------------------
                    # Total loss
                    # L_total = L_REINFORCE + β · KL
                    # Both terms have grad through noise_pred.
                    # ------------------------------------------------
                    loss = L_reinforce + args.kl_beta * kl_penalty

                    avg_loss = self.accelerator.gather(
                        loss.repeat(args.train_batch_size)
                    ).mean()
                    train_loss_accum += avg_loss.item() / args.gradient_accumulation_steps

                    # Backward — hooks scale gradients per block automatically
                    self.accelerator.backward(loss)

                    # ------------------------------------------------
                    # Gradient norm logging (this MUST be > 0 now)
                    # ------------------------------------------------
                    grad_norm = 0.0
                    if self.accelerator.sync_gradients:
                        raw = self.unet.module if hasattr(self.unet, 'module') else self.unet
                        for p in raw.parameters():
                            if p.grad is not None:
                                grad_norm += p.grad.data.norm(2).item() ** 2
                        grad_norm = grad_norm ** 0.5

                        self.accelerator.clip_grad_norm_(
                            self.unet.parameters(), args.max_grad_norm
                        )

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # --------------------------------------------------------
                # Logging
                # --------------------------------------------------------
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    log_dict = {
                        "train_loss":      train_loss_accum,
                        "reward":          R_scalar,
                        "advantage":       advantage,
                        "reward_baseline": self.reward_baseline,
                        "log_prob":        log_prob.item(),
                        "reinforce_loss":  L_reinforce.item(),
                        "kl_penalty":      kl_penalty.item(),
                        "grad_norm":       grad_norm,
                        "skipped_steps":   skipped_steps,
                        "w_clip":          w_clip,
                        "w_aes":           w_aes,
                        "w_pick":          w_pick,
                        "w_quality":       w_quality,
                        "mid_timestep":    mid_timestep,
                        **{f"head/{k}": v for k, v in head_scores.items()},
                    }
                    self.accelerator.log(log_dict, step=global_step)
                    train_loss_accum = 0.0

                    if self.accelerator.is_main_process and self.graph_logger is not None:
                        self.graph_logger.update(global_step, {
                            'train_loss':    log_dict['train_loss'],
                            'reward':        log_dict['reward'],
                            'kl_penalty':    log_dict['kl_penalty'],
                            'log_prob':      log_dict['log_prob'],
                            'reinforce_loss':log_dict['reinforce_loss'],
                            'w_clip':        log_dict['w_clip'],
                            'w_aes':         log_dict['w_aes'],
                            'w_pick':        log_dict['w_pick'],
                            'w_quality':     log_dict['w_quality'],
                            'mid_timestep':  log_dict['mid_timestep'],
                            'r_clip':        log_dict['head/r_clip'],
                            'r_aesthetic':   log_dict['head/r_aesthetic'],
                            'r_pick':        log_dict['head/r_pick'],
                            'r_laplacian':   log_dict['head/r_laplacian'],
                            'grad_norm':     log_dict['grad_norm'],
                        })

                        if global_step % args.graph_every == 0:
                            # Clear CUDA cache before image generation to avoid OOM
                            # (graph generation spikes VRAM on top of existing allocations)
                            torch.cuda.empty_cache()
                            import gc
                            gc.collect()
                            try:
                                sample_images = self._generate_sample_images(
                                    batch["prompts"][:4], args
                                )
                                self.graph_logger.save_all(
                                    step=global_step,
                                    generated_images=sample_images,
                                    prompts=batch["prompts"][:4]
                                )
                            except torch.cuda.OutOfMemoryError:
                                logger.warning(
                                    f"OOM during graph generation at step {global_step} "
                                    f"— skipping images, saving curves only."
                                )
                                torch.cuda.empty_cache()
                                gc.collect()
                                self.graph_logger.save_all(
                                    step=global_step,
                                    generated_images=None,
                                    prompts=None
                                )

                    if global_step % args.checkpointing_steps == 0:
                        if self.accelerator.is_main_process:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            self.accelerator.save_state(save_path)
                            logger.info(f"Saved checkpoint to {save_path}")

                progress_bar.set_postfix({
                    "loss":  loss.detach().item(),
                    "R":     f"{R_scalar:.4f}",
                    "adv":   f"{advantage:.4f}",
                    "base":  f"{self.reward_baseline:.4f}",
                    "KL":    f"{kl_penalty.item():.4f}",
                    "||g||": f"{grad_norm:.4f}",
                    "t":     mid_timestep,
                })

                if global_step >= args.max_train_steps:
                    break

            if global_step >= args.max_train_steps:
                break

        # Save final graphs
        if self.accelerator.is_main_process and self.graph_logger is not None:
            prompts_final = list(self.train_dataset[:4][args.caption_column])
            final_images  = self._generate_sample_images(prompts_final, args)
            self.graph_logger.save_all(
                step=global_step,
                generated_images=final_images,
                prompts=prompts_final
            )
            logger.info("Final graphs saved.")

        # Save final pipeline
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unet_final = self.accelerator.unwrap_model(self.unet)
            pipeline   = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=self.text_encoder,
                vae=self.vae,
                unet=unet_final,
                revision=args.revision,
            )
            pipeline.save_pretrained(args.output_dir)
            logger.info(f"Final pipeline saved to {args.output_dir}")

        self.layer_router.remove()
        self.accelerator.end_training()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args    = parse_args()
    trainer = Trainer(args)
    trainer.train(args)