"""
DDPO v2 — All 10 fixes applied
Run:
  !PYTORCH_ALLOC_CONF=expandable_segments:True \
   python /kaggle/working/train_ddpo_v2.py --max_steps 500 --max_samples 3000
"""

import os, json, copy, random, shutil, argparse, gc
import numpy as np
import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    "model_id":            "CompVis/stable-diffusion-v1-4",
    "train_data":          "/kaggle/input/datasets/mubasharalidataai/multirewards-v1/refl_data.json",
    "num_denoising_steps": 50,       # change 6
    "batch_size":          2,        # change 8
    "lr":                  3e-6,     # change 5
    "kl_beta":             0.1,      # change 4
    "grad_clip":           1.0,
    "output_dir":          "/kaggle/working/checkpoints",
    "log_path":            "/kaggle/working/train_log_v2.jsonl",
    "hf_repo":             "mubasharjaved/strw-4head-layer-routing",
    "upload_every":        100,
}

# Change 9 — Fixed reward weights (all 4 always active)
FIXED_W = {
    "r_align":      0.35,
    "r_aesthetic":  0.30,
    "r_preference": 0.25,
    "r_quality":    0.10,
}

NORM = {
    "r_align":      30.0,
    "r_aesthetic":   8.0,
    "r_preference":  0.2,
    "r_quality":  5000.0,
}

# Layer routing — unchanged
ROUTING = {
    "r_align":      {"primary": ["cross_attn"], "secondary": ["self_attn"]},
    "r_aesthetic":  {"primary": ["self_attn", "conv"], "secondary": ["cross_attn"]},
    "r_preference": {"primary": ["self_attn", "cross_attn"], "secondary": ["conv"]},
    "r_quality":    {"primary": ["resnets"], "secondary": ["conv"]},
}
SECONDARY_SCALE = 0.3

def is_group(name, group):
    if group == "cross_attn": return "attn2" in name
    if group == "self_attn":  return "attn1" in name
    if group == "conv":       return "conv" in name and "attn" not in name and "norm" not in name
    if group == "resnets":    return "resnets" in name
    return False

# ── Reward heads ──────────────────────────────────────────────────────────────
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
        print("  [OK] r_align — CLIP ViT-B/32")

    def _load_aesthetic(self):
        import clip, urllib.request

        class AestheticMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(768, 1024), torch.nn.Dropout(0.2),
                    torch.nn.Linear(1024, 128), torch.nn.Dropout(0.2),
                    torch.nn.Linear(128,   64), torch.nn.Dropout(0.1),
                    torch.nn.Linear(64,    16),
                    torch.nn.Linear(16,     1),
                )
            def forward(self, x): return self.layers(x)

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
        from transformers import AutoProcessor, AutoModel
        self.pick_proc  = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.pick_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").to("cpu").eval()
        self.pick_model.requires_grad_(False)
        print("  [OK] r_preference — PickScore (CPU)")

    def to_cpu(self):
        self.clip_model.cpu(); self.aes_clip.cpu(); self.aes_mlp.cpu()
        gc.collect(); torch.cuda.empty_cache()

    def to_gpu(self):
        self.clip_model.to(self.device)
        self.aes_clip.to(self.device)
        self.aes_mlp.to(self.device)

    @torch.no_grad()
    def score(self, prompt, image):
        import clip as C
        i   = self.clip_prep(image).unsqueeze(0).to(self.device)
        t   = C.tokenize([prompt], truncate=True).to(self.device)
        if_ = self.clip_model.encode_image(i);  if_ = if_ / if_.norm(dim=-1, keepdim=True)
        tf_ = self.clip_model.encode_text(t);   tf_ = tf_ / tf_.norm(dim=-1, keepdim=True)
        r_align = (if_ * tf_).sum().item()

        a   = self.aes_prep(image).unsqueeze(0).to(self.device)
        f   = self.aes_clip.encode_image(a).float()
        f   = f / f.norm(dim=-1, keepdim=True)
        r_aesthetic = self.aes_mlp(f).item()

        pi = self.pick_proc(images=[image], return_tensors="pt", padding=True)
        pt = self.pick_proc(text=[prompt],  return_tensors="pt", padding=True, truncation=True)
        ie = self.pick_model.get_image_features(**pi)
        if hasattr(ie, 'pooler_output'): ie = ie.pooler_output
        ie = ie / ie.norm(dim=-1, keepdim=True)
        te = self.pick_model.get_text_features(**pt)
        if hasattr(te, 'pooler_output'): te = te.pooler_output
        te = te / te.norm(dim=-1, keepdim=True)
        r_preference = (ie * te).sum().item()

        g      = np.array(image.convert("L")).astype(np.float32)
        gy, gx = np.gradient(g)
        lap    = np.gradient(gx)[1] + np.gradient(gy)[0]
        r_quality = float(np.var(lap))

        return dict(r_align=r_align, r_aesthetic=r_aesthetic,
                    r_preference=r_preference, r_quality=r_quality)

def normalize(sc):
    return {
        "r_align":      sc["r_align"]      / NORM["r_align"],
        "r_aesthetic":  sc["r_aesthetic"]  / NORM["r_aesthetic"],
        "r_preference": max(sc["r_preference"] - 0.15, 0.0) / NORM["r_preference"],
        "r_quality":    min(sc["r_quality"] / NORM["r_quality"], 1.0),
    }

def decode(vae, latent):
    with torch.no_grad():
        lat = latent.detach() / vae.config.scaling_factor
        lat = lat.to(next(vae.parameters()).device).to(next(vae.parameters()).dtype)
        px  = vae.decode(lat).sample
    px = (px / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
    px = np.nan_to_num(px, nan=0.0, posinf=1.0, neginf=0.0)
    return Image.fromarray((px[0] * 255).astype(np.uint8))

# ── DDPO step ─────────────────────────────────────────────────────────────────
def ddpo_step(unet, noise_preds, ref_preds, kls, ns_list, opt):
    """
    Minibatch DDPO step.
    noise_preds, ref_preds, kls: lists of batch_size tensors
    ns_list: list of normalized score dicts
    """
    keys = ["r_align", "r_aesthetic", "r_preference", "r_quality"]

    # Average composite reward across batch
    composite_Rs = [sum(ns[k] * FIXED_W[k] for k in keys) for ns in ns_list]
    composite_R  = sum(composite_Rs) / len(composite_Rs)

    # Layer routing weights (based on average reward)
    avg_ns  = {k: sum(ns[k] for ns in ns_list) / len(ns_list) for k in keys}
    param_w = {}
    for k, routing in ROUTING.items():
        r_val = avg_ns[k] * FIXED_W[k]
        for name, param in unet.named_parameters():
            if not param.requires_grad:
                continue
            for g in routing["primary"]:
                if is_group(name, g):
                    param_w[name] = param_w.get(name, 0.0) + r_val
            for g in routing["secondary"]:
                if is_group(name, g):
                    param_w[name] = param_w.get(name, 0.0) + r_val * SECONDARY_SCALE

    # Average loss across batch
    opt.zero_grad()
    total_loss = torch.tensor(0.0, device=noise_preds[0].device)
    total_kl   = torch.tensor(0.0, device=noise_preds[0].device)

    for noise_pred, ref_pred, kl in zip(noise_preds, ref_preds, kls):
        log_prob    = -0.5 * F.mse_loss(noise_pred.float(), ref_pred.float(), reduction="mean")
        step_loss   = -composite_R * log_prob + CFG["kl_beta"] * kl.float()
        total_loss  = total_loss + step_loss
        total_kl    = total_kl + kl.float()

    # Average over batch
    loss    = total_loss / len(noise_preds)
    avg_kl  = total_kl  / len(noise_preds)

    if torch.isfinite(loss) and composite_R > 1e-6:
        loss.backward()

        # Layer routing — zero non-routed grads
        for name, param in unet.named_parameters():
            if param.grad is not None and name not in param_w:
                param.grad.zero_()

        # Grad norms per group
        group_gnorms = {}
        for group in ["cross_attn", "self_attn", "conv", "resnets"]:
            norms = [p.grad.float().norm().item()
                     for n, p in unet.named_parameters()
                     if p.grad is not None and p.grad.abs().max() > 0 and is_group(n, group)]
            group_gnorms[f"gnorm_{group}"] = round(sum(norms)/len(norms), 6) if norms else 0.0

        total_gnorm = torch.nn.utils.clip_grad_norm_(
            unet.parameters(), CFG["grad_clip"]
        ).item()
        opt.step()
    else:
        group_gnorms = {f"gnorm_{g}": 0.0 for g in ["cross_attn","self_attn","conv","resnets"]}
        total_gnorm  = 0.0

    return {
        "composite_R":      round(composite_R, 4),
        "loss":             round(loss.item(), 6) if torch.isfinite(loss) else 0.0,
        "kl":               round(avg_kl.item(), 6),
        "total_gnorm":      round(total_gnorm, 6),
        "gnorm_cross_attn": group_gnorms["gnorm_cross_attn"],
        "gnorm_self_attn":  group_gnorms["gnorm_self_attn"],
        "gnorm_conv":       group_gnorms["gnorm_conv"],
        "gnorm_resnets":    group_gnorms["gnorm_resnets"],
    }

# ── HF upload ─────────────────────────────────────────────────────────────────
def upload(pipe, gs, api):
    ckpt = f"{CFG['output_dir']}/v2_step_{gs}"
    pipe.save_pretrained(ckpt)
    print(f"\n[Step {gs}] Uploading to HF...")
    try:
        api.create_repo(repo_id=CFG["hf_repo"], repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=ckpt, repo_id=CFG["hf_repo"],
                          repo_type="model", path_in_repo=f"v2_step_{gs}")
        shutil.rmtree(ckpt)
        print(f"[Step {gs}] Uploaded + deleted")
    except Exception as e:
        print(f"[Step {gs}] Upload failed: {e}")
    try:
        api.upload_file(path_or_fileobj=CFG["log_path"],
                        path_in_repo="train_log_v2.jsonl",
                        repo_id=CFG["hf_repo"])
    except Exception as e:
        print(f"[Step {gs}] Log upload failed: {e}")

# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    from diffusers import StableDiffusionPipeline, DDIMScheduler
    from huggingface_hub import HfApi

    Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\nLoading SD v1.4...")
    pipe = StableDiffusionPipeline.from_pretrained(
        CFG["model_id"], torch_dtype=torch.float16, safety_checker=None,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(CFG["num_denoising_steps"])

    unet = pipe.unet
    vae  = pipe.vae
    tok  = pipe.tokenizer
    enc  = pipe.text_encoder

    # Change 1+2+3: full fp32 UNet + gradient checkpointing + AdamW8bit
    unet = unet.float()
    unet.requires_grad_(True)
    unet.enable_gradient_checkpointing()
    vae.requires_grad_(False)
    enc.requires_grad_(False)

    print(f"UNet: fp32 | All params trainable: {sum(p.numel() for p in unet.parameters()):,}")
    print(f"Gradient checkpointing: ON")

    ref_unet = copy.deepcopy(unet).cpu().eval()
    ref_unet.requires_grad_(False)

    opt = bnb.optim.AdamW8bit(unet.parameters(), lr=CFG["lr"])
    print(f"Optimizer: AdamW8bit | lr={CFG['lr']} | kl_beta={CFG['kl_beta']}")
    gc.collect(); torch.cuda.empty_cache()

    print("\nLoading reward heads...")
    heads = RewardHeads(device)
    gc.collect(); torch.cuda.empty_cache()

    print("\nLoading prompts...")
    with open(CFG["train_data"]) as f:
        data = json.load(f)
    prompts = [x.get("text", x.get("prompt", "")) for x in data]
    if args.max_samples:
        prompts = prompts[:args.max_samples]
    print(f"{len(prompts)} prompts")

    T          = CFG["num_denoising_steps"]
    batch_size = CFG["batch_size"]
    gs         = 0
    api        = HfApi()
    log_f      = open(CFG["log_path"], "a")

    print(f"\nStarting — max_steps={args.max_steps} | batch={batch_size}")
    print(f"Denoising steps: {T} | Reward from steps {T-10} to {T-1}")
    print(f"Fixed weights: {FIXED_W}\n")

    prompt_pool = prompts * 100
    idx = 0

    while True:
        if args.max_steps and gs >= args.max_steps:
            break

        # Get batch of prompts
        batch_prompts = [prompt_pool[(idx + i) % len(prompt_pool)] for i in range(batch_size)]
        idx += batch_size

        noise_preds, ref_preds, kls, ns_list, sc_list, reward_idxs = [], [], [], [], [], []

        for prompt in batch_prompts:
            # Encode prompt
            t_ = tok([prompt], padding="max_length", max_length=tok.model_max_length,
                     truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = enc(t_.input_ids).last_hidden_state.float()

            # Starting latent
            lat = (torch.randn(1, unet.config.in_channels, 64, 64, device=device).float()
                   * pipe.scheduler.init_noise_sigma)

            # Change 7: last 10 steps dynamically
            reward_idx = random.randint(T - 10, T - 1)
            reward_idxs.append(reward_idx)
            lat_at_r = None

            # Full denoise WITHOUT grad
            with torch.no_grad():
                for i, t in enumerate(pipe.scheduler.timesteps):
                    tb  = t.unsqueeze(0).to(device)
                    np_ = unet(lat, tb, emb).sample
                    out = pipe.scheduler.step(np_.detach().cpu().float(),
                                              t.cpu(), lat.detach().cpu().float())
                    lat = out.prev_sample.to(device).float()
                    if i == reward_idx:
                        lat_at_r = lat.clone()

            gc.collect(); torch.cuda.empty_cache()

            # Re-run reward step WITH grad
            tb_r       = pipe.scheduler.timesteps[reward_idx].unsqueeze(0).to(device)
            noise_pred = unet(lat_at_r, tb_r, emb).sample

            # Ref pred on CPU
            with torch.no_grad():
                ref_pred = ref_unet(
                    lat_at_r.detach().cpu().float(),
                    tb_r.cpu(), emb.cpu().float(),
                ).sample.to(device).float()

            kl = 0.5 * F.mse_loss(noise_pred.float(), ref_pred.float())

            # Decode
            with torch.no_grad():
                out_r = pipe.scheduler.step(
                    noise_pred.detach().cpu().float(),
                    tb_r.cpu(), lat_at_r.detach().cpu().float()
                )
                img = decode(vae, out_r.prev_sample.to(device).float())

            gc.collect(); torch.cuda.empty_cache()

            sc = heads.score(prompt, img)
            ns = normalize(sc)
            del img

            noise_preds.append(noise_pred)
            ref_preds.append(ref_pred)
            kls.append(kl)
            ns_list.append(ns)
            sc_list.append(sc)

        # Offload heads before backward
        heads.to_cpu()

        # DDPO step on batch
        step_log = ddpo_step(unet, noise_preds, ref_preds, kls, ns_list, opt)

        for np_, rp_, kl_ in zip(noise_preds, ref_preds, kls):
            del np_, rp_, kl_
        gc.collect(); torch.cuda.empty_cache()
        heads.to_gpu()
        gs += 1

        # Average scores across batch for logging
        avg_sc = {k: sum(sc[k] for sc in sc_list)/len(sc_list) for k in sc_list[0]}
        avg_ns = {k: sum(ns[k] for ns in ns_list)/len(ns_list) for k in ns_list[0]}

        # Log
        entry = {
            "step":              gs,
            "prompts":           [p[:40] for p in batch_prompts],
            "reward_idxs":       reward_idxs,
            "batch_size":        batch_size,
            # raw scores (averaged)
            "r_align_raw":       round(avg_sc["r_align"],      4),
            "r_aesthetic_raw":   round(avg_sc["r_aesthetic"],  4),
            "r_preference_raw":  round(avg_sc["r_preference"], 4),
            "r_quality_raw":     round(avg_sc["r_quality"],    2),
            # normalised (averaged)
            "r_align_norm":      round(avg_ns["r_align"],      4),
            "r_aesthetic_norm":  round(avg_ns["r_aesthetic"],  4),
            "r_preference_norm": round(avg_ns["r_preference"], 4),
            "r_quality_norm":    round(avg_ns["r_quality"],    4),
            # fixed weights
            "fw_align":          FIXED_W["r_align"],
            "fw_aesthetic":      FIXED_W["r_aesthetic"],
            "fw_preference":     FIXED_W["r_preference"],
            "fw_quality":        FIXED_W["r_quality"],
            # training signal
            "composite_R":       step_log["composite_R"],
            "loss":              step_log["loss"],
            "kl":                step_log["kl"],
            "total_gnorm":       step_log["total_gnorm"],
            "gnorm_cross_attn":  step_log["gnorm_cross_attn"],
            "gnorm_self_attn":   step_log["gnorm_self_attn"],
            "gnorm_conv":        step_log["gnorm_conv"],
            "gnorm_resnets":     step_log["gnorm_resnets"],
        }
        log_f.write(json.dumps(entry) + "\n")
        log_f.flush()

        if gs % 10 == 0:
            print(f"  Step {gs:4d} | R={step_log['composite_R']:.4f} | "
                  f"loss={step_log['loss']:.6f} | kl={step_log['kl']:.6f} | "
                  f"gnorm={step_log['total_gnorm']:.5f} | "
                  f"cross={step_log['gnorm_cross_attn']:.5f} "
                  f"self={step_log['gnorm_self_attn']:.5f} "
                  f"conv={step_log['gnorm_conv']:.5f} "
                  f"res={step_log['gnorm_resnets']:.5f}")

        is_final = args.max_steps and gs >= args.max_steps
        if gs % CFG["upload_every"] == 0 and not is_final:
            upload(pipe, gs, api)

    log_f.close()

    print(f"\nSaving final checkpoint at step {gs}...")
    upload(pipe, gs, api)

    leftover = sorted(Path(CFG["output_dir"]).glob("v2_step_*"))
    for ckpt in leftover:
        try:
            api.upload_folder(folder_path=str(ckpt), repo_id=CFG["hf_repo"],
                              repo_type="model", path_in_repo=ckpt.name)
            shutil.rmtree(ckpt)
        except Exception as e:
            print(f"  {ckpt.name} failed: {e}")

    try:
        api.upload_file(path_or_fileobj=CFG["log_path"],
                        path_in_repo="train_log_v2.jsonl",
                        repo_id=CFG["hf_repo"])
        print("Final log uploaded")
    except Exception as e:
        print(f"Log upload failed: {e}")

    print(f"\nDone. Total steps: {gs}")
    print(f"Model: https://huggingface.co/{CFG['hf_repo']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_steps",   type=int, default=500)
    p.add_argument("--max_samples", type=int, default=None)
    train(p.parse_args())
