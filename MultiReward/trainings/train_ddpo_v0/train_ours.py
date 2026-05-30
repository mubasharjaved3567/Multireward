import os, sys, json, copy, argparse, random, shutil
import numpy as np
import torch, torch.nn.functional as F
import bitsandbytes as bnb
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from diffusers import StableDiffusionPipeline, DDIMScheduler
from huggingface_hub import HfApi, login

from MultiReward.trainings.train_ddpo_v0.reward_heads import RewardHeads, norm_scores, smooth_weights
from layer_router import LayerRouter


CFG = {
    "model_id":            "CompVis/stable-diffusion-v1-4",
    "train_data":          "data/refl_data.json",   # local path
    "num_epochs":          1,
    "lr":                  1e-5,
    "grad_clip":           1.0,
    "num_inference_steps": 30,
    "kl_beta":             0.01,
    "output_dir":          "checkpoints",
    "log_path":            "train_log.jsonl",
    "hf_repo":             "mubasharjaved/strw-4head-layer-routing",
    "upload_every":        100,   # upload checkpoint every N steps
}


def write_log(r):
    with open(CFG["log_path"], "a") as f:
        f.write(json.dumps(r) + "\n")


def decode(vae, latents):
    latents = latents / vae.config.scaling_factor
    latents = latents.to(next(vae.parameters()).device).to(next(vae.parameters()).dtype)
    with torch.no_grad():
        px = vae.decode(latents).sample
    px = (px / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
    return Image.fromarray((px[0] * 255).astype(np.uint8))


def offload_heads(heads):
    """Move CLIP+LAION to CPU before backward to free ~2.3 GB VRAM."""
    heads.clip_model.cpu()
    heads.aes_clip.cpu()
    heads.aes_mlp.cpu()
    torch.cuda.empty_cache()


def reload_heads(heads, device):
    """Move CLIP+LAION back to GPU for next scoring step."""
    heads.clip_model.to(device)
    heads.aes_clip.to(device)
    heads.aes_mlp.to(device)


def upload_checkpoint(pipe, gs, hf_api):
    """Save pipeline, upload to HF, delete from disk."""
    ckpt_path = f"{CFG['output_dir']}/checkpoint_step{gs}"
    pipe.save_pretrained(ckpt_path)
    print(f"\n[Step {gs}] Uploading checkpoint_step{gs} to HF...")
    try:
        hf_api.create_repo(repo_id=CFG["hf_repo"], repo_type="model", exist_ok=True)
        hf_api.upload_folder(
            folder_path=ckpt_path,
            repo_id=CFG["hf_repo"],
            repo_type="model",
            path_in_repo=f"checkpoint_step{gs}",
        )
        shutil.rmtree(ckpt_path)
        print(f"[Step {gs}] Uploaded + deleted from disk")
    except Exception as e:
        print(f"[Step {gs}] Upload failed: {e} — checkpoint kept at {ckpt_path}")


def train(args):
    Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    max_steps = getattr(args, "max_steps", None)
    if max_steps:
        print(f"Stopping at {max_steps} steps")

    # HF login + API
    if args.hf_token:
        login(token=args.hf_token)
    hf_api = HfApi()

    print("\nLoading SD v1.4...")
    pipe = StableDiffusionPipeline.from_pretrained(
        CFG["model_id"], torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(CFG["num_inference_steps"])

    unet, vae, tok, enc = pipe.unet, pipe.vae, pipe.tokenizer, pipe.text_encoder
    vae.requires_grad_(False)
    enc.requires_grad_(False)

    # FIX: Ref UNet on CPU — saves 3.4 GB VRAM
    print("Creating frozen reference UNet on CPU...")
    ref = copy.deepcopy(unet).to("cpu").float()
    ref.requires_grad_(False).eval()

    unet.requires_grad_(True).train()

    # FIX: 8-bit Adam — saves ~4.8 GB vs standard AdamW
    opt = bnb.optim.AdamW8bit(unet.parameters(), lr=CFG["lr"])
    print(f"Trainable params: {sum(p.numel() for p in unet.parameters()):,}")
    torch.cuda.empty_cache()

    print("\nBuilding LayerRouter...")
    router = LayerRouter(unet)

    print("\nLoading reward heads...")
    heads = RewardHeads(device)
    torch.cuda.empty_cache()

    print("\nLoading prompts...")
    with open(CFG["train_data"]) as f:
        data = json.load(f)
    prompts = [x.get("text", x.get("prompt", "")) for x in data]
    if args.max_samples:
        prompts = prompts[:args.max_samples]
    print(f"{len(prompts)} prompts loaded")

    T             = CFG["num_inference_steps"]
    gs            = 0
    stop_training = False

    for epoch in range(CFG["num_epochs"]):
        if stop_training:
            break
        epoch_R, epoch_kl = [], []

        for prompt in tqdm(prompts, desc=f"Epoch {epoch+1}/{CFG['num_epochs']}"):

            # Encode prompt
            t_ = tok([prompt], padding="max_length",
                     max_length=tok.model_max_length,
                     truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                e = enc(t_.input_ids).last_hidden_state

            # Starting noise
            lat = (torch.randn((1, unet.config.in_channels, 64, 64),
                               device=device, dtype=torch.float16)
                   * pipe.scheduler.init_noise_sigma)

            # Sample random denoising step index
            reward_idx = random.randint(0, T - 1)

            # Full denoising WITHOUT grad
            with torch.no_grad():
                for i, t in enumerate(pipe.scheduler.timesteps):
                    tb   = t.unsqueeze(0).to(device)
                    np_  = unet(lat, tb, e).sample
                    # FIX: scheduler step on CPU to avoid device mismatch
                    step = pipe.scheduler.step(
                        np_.cpu().float(), t.cpu(), lat.cpu().float()
                    )
                    lat = step.prev_sample.to(device).half()
                    if i == reward_idx:
                        lat_at_reward = lat.clone()

            del lat
            torch.cuda.empty_cache()

            # Re-run reward timestep WITH grad
            tb_r       = pipe.scheduler.timesteps[reward_idx].unsqueeze(0).to(device)
            noise_pred = unet(lat_at_reward, tb_r, e).sample

            # Ref pred on CPU
            with torch.no_grad():
                ref_pred = ref(
                    lat_at_reward.detach().cpu().float(),
                    tb_r.cpu(), e.cpu().float()
                ).sample.to(device).half()

            kl = 0.5 * F.mse_loss(noise_pred.float(), ref_pred.float())

            # Decode with detached latent — no graph held
            with torch.no_grad():
                lat_r = pipe.scheduler.step(
                    noise_pred.detach().cpu().float(),
                    tb_r.cpu(), lat_at_reward.cpu().float()
                ).prev_sample.to(device).half()
            img = decode(vae, lat_r)
            del lat_r
            torch.cuda.empty_cache()

            # Score
            sc = heads.score(prompt, img)
            ns = norm_scores(sc)
            del img

            # Compute smooth_weights for this denoising step index
            # These NOW drive the actual loss — not just logging
            sw = smooth_weights(reward_idx, T)

            # FIX: offload reward heads to CPU before backward (~2.3 GB freed)
            offload_heads(heads)

            # Trajectory-aware routed backward
            # smooth_w passed into router — drives composite_R proportions
            route_log = router.step(
                norm_scores=ns,
                smooth_w=sw,          # ← trajectory-aware weights in loss
                noise_pred=noise_pred,
                ref_pred=ref_pred,
                kl=kl,
                opt=opt,
                kl_beta=CFG["kl_beta"],
                grad_clip=CFG["grad_clip"],
            )

            del noise_pred, ref_pred, kl
            torch.cuda.empty_cache()

            reload_heads(heads, device)

            epoch_R.append(route_log["composite_R"])
            epoch_kl.append(route_log["kl"])
            gs += 1

            log_entry = {
                "epoch":             epoch + 1,
                "global_step":       gs,
                "text":              prompt[:60],
                "reward_idx":        reward_idx,
                # raw scores
                "r_align_raw":       round(sc["r_align"],      4),
                "r_aesthetic_raw":   round(sc["r_aesthetic"],  4),
                "r_preference_raw":  round(sc["r_preference"], 4),
                "r_quality_raw":     round(sc["r_quality"],    2),
                # normalised scores
                "r_align_norm":      round(ns["r_align"],      4),
                "r_aesthetic_norm":  round(ns["r_aesthetic"],  4),
                "r_preference_norm": round(ns["r_preference"], 4),
                "r_quality_norm":    round(ns["r_quality"],    4),
                # smooth weights (now used in loss)
                "sw_align":          round(sw[0].item(),       4),
                "sw_aesthetic":      round(sw[1].item(),       4),
                "sw_preference":     round(sw[2].item(),       4),
                "sw_quality":        round(sw[3].item(),       4),
            }
            log_entry.update(route_log)
            write_log(log_entry)

            # Upload checkpoint every N steps
            if gs % CFG["upload_every"] == 0:
                upload_checkpoint(pipe, gs, hf_api)
                # Also upload log
                try:
                    hf_api.upload_file(
                        path_or_fileobj=CFG["log_path"],
                        path_in_repo="train_log.jsonl",
                        repo_id=CFG["hf_repo"],
                    )
                except Exception as e:
                    print(f"  Log upload failed: {e}")

            # Stop at max_steps
            if max_steps and gs >= max_steps:
                print(f"\nReached max_steps={max_steps} — stopping.")
                stop_training = True
                break

        avg_R  = sum(epoch_R)  / len(epoch_R)
        avg_kl = sum(epoch_kl) / len(epoch_kl)
        print(f"Epoch {epoch+1}  avg_composite_R={avg_R:.4f}  avg_kl={avg_kl:.6f}  steps={gs}")

    # Final upload of anything remaining
    if os.path.exists(CFG["output_dir"]):
        remaining = [d for d in os.listdir(CFG["output_dir"]) if d.startswith("checkpoint")]
        for ckpt in sorted(remaining):
            path = os.path.join(CFG["output_dir"], ckpt)
            print(f"Final upload: {ckpt}")
            try:
                hf_api.upload_folder(folder_path=path, repo_id=CFG["hf_repo"],
                                     repo_type="model", path_in_repo=ckpt)
                shutil.rmtree(path)
            except Exception as e:
                print(f"  Failed: {e}")

    # Final log upload
    try:
        hf_api.upload_file(path_or_fileobj=CFG["log_path"],
                           path_in_repo="train_log.jsonl",
                           repo_id=CFG["hf_repo"])
        print("Final log uploaded")
    except Exception as e:
        print(f"Log upload failed: {e}")

    print(f"\nTraining complete. Total steps: {gs}")
    print(f"Model: https://huggingface.co/{CFG['hf_repo']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_steps",   type=int, default=None)
    p.add_argument("--hf_token",    type=str, default=None,
                   help="HuggingFace token for uploads")
    train(p.parse_args())