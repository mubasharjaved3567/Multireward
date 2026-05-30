"""
eval_fid.py
===========
Generate 1000 images from checkpoint (or any checkpoint) and compute
FID score against COCO val2017.

USAGE:
    # Generate + FID vs COCO (full pipeline using your final checkpoint)
    modal run eval_fid.py --checkpoint checkpoint-final
"""

import os
import sys
import modal

app   = modal.App("ddpo-altaf-eval")
volume = modal.Volume.from_name("ddpo-altaf-outputs", create_if_missing=True)
VOLUME_PATH = "/outputs"

# Separate eval volume for generated images + FID results
eval_volume = modal.Volume.from_name("ddpo-altaf-eval", create_if_missing=True)
EVAL_PATH   = "/eval"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "wget", "libgl1", "libglib2.0-0"])
    .pip_install(["numpy==1.26.4"])
    .pip_install([
        "torch==2.1.2",
        "torchvision==0.16.2",
    ])
    .pip_install([
        "huggingface_hub==0.20.3",
        "diffusers==0.21.4",
        "transformers==4.36.2",
        "accelerate==0.25.0",
        "Pillow",
        "tqdm",
        "scipy",
        "pytorch-fid==0.3.0",
        "safetensors",
        "requests",
    ])
    .add_local_dir("data", "/root/data")
)


# ---------------------------------------------------------------------------
# Step 1: Generate images from checkpoint (With 77-token truncation)
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",
    timeout=7200,
    image=image,
    volumes={
        VOLUME_PATH: volume,
        EVAL_PATH:   eval_volume,
    },
)
def generate_images(
    checkpoint: str  = "checkpoint-final",
    num_images: int  = 1000,
    steps: int       = 50,
    cfg_scale: float = 7.5,
    seed: int        = 42,
):
    import torch
    import json
    import random
    import glob
    from PIL import Image
    from tqdm import tqdm
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    from transformers import CLIPTokenizer

    device = torch.device("cuda")

    # Load prompts
    with open("/root/data/refl_data.json", "r") as f:
        data = json.load(f)

    # Consistent shuffling across runs using the explicit evaluation seed
    random.seed(seed)
    prompts = [item["text"] for item in data]
    random.shuffle(prompts)
    prompts = prompts[:num_images]
    print(f"Loaded {len(prompts)} prompts for evaluation.")

    ckpt_path = os.path.join(VOLUME_PATH, "checkpoint", checkpoint)

    print(f"Loading base SD v1.4 pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None,
    )

    if os.path.exists(ckpt_path):
        print(f"Found checkpoint at {ckpt_path}")
        loaded = False

        sf_path = os.path.join(ckpt_path, "model.safetensors")
        if os.path.exists(sf_path):
            from safetensors.torch import load_file
            state_dict = load_file(sf_path)
            pipe.unet.load_state_dict(state_dict, strict=False)
            print(f"Loaded UNet weights from {sf_path}")
            loaded = True

        if not loaded:
            bin_path = os.path.join(ckpt_path, "pytorch_model.bin")
            if os.path.exists(bin_path):
                state_dict = torch.load(bin_path, map_location="cpu")
                pipe.unet.load_state_dict(state_dict, strict=False)
                print(f"Loaded UNet weights from {bin_path}")
                loaded = True

        if not loaded:
            shards = sorted(glob.glob(os.path.join(ckpt_path, "pytorch_model*.bin")))
            if shards:
                import collections
                state_dict = collections.OrderedDict()
                for shard in shards:
                    state_dict.update(torch.load(shard, map_location="cpu"))
                pipe.unet.load_state_dict(state_dict, strict=False)
                print(f"Loaded UNet weights from {len(shards)} shards")
                loaded = True

        if not loaded:
            print("WARNING: Could not find UNet weights in checkpoint. Falling back to base model.")
    else:
        print(f"Checkpoint path not found: {ckpt_path}. Using base model.")

    pipe = pipe.to(device, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()

    # Setup tokenizer truncation to completely prevent 77+ token warnings
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    out_dir = os.path.join(EVAL_PATH, "generated", checkpoint.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating {num_images} images → {out_dir}")

    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        out_path = os.path.join(out_dir, f"img_{i:04d}.png")
        if os.path.exists(out_path):
            continue
        
        # Enforce exact 77-token clipping before generation
        tokens = tokenizer(prompt, truncation=True, max_length=77, return_tensors="pt")
        clean_prompt = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

        with torch.no_grad():
            image = pipe(
                clean_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=torch.Generator(device=device).manual_seed(seed + i),
                height=512,
                width=512,
            ).images[0]
        image.save(out_path)

    with open(os.path.join(out_dir, "prompts_used.json"), "w") as f:
        json.dump(prompts, f, indent=2)

    eval_volume.commit()
    print(f"\nDone. {num_images} images saved to {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Step 2: Download COCO val2017 reference images
# ---------------------------------------------------------------------------
@app.function(
    timeout=3600,
    image=image,
    volumes={EVAL_PATH: eval_volume},
)
def download_coco_reference(num_ref: int = 2000):
    import urllib.request
    import zipfile
    from PIL import Image
    from tqdm import tqdm
    import random
    import json

    coco_dir = os.path.join(EVAL_PATH, "coco_ref")
    os.makedirs(coco_dir, exist_ok=True)

    existing = [f for f in os.listdir(coco_dir) if f.endswith(".png")]
    if len(existing) >= num_ref:
        print(f"COCO reference already downloaded ({len(existing)} images). Skipping.")
        return coco_dir

    print(f"Downloading COCO val2017 annotations to get image URLs...")
    ann_url  = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ann_path = "/tmp/coco_annotations.zip"

    if not os.path.exists("/tmp/coco_captions_val2017.json"):
        urllib.request.urlretrieve(ann_url, ann_path)
        with zipfile.ZipFile(ann_path, "r") as z:
            z.extract("annotations/captions_val2017.json", "/tmp/")
        os.rename("/tmp/annotations/captions_val2017.json", "/tmp/coco_captions_val2017.json")

    with open("/tmp/coco_captions_val2017.json") as f:
        coco = json.load(f)

    images  = coco["images"]
    random.seed(42)
    random.shuffle(images)
    images  = images[:num_ref]

    print(f"Downloading {num_ref} COCO val2017 images...")
    failed = 0
    for item in tqdm(images, desc="COCO download"):
        img_id   = item["id"]
        filename = item["file_name"]
        url      = f"http://images.cocodataset.org/val2017/{filename}"
        out_path = os.path.join(coco_dir, f"{img_id}.png")

        if os.path.exists(out_path):
            continue
        try:
            urllib.request.urlretrieve(url, "/tmp/coco_tmp.jpg")
            img = Image.open("/tmp/coco_tmp.jpg").convert("RGB")
            img = img.resize((512, 512), Image.LANCZOS)
            img.save(out_path)
        except Exception as e:
            failed += 1
            if failed > 50:
                break

    downloaded = len([f for f in os.listdir(coco_dir) if f.endswith(".png")])
    eval_volume.commit()
    print(f"Downloaded {downloaded} COCO reference images → {coco_dir}")
    return coco_dir


# ---------------------------------------------------------------------------
# Step 3: Compute FID
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",
    timeout=3600,
    image=image,
    volumes={EVAL_PATH: eval_volume},
)
def compute_fid(
    checkpoint: str = "checkpoint-final",
    num_ref: int    = 2000,
):
    import subprocess
    import json

    gen_dir  = os.path.join(EVAL_PATH, "generated", checkpoint.replace("/", "_"))
    coco_dir = os.path.join(EVAL_PATH, "coco_ref")

    gen_images  = [f for f in os.listdir(gen_dir)  if f.endswith(".png")]
    coco_images = [f for f in os.listdir(coco_dir) if f.endswith(".png")]

    print(f"Generated images : {len(gen_images)}")
    print(f"COCO ref images  : {len(coco_images)}")

    print(f"\nComputing FID: {gen_dir} vs {coco_dir}")

    result = subprocess.run(
        ["python", "-m", "pytorch_fid", gen_dir, coco_dir,
         "--device", "cuda", "--num-workers", "4"],
        capture_output=True, text=True
    )

    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    fid_score = None
    for line in result.stdout.split("\n"):
        if "FID" in line:
            try:
                fid_score = float(line.split(":")[-1].strip())
            except:
                pass

    results = {
        "checkpoint":    checkpoint,
        "num_generated": len(gen_images),
        "num_reference": len(coco_images),
        "fid_score":     fid_score,
        "raw_output":    result.stdout,
    }

    results_path = os.path.join(EVAL_PATH, f"fid_results_{checkpoint.replace('/', '_')}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    eval_volume.commit()
    return fid_score


# ---------------------------------------------------------------------------
# Compute baseline FID (base SD v1.4, no finetuning) for comparison
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",
    timeout=7200,
    image=image,
    volumes={EVAL_PATH: eval_volume},
)
def generate_baseline_images(
    num_images: int  = 1000,
    steps: int       = 50,
    cfg_scale: float = 7.5,
    seed: int        = 42,
):
    import torch
    import json
    import random
    from tqdm import tqdm
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    from transformers import CLIPTokenizer

    device = torch.device("cuda")

    with open("/root/data/refl_data.json", "r") as f:
        data = json.load(f)

    random.seed(seed)
    prompts = [item["text"] for item in data]
    random.shuffle(prompts)
    prompts = prompts[:num_images]

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    out_dir = os.path.join(EVAL_PATH, "generated", "baseline_sd14")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating {num_images} baseline images → {out_dir}")
    for i, prompt in enumerate(tqdm(prompts, desc="Baseline generation")):
        out_path = os.path.join(out_dir, f"img_{i:04d}.png")
        if os.path.exists(out_path):
            continue

        # Enforce exact 77-token clipping before generation
        tokens = tokenizer(prompt, truncation=True, max_length=77, return_tensors="pt")
        clean_prompt = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

        with torch.no_grad():
            img = pipe(
                clean_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=torch.Generator(device=device).manual_seed(seed + i),
                height=512, width=512,
            ).images[0]
        img.save(out_path)

    with open(os.path.join(out_dir, "prompts_used.json"), "w") as f:
        json.dump(prompts, f, indent=2)

    eval_volume.commit()
    print(f"Baseline images saved to {out_dir}")
    return out_dir


@app.function(
    image=modal.Image.debian_slim(),
    volumes={EVAL_PATH: eval_volume},
)
def download_eval():
    results = {}
    for root, dirs, files in os.walk(EVAL_PATH):
        for fname in files:
            if "coco_ref" in root:
                continue
            fpath = os.path.join(root, fname)
            rel   = os.path.relpath(fpath, EVAL_PATH)
            with open(fpath, "rb") as f:
                results[rel] = f.read()
    return results


# ---------------------------------------------------------------------------
# Local entrypoint (Now defaulting to 1000 images)
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    mode:       str   = "all",
    checkpoint: str   = "checkpoint-final", 
    num_images: int   = 1000,                # Upgraded default to 1000
    num_ref:    int   = 2000,
):
    if mode == "all":
        print(f"\n{'='*60}")
        print(f"Full FID evaluation pipeline (1,000 Image Upgrade)")
        print(f"Checkpoint : {checkpoint}")
        print(f"Images     : {num_images} generated vs {num_ref} COCO ref")
        print(f"{'='*60}\n")

        print("Step 1/5: Generating images from checkpoint...")
        generate_images.remote(checkpoint=checkpoint, num_images=num_images)

        print("Step 2/5: Generating baseline images from SD v1.4...")
        generate_baseline_images.remote(num_images=num_images)

        print("Step 3/5: Downloading COCO val2017 reference...")
        download_coco_reference.remote(num_ref=num_ref)

        print("Step 4/5: Computing FID for checkpoint...")
        fid_ckpt = compute_fid.remote(checkpoint=checkpoint, num_ref=num_ref)

        print("Step 5/5: Computing FID for baseline...")
        fid_base = compute_fid.remote(checkpoint="baseline_sd14", num_ref=num_ref)

        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"SD v1.4 baseline FID (no finetuning) : {fid_base:.2f}")
        print(f"{checkpoint} FID                      : {fid_ckpt:.2f}")
        delta = fid_base - fid_ckpt
        direction = "BETTER" if delta > 0 else "WORSE"
        print(f"Delta                                 : {delta:+.2f} ({direction})")
        print(f"{'='*60}")

    elif mode == "generate":
        generate_images.remote(checkpoint=checkpoint, num_images=num_images)
    elif mode == "baseline":
        generate_baseline_images.remote(num_images=num_images)
    elif mode == "coco":
        download_coco_reference.remote(num_ref=num_ref)
    elif mode == "fid":
        fid = compute_fid.remote(checkpoint=checkpoint, num_ref=num_ref)
        print(f"\nFID Score: {fid}")
    elif mode == "download":
        results = download_eval.remote()
        for rel_path, data in results.items():
            local_path = os.path.join("eval_outputs", rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(data)