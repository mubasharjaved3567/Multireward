# reproduce_benchmark.py
# Table 1 reproduction — uses authors' exact evaluation protocol
import modal

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
        add_python="3.10"
    )
    .apt_install("git", "wget", "unzip")
    .pip_install(
        "transformers==4.38.2",
        "tokenizers==0.15.2",
        "safetensors",
        "diffusers==0.24.0",
        "accelerate==0.27.2",
        "peft==0.7.1",
        "fairscale==0.4.13",
        "timm==0.9.12",
        "ftfy",
        "regex",
        "Pillow",
        "pandas",
        "tqdm",
        "numpy<2.0",
    )
    .pip_install("git+https://github.com/openai/CLIP.git")
    .pip_install("image-reward")
    .run_commands("pip install --force-reinstall --no-deps huggingface_hub==0.22.2")
    .run_commands("git clone https://github.com/zai-org/ImageReward.git /root/ImageReward")
)

app = modal.App("ir-table1", image=image)


def compute_accuracy(score_sample, target_sample):
    """
    Exact copy of the authors' acc() function from test.py.
    
    For each pair (i, j) with i < j:
      - If human ranked i better than j (item_base[i] > item_base[j]):
          model agrees if rewards[i] >= rewards[j]
      - If human ranked j better than i (item_base[i] < item_base[j]):
          model agrees if rewards[j] > rewards[i]
      - If tied: skip
    """
    tol_cnt = 0.0  # total pairs considered
    true_cnt = 0.0  # pairs where model and human disagree on direction
    
    for idx in range(len(score_sample)):
        item_base = score_sample[idx]["ranking"]   # human ranks (lower = better)
        item = target_sample[idx]["rewards"]        # ImageReward scores (higher = better)
        
        for i in range(len(item_base)):
            for j in range(i + 1, len(item_base)):
                # Note: authors use ranks where LOWER number = BETTER rank
                # So item_base[i] > item_base[j] means j is ranked better by humans
                # Reading the original code carefully:
                if item_base[i] > item_base[j]:
                    # Human: j > i (j is better)
                    if item[i] >= item[j]:
                        tol_cnt += 1          # wrong direction (counted but not "true")
                    elif item[i] < item[j]:
                        tol_cnt += 1
                        true_cnt += 1          # right: model also says j > i
                elif item_base[i] < item_base[j]:
                    # Human: i > j (i is better)
                    if item[i] > item[j]:
                        tol_cnt += 1
                        true_cnt += 1          # right: model also says i > j
                    elif item[i] <= item[j]:
                        tol_cnt += 1           # wrong direction
                # tied ranks (item_base[i] == item_base[j]) are skipped
    
    return true_cnt / tol_cnt if tol_cnt > 0 else 0.0


@app.function(gpu="T4", timeout=3600)
def reproduce_table1():
    import os, json, zipfile
    import torch
    from tqdm import tqdm
    from huggingface_hub import hf_hub_download
    import ImageReward as RM
    
    # ===== STEP 1: Set up model weights =====
    print("=" * 70)
    print("[1] Setting up ImageReward weights")
    print("=" * 70)
    cache_dir = "/root/.cache/ImageReward"
    os.makedirs(cache_dir, exist_ok=True)
    for fname in ["ImageReward.pt", "med_config.json"]:
        target = os.path.join(cache_dir, fname)
        if os.path.exists(target):
            print(f"    {fname}: cached")
            continue
        dl = hf_hub_download(repo_id="THUDM/ImageReward", filename=fname)
        real = os.path.realpath(dl)
        with open(real, "rb") as s, open(target, "wb") as d:
            while chunk := s.read(16 * 1024 * 1024):
                d.write(chunk)
        print(f"    {fname}: copied ({os.path.getsize(target)/1e6:.1f} MB)")
    
    # ===== STEP 2: Download and extract test images =====
    print("\n" + "=" * 70)
    print("[2] Setting up benchmark images")
    print("=" * 70)
    extract_dir = "/root/benchmark_images"
    if not os.path.exists(extract_dir) or not os.listdir(extract_dir):
        zip_path = hf_hub_download(repo_id="THUDM/ImageReward", filename="test_images.zip")
        os.makedirs(extract_dir, exist_ok=True)
        print(f"    Extracting test_images.zip...")
        with zipfile.ZipFile(os.path.realpath(zip_path), "r") as z:
            z.extractall(extract_dir)
        print(f"    Extraction complete")
    else:
        print(f"    Images already extracted")
    
    img_prefix = os.path.join(extract_dir, "test_images")
    print(f"    Image prefix: {img_prefix}")
    print(f"    Number of prompt folders: {len([d for d in os.listdir(img_prefix) if d.startswith('part-')])}")
    
    # ===== STEP 3: Load test.json =====
    print("\n" + "=" * 70)
    print("[3] Loading test.json")
    print("=" * 70)
    with open("/root/ImageReward/data/test.json") as f:
        score_sample = json.load(f)
    print(f"    Loaded {len(score_sample)} test samples")
    print(f"    Example: id={score_sample[0]['id']}, n_generations={len(score_sample[0]['generations'])}")
    
    # ===== STEP 4: Load model =====
    print("\n" + "=" * 70)
    print("[4] Loading ImageReward model")
    print("=" * 70)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Device: {device}")
    model = RM.load("ImageReward-v1.0", device=device)
    print(f"    Loaded: {type(model).__name__}")
    
    # ===== STEP 5: Score all samples =====
    print("\n" + "=" * 70)
    print("[5] Running inference on all 466 samples")
    print("=" * 70)
    target_sample = []
    skipped = 0
    
    with torch.no_grad():
        for item in tqdm(score_sample, desc="Scoring"):
            img_list = [os.path.join(img_prefix, img) for img in item["generations"]]
            
            # Verify all images exist
            missing = [p for p in img_list if not os.path.exists(p)]
            if missing:
                if skipped < 3:
                    print(f"    [WARN] Missing images for id={item['id']}: {missing[0]}")
                skipped += 1
                continue
            
            try:
                ranking, rewards = model.inference_rank(item["prompt"], img_list)
            except Exception as e:
                print(f"    [ERROR] {item['id']}: {e}")
                skipped += 1
                continue
            
            target_sample.append({
                "id": item["id"],
                "prompt": item["prompt"],
                "ranking": ranking,
                "rewards": rewards,
            })
    
    print(f"\n    Scored: {len(target_sample)} / {len(score_sample)} samples")
    print(f"    Skipped: {skipped}")
    
    # ===== STEP 6: Compute accuracy =====
    print("\n" + "=" * 70)
    print("[6] Computing preference accuracy")
    print("=" * 70)
    
    # Only compare samples we actually scored
    scored_ids = {t["id"] for t in target_sample}
    filtered_score_sample = [s for s in score_sample if s["id"] in scored_ids]
    
    accuracy = compute_accuracy(filtered_score_sample, target_sample)
    
    print(f"\n    ╔══════════════════════════════════════════════╗")
    print(f"    ║  ImageReward-v1.0 Test Accuracy: {100 * accuracy:6.2f}%  ║")
    print(f"    ║  Paper reported:                  65.14%     ║")
    print(f"    ╚══════════════════════════════════════════════╝")
    
    diff = 100 * accuracy - 65.14
    if abs(diff) < 1.0:
        print(f"\n    ✅ REPRODUCTION SUCCESSFUL (within 1% of paper)")
    elif abs(diff) < 3.0:
        print(f"\n    🟡 Close reproduction (off by {diff:+.2f} percentage points)")
    else:
        print(f"\n    ⚠️  Off by {diff:+.2f} percentage points — worth investigating")
    
    return {
        "accuracy": accuracy,
        "accuracy_pct": 100 * accuracy,
        "paper_pct": 65.14,
        "samples_scored": len(target_sample),
        "samples_skipped": skipped,
    }


@app.local_entrypoint()
def main():
    result = reproduce_table1.remote()
    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    import json
    print(json.dumps(result, indent=2))