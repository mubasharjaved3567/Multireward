# Reproducing ImageReward — Table 1 Preference Accuracy

This repo documents a reproduction of **Table 1** from the NeurIPS 2023 paper [*ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation*](https://arxiv.org/abs/2304.05977) by Xu et al.

We use the authors' publicly released pretrained checkpoint (`ImageReward-v1.0`) and evaluate it on their published 466-sample preference test set. No training required.

## Result

| Metric | Paper | This Reproduction |
|---|---|---|
| Preference Accuracy | **65.14%** | **65.15%** |
| Test samples | 466 | 466 |
| Difference | — | +0.01 percentage points |

The reproduction matches the paper's reported number to the second decimal place. Because we use the authors' released weights and their exact evaluation script, this is a faithful replication of their metric on their data.

## What This Reproduction Covers

-  Downloading and loading the pretrained `ImageReward-v1.0` checkpoint
-  Downloading the 1.18 GB benchmark test images from HuggingFace
-  Running inference across all 466 test prompts (~2,500 image scorings)
-  Computing preference accuracy using the authors' `acc()` function from `test.py`

Not covered (reproduction only validates their evaluation — not training):
- Training the reward model from scratch
- ReFL fine-tuning of Stable Diffusion
- Other tables/ablations from the paper

## Requirements

### Hardware

| Component | Minimum | Recommended | Notes |
|---|---|---|---|
| GPU | None (CPU works) | NVIDIA GPU with ≥8 GB VRAM | CPU is 30-50× slower |
| RAM | 8 GB | 16 GB | Model weights are ~1.8 GB |
| Disk | 5 GB free | 10 GB free | Model + test images + extraction |
| Internet | Required | — | Downloads ~3 GB on first run |

**On CPU:** The full 466-sample benchmark takes 2–4 hours. A single image takes ~10–30 seconds. Use CPU only if you're testing with a few images or have no GPU available.

**On GPU:** The full benchmark runs in 4–8 minutes on a T4, 2–4 minutes on an A100.

### Software

- Python 3.10 (3.9 and 3.11 also work but untested for this reproduction)
- Git
- pip

## The Dependency Gotcha (Read This First)

This is the reason this README exists. The official `image-reward` package has transitive dependencies that break in very specific ways on modern pip. If you try to `pip install image-reward` and follow the repo's `requirements.txt` as-is (in 2026 and later), you will hit a cascade of errors:

1. `ImportError: cannot import name 'apply_chunking_to_forward'` — newer transformers moved the function
2. `ImportError: cannot import name 'cached_download' from 'huggingface_hub'` — removed in hub ≥0.26
3. `AttributeError: module 'torch' has no attribute 'xpu'` — newer diffusers needs newer torch
4. `ResolutionImpossible: Cannot install tokenizers and transformers because...` — the `>=` pins resolve to incompatible versions

The root cause is that the original `requirements.txt` uses `>=` version pins. Three years of ecosystem changes mean those pins now resolve to mutually incompatible packages.

**The fix is a specific, tested combination of exact versions that we've verified works.** Use the install recipe below exactly as written. Do not "upgrade a few packages" — that breaks it again.

## Installation

### 1. Clone the official ImageReward repo

```bash
git clone https://github.com/zai-org/ImageReward.git
cd ImageReward
```

You need this for `data/test.json` (the labels file) and the data directory structure.

### 2. Create a virtual environment

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
### 3. Install the tested dependency stack

Install exactly these versions in exactly this order. The order matters because `image-reward` silently upgrades `huggingface_hub`, and we need to force it back down afterwards.

```bash
# PyTorch stack (CPU version — for GPU, see the note below)
pip install torch==2.0.1 torchvision==0.15.2

# ML core libraries (pinned versions that work together)
pip install \
  transformers==4.38.2 \
  tokenizers==0.15.2 \
  safetensors \
  diffusers==0.24.0 \
  accelerate==0.27.2 \
  peft==0.7.1 \
  fairscale==0.4.13 \
  timm==0.9.12 \
  ftfy regex Pillow pandas tqdm "numpy<2.0"

# OpenAI CLIP (from GitHub — the PyPI package named "clip" is a DIFFERENT thing)
pip install git+https://github.com/openai/CLIP.git

# The ImageReward package itself
pip install image-reward

# CRITICAL: force huggingface_hub to 0.22.2 AFTER image-reward
# (image-reward pulls in a newer version that breaks diffusers 0.24)
pip install --force-reinstall --no-deps huggingface_hub==0.22.2
```

**For GPU installation on Linux with CUDA 11.7:** replace the first `pip install torch==2.0.1 torchvision==0.15.2` with:
```bash
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
```

For other CUDA versions or Windows GPU setup, see [pytorch.org/get-started/previous-versions](https://pytorch.org/get-started/previous-versions/).

### 4. Verify the installation

Run the test_everyting.py file to confirm everything is ready.

```

On first run, this downloads the 1.8 GB model to `~/.cache/ImageReward/` (or the Windows equivalent). Expect 1-3 minutes for download + load. If you see `✓ Success`, you're ready.

## Reproducing Table 1

### 1. Download the benchmark images

The test images (1.18 GB) are hosted on HuggingFace. Download and extract them into the repo's `data/` directory:

```bash
# From inside the ImageReward repo root
mkdir -p data
cd data

# Download test_images.zip from HuggingFace
# Option A: via huggingface_hub (recommended)
python -c "from huggingface_hub import hf_hub_download; import shutil; \
  p = hf_hub_download('THUDM/ImageReward', 'test_images.zip'); \
  shutil.copy(p, 'test_images.zip')"

# Option B: via wget
# wget https://huggingface.co/THUDM/ImageReward/resolve/main/test_images.zip

# Extract
unzip test_images.zip   # Linux/macOS
# On Windows use: Expand-Archive -Path test_images.zip -DestinationPath .

cd ..
```

After extraction, you should have:
```
ImageReward/
├── data/
│   ├── test.json              # 466 samples with rankings
│   └── test_images/           # 417 prompt folders
│       ├── part-001130/
│       │   └── images/
│       └── ...
```

### 2. Run the authors' evaluation script

The authors already wrote the evaluation code. It's in `test.py` at the repo root:

```bash
python test.py \
  --model_type ImageReward-v1.0 \
  --source_path data/test.json \
  --img_prefix data/test_images \
  --target_dir data/ \
  --rm_path checkpoint/
```

**Expected output:**
```
ImageReward-v1.0 Test begin:
   ImageReward-v1.0 Test Acc: 65.15%
```

**Expected timing:**
- GPU (T4 or better): 4–8 minutes
- CPU: 2–4 hours

### 3. Interpreting the result

The paper reports **65.14%** in Table 1 (row: "Preference accuracy", column: "ImageReward-v1.0"). A match within ~1 percentage point is a successful reproduction. Small variations can come from floating-point nondeterminism on different GPUs.

## How the Metric Works

The `acc()` function in the authors' `test.py` computes preference accuracy as follows. For each prompt:

1. Humans ranked the 4–9 generated images (rank 1 = best, higher numbers = worse, ties allowed).
2. ImageReward scored all images (higher score = better).
3. For every pair of images `(i, j)` under the same prompt:
   - If humans ranked them differently (no tie), check whether the model agrees on the direction.
   - If humans tied, skip the pair.
4. Accuracy = (agreements) / (total non-tied pairs).

Across 466 prompts with ~4–9 images each, this produces roughly 8,000–15,000 pairwise comparisons.

## Running Without a GPU

If you have no GPU at all, the code still works on CPU — you just need to be patient. `test.py` automatically detects CUDA and falls back to CPU:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

On CPU you have three options:
1. **Full benchmark on CPU:** runs as written, takes 2–4 hours. You'll still get the 65.15% number.
2. **Subset evaluation:** edit `test.py` to only process the first 50 samples (change `for item in score_sample:` to `for item in score_sample[:50]:`). Runs in ~15 minutes. Accuracy will be close to 65% but with more variance.
3. **Free GPU alternatives:** Google Colab (free T4, ~12 hr/week), Kaggle Notebooks (free P100, 30 hr/week), or Modal.com (free $30/month credit). These are all practical for this workload.

## Troubleshooting

### `cannot import name 'cached_download' from 'huggingface_hub'`
Your `huggingface_hub` version is too new. Run the force-reinstall step:
```bash
pip install --force-reinstall --no-deps huggingface_hub==0.22.2
```

### `AttributeError: module 'torch' has no attribute 'xpu'`
Your `diffusers` is too new for torch 2.0.1. Pin it:
```bash
pip install diffusers==0.24.0
```

### `FileNotFoundError: ImageReward.pt`
The automatic download path ImageReward uses can break on certain setups (especially with mounted volumes or symlinks). Download the file manually:
```python
from huggingface_hub import hf_hub_download
import shutil, os
os.makedirs(os.path.expanduser("~/.cache/ImageReward"), exist_ok=True)
path = hf_hub_download("THUDM/ImageReward", "ImageReward.pt")
shutil.copyfile(os.path.realpath(path), os.path.expanduser("~/.cache/ImageReward/ImageReward.pt"))
# Repeat for med_config.json
```

### `ImportError: No module named 'clip'`
You need OpenAI's CLIP, not the PyPI package named `clip` (which is unrelated). Install from GitHub:
```bash
pip uninstall clip -y
pip install git+https://github.com/openai/CLIP.git
```

### CUDA out of memory
ImageReward is small enough to run on 4 GB VRAM, but if you're on a shared GPU, other processes might be occupying memory. Try setting `CUDA_VISIBLE_DEVICES=0` and closing other GPU apps. Or fall back to CPU — it's slow but works.

### Installation succeeds but import hangs
Some pip installations pull in the wrong `numpy` (2.x) which breaks older torch. Pin it:
```bash
pip install "numpy<2.0" --force-reinstall
```

## File Layout After Setup

```
ImageReward/                    # cloned repo
├── data/
│   ├── test.json              # 466 test samples with rankings
│   └── test_images/           # extracted from test_images.zip
│       └── part-XXXXXX/
│           └── images/
│               └── XXXX/
│                   └── *.webp
├── test.py                    # authors' evaluation script
├── ImageReward/               # package source
└── ...

~/.cache/ImageReward/          # created automatically
├── ImageReward.pt             # 1.8 GB model weights
└── med_config.json            # BLIP config

venv/                          # your virtualenv (wherever you put it)
```

## Why This Reproduction Matters

Reproducing the headline metric of a peer-reviewed paper using the authors' released artifacts validates that:

1. The released checkpoint matches what they evaluated for the paper
2. The evaluation script correctly computes the reported number
3. The test set and labels are as described

This is a baseline reproduction — it does not retrain anything. A stronger reproduction would include training from scratch on `ImageRewardDB` and matching the number (the paper reports that training multiple random seeds yields 65.14% ± ~0.3%).

## Citation

If you use ImageReward in your own work, cite the original paper:

```bibtex
@inproceedings{xu2023imagereward,
  title={ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation},
  author={Xu, Jiazheng and Liu, Xiao and Wu, Yuchen and Tong, Yuxuan and Li, Qinkai and Ding, Ming and Tang, Jie and Dong, Yuxiao},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

## License

This reproduction guide is released under the MIT License. The original ImageReward code and weights are under their respective licenses (see [the main repo](https://github.com/zai-org/ImageReward)).

## Tested Environment

This guide was validated on:
- **OS:** Debian 11 (container), Ubuntu 20.04 (container), Windows 11 (host)
- **Python:** 3.10.13
- **GPU:** NVIDIA T4 (16 GB) via Modal.com
- **Date:** 2026-04

If you reproduce this in a different environment and get different results, please open an issue or share your setup.
