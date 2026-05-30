# hello_modal.py
import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.0.1")
)

app = modal.App("hello-gpu", image=image)

@app.function(gpu="T4", timeout=300)
def check_gpu():
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return "success"

@app.local_entrypoint()
def main():
    print("Calling Modal...")
    result = check_gpu.remote()
    print(f"Result: {result}")