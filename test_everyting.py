# test_install.py
print("Testing imports...")

try:
    import modal
    print(f"  [OK] modal version: {modal.__version__}")
except ImportError as e:
    print(f"  [FAIL] modal: {e}")

try:
    import ImageReward as RM
    print(f"  [OK] ImageReward package imported")
except ImportError as e:
    print(f"  [FAIL] ImageReward: {e}")

try:
    import torch
    print(f"  [OK] torch version: {torch.__version__}")
    print(f"  [OK] CUDA available (should be False on your laptop): {torch.cuda.is_available()}")
except ImportError as e:
    print(f"  [FAIL] torch: {e}")

print("\nAll good if you see [OK] for all three.")