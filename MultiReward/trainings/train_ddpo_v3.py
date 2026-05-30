"""
V4 Patch Script
Run this to convert train_ddpo_v2.py to V4 config (kl_beta=0.01, lr=1e-5)

Usage:
  python train_ddpo_v4_patch.py
  Then run: python train_ddpo_v4.py --max_steps 500 --max_samples 3000
"""
import shutil

shutil.copy('train_ddpo_v2.py', 'train_ddpo_v4.py')
content = open('train_ddpo_v4.py').read()

content = content.replace('"kl_beta":             0.1,',  '"kl_beta":             0.01,')
content = content.replace('"lr":                  3e-6,', '"lr":                  1e-5,')
content = content.replace('v2_step_', 'v4_step_')
content = content.replace('train_log_v2.jsonl', 'train_log_v4.jsonl')

open('train_ddpo_v4.py', 'w').write(content)

c = open('train_ddpo_v4.py').read()
print("V4 patch applied:")
print(f"  kl_beta=0.01: {chr(34)}kl_beta{chr(34):>12}             0.01,{chr(34) in c}")
print(f"  lr=1e-5:      {chr(34)}lr{chr(34):>12}                  1e-5,{chr(34) in c}")
print(f"  v4 prefix:    {'v4_step_' in c}")
print("\nRun: python train_ddpo_v4.py --max_steps 500 --max_samples 3000")
