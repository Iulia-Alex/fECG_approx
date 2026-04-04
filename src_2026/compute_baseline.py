"""
Compute the zero-prediction baseline loss on a sample of the validation set.

zero_loss = E[y_real² + y_imag²]
         = loss our model would get if it always predicted 0

If model_val_loss < zero_loss  → model is useful (beats doing nothing)
If model_val_loss > zero_loss  → model is worse than predicting zeros
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from movement_dataset import MovementECGDataset

SEED       = 42
VAL_SPLIT  = 0.15
BATCH_SIZE = 32
N_BATCHES  = 50   # 50 × 32 = 1600 samples — fast but representative

torch.manual_seed(SEED)

print("Loading dataset index...")
full_dataset = MovementECGDataset('../data/movement_ecg')
n_total = len(full_dataset)
n_val   = max(1, int(n_total * VAL_SPLIT))
n_train = n_total - n_val
print(f"  Total windows : {n_total}  (train={n_train}, val={n_val})")

generator = torch.Generator().manual_seed(SEED)
_, val_set = random_split(full_dataset, [n_train, n_val], generator=generator)

val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

zero_loss_acc  = 0.0
mix_power_acc  = 0.0
fecg_power_acc = 0.0
n = 0

print(f"\nEvaluating on {N_BATCHES} batches ({N_BATCHES * BATCH_SIZE} samples)...")
for batch_idx, (x, y) in enumerate(val_loader):
    if batch_idx >= N_BATCHES:
        break

    # Zero-prediction loss: E[|y - 0|²] = E[y_real² + y_imag²]
    zero_loss = y.real.pow(2).mean() + y.imag.pow(2).mean()

    # Mixture power (input energy, for scale reference)
    mix_power  = x.real.pow(2).mean() + x.imag.pow(2).mean()

    # fECG power (target energy)
    fecg_power = y.real.pow(2).mean() + y.imag.pow(2).mean()

    zero_loss_acc  += zero_loss.item()
    mix_power_acc  += mix_power.item()
    fecg_power_acc += fecg_power.item()
    n += 1

    if (batch_idx + 1) % 10 == 0:
        print(f"  batch {batch_idx+1:3d}/{N_BATCHES} | "
              f"zero_loss={zero_loss_acc/n:.4f} | "
              f"mix_power={mix_power_acc/n:.4f}")
        sys.stdout.flush()

zero_loss  = zero_loss_acc  / n
mix_power  = mix_power_acc  / n
fecg_power = fecg_power_acc / n

MODEL_VAL_E4 = 2.342411  # best val loss so far (epoch 4)

print("\n" + "="*55)
print(f"  Zero-prediction baseline loss : {zero_loss:.6f}")
print(f"  Mixture input power           : {mix_power:.6f}")
print(f"  fECG target power             : {fecg_power:.6f}")
print(f"  fECG/mixture power ratio      : {fecg_power/mix_power:.4f}  "
      f"({10*np.log10(fecg_power/mix_power):.1f} dB)")
print("-"*55)
print(f"  Model val loss (epoch 4)      : {MODEL_VAL_E4:.6f}")
print(f"  Model / zero-baseline ratio   : {MODEL_VAL_E4/zero_loss:.3f}")
if MODEL_VAL_E4 < zero_loss:
    improvement_db = 10 * np.log10(zero_loss / MODEL_VAL_E4)
    print(f"  → Model BEATS zero pred by    : {improvement_db:.2f} dB  ✓")
else:
    print(f"  → Model is WORSE than zero pred (ratio > 1)  ✗")
print("="*55)
