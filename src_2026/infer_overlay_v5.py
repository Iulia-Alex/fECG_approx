"""
Overlay plot v5 — predicted fECG vs ground truth.
v1 architecture + AmpWeightedMSE from scratch (128×400, 1000Hz). Runs on CPU.
Uses a window from the VALIDATION set (same split as training, SEED=42).
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from movement_dataset import (
    MovementECGDataset,
    _stft_multichannel, _to_resized_complex_tensor,
    NFFT, HOP_LENGTH, WIN_LENGTH, TARGET_SIZE_F, TARGET_SIZE_T, FS,
)
from complex_network import ComplexUNet

DATA_DIR     = '../data/movement_ecg'
MODEL_PATH   = '../models/movement_CUNet_128x400_ampw_scratch.pth'
HISTORY_PATH = MODEL_PATH.replace('.pth', '_history.json')

WINDOW      = 4 * FS
CHANNELS    = [0, 1, 2]
DIMENSION   = TARGET_SIZE_F * TARGET_SIZE_T
IN_CHANNELS = 6
ORIG_F      = NFFT // 2 + 1
ORIG_T      = 1 + (WINDOW // HOP_LENGTH)
SEED        = 42
VAL_SPLIT   = 0.15
VAL_IDX     = 42   # which sample from the val set to use

# ── Load model ────────────────────────────────────────────────────────────────
model = ComplexUNet(DIMENSION, in_channels=IN_CHANNELS)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

EPOCH = '?'
if os.path.exists(HISTORY_PATH):
    hist = json.load(open(HISTORY_PATH))
    vals = hist.get('val_loss', [])
    if vals:
        EPOCH = vals.index(min(vals)) + 1
print(f'Model loaded — best epoch: {EPOCH}')

# ── Pick a validation window ──────────────────────────────────────────────────
full_dataset = MovementECGDataset(DATA_DIR)
n_total = len(full_dataset)
n_val   = max(1, int(n_total * VAL_SPLIT))
n_train = n_total - n_val
generator = torch.Generator().manual_seed(SEED)
_, val_set = random_split(full_dataset, [n_train, n_val], generator=generator)

idx = VAL_IDX % len(val_set)
mix_spec_t, fecg_spec_t, fecg_time_t = val_set[idx]   # tensors

print(f'Val set size: {len(val_set)}  |  Using val index {idx}')

# ── Run inference ─────────────────────────────────────────────────────────────
x = mix_spec_t.unsqueeze(0)   # (1, 6, 128, 400) complex
with torch.no_grad():
    pred = model(x)

def spec_to_time(tensor):
    tensor = tensor.squeeze(0)
    real = F.interpolate(tensor.real.unsqueeze(0), size=(ORIG_F, ORIG_T),
                         mode='bilinear', align_corners=False).squeeze(0)
    imag = F.interpolate(tensor.imag.unsqueeze(0), size=(ORIG_F, ORIG_T),
                         mode='bilinear', align_corners=False).squeeze(0)
    out = []
    for ch in range(tensor.shape[0]):
        spec = real[ch].numpy() + 1j * imag[ch].numpy()
        sig  = librosa.istft(spec, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                             n_fft=NFFT, length=WINDOW)
        out.append(sig)
    return np.stack(out, axis=0)

pred_time = spec_to_time(pred)
fecg_time = fecg_time_t.numpy()   # (6, 4000) — already normalised

# ── Plot ──────────────────────────────────────────────────────────────────────
os.makedirs('../plots_2026', exist_ok=True)
OUT_PATH = f'../plots_2026/inference_overlay_v5_ep{EPOCH}_val{idx}.png'

t = np.arange(WINDOW) / FS
n_ch = len(CHANNELS)
fig, axes = plt.subplots(n_ch, 1, figsize=(14, 3 * n_ch), sharex=True)
if n_ch == 1:
    axes = [axes]

for i, ch in enumerate(CHANNELS):
    ax = axes[i]
    ax.plot(t, fecg_time[ch], color='seagreen',  linewidth=1.0,
            label='Ground truth fECG', alpha=0.9)
    ax.plot(t, pred_time[ch], color='darkorange', linewidth=0.9,
            label='Predicted fECG', alpha=0.85)
    ax.set_ylabel(f'Channel {ch + 1}', fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    if i == 0:
        ax.legend(loc='upper right', fontsize=9)

axes[-1].set_xlabel('Time (s)', fontsize=10)
fig.suptitle(
    f'fECG Extraction v5 (SignalMSE + 3×AmpWeightedMSE, from scratch) — epoch {EPOCH}\n'
    f'Validation window #{idx}  |  4s @ 1000Hz  |  normalised amplitude',
    fontsize=11
)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f'Plot saved -> {OUT_PATH}')
