"""
Overlay plot v2 — predicted fECG vs ground truth on same axes.
Paper architecture (500Hz, 128×128). Runs on CPU.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from movement_dataset_paper import (
    _load_signals_500hz, _stft_multichannel,
    NFFT, HOP_LENGTH, WIN_LENGTH, TARGET_SIZE_F, TARGET_SIZE_T, FS, WINDOW,
)
from complex_network_paper import ComplexUNet

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR    = '../data/movement_ecg'
MODEL_PATH  = '../models/movement_CUNet_128x128_paper.pth'
HISTORY_PATH = MODEL_PATH.replace('.pth', '_history.json')
CORRUPTED   = 'fecgsyn_Long_time_segment_var13_23_snr3dB.mat'

START       = 10 * FS
CHANNELS    = [0, 1, 2]
DIMENSION   = TARGET_SIZE_F * TARGET_SIZE_T
IN_CHANNELS = 6
ORIG_F      = NFFT // 2 + 1
ORIG_T      = TARGET_SIZE_T

# ---------------------------------------------------------------------------
# Pick test file
# ---------------------------------------------------------------------------
all_files = sorted(
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR) if f.endswith('.mat')
)
test_file = next(f for f in all_files if os.path.basename(f) != CORRUPTED)
print(f'Test file : {os.path.basename(test_file)}')

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model = ComplexUNet(DIMENSION, in_channels=IN_CHANNELS)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

EPOCH = '?'
if os.path.exists(HISTORY_PATH):
    hist = json.load(open(HISTORY_PATH))
    vals = hist.get('val_loss', [])
    if vals:
        EPOCH = vals.index(min(vals)) + 1

OUT_PATH = f'../plots_2026/inference_overlay_v2_ep{EPOCH}.png'
print(f'Model loaded from {MODEL_PATH}  (best epoch: {EPOCH})')

# ---------------------------------------------------------------------------
# Load & normalise signals (500 Hz)
# ---------------------------------------------------------------------------
mixture, fecg = _load_signals_500hz(test_file)
mix_win  = mixture[:, START:START + WINDOW].copy()
fecg_win = fecg[:,   START:START + WINDOW].copy()

stds = mix_win.std(axis=1, keepdims=True)
stds = np.where(stds < 1e-8, 1.0, stds)
mix_norm  = mix_win  / stds
fecg_norm = fecg_win / stds

# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------
mix_spec = _stft_multichannel(mix_norm, NFFT, HOP_LENGTH, WIN_LENGTH)
mix_spec = mix_spec[:, :-1, :]  # drop last freq bin

x = torch.complex(
    torch.from_numpy(np.real(mix_spec).copy()),
    torch.from_numpy(np.imag(mix_spec).copy()),
).unsqueeze(0)

with torch.no_grad():
    pred = model(x)

# ---------------------------------------------------------------------------
# iSTFT -> time domain
# ---------------------------------------------------------------------------
def spec_to_time(tensor):
    tensor = tensor.squeeze(0)
    C = tensor.shape[0]
    zeros = torch.zeros(C, 1, ORIG_T, dtype=tensor.dtype)
    full = torch.cat([tensor, zeros], dim=1)
    out = []
    for ch in range(C):
        spec = full[ch].numpy()
        sig = librosa.istft(spec, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                            n_fft=NFFT, length=WINDOW)
        out.append(sig)
    return np.stack(out, axis=0)

pred_time = spec_to_time(pred)
fecg_time = fecg_norm

# ---------------------------------------------------------------------------
# Overlay plot
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

t = np.linspace(START / FS, (START + WINDOW) / FS, WINDOW)
n_ch = len(CHANNELS)

fig, axes = plt.subplots(n_ch, 1, figsize=(14, 3 * n_ch), sharex=True)
if n_ch == 1:
    axes = [axes]

for i, ch in enumerate(CHANNELS):
    ax = axes[i]
    ax.plot(t, fecg_time[ch], color='seagreen',  linewidth=1.0, label='Ground truth fECG', alpha=0.9)
    ax.plot(t, pred_time[ch], color='darkorange', linewidth=0.9, label='Predicted fECG',    alpha=0.85)
    ax.set_ylabel(f'Channel {ch + 1}', fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    if i == 0:
        ax.legend(loc='upper right', fontsize=9)

axes[-1].set_xlabel('Time (s)', fontsize=10)
fig.suptitle(
    f'fECG Extraction v2 (paper arch) — Predicted vs Ground Truth  (epoch {EPOCH})\n'
    f'{os.path.basename(test_file)}   |   Window: {START / FS:.1f}–{(START + WINDOW) / FS:.1f} s   '
    f'(500 Hz, 128x128, SignalMSE)',
    fontsize=11
)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f'Plot saved -> {OUT_PATH}')
