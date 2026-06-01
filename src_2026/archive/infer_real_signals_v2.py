"""
Inference + plotting for v2 (paper architecture, 500Hz, 128×128).
Runs on CPU — does NOT touch the training job.
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

START       = 10 * FS             # start at 10 s (5000 samples at 500Hz)
CHANNELS    = [0, 1, 2]
DIMENSION   = TARGET_SIZE_F * TARGET_SIZE_T   # 128 × 128 = 16384
IN_CHANNELS = 6

# STFT original output shape (before dropping last freq bin)
ORIG_F = NFFT // 2 + 1           # 129
ORIG_T = TARGET_SIZE_T            # 128 (natural, no resize)

# ---------------------------------------------------------------------------
# Pick test file
# ---------------------------------------------------------------------------
all_files = sorted(
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR) if f.endswith('.mat')
)
test_file = next(
    f for f in all_files if os.path.basename(f) != CORRUPTED
)
print(f'Test file : {os.path.basename(test_file)}')

# ---------------------------------------------------------------------------
# Load model on CPU
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

OUT_PATH = f'../plots_2026/inference_v2_ep{EPOCH}.png'
print(f'Model loaded from {MODEL_PATH}  (best epoch: {EPOCH})')

# ---------------------------------------------------------------------------
# Load signals (500 Hz) and extract window
# ---------------------------------------------------------------------------
mixture, fecg = _load_signals_500hz(test_file)   # (6, 300000) at 500Hz

mix_win  = mixture[:, START:START + WINDOW].copy()   # (6, 1915)
fecg_win = fecg[:,   START:START + WINDOW].copy()

# Per-channel normalization (same as dataset)
stds = mix_win.std(axis=1, keepdims=True)
stds = np.where(stds < 1e-8, 1.0, stds)
mix_norm  = mix_win  / stds
fecg_norm = fecg_win / stds

# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------
mix_spec = _stft_multichannel(mix_norm, NFFT, HOP_LENGTH, WIN_LENGTH)  # (6, 129, 128)

# Drop last freq bin → (6, 128, 128)
mix_spec = mix_spec[:, :-1, :]

x = torch.complex(
    torch.from_numpy(np.real(mix_spec).copy()),
    torch.from_numpy(np.imag(mix_spec).copy()),
).unsqueeze(0)  # (1, 6, 128, 128)

with torch.no_grad():
    pred = model(x)   # (1, 6, 128, 128)

# ---------------------------------------------------------------------------
# Invert predicted spectrogram → time domain
# ---------------------------------------------------------------------------
def spec_to_time(tensor):
    """(1, C, 128, 128) complex → time-domain (C, WINDOW) via iSTFT."""
    tensor = tensor.squeeze(0)  # (C, 128, 128)
    C = tensor.shape[0]
    # Add back the dropped 129th freq bin as zeros
    zeros = torch.zeros(C, 1, ORIG_T, dtype=tensor.dtype)
    full = torch.cat([tensor, zeros], dim=1)  # (C, 129, 128)
    out = []
    for ch in range(C):
        spec = full[ch].numpy()
        sig = librosa.istft(spec, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                            n_fft=NFFT, length=WINDOW)
        out.append(sig)
    return np.stack(out, axis=0)


mix_time  = mix_norm
fecg_time = fecg_norm
pred_time = spec_to_time(pred)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

t = np.linspace(START / FS, (START + WINDOW) / FS, WINDOW)

n_ch = len(CHANNELS)
fig, axes = plt.subplots(n_ch, 3, figsize=(18, 3 * n_ch), sharex=True)

COLS   = ['Mixture (input X)', 'Predicted fECG', 'Ground truth fECG']
COLORS = ['steelblue', 'darkorange', 'seagreen']
SIGS   = [mix_time, pred_time, fecg_time]

for i, ch in enumerate(CHANNELS):
    for j, (sig, col, title) in enumerate(zip(SIGS, COLORS, COLS)):
        ax = axes[i, j]
        ax.plot(t, sig[ch], linewidth=0.7, color=col)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        if i == 0:
            ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Channel {ch + 1}', fontsize=9)
        ax.set_xlim(t[0], t[-1])

for j in range(3):
    axes[-1, j].set_xlabel('Time (s)', fontsize=9)

fig.suptitle(
    f'fECG Extraction v2 (paper arch) — {os.path.basename(test_file)}  (epoch {EPOCH})\n'
    f'Window: {START / FS:.1f}–{(START + WINDOW) / FS:.1f} s   '
    f'(500 Hz, 128x128, SignalMSE)',
    fontsize=11
)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f'Plot saved -> {OUT_PATH}')
