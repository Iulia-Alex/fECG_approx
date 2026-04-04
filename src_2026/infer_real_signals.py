"""
Inference + plotting script — runs on CPU, does NOT touch the training job.

Loads one test .mat file, runs the trained ComplexUNet on a short window,
and plots mixture / predicted fECG / ground-truth fECG in the time domain.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import matplotlib
matplotlib.use('Agg')          # headless — no display needed
import matplotlib.pyplot as plt

from movement_dataset import (
    _load_signals, _stft_multichannel, _to_resized_complex_tensor,
    NFFT, HOP_LENGTH, WIN_LENGTH, TARGET_SIZE_F, TARGET_SIZE_T, FS,
)
from complex_network import ComplexUNet

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR    = '../data/movement_ecg'
MODEL_PATH  = '../models/movement_CUNet_128x400_composed.pth'
HISTORY_PATH = MODEL_PATH.replace('.pth', '_history.json')
CORRUPTED   = 'fecgsyn_Long_time_segment_var13_23_snr3dB.mat'

WINDOW      = 4 * FS          # 4-second window = 4000 samples
START       = 10 * FS         # start at 10 s (skip initial transients)
CHANNELS    = [0, 1, 2]       # which of the 6 channels to plot
DIMENSION   = TARGET_SIZE_F * TARGET_SIZE_T   # 128 × 400 = 51200
IN_CHANNELS = 6

# STFT original output shape (before resize)
ORIG_F = NFFT // 2 + 1        # 129 frequency bins
ORIG_T = 1 + (WINDOW // HOP_LENGTH)  # ~401 time frames

# ---------------------------------------------------------------------------
# Pick test file (skip the corrupted one)
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

# Detect best epoch from history
EPOCH = '?'
if os.path.exists(HISTORY_PATH):
    hist = json.load(open(HISTORY_PATH))
    vals = hist.get('val_loss', [])
    if vals:
        EPOCH = vals.index(min(vals)) + 1

OUT_PATH = f'../plots_2026/inference_plot_ep{EPOCH}.png'
print(f'Model loaded from {MODEL_PATH}  (best epoch: {EPOCH})')

# ---------------------------------------------------------------------------
# Load signals and extract window
# ---------------------------------------------------------------------------
mixture, fecg = _load_signals(test_file)          # (6, 600000) each

mix_win  = mixture[:, START:START + WINDOW].copy()  # (6, 4000)
fecg_win = fecg[:,   START:START + WINDOW].copy()

# Same normalisation as MovementECGDataset.__getitem__
stds = mix_win.std(axis=1, keepdims=True)           # (6, 1)
stds = np.where(stds < 1e-8, 1.0, stds)
mix_norm  = mix_win  / stds
fecg_norm = fecg_win / stds

# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------
mix_spec  = _stft_multichannel(mix_norm,  NFFT, HOP_LENGTH, WIN_LENGTH)
fecg_spec = _stft_multichannel(fecg_norm, NFFT, HOP_LENGTH, WIN_LENGTH)

x = _to_resized_complex_tensor(mix_spec,  TARGET_SIZE_F, TARGET_SIZE_T).unsqueeze(0)  # (1,6,128,400)
y = _to_resized_complex_tensor(fecg_spec, TARGET_SIZE_F, TARGET_SIZE_T).unsqueeze(0)

with torch.no_grad():
    pred = model(x)   # (1, 6, 128, 128)

# ---------------------------------------------------------------------------
# Invert predicted spectrogram → time domain
# ---------------------------------------------------------------------------
def spec_to_time(tensor):
    """(1, C, 128, 128) complex tensor → time-domain signals (C, WINDOW)."""
    tensor = tensor.squeeze(0)          # (C, 128, 128)
    C = tensor.shape[0]
    # Resize back to original STFT shape
    real = F.interpolate(tensor.real.unsqueeze(0), size=(ORIG_F, ORIG_T),
                         mode='bilinear', align_corners=False).squeeze(0)
    imag = F.interpolate(tensor.imag.unsqueeze(0), size=(ORIG_F, ORIG_T),
                         mode='bilinear', align_corners=False).squeeze(0)
    out = []
    for ch in range(C):
        spec = real[ch].numpy() + 1j * imag[ch].numpy()
        sig  = librosa.istft(spec, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                             n_fft=NFFT, length=WINDOW)
        out.append(sig)
    return np.stack(out, axis=0)    # (C, WINDOW)


mix_time  = mix_norm                  # already in time domain
fecg_time = fecg_norm                 # ground truth
pred_time = spec_to_time(pred)        # reconstructed from predicted spectrogram

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
t = np.linspace(START / FS, (START + WINDOW) / FS, WINDOW)

n_ch  = len(CHANNELS)
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
    f'fECG Extraction — {os.path.basename(test_file)}  (epoch {EPOCH})\n'
    f'Window: {START // FS}–{(START + WINDOW) // FS} s   '
    f'(normalised by mixture std per channel)',
    fontsize=11
)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f'Plot saved → {OUT_PATH}')
