"""
SNR comparison inference — 3 rows (snr0dB / snr3dB / snr6dB), one channel each,
predicted fECG vs ground truth overlaid, on a 12s continuous segment.

Runs on CPU only — does NOT affect training.
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

CHUNK       = 4 * FS          # 4 s per chunk (matches training)
N_CHUNKS    = 3               # → 12 s total display
START       = 10 * FS         # skip first 10 s (transients)
CHANNEL     = 1               # channel index to plot (0-based)
DIMENSION   = TARGET_SIZE_F * TARGET_SIZE_T
IN_CHANNELS = 6
ORIG_F      = NFFT // 2 + 1
ORIG_T      = 1 + (CHUNK // HOP_LENGTH)

# One file per SNR level — picked from different variants for variety
FILES = {
    'SNR 0 dB (hard)':   'fecgsyn_Long_time_segment_var6_25_snr0dB.mat',
    'SNR 3 dB (medium)': 'fecgsyn_Long_time_segment_var9_25_snr3dB.mat',
    'SNR 6 dB (easy)':   'fecgsyn_Long_time_segment_var13_15_snr6dB.mat',
}

# ---------------------------------------------------------------------------
# Load model once
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

OUT_PATH = f'../plots_2026/inference_snr_comparison_ep{EPOCH}.png'
print(f'Model loaded from {MODEL_PATH}  (best epoch: {EPOCH})')

# ---------------------------------------------------------------------------
# iSTFT helper
# ---------------------------------------------------------------------------
def spec_to_time(tensor):
    """(1, C, 128, 400) complex tensor → (C, CHUNK) numpy."""
    tensor = tensor.squeeze(0)
    C = tensor.shape[0]
    real = F.interpolate(tensor.real.unsqueeze(0), size=(ORIG_F, ORIG_T),
                         mode='bilinear', align_corners=False).squeeze(0)
    imag = F.interpolate(tensor.imag.unsqueeze(0), size=(ORIG_F, ORIG_T),
                         mode='bilinear', align_corners=False).squeeze(0)
    out = []
    for ch in range(C):
        spec = real[ch].numpy() + 1j * imag[ch].numpy()
        sig  = librosa.istft(spec, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                             n_fft=NFFT, length=CHUNK)
        out.append(sig)
    return np.stack(out, axis=0)


def infer_long(filepath, start, n_chunks, channel):
    """Run inference on n_chunks consecutive 4s windows, return concatenated signals."""
    mixture, fecg = _load_signals(filepath)
    pred_segs, gt_segs = [], []

    for i in range(n_chunks):
        s = start + i * CHUNK
        mix_win  = mixture[:, s:s + CHUNK].copy()
        fecg_win = fecg[:,   s:s + CHUNK].copy()

        # Normalise per chunk (same as dataset)
        stds = mix_win.std(axis=1, keepdims=True)
        stds = np.where(stds < 1e-8, 1.0, stds)
        mix_norm  = mix_win  / stds
        fecg_norm = fecg_win / stds

        mix_spec = _stft_multichannel(mix_norm, NFFT, HOP_LENGTH, WIN_LENGTH)
        x = _to_resized_complex_tensor(mix_spec, TARGET_SIZE_F, TARGET_SIZE_T).unsqueeze(0)

        with torch.no_grad():
            pred = model(x)

        pred_time = spec_to_time(pred)   # (C, CHUNK)
        pred_segs.append(pred_time[channel])
        gt_segs.append(fecg_norm[channel])

    return np.concatenate(pred_segs), np.concatenate(gt_segs)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
total_samples = N_CHUNKS * CHUNK
t = np.linspace(START / FS, (START + total_samples) / FS, total_samples)

fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)

for ax, (label, fname) in zip(axes, FILES.items()):
    fpath = os.path.join(DATA_DIR, fname)
    print(f'Processing {fname} ...')
    pred_sig, gt_sig = infer_long(fpath, START, N_CHUNKS, CHANNEL)

    ax.plot(t, gt_sig,   color='seagreen',  linewidth=1.0,  label='Ground truth fECG', alpha=0.9)
    ax.plot(t, pred_sig, color='darkorange', linewidth=0.85, label='Predicted fECG',    alpha=0.85)
    ax.set_ylabel(label, fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel('Time (s)', fontsize=10)
fig.suptitle(
    f'fECG Extraction — Predicted vs Ground Truth across SNR levels  (epoch {EPOCH})\n'
    f'Channel {CHANNEL + 1}  |  Window: {START // FS}–{(START + total_samples) // FS} s  '
    f'(3 × 4 s chunks, normalised per chunk)',
    fontsize=11
)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f'Plot saved → {OUT_PATH}')
