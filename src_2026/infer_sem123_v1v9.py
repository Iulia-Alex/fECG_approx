"""
Inferenta pe Sem1, Sem2, Sem3 din Test_DB.
Comparatie: GT (verde) vs v1 ep199 (albastru) vs v9 ep_best (portocaliu).

v9 la ep28+ — cel mai bun checkpoint curent.
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
import scipy.io as sio

from movement_dataset import (
    _stft_multichannel, _to_resized_complex_tensor, _extract_fecg,
    NFFT, HOP_LENGTH, WIN_LENGTH, TARGET_SIZE_F, TARGET_SIZE_T, FS,
)
from complex_network import ComplexUNet
from complex_network_v9 import ComplexUNetV9

TEST_DIR   = '/shared_storage/stan.edward/fECG_mvm_DB/Test_DB'
OUT_DIR    = '/shared_storage/iulia.orvas/paper/fECG_approx/plots_2026/testdb'

MODEL_V1   = '/shared_storage/iulia.orvas/paper/fECG_approx/models/movement_CUNet_128x400_composed.pth'
HIST_V1    = '/shared_storage/iulia.orvas/paper/fECG_approx/models/movement_CUNet_128x400_composed_history.json'

MODEL_V9   = '/shared_storage/iulia.orvas/paper/fECG_approx/models/movement_CUNet_v9_mask.pth'
HIST_V9    = '/shared_storage/iulia.orvas/paper/fECG_approx/models/movement_CUNet_v9_mask_history.json'

TARGET_FILES = ['Sem1.mat', 'Sem2.mat', 'Sem3.mat']

WINDOW      = 4 * FS
START       = 10 * FS
CHANNELS    = [0, 1, 2]
DIMENSION   = TARGET_SIZE_F * TARGET_SIZE_T
IN_CHANNELS = 6
ORIG_F      = NFFT // 2 + 1
ORIG_T      = 1 + (WINDOW // HOP_LENGTH)

os.makedirs(OUT_DIR, exist_ok=True)


def best_epoch(hist_path):
    if os.path.exists(hist_path):
        h = json.load(open(hist_path))
        vals = h.get('val_loss', [])
        if vals:
            return vals.index(min(vals)) + 1
    return '?'


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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Load v1
model_v1 = ComplexUNet(DIMENSION, in_channels=IN_CHANNELS).to(device)
model_v1.load_state_dict(torch.load(MODEL_V1, map_location=device))
model_v1.eval()
ep_v1 = best_epoch(HIST_V1)
print(f'v1 loaded (best ep {ep_v1})')

# Load v9
model_v9 = ComplexUNetV9(DIMENSION, in_channels=IN_CHANNELS).to(device)
model_v9.load_state_dict(torch.load(MODEL_V9, map_location=device))
model_v9.eval()
ep_v9 = best_epoch(HIST_V9)
print(f'v9 loaded (best ep {ep_v9})')

for fname in TARGET_FILES:
    fpath = os.path.join(TEST_DIR, fname)
    if not os.path.exists(fpath):
        print(f'  SKIP — not found: {fpath}')
        continue

    stem = os.path.splitext(fname)[0]
    print(f'\nProcessing {fname} ...')

    mat     = sio.loadmat(fpath)
    mixture = mat['out']['mixture'][0][0].astype(np.float32)
    fecg    = _extract_fecg(mat['out'])
    del mat

    if fecg.shape[0] < mixture.shape[0]:
        fecg = np.repeat(fecg, mixture.shape[0] // fecg.shape[0], axis=0)

    mix_win  = mixture[:, START:START + WINDOW].copy()
    fecg_win = fecg[:,   START:START + WINDOW].copy()

    stds      = mix_win.std(axis=1, keepdims=True)
    stds      = np.where(stds < 1e-8, 1.0, stds)
    mix_norm  = mix_win  / stds
    fecg_norm = fecg_win / stds

    mix_spec = _stft_multichannel(mix_norm, NFFT, HOP_LENGTH, WIN_LENGTH)
    x = _to_resized_complex_tensor(mix_spec, TARGET_SIZE_F, TARGET_SIZE_T).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_v1 = model_v1(x).cpu()
        pred_v9 = model_v9(x).cpu()

    time_v1 = spec_to_time(pred_v1)
    time_v9 = spec_to_time(pred_v9)
    t = np.linspace(START / FS, (START + WINDOW) / FS, WINDOW)

    n_ch = len(CHANNELS)
    fig, axes = plt.subplots(n_ch, 1, figsize=(14, 3.2 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    for i, ch in enumerate(CHANNELS):
        ax = axes[i]
        ax.plot(t, fecg_norm[ch], color='seagreen',   lw=1.2, label='Ground truth',           alpha=0.9, zorder=3)
        ax.plot(t, time_v1[ch],   color='steelblue',  lw=0.9, label=f'v1 ep{ep_v1} (direct)', alpha=0.8, zorder=2)
        ax.plot(t, time_v9[ch],   color='darkorange', lw=0.9, label=f'v9 ep{ep_v9} (mask)',   alpha=0.8, zorder=2)
        ax.set_ylabel(f'Ch {ch + 1}', fontsize=10)
        ax.grid(True, alpha=0.3, lw=0.5)
        if i == 0:
            ax.legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel('Time (s)', fontsize=10)
    fig.suptitle(
        f'fECG Extraction — v1 (0.59M, SignalMSE) vs v9 (1.87M, SignalMSE+CplxMSE+mask) — {stem}\n'
        f'Window: {START//FS}–{(START+WINDOW)//FS} s  |  best val v1={ep_v1}, v9={ep_v9}',
        fontsize=10,
    )
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'{stem}_v1_vs_v9_ep{ep_v9}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {out_path}')

print('\nDone.')
