"""
Inference on Test_DB — v7 model (paper arch, L1, BS=16), epoch in filename.
Saves: {stem}_v7_ep{N}.png
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
    _stft_multichannel, _to_resized_complex_tensor,
    _extract_fecg,
    NFFT, HOP_LENGTH, WIN_LENGTH, TARGET_SIZE_F, TARGET_SIZE_T, FS,
)
from complex_network_v7 import ComplexUNetV7

TEST_DIR = '/shared_storage/stan.edward/fECG_mvm_DB/Test_DB'
OUT_DIR  = '../plots_2026/testdb'

MODEL_V7 = '../models/movement_CUNet_v7_paper_direct.pth'
HIST_V7  = MODEL_V7.replace('.pth', '_history.json')

WINDOW      = 4 * FS
START       = 10 * FS
CHANNELS    = [0, 1, 2]
DIMENSION   = TARGET_SIZE_F * TARGET_SIZE_T
IN_CHANNELS = 6
ORIG_F      = NFFT // 2 + 1
ORIG_T      = 1 + (WINDOW // HOP_LENGTH)

os.makedirs(OUT_DIR, exist_ok=True)


def load_model(path):
    m = ComplexUNetV7(DIMENSION, in_channels=IN_CHANNELS)
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval()
    return m


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


model_v7 = load_model(MODEL_V7)
ep_v7    = best_epoch(HIST_V7)
print(f'v7 loaded (best ep {ep_v7})')

files = sorted(f for f in os.listdir(TEST_DIR) if f.endswith('.mat'))
print(f'Found {len(files)} test files')

for fname in files:
    fpath = os.path.join(TEST_DIR, fname)
    stem  = os.path.splitext(fname)[0]
    print(f'\nProcessing {fname} ...')

    mat     = sio.loadmat(fpath)
    mixture = mat['out']['mixture'][0][0].astype(np.float32)
    fecg    = _extract_fecg(mat['out'])
    del mat

    if fecg.shape[0] < mixture.shape[0]:
        fecg = np.repeat(fecg, mixture.shape[0] // fecg.shape[0], axis=0)

    mix_win  = mixture[:, START:START + WINDOW].copy()
    fecg_win = fecg[:,   START:START + WINDOW].copy()

    stds = mix_win.std(axis=1, keepdims=True)
    stds = np.where(stds < 1e-8, 1.0, stds)
    mix_norm  = mix_win  / stds
    fecg_norm = fecg_win / stds

    mix_spec = _stft_multichannel(mix_norm, NFFT, HOP_LENGTH, WIN_LENGTH)
    x = _to_resized_complex_tensor(mix_spec, TARGET_SIZE_F, TARGET_SIZE_T).unsqueeze(0)

    with torch.no_grad():
        pred_v7 = model_v7(x)

    pred_v7_time = spec_to_time(pred_v7)
    t = np.linspace(START / FS, (START + WINDOW) / FS, WINDOW)

    n_ch = len(CHANNELS)
    fig, axes = plt.subplots(n_ch, 1, figsize=(14, 3 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]
    for i, ch in enumerate(CHANNELS):
        ax = axes[i]
        ax.plot(t, fecg_norm[ch], color='seagreen',   linewidth=1.0,
                label='Ground truth fECG', alpha=0.9)
        ax.plot(t, pred_v7_time[ch], color='darkorange', linewidth=0.9,
                label='Predicted fECG',    alpha=0.85)
        ax.set_ylabel(f'Channel {ch + 1}', fontsize=10)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        if i == 0:
            ax.legend(loc='upper right', fontsize=9)
    axes[-1].set_xlabel('Time (s)', fontsize=10)
    fig.suptitle(
        f'fECG Extraction (v7 paper arch+L1, ep {ep_v7}) — {stem}\n'
        f'Window: {START // FS}–{(START + WINDOW) // FS} s  |  normalised amplitude',
        fontsize=11,
    )
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'{stem}_v7_ep{ep_v7}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out_path}')

print('\nDone.')
