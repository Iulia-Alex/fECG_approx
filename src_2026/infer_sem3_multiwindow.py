"""
Sem3 — v15 / v16 / v17 pe 5 intervale temporale diferite, Canal 1.
Grid: 4 randuri (GT / v15 / v16 / v17) x 5 coloane (ferestre de timp).
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import scipy.io as sio
import scipy.signal
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from movement_dataset import _extract_fecg, FS as FS1k
from movement_dataset_v15 import (
    NFFT as NFFT5, HOP_LENGTH as HOP5, WIN_LENGTH as WIN5,
    FS as FS5, WINDOW as WINDOW5,
)
from complex_network_v15 import ComplexUNet as ComplexUNetV15
from complex_network_v16 import ComplexAttentionUNet

SIGNAL_PATH = '/shared_storage/stan.edward/fECG_mvm_DB/Test_DB/Sem3.mat'
OUT_PATH    = '/shared_storage/iulia.orvas/paper/fECG_approx/plots_2026/testdb/Sem3_multiwindow.png'
MODELS_DIR  = '/shared_storage/iulia.orvas/paper/fECG_approx/models'

# intervale de start (in secunde), la 1 kHz
START_TIMES_S = [5, 30, 60, 120, 270]
CHANNEL       = 0   # Ch1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

_hann = torch.hann_window(WIN5)


def best_epoch(hist_path):
    if os.path.exists(hist_path):
        h = json.load(open(hist_path))
        vl = h.get('val_loss', [])
        if vl:
            return vl.index(min(vl)) + 1
    return '?'


def load_model(arch_cls, pth, hist):
    m = arch_cls(128 * 128, in_channels=6)
    m.load_state_dict(torch.load(pth, map_location=device))
    return m.to(device).eval(), best_epoch(hist)


print('Loading models...')
model_v15, ep_v15 = load_model(
    ComplexUNetV15,
    f'{MODELS_DIR}/movement_CUNet_v15_paper.pth',
    f'{MODELS_DIR}/movement_CUNet_v15_paper_history.json',
)
model_v16, ep_v16 = load_model(
    ComplexAttentionUNet,
    f'{MODELS_DIR}/movement_CUNet_v16_attention.pth',
    f'{MODELS_DIR}/movement_CUNet_v16_attention_history.json',
)
model_v17, ep_v17 = load_model(
    ComplexAttentionUNet,
    f'{MODELS_DIR}/movement_CUNet_v17_attsup.pth',
    f'{MODELS_DIR}/movement_CUNet_v17_attsup_history.json',
)
print(f'  v15 ep{ep_v15} | v16 ep{ep_v16} | v17 ep{ep_v17}')


def infer_500(model, mix_win_norm):
    specs = []
    for ch in range(mix_win_norm.shape[0]):
        S = librosa.stft(mix_win_norm[ch], n_fft=NFFT5, hop_length=HOP5,
                         win_length=WIN5, center=True)
        specs.append(S)
    spec = np.stack(specs)[:, :-1, :]
    x = (torch.from_numpy(spec.real.copy()) +
         1j * torch.from_numpy(spec.imag.copy())).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x).squeeze(0).cpu()
    zeros    = torch.zeros(out.shape[0], 1, out.shape[2], dtype=out.dtype)
    out_full = torch.cat([out, zeros], dim=1)
    pred = []
    for ch in range(out_full.shape[0]):
        sig = torch.istft(out_full[ch], n_fft=NFFT5, hop_length=HOP5,
                          win_length=WIN5, window=_hann,
                          center=True, length=WINDOW5).numpy()
        pred.append(sig)
    return np.stack(pred)


print(f'Loading {SIGNAL_PATH} ...')
mat     = sio.loadmat(SIGNAL_PATH)
mixture = mat['out']['mixture'][0][0].astype(np.float32)
fecg    = _extract_fecg(mat['out']).astype(np.float32)
del mat

if fecg.shape[0] < mixture.shape[0]:
    fecg = np.repeat(fecg, mixture.shape[0] // fecg.shape[0], axis=0)

mix_500  = scipy.signal.decimate(mixture, 2, axis=1).astype(np.float32)
fecg_500 = scipy.signal.decimate(fecg,    2, axis=1).astype(np.float32)

ROWS = [
    ('GT',              'seagreen',     None),
    (f'v15 ep{ep_v15}', 'steelblue',   model_v15),
    (f'v16 ep{ep_v16}', 'mediumpurple', model_v16),
    (f'v17 ep{ep_v17}', 'tomato',       model_v17),
]

n_wins = len(START_TIMES_S)
n_rows = len(ROWS)
fig, axes = plt.subplots(n_rows, n_wins,
                         figsize=(4.2 * n_wins, 2.4 * n_rows),
                         sharex='col')

for col_i, t_start_s in enumerate(START_TIMES_S):
    start_500 = int(t_start_s * FS5)
    end_500   = start_500 + WINDOW5

    mix_win  = mix_500[:,  start_500:end_500].copy()
    fecg_win = fecg_500[:, start_500:end_500].copy()

    stds = mix_win.std(axis=1, keepdims=True)
    stds = np.where(stds < 1e-8, 1.0, stds)
    mix_norm  = (mix_win  / stds).astype(np.float32)
    fecg_norm = (fecg_win / stds).astype(np.float32)

    t = np.linspace(t_start_s, t_start_s + WINDOW5 / FS5, WINDOW5)

    print(f'  Inferenta fereastra {t_start_s}s–{t_start_s + WINDOW5/FS5:.1f}s ...')

    preds = {}
    for label, color, model in ROWS:
        if model is not None:
            preds[label] = infer_500(model, mix_norm)

    for row_i, (label, color, model) in enumerate(ROWS):
        ax = axes[row_i][col_i]
        ax.plot(t, fecg_norm[CHANNEL], color='seagreen', lw=0.7, alpha=0.35)
        if model is not None:
            ax.plot(t, preds[label][CHANNEL], color=color, lw=1.0, alpha=0.9)
        else:
            ax.plot(t, fecg_norm[CHANNEL], color=color, lw=1.0, alpha=0.9)
        ax.grid(True, alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=7)
        if col_i == 0:
            ax.set_ylabel(label, fontsize=8, color=color, fontweight='bold')
        if row_i == 0:
            ax.set_title(f'{t_start_s}–{t_start_s + WINDOW5/FS5:.1f}s', fontsize=9)
        if row_i == n_rows - 1:
            ax.set_xlabel('t (s)', fontsize=8)

fig.suptitle(
    f'Sem3  —  v15 / v16 / v17  |  Ch1  |  5 ferestre temporale\n'
    f'[verde pal = GT]   v16 = v15+AG   v17 = v16+L_att',
    fontsize=10,
)
plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved -> {OUT_PATH}')
