"""
Inferenta v15/v16/v17 pe 3 ferestre din setul de validare.
Reproduce split-ul de antrenare (seed=42, val=15%) si alege 3 ferestre
din fisiere diferite.

Grid: 5 randuri (Mixture / GT / v15 / v16 / v17) x 3 coloane (ferestre val).
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import random_split
import scipy.io as sio
import scipy.signal
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from movement_dataset_v15 import (
    MovementECGDatasetPaper,
    NFFT as NFFT5, HOP_LENGTH as HOP5, WIN_LENGTH as WIN5,
    FS as FS5, WINDOW as WINDOW5,
)
from complex_network_v15 import ComplexUNet as ComplexUNetV15
from complex_network_v16 import ComplexAttentionUNet

DATA_DIR   = '/shared_storage/iulia.orvas/paper/fECG_approx/data/movement_ecg'
OUT_PATH   = '/shared_storage/iulia.orvas/paper/fECG_approx/plots_2026/testdb/valset_v15_v16_v17.png'
MODELS_DIR = '/shared_storage/iulia.orvas/paper/fECG_approx/models'
VAL_SPLIT  = 0.15
SEED       = 42
CHANNEL    = 0   # Ch1

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

# --- reproduce split de validare ---
print('Building dataset index...')
dataset = MovementECGDatasetPaper(DATA_DIR)
n_val   = int(len(dataset) * VAL_SPLIT)
n_train = len(dataset) - n_val
_, val_set = random_split(
    dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED),
)
val_indices = list(val_set.indices)
print(f'  Total windows: {len(dataset)}  |  Val: {len(val_indices)}')

# alege 3 ferestre din fisiere diferite
chosen = []
seen_files = set()
for idx in val_indices:
    file_idx, start = dataset.windows[idx]
    if file_idx not in seen_files:
        seen_files.add(file_idx)
        chosen.append((idx, file_idx, start))
    if len(chosen) == 3:
        break

print('Ferestre alese:')
for idx, file_idx, start in chosen:
    fname = os.path.basename(dataset.files[file_idx])
    print(f'  idx={idx}  file={fname}  start={start} ({start/FS5:.1f}s)')


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


ROWS = [
    ('Mixture',         '#444444',     'mixture'),
    ('GT fECG',         'seagreen',    'fecg'),
    (f'v15 ep{ep_v15}', 'steelblue',   'v15'),
    (f'v16 ep{ep_v16}', 'mediumpurple','v16'),
    (f'v17 ep{ep_v17}', 'tomato',      'v17'),
]

n_cols = len(chosen)
n_rows = len(ROWS)
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(5.5 * n_cols, 2.2 * n_rows),
                         sharex='col')

for col_i, (idx, file_idx, start) in enumerate(chosen):
    fname = os.path.basename(dataset.files[file_idx])
    mixture, fecg = dataset._get_signals(file_idx)

    mix_win  = mixture[:, start:start + WINDOW5].copy()
    fecg_win = fecg[:,   start:start + WINDOW5].copy()

    stds = mix_win.std(axis=1, keepdims=True)
    stds = np.where(stds < 1e-8, 1.0, stds)
    mix_norm  = (mix_win  / stds).astype(np.float32)
    fecg_norm = (fecg_win / stds).astype(np.float32)

    t_start_s = start / FS5
    t = np.linspace(t_start_s, t_start_s + WINDOW5 / FS5, WINDOW5)

    print(f'  Inferenta [{col_i+1}] {fname[:40]} t={t_start_s:.1f}s ...')
    pred_v15 = infer_500(model_v15, mix_norm)
    pred_v16 = infer_500(model_v16, mix_norm)
    pred_v17 = infer_500(model_v17, mix_norm)

    signals = {
        'mixture': mix_norm,
        'fecg':    fecg_norm,
        'v15':     pred_v15,
        'v16':     pred_v16,
        'v17':     pred_v17,
    }

    for row_i, (label, color, key) in enumerate(ROWS):
        ax = axes[row_i][col_i]
        sig = signals[key]

        if key in ('v15', 'v16', 'v17'):
            ax.plot(t, fecg_norm[CHANNEL], color='seagreen', lw=0.7, alpha=0.35)

        ax.plot(t, sig[CHANNEL], color=color, lw=0.9, alpha=0.9)
        ax.grid(True, alpha=0.20, lw=0.4)
        ax.tick_params(labelsize=7)

        if col_i == 0:
            ax.set_ylabel(label, fontsize=8, color=color, fontweight='bold')
        if row_i == 0:
            short = fname.replace('fecgsyn_Long_time_segment_', '').replace('_snr6dB.mat', '')
            ax.set_title(f'{short}\nt={t_start_s:.1f}–{t_start_s + WINDOW5/FS5:.1f}s', fontsize=8)
        if row_i == n_rows - 1:
            ax.set_xlabel('t (s)', fontsize=8)

fig.suptitle(
    f'Validation set  —  v15 / v16 / v17  |  Ch1\n'
    f'Rand 1 = amestec (dificultatea task-ului)   [verde pal = GT]   v16=v15+AG   v17=v16+L_att',
    fontsize=10,
)
plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved -> {OUT_PATH}')
