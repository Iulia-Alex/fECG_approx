"""
Sem10 — v15 / v16 / v17 pe 5 ferestre temporale, Ch1.
Fiecare fereastra e aleasa sa fie reprezentativa pentru o clasa de miscare.
Grid: 4 randuri (GT / v15 / v16 / v17) x 5 coloane (clase).
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
import matplotlib.patches as mpatches

from movement_dataset import _extract_fecg, FS as FS1k
from movement_dataset_v15 import (
    NFFT as NFFT5, HOP_LENGTH as HOP5, WIN_LENGTH as WIN5,
    FS as FS5, WINDOW as WINDOW5,
)
from complex_network_v15 import ComplexUNet as ComplexUNetV15
from complex_network_v16 import ComplexAttentionUNet

SIGNAL_PATH = '/shared_storage/stan.edward/fECG_mvm_DB/Test_DB/Sem10.mat'
OUT_PATH    = '/shared_storage/iulia.orvas/paper/fECG_approx/plots_2026/testdb/Sem10_multiwindow_classes.png'
MODELS_DIR  = '/shared_storage/iulia.orvas/paper/fECG_approx/models'

CLASS_NAMES   = ['Stationary', 'Linear', 'Helical', 'Screw']
CLASS_COLORS  = ['#888888', '#2196F3', '#FF9800', '#E53935']

# ferestre alese manual: (start_s, eticheta descriptiva)
WINDOWS = [
    (32.5,  0),   # Stationary pur
    (154.9, 1),   # Linear pur
    (574.6, 2),   # Helical pur
    (429.2, 3),   # Screw pur
    (177.8, None) # Tranzitie Screw(33%) + Stationary(67%)
]

CHANNEL = 0  # Ch1
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
cat_mask = mat['out']['category_mask'][0][0].flatten()
del mat

if fecg.shape[0] < mixture.shape[0]:
    fecg = np.repeat(fecg, mixture.shape[0] // fecg.shape[0], axis=0)

mix_500  = scipy.signal.decimate(mixture, 2, axis=1).astype(np.float32)
fecg_500 = scipy.signal.decimate(fecg,    2, axis=1).astype(np.float32)
cat_500  = cat_mask[::2]

ROWS = [
    ('GT',              'seagreen',     None),
    (f'v15 ep{ep_v15}', 'steelblue',   model_v15),
    (f'v16 ep{ep_v16}', 'mediumpurple', model_v16),
    (f'v17 ep{ep_v17}', 'tomato',       model_v17),
]

n_wins = len(WINDOWS)
n_rows = len(ROWS)
fig, axes = plt.subplots(n_rows, n_wins,
                         figsize=(4.2 * n_wins, 2.4 * n_rows),
                         sharex='col')

for col_i, (t_start_s, cls_hint) in enumerate(WINDOWS):
    start_500 = int(t_start_s * FS5)
    end_500   = start_500 + WINDOW5

    mix_win  = mix_500[:,  start_500:end_500].copy()
    fecg_win = fecg_500[:, start_500:end_500].copy()
    cat_win  = cat_500[start_500:end_500]

    stds = mix_win.std(axis=1, keepdims=True)
    stds = np.where(stds < 1e-8, 1.0, stds)
    mix_norm  = (mix_win  / stds).astype(np.float32)
    fecg_norm = (fecg_win / stds).astype(np.float32)

    t = np.linspace(t_start_s, t_start_s + WINDOW5 / FS5, WINDOW5)

    # distributia claselor in fereastra
    counts = [(np.mean(cat_win == c) * 100, CLASS_NAMES[c], CLASS_COLORS[c])
              for c in range(4) if np.any(cat_win == c)]
    counts.sort(reverse=True)
    cls_str = '  '.join([f'{pct:.0f}% {name}' for pct, name, _ in counts[:3]])

    print(f'  [{col_i+1}] t={t_start_s:.1f}s  {cls_str}')

    preds = {}
    for label, color, model in ROWS:
        if model is not None:
            preds[label] = infer_500(model, mix_norm)

    for row_i, (label, color, model) in enumerate(ROWS):
        ax = axes[row_i][col_i]

        # colorbar de clase in background
        for c in range(4):
            mask = (cat_win == c)
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            # deseneaza dreptunghiuri colorate pentru fiecare segment
            in_seg = False
            for i, idx in enumerate(range(len(cat_win))):
                cur = cat_win[idx] == c
                if cur and not in_seg:
                    seg_start = t[idx]
                    in_seg = True
                elif not cur and in_seg:
                    ax.axvspan(seg_start, t[idx], alpha=0.10,
                               color=CLASS_COLORS[c], lw=0)
                    in_seg = False
            if in_seg:
                ax.axvspan(seg_start, t[-1], alpha=0.10,
                           color=CLASS_COLORS[c], lw=0)

        ax.plot(t, fecg_norm[CHANNEL], color='seagreen', lw=0.7, alpha=0.35)
        if model is not None:
            ax.plot(t, preds[label][CHANNEL], color=color, lw=1.0, alpha=0.9)
        else:
            ax.plot(t, fecg_norm[CHANNEL], color=color, lw=1.0, alpha=0.9)
        ax.grid(True, alpha=0.20, lw=0.4)
        ax.tick_params(labelsize=7)
        if col_i == 0:
            ax.set_ylabel(label, fontsize=8, color=color, fontweight='bold')
        if row_i == 0:
            ax.set_title(f'{cls_str}\n({t_start_s:.0f}–{t_start_s + WINDOW5/FS5:.1f}s)',
                         fontsize=8)
        if row_i == n_rows - 1:
            ax.set_xlabel('t (s)', fontsize=8)

# legenda clase
patches = [mpatches.Patch(color=CLASS_COLORS[c], alpha=0.7, label=f'{c} {CLASS_NAMES[c]}')
           for c in range(4)]
fig.legend(handles=patches, loc='lower center', ncol=4,
           fontsize=9, framealpha=0.8,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    f'Sem10  —  v15 / v16 / v17  |  Ch1  |  ferestre per tip miscare\n'
    f'[verde pal = GT]   v16 = v15+AG   v17 = v16+L_att   '
    f'[fundal colorat = clasa miscare]',
    fontsize=10,
)
plt.tight_layout(rect=[0, 0.04, 1, 1])
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved -> {OUT_PATH}')
