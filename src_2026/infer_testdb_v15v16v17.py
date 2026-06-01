"""
Inferenta comparativa pe Sem1, Sem2, Sem3 — v15, v16, v17.

Toate folosesc pipeline-ul 500 Hz (scipy.signal.decimate, STFT 128×128).
v16 = v15 + Attention Gates (fara supervizare).
v17 = v16 + L_att (attention supervision pe fQRS mask).

Grid: randuri = [GT | v15 | v16 | v17], coloane = [Ch1, Ch2, Ch3]
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
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

TEST_DIR   = '/shared_storage/stan.edward/fECG_mvm_DB/Test_DB'
OUT_DIR    = '/shared_storage/iulia.orvas/paper/fECG_approx/plots_2026/testdb'
MODELS_DIR = '/shared_storage/iulia.orvas/paper/fECG_approx/models'

TARGET_FILES = [f'Sem{i}.mat' for i in range(1, 12)]
CHANNELS     = [0, 1, 2]
IN_CHANNELS  = 6
DIM_V15      = 128 * 128

START_1k = 10 * FS1k   # 10 s @ 1000 Hz

os.makedirs(OUT_DIR, exist_ok=True)
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
    m = arch_cls(DIM_V15, in_channels=IN_CHANNELS)
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
    """mix_win_norm: (6, WINDOW5) float32 — normalized. Returns (6, WINDOW5) float32."""
    specs = []
    for ch in range(mix_win_norm.shape[0]):
        S = librosa.stft(mix_win_norm[ch], n_fft=NFFT5, hop_length=HOP5,
                         win_length=WIN5, center=True)
        specs.append(S)
    spec = np.stack(specs)[:, :-1, :]   # (6, 128, 128)
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


for fname in TARGET_FILES:
    fpath = os.path.join(TEST_DIR, fname)
    if not os.path.exists(fpath):
        print(f'SKIP: {fpath}')
        continue
    stem = os.path.splitext(fname)[0]
    print(f'\nProcessing {fname} ...')

    mat     = sio.loadmat(fpath)
    mixture = mat['out']['mixture'][0][0].astype(np.float32)   # (6, 600000)
    fecg    = _extract_fecg(mat['out']).astype(np.float32)
    del mat

    if fecg.shape[0] < mixture.shape[0]:
        fecg = np.repeat(fecg, mixture.shape[0] // fecg.shape[0], axis=0)

    # Decimate to 500 Hz
    mix_500  = scipy.signal.decimate(mixture, 2, axis=1).astype(np.float32)
    fecg_500 = scipy.signal.decimate(fecg,    2, axis=1).astype(np.float32)

    start_500 = START_1k // 2
    mix_win   = mix_500[:,  start_500:start_500 + WINDOW5].copy()
    fecg_win  = fecg_500[:, start_500:start_500 + WINDOW5].copy()

    stds = mix_win.std(axis=1, keepdims=True)
    stds = np.where(stds < 1e-8, 1.0, stds)
    mix_norm  = (mix_win  / stds).astype(np.float32)
    fecg_norm = (fecg_win / stds).astype(np.float32)

    pred_v15 = infer_500(model_v15, mix_norm)
    pred_v16 = infer_500(model_v16, mix_norm)
    pred_v17 = infer_500(model_v17, mix_norm)

    t = np.linspace(START_1k / FS1k,
                    (START_1k / FS1k) + WINDOW5 / FS5,
                    WINDOW5)

    ROWS = [
        ('GT',              'seagreen',     None,      fecg_norm),
        (f'v15 ep{ep_v15}', 'steelblue',   pred_v15,  fecg_norm),
        (f'v16 ep{ep_v16}', 'mediumpurple', pred_v16, fecg_norm),
        (f'v17 ep{ep_v17}', 'tomato',       pred_v17, fecg_norm),
    ]

    n_ch = len(CHANNELS)
    fig, axes = plt.subplots(len(ROWS), n_ch,
                             figsize=(5 * n_ch, 2.6 * len(ROWS)), sharex=True)

    for row_i, (label, color, pred, gt) in enumerate(ROWS):
        for col_i, ch in enumerate(CHANNELS):
            ax = axes[row_i][col_i]
            ax.plot(t, gt[ch], color='seagreen', lw=0.7, alpha=0.35)
            if pred is not None:
                ax.plot(t, pred[ch], color=color, lw=1.0, alpha=0.9)
            else:
                ax.plot(t, gt[ch], color=color, lw=1.0, alpha=0.9)
            ax.grid(True, alpha=0.25, lw=0.4)
            if col_i == 0:
                ax.set_ylabel(label, fontsize=8, color=color, fontweight='bold')
            if row_i == 0:
                ax.set_title(f'Ch {ch + 1}', fontsize=10)
            if row_i == len(ROWS) - 1:
                ax.set_xlabel('Time (s)', fontsize=9)

    fig.suptitle(
        f'{stem}  —  v15 / v16 / v17  |  window {START_1k // FS1k}–'
        f'{START_1k // FS1k + WINDOW5 // FS5:.2f} s @ 500 Hz\n'
        f'[verde pal = GT]   v16 = v15+AG   v17 = v16+L_att',
        fontsize=10,
    )
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'{stem}_v15_v16_v17.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {out_path}')

print('\nDone.')
