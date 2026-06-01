"""
Inferenta comparativa pe Sem1, Sem2, Sem3 — v12, v13, v15.

v12/v13: pipeline 1000 Hz, STFT 128×400 (identic cu dataset-ul de antrenament)
v15:     pipeline 500 Hz, STFT 128×128 (paper-faithful, scipy.signal.decimate)

Grid: randuri = [GT | v12 | v13 | v15], coloane = [Ch1, Ch2, Ch3]
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

from movement_dataset import (
    _stft_multichannel, _to_resized_complex_tensor, _extract_fecg,
    NFFT as NFFT1k, HOP_LENGTH as HOP1k, WIN_LENGTH as WIN1k,
    TARGET_SIZE_F, TARGET_SIZE_T, FS as FS1k,
)
from movement_dataset_v15 import (
    NFFT as NFFT5, HOP_LENGTH as HOP5, WIN_LENGTH as WIN5,
    FS as FS5, WINDOW as WINDOW5,
)
from complex_network_v12 import ComplexUNetV12
from complex_network_v13 import ComplexUNetV13
from complex_network_v15 import ComplexUNet as ComplexUNetV15

# ---------------------------------------------------------------------------
TEST_DIR    = '/shared_storage/stan.edward/fECG_mvm_DB/Test_DB'
OUT_DIR     = '/shared_storage/iulia.orvas/paper/fECG_approx/plots_2026/testdb'
MODELS_DIR  = '/shared_storage/iulia.orvas/paper/fECG_approx/models'

TARGET_FILES = ['Sem1.mat', 'Sem2.mat', 'Sem3.mat']
CHANNELS     = [0, 1, 2]

WINDOW_1k   = 4 * FS1k           # 4000 samples @ 1000 Hz
START_1k    = 10 * FS1k          # start at 10 s
DIMENSION   = TARGET_SIZE_F * TARGET_SIZE_T
IN_CHANNELS = 6

ORIG_F_1k   = NFFT1k // 2 + 1   # 129
ORIG_T_1k   = 1 + (WINDOW_1k // HOP1k)  # 401

os.makedirs(OUT_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# ---------------------------------------------------------------------------
# Helpers — 1000 Hz pipeline (v12, v13)
# ---------------------------------------------------------------------------

def preprocess_1k(mixture_1k):
    """mixture_1k: (6, N_samples) float32, normalized → complex tensor (1, 6, 128, 400)"""
    spec = _stft_multichannel(mixture_1k, NFFT1k, HOP1k, WIN1k)
    x = _to_resized_complex_tensor(spec, TARGET_SIZE_F, TARGET_SIZE_T)
    return x.unsqueeze(0).to(device)


def spec_to_time_1k(tensor):
    """tensor: (1, 6, 128, 400) complex → (6, 4000) float32"""
    t = tensor.squeeze(0).cpu()
    real = F.interpolate(t.real.unsqueeze(0), size=(ORIG_F_1k, ORIG_T_1k),
                         mode='bilinear', align_corners=False).squeeze(0)
    imag = F.interpolate(t.imag.unsqueeze(0), size=(ORIG_F_1k, ORIG_T_1k),
                         mode='bilinear', align_corners=False).squeeze(0)
    out = []
    for ch in range(t.shape[0]):
        spec = real[ch].numpy() + 1j * imag[ch].numpy()
        sig  = librosa.istft(spec, hop_length=HOP1k, win_length=WIN1k,
                             n_fft=NFFT1k, length=WINDOW_1k)
        out.append(sig)
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Helpers — 500 Hz pipeline (v15)
# ---------------------------------------------------------------------------

_hann_v15 = torch.hann_window(WIN5)

def preprocess_500(mixture_1k):
    """mixture_1k: (6, N_samples) — downsample + STFT 128×128 → (1, 6, 128, 128)"""
    mix_500 = scipy.signal.decimate(mixture_1k, 2, axis=1).astype(np.float32)
    start_500 = START_1k // 2
    win_500   = WINDOW5
    mix_win   = mix_500[:, start_500:start_500 + win_500]
    stds = mix_win.std(axis=1, keepdims=True)
    stds = np.where(stds < 1e-8, 1.0, stds)
    mix_win = mix_win / stds
    # STFT per channel
    specs = []
    for ch in range(mix_win.shape[0]):
        S = librosa.stft(mix_win[ch], n_fft=NFFT5, hop_length=HOP5,
                         win_length=WIN5, center=True)
        specs.append(S)
    spec = np.stack(specs, axis=0)   # (6, 129, 128)
    spec = spec[:, :-1, :]           # (6, 128, 128)
    t = torch.from_numpy(np.stack([spec.real, spec.imag], axis=0))
    x = t[0] + 1j * t[1]            # (6, 128, 128) complex
    return x.unsqueeze(0).to(device), stds


def spec_to_time_500(tensor):
    """tensor: (1, 6, 128, 128) complex → (6, 1915) float32"""
    t = tensor.squeeze(0).cpu()
    # zero-pad last freq bin back to 129
    B_C, Fq, T = t.shape[0], t.shape[1], t.shape[2]
    zeros = torch.zeros(B_C, 1, T, dtype=t.dtype)
    t_full = torch.cat([t, zeros], dim=1)   # (6, 129, 128)
    out = []
    for ch in range(t_full.shape[0]):
        sig = torch.istft(
            t_full[ch], n_fft=NFFT5, hop_length=HOP5,
            win_length=WIN5, window=_hann_v15,
            center=True, length=WINDOW5,
        ).numpy()
        out.append(sig)
    return np.stack(out, axis=0)


def best_epoch(hist_path):
    if os.path.exists(hist_path):
        h = json.load(open(hist_path))
        vl = h.get('val_loss', [])
        if vl:
            return vl.index(min(vl)) + 1
    return '?'


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

def load_v12():
    pth  = f'{MODELS_DIR}/movement_CUNet_v12_gainmask.pth'
    hist = f'{MODELS_DIR}/movement_CUNet_v12_gainmask_history.json'
    m = ComplexUNetV12(DIMENSION, in_channels=IN_CHANNELS)
    m.load_state_dict(torch.load(pth, map_location=device))
    return m.to(device).eval(), best_epoch(hist)


def load_v13():
    pth  = f'{MODELS_DIR}/movement_CUNet_v13_instanorm.pth'
    hist = f'{MODELS_DIR}/movement_CUNet_v13_instanorm_history.json'
    m = ComplexUNetV13(DIMENSION, in_channels=IN_CHANNELS)
    m.load_state_dict(torch.load(pth, map_location=device))
    return m.to(device).eval(), best_epoch(hist)


def load_v15():
    pth  = f'{MODELS_DIR}/movement_CUNet_v15_paper.pth'
    hist = f'{MODELS_DIR}/movement_CUNet_v15_paper_history.json'
    dim5 = 128 * 128
    m = ComplexUNetV15(dim5, in_channels=IN_CHANNELS)
    m.load_state_dict(torch.load(pth, map_location=device))
    return m.to(device).eval(), best_epoch(hist)


print('Loading models...')
model_v12, ep_v12 = load_v12()
model_v13, ep_v13 = load_v13()
model_v15, ep_v15 = load_v15()
print(f'  v12 ep{ep_v12} | v13 ep{ep_v13} | v15 ep{ep_v15}')

# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

for fname in TARGET_FILES:
    fpath = os.path.join(TEST_DIR, fname)
    if not os.path.exists(fpath):
        print(f'SKIP: {fpath}')
        continue
    stem = os.path.splitext(fname)[0]
    print(f'\nProcessing {fname} ...')

    mat     = sio.loadmat(fpath)
    mixture = mat['out']['mixture'][0][0].astype(np.float32)  # (6, 600000)
    fecg    = _extract_fecg(mat['out'])                        # (6, 600000) or (1, 600000)
    del mat

    if fecg.shape[0] < mixture.shape[0]:
        fecg = np.repeat(fecg, mixture.shape[0] // fecg.shape[0], axis=0)

    # Window at 1000 Hz
    mix_win_1k  = mixture[:, START_1k:START_1k + WINDOW_1k].copy()
    fecg_win_1k = fecg[:,   START_1k:START_1k + WINDOW_1k].copy()

    stds_1k  = mix_win_1k.std(axis=1, keepdims=True)
    stds_1k  = np.where(stds_1k < 1e-8, 1.0, stds_1k)
    mix_norm_1k  = mix_win_1k  / stds_1k
    fecg_norm_1k = fecg_win_1k / stds_1k

    # Window at 500 Hz (for GT of v15)
    mix_full_1k = mixture / stds_1k
    mix_500 = scipy.signal.decimate(mix_full_1k, 2, axis=1).astype(np.float32)
    fecg_500 = scipy.signal.decimate(fecg / stds_1k, 2, axis=1).astype(np.float32)
    start_500 = START_1k // 2
    fecg_norm_500 = fecg_500[:, start_500:start_500 + WINDOW5]

    # Preprocess inputs
    x_1k    = preprocess_1k(mix_norm_1k)
    x_500, _ = preprocess_500(mix_full_1k)

    with torch.no_grad():
        pred_v12 = spec_to_time_1k(model_v12(x_1k).cpu())
        pred_v13 = spec_to_time_1k(model_v13(x_1k).cpu())
        pred_v15 = spec_to_time_500(model_v15(x_500).cpu())

    t_1k  = np.linspace(START_1k / FS1k, (START_1k + WINDOW_1k) / FS1k, WINDOW_1k)
    t_500 = np.linspace(START_1k / FS1k, (START_1k + WINDOW_1k) / FS1k, WINDOW5)

    n_ch   = len(CHANNELS)
    ROWS   = [
        ('GT (1kHz)',  'seagreen',    None,       fecg_norm_1k, t_1k),
        (f'v12 ep{ep_v12}', 'chocolate', pred_v12, fecg_norm_1k, t_1k),
        (f'v13 ep{ep_v13}', 'steelblue', pred_v13, fecg_norm_1k, t_1k),
        (f'v15 ep{ep_v15} (500Hz)', 'mediumpurple', pred_v15, fecg_norm_500, t_500),
    ]

    fig, axes = plt.subplots(len(ROWS), n_ch,
                             figsize=(5 * n_ch, 2.6 * len(ROWS)), sharex=False)

    for row_i, (label, color, pred, gt, t_ax) in enumerate(ROWS):
        for col_i, ch in enumerate(CHANNELS):
            ax = axes[row_i][col_i]
            ax.plot(t_ax, gt[ch], color='seagreen', lw=0.7, alpha=0.35)
            if pred is not None:
                ax.plot(t_ax, pred[ch], color=color, lw=1.0, alpha=0.9)
            else:
                ax.plot(t_ax, gt[ch], color=color, lw=1.0, alpha=0.9)
            ax.grid(True, alpha=0.25, lw=0.4)
            if col_i == 0:
                ax.set_ylabel(label, fontsize=8, color=color, fontweight='bold')
            if row_i == 0:
                ax.set_title(f'Ch {ch + 1}', fontsize=10)
            if row_i == len(ROWS) - 1:
                ax.set_xlabel('Time (s)', fontsize=9)

    fig.suptitle(
        f'{stem}  —  v12 / v13 / v15  |  window {START_1k // FS1k}–{(START_1k + WINDOW_1k) // FS1k} s\n'
        f'[verde pal = GT]  v15 la 500 Hz (decimat)',
        fontsize=10,
    )
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'{stem}_v12v13v15_grid.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {out_path}')

print('\nDone.')
