"""
Inferenta completa pe toate semnalele din Test_DB (Sem1-Sem11) cu modelele
v1, v11, v13 (1 kHz) si v15, v16 (500 Hz).

Salvează pentru fiecare (semnal, model) un fisier .npz in:
  results_fECG_extraction/{semnal}_model_v{N}_ep{E}.npz

Fiecare fisier contine:
  prediction  : (6, N_samples) float32  — fECG prezis, in unitati originale
  ground_truth: (6, N_samples) float32  — fECG GT, in unitati originale
  fs          : int  — 1000 sau 500
  model       : str  — 'v1', 'v11', ...
  epoch       : int  — best epoch al checkpointului
  signal      : str  — 'Sem1', ...
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
import scipy.signal
from scipy.signal import resample_poly
import librosa

from movement_dataset import (
    _stft_multichannel, _to_resized_complex_tensor, _extract_fecg,
    NFFT as NFFT1k, HOP_LENGTH as HOP1k, WIN_LENGTH as WIN1k,
    TARGET_SIZE_F, TARGET_SIZE_T, FS as FS1k,
)
from movement_dataset_v15 import (
    NFFT as NFFT5, HOP_LENGTH as HOP5, WIN_LENGTH as WIN5,
    FS as FS5, WINDOW as WINDOW5,
)

from complex_network     import ComplexUNet       as CUNetV1
from complex_network_v15 import ComplexUNet       as CUNetV15
from complex_network_v16 import ComplexAttentionUNet as CUNetV16

MODELS_DIR = '/shared_storage/iulia.orvas/paper/fECG_approx/models'
TEST_DIR   = '/shared_storage/stan.edward/fECG_mvm_DB/Test_DB'
OUT_DIR    = '/shared_storage/iulia.orvas/paper/fECG_approx/results_fECG_extraction'

SIGNALS = [f'Sem{i}' for i in range(1, 12)]

WINDOW1k  = 4 * FS1k       # 4000 samples @ 1 kHz
DIMENSION = TARGET_SIZE_F * TARGET_SIZE_T   # 128*400
DIM_V15   = 128 * 128
ORIG_F1k  = NFFT1k // 2 + 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}', flush=True)

_hann5 = torch.hann_window(WIN5)


def best_epoch(hist_path):
    if not os.path.exists(hist_path):
        return 0
    h = json.load(open(hist_path))
    vals = h.get('val_loss', h.get('val', []))
    return vals.index(min(vals)) + 1 if vals else 0


MODEL_CFGS = {
    'v1': {
        'pth':  f'{MODELS_DIR}/movement_CUNet_128x400_composed.pth',
        'hist': f'{MODELS_DIR}/movement_CUNet_128x400_composed_history.json',
        'fs':   FS1k,
    },
    'v15': {
        'pth':  f'{MODELS_DIR}/movement_CUNet_v15_paper.pth',
        'hist': f'{MODELS_DIR}/movement_CUNet_v15_paper_history.json',
        'fs':   FS5,
    },
    'v16': {
        'pth':  f'{MODELS_DIR}/movement_CUNet_v16_attention.pth',
        'hist': f'{MODELS_DIR}/movement_CUNet_v16_attention_history.json',
        'fs':   FS5,
    },
    'v17': {
        'pth':  f'{MODELS_DIR}/movement_CUNet_v17_attsup.pth',
        'hist': f'{MODELS_DIR}/movement_CUNet_v17_attsup_history.json',
        'fs':   FS5,
    },
}

# --- incarca modele ---
m_v1  = CUNetV1(DIMENSION,  in_channels=6)
m_v15 = CUNetV15(DIM_V15,  in_channels=6)
m_v16 = CUNetV16(DIM_V15,  in_channels=6)
m_v17 = CUNetV16(DIM_V15,  in_channels=6)

MODEL_OBJS = {'v1': m_v1, 'v15': m_v15, 'v16': m_v16, 'v17': m_v17}

for v, m in MODEL_OBJS.items():
    cfg = MODEL_CFGS[v]
    m.load_state_dict(torch.load(cfg['pth'], map_location=device))
    m.to(device).eval()
    MODEL_CFGS[v]['epoch'] = best_epoch(cfg['hist'])
    print(f'  loaded {v}  (best ep {MODEL_CFGS[v]["epoch"]})', flush=True)


# ---------------------------------------------------------------------------
# Inferenta pe fereastra unica, 1 kHz
# ---------------------------------------------------------------------------
def infer_window_1k(model, mix_win_norm):
    """mix_win_norm: (6, WINDOW1k) float32 normalizat."""
    spec = _stft_multichannel(mix_win_norm, NFFT1k, HOP1k, WIN1k)
    x = _to_resized_complex_tensor(spec, TARGET_SIZE_F, TARGET_SIZE_T).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x).squeeze(0).cpu()
    real = F.interpolate(out.real.unsqueeze(0), size=(ORIG_F1k, 1 + WINDOW1k // HOP1k),
                         mode='bilinear', align_corners=False).squeeze(0)
    imag = F.interpolate(out.imag.unsqueeze(0), size=(ORIG_F1k, 1 + WINDOW1k // HOP1k),
                         mode='bilinear', align_corners=False).squeeze(0)
    pred = []
    for ch in range(out.shape[0]):
        s = real[ch].numpy() + 1j * imag[ch].numpy()
        pred.append(librosa.istft(s, hop_length=HOP1k, win_length=WIN1k,
                                  n_fft=NFFT1k, length=WINDOW1k))
    return np.stack(pred)   # (6, WINDOW1k)


# ---------------------------------------------------------------------------
# Inferenta pe fereastra unica, 500 Hz
# ---------------------------------------------------------------------------
def infer_window_500(model, mix_win_norm):
    """mix_win_norm: (6, WINDOW5) float32 normalizat."""
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
                          win_length=WIN5, window=_hann5,
                          center=True, length=WINDOW5).numpy()
        pred.append(sig)
    return np.stack(pred)   # (6, WINDOW5)


# ---------------------------------------------------------------------------
# Inferenta pe semnal complet (sliding non-overlapping windows)
# ---------------------------------------------------------------------------
def infer_full_1k(model, mixture_1k):
    """mixture_1k: (6, T) float32. Returneaza (6, T_out) la 1 kHz."""
    n_ch, total = mixture_1k.shape
    n_win = total // WINDOW1k
    out_total = n_win * WINDOW1k
    result = np.zeros((n_ch, out_total), dtype=np.float32)
    for i in range(n_win):
        s = i * WINDOW1k
        e = s + WINDOW1k
        win = mixture_1k[:, s:e].copy()
        stds = win.std(axis=1, keepdims=True)
        stds = np.where(stds < 1e-8, 1.0, stds)
        pred_norm = infer_window_1k(model, (win / stds).astype(np.float32))
        result[:, s:e] = pred_norm * stds
    return result


def infer_full_500(model, mixture_1k):
    """Decimează la 500 Hz, inferenta, returnează (6, T_out) la 500 Hz."""
    mix_500 = scipy.signal.decimate(mixture_1k, 2, axis=1).astype(np.float32)
    n_ch, total = mix_500.shape
    n_win = total // WINDOW5
    out_total = n_win * WINDOW5
    result = np.zeros((n_ch, out_total), dtype=np.float32)
    for i in range(n_win):
        s = i * WINDOW5
        e = s + WINDOW5
        win = mix_500[:, s:e].copy()
        stds = win.std(axis=1, keepdims=True)
        stds = np.where(stds < 1e-8, 1.0, stds)
        pred_norm = infer_window_500(model, (win / stds).astype(np.float32))
        result[:, s:e] = pred_norm * stds
    return result


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

for sig_name in SIGNALS:
    mat_path = os.path.join(TEST_DIR, f'{sig_name}.mat')
    if not os.path.exists(mat_path):
        print(f'SKIP {sig_name} — not found', flush=True)
        continue

    print(f'\n=== {sig_name} ===', flush=True)
    mat     = sio.loadmat(mat_path)
    mixture = mat['out']['mixture'][0][0].astype(np.float32)   # (6, 600000)
    fecg    = _extract_fecg(mat['out']).astype(np.float32)      # (6, 600000)
    del mat

    if fecg.shape[0] < mixture.shape[0]:
        fecg = np.repeat(fecg, mixture.shape[0] // fecg.shape[0], axis=0)

    for ver, cfg in MODEL_CFGS.items():
        model = MODEL_OBJS[ver]
        ep    = cfg['epoch']
        fs    = cfg['fs']
        fname = f'{sig_name}_model_{ver}_ep{ep}.npz'
        out_path = os.path.join(OUT_DIR, fname)

        if os.path.exists(out_path):
            print(f'  {ver}: already exists, skip', flush=True)
            continue

        print(f'  {ver} (ep {ep}, fs={fs}) ...', end=' ', flush=True)

        if fs == FS1k:
            pred = infer_full_1k(model, mixture)
            n_out = pred.shape[1]
            gt = fecg[:, :n_out].copy()
            save_fs = FS1k
            fs_native = FS1k
        else:
            pred = infer_full_500(model, mixture)
            fecg_500 = scipy.signal.decimate(fecg, 2, axis=1).astype(np.float32)
            n_out = pred.shape[1]
            gt = fecg_500[:, :n_out].copy()
            # upsample la 1 kHz pentru consistenta cu GT original
            pred = resample_poly(pred, 2, 1, axis=1).astype(np.float32)
            gt   = resample_poly(gt,   2, 1, axis=1).astype(np.float32)
            save_fs = FS1k
            fs_native = FS5

        np.savez_compressed(
            out_path,
            prediction   = pred,
            ground_truth = gt,
            fs           = np.int32(save_fs),
            fs_native    = np.int32(fs_native),
            model        = np.str_(ver),
            epoch        = np.int32(ep),
            signal       = np.str_(sig_name),
        )
        print(f'saved {fname}  shape={pred.shape}', flush=True)

print('\nDone.', flush=True)
