"""
Inferenta comparativa pe Sem1, Sem2, Sem3 din Test_DB.
Afiseaza: GT (verde) | v1 (albastru) | v9 (portocaliu) | v10 (rosu).

Rulat cu cel mai bun checkpoint disponibil pentru fiecare model.
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
from complex_network    import ComplexUNet
from complex_network_v9 import ComplexUNetV9
from complex_network_v10 import ComplexUNetV10

TEST_DIR = '/shared_storage/stan.edward/fECG_mvm_DB/Test_DB'
OUT_DIR  = '/shared_storage/iulia.orvas/paper/fECG_approx/plots_2026/testdb'

MODELS = {
    'v1': {
        'pth':   '../models/movement_CUNet_128x400_composed.pth',
        'hist':  '../models/movement_CUNet_128x400_composed_history.json',
        'color': 'steelblue',
        'cls':   'ComplexUNet',
    },
    'v9': {
        'pth':   '../models/movement_CUNet_v9_mask.pth',
        'hist':  '../models/movement_CUNet_v9_mask_history.json',
        'color': 'darkorange',
        'cls':   'ComplexUNetV9',
    },
    'v10': {
        'pth':   '../models/movement_CUNet_v10_fqrs.pth',
        'hist':  '../models/movement_CUNet_v10_fqrs_history.json',
        'color': 'crimson',
        'cls':   'ComplexUNetV10',
    },
}

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
    hist_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), hist_path)
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


def load_model(cfg, device):
    pth = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['pth'])
    cls = cfg['cls']
    if cls == 'ComplexUNet':
        m = ComplexUNet(DIMENSION, in_channels=IN_CHANNELS)
    elif cls == 'ComplexUNetV9':
        m = ComplexUNetV9(DIMENSION, in_channels=IN_CHANNELS)
    elif cls == 'ComplexUNetV10':
        m = ComplexUNetV10(DIMENSION, in_channels=IN_CHANNELS)
    else:
        raise ValueError(f'Unknown class: {cls}')
    m.load_state_dict(torch.load(pth, map_location=device))
    m.to(device).eval()
    return m


# ---------------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

loaded = {}
for name, cfg in MODELS.items():
    ep = best_epoch(cfg['hist'])
    loaded[name] = {'model': load_model(cfg, device), 'ep': ep, 'color': cfg['color']}
    print(f'{name} loaded  (best ep {ep})')

# ---------------------------------------------------------------------------
for fname in TARGET_FILES:
    fpath = os.path.join(TEST_DIR, fname)
    if not os.path.exists(fpath):
        print(f'SKIP — not found: {fpath}')
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

    stds     = mix_win.std(axis=1, keepdims=True)
    stds     = np.where(stds < 1e-8, 1.0, stds)
    mix_norm = mix_win  / stds
    fecg_norm= fecg_win / stds

    mix_spec = _stft_multichannel(mix_norm, NFFT, HOP_LENGTH, WIN_LENGTH)
    x = _to_resized_complex_tensor(mix_spec, TARGET_SIZE_F, TARGET_SIZE_T).unsqueeze(0).to(device)

    preds = {}
    with torch.no_grad():
        for name, info in loaded.items():
            preds[name] = spec_to_time(info['model'](x).cpu())

    t = np.linspace(START / FS, (START + WINDOW) / FS, WINDOW)
    n_ch = len(CHANNELS)

    # Layout: rânduri = [GT, v1, v9, v10], coloane = canale
    row_labels = [('GT', 'seagreen', None)] + [
        (name, info['color'], info) for name, info in loaded.items()
    ]
    n_rows = len(row_labels)  # 4

    fig, axes = plt.subplots(
        n_rows, n_ch,
        figsize=(5 * n_ch, 2.8 * n_rows),
        sharex=True,
    )
    # axes shape: (n_rows, n_ch)

    for row_i, (label, color, info) in enumerate(row_labels):
        for col_i, ch in enumerate(CHANNELS):
            ax = axes[row_i][col_i]

            if label == 'GT':
                ax.plot(t, fecg_norm[ch], color=color, lw=1.0, alpha=0.9)
            else:
                # fond gri: GT
                ax.plot(t, fecg_norm[ch], color='seagreen', lw=0.7, alpha=0.4)
                ax.plot(t, preds[label][ch], color=color, lw=1.0, alpha=0.9)

            ax.grid(True, alpha=0.25, lw=0.4)

            # eticheta rând (prima coloana)
            if col_i == 0:
                if label == 'GT':
                    ax.set_ylabel('Ground truth', fontsize=9, color=color, fontweight='bold')
                else:
                    ep = info['ep']
                    ax.set_ylabel(f'{label} ep{ep}', fontsize=9, color=color, fontweight='bold')

            # titlu coloana (primul rand)
            if row_i == 0:
                ax.set_title(f'Ch {ch + 1}', fontsize=10)

            # xlabel (ultimul rand)
            if row_i == n_rows - 1:
                ax.set_xlabel('Time (s)', fontsize=9)

    ep_v1  = loaded['v1']['ep']
    ep_v9  = loaded['v9']['ep']
    ep_v10 = loaded['v10']['ep']
    fig.suptitle(
        f'{stem}  —  v1 ep{ep_v1} (direct, SignalMSE) | '
        f'v9 ep{ep_v9} (mask, Sig+Cpl) | '
        f'v10 ep{ep_v10} (mask, Sig+Cpl+PeakMSE)\n'
        f'Window: {START//FS}–{(START+WINDOW)//FS} s  '
        f'[verde pal în fundal = GT]',
        fontsize=10,
    )
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'{stem}_v1v9v10_ep{ep_v9}_grid.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {out_path}')

print('\nDone.')
