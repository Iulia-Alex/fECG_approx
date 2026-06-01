"""
Inference on Test_DB — v5 and v6 models, epoch number in filename.
Saves: {stem}_v5_ep{N}.png  and  {stem}_v6_ep{N}.png
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
from complex_network import ComplexUNet

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TEST_DIR = '/shared_storage/stan.edward/fECG_mvm_DB/Test_DB'
OUT_DIR  = '../plots_2026/testdb'

MODEL_V5 = '../models/movement_CUNet_128x400_ampw_scratch.pth'
MODEL_V6 = '../models/movement_CUNet_128x400_v6_baseline.pth'

HIST_V5  = MODEL_V5.replace('.pth', '_history.json')
HIST_V6  = MODEL_V6.replace('.pth', '_history.json')

WINDOW      = 4 * FS       # 4000 samples
START       = 10 * FS      # start at 10 s
CHANNELS    = [0, 1, 2]
DIMENSION   = TARGET_SIZE_F * TARGET_SIZE_T
IN_CHANNELS = 6
ORIG_F      = NFFT // 2 + 1
ORIG_T      = 1 + (WINDOW // HOP_LENGTH)

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model(path):
    m = ComplexUNet(DIMENSION, in_channels=IN_CHANNELS)
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

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
model_v5 = load_model(MODEL_V5)
model_v6 = load_model(MODEL_V6)
ep_v5    = best_epoch(HIST_V5)
ep_v6    = best_epoch(HIST_V6)
print(f'v5 loaded (best ep {ep_v5}),  v6 loaded (best ep {ep_v6})')

# ---------------------------------------------------------------------------
# Process each file
# ---------------------------------------------------------------------------
files = sorted(f for f in os.listdir(TEST_DIR) if f.endswith('.mat'))
print(f'Found {len(files)} test files: {files}')

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
        pred_v5 = model_v5(x)
        pred_v6 = model_v6(x)

    pred_v5_time = spec_to_time(pred_v5)
    pred_v6_time = spec_to_time(pred_v6)
    t = np.linspace(START / FS, (START + WINDOW) / FS, WINDOW)

    def save_plot(pred_time, model_label, epoch, out_path):
        n_ch = len(CHANNELS)
        fig, axes = plt.subplots(n_ch, 1, figsize=(14, 3 * n_ch), sharex=True)
        if n_ch == 1:
            axes = [axes]
        for i, ch in enumerate(CHANNELS):
            ax = axes[i]
            ax.plot(t, fecg_norm[ch], color='seagreen',   linewidth=1.0,
                    label='Ground truth fECG', alpha=0.9)
            ax.plot(t, pred_time[ch], color='darkorange',  linewidth=0.9,
                    label='Predicted fECG',    alpha=0.85)
            ax.set_ylabel(f'Channel {ch + 1}', fontsize=10)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            if i == 0:
                ax.legend(loc='upper right', fontsize=9)
        axes[-1].set_xlabel('Time (s)', fontsize=10)
        fig.suptitle(
            f'fECG Extraction ({model_label}, ep {epoch}) — {stem}\n'
            f'Window: {START // FS}–{(START + WINDOW) // FS} s  |  normalised amplitude',
            fontsize=11,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved → {out_path}')

    save_plot(pred_v5_time, 'v5 SignalMSE+3×AmpW', ep_v5,
              os.path.join(OUT_DIR, f'{stem}_v5_ep{ep_v5}.png'))
    save_plot(pred_v6_time, 'v6 SignalMSE+AmpW+Baseline', ep_v6,
              os.path.join(OUT_DIR, f'{stem}_v6_ep{ep_v6}.png'))

print('\nDone.')
