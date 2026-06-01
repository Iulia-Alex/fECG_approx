"""
Generates:
  1. Loss curves for v1 (ep1-200) and v5 (current) from history JSON files.
  2. inference_plot (3-panel: Mixture | Predicted | Ground Truth) for all Test_DB files,
     for both v1 and v5.

Output dirs:
  ../plots_2026/loss_v1.png
  ../plots_2026/loss_v5.png
  ../plots_2026/testdb/{stem}_v1_plot.png
  ../plots_2026/testdb/{stem}_v5_plot.png
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from movement_dataset import (
    _stft_multichannel, _to_resized_complex_tensor, _extract_fecg,
    NFFT, HOP_LENGTH, WIN_LENGTH, TARGET_SIZE_F, TARGET_SIZE_T, FS,
)
from complex_network import ComplexUNet

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TEST_DIR  = '/shared_storage/stan.edward/fECG_mvm_DB/Test_DB'
OUT_DIR   = '../plots_2026/testdb'
LOSS_DIR  = '../plots_2026'

MODEL_V1  = '../models/movement_CUNet_128x400_composed.pth'
MODEL_V5  = '../models/movement_CUNet_128x400_ampw_scratch.pth'
HIST_V1   = MODEL_V1.replace('.pth', '_history.json')
HIST_V5   = MODEL_V5.replace('.pth', '_history.json')

WINDOW      = 4 * FS
START       = 10 * FS
CHANNELS    = [0, 1, 2]
DIMENSION   = TARGET_SIZE_F * TARGET_SIZE_T
IN_CHANNELS = 6
ORIG_F      = NFFT // 2 + 1
ORIG_T      = 1 + (WINDOW // HOP_LENGTH)

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Loss plots
# ---------------------------------------------------------------------------
def plot_loss(hist_path, out_path, title):
    hist = json.load(open(hist_path))
    train_loss = hist['train_loss']
    val_loss   = hist['val_loss']
    epochs     = list(range(1, len(train_loss) + 1))
    best_ep    = int(np.argmin(val_loss)) + 1
    best_val   = min(val_loss)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(epochs, train_loss, color='steelblue',  linewidth=1.2, label='Train loss')
    ax.plot(epochs, val_loss,   color='darkorange', linewidth=1.2, label='Val loss')
    ax.axvline(best_ep, color='seagreen', linestyle='--', linewidth=1.0,
               label=f'Best val={best_val:.6f} @ ep {best_ep}')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss (SignalMSE)', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Loss plot saved → {out_path}')

plot_loss(HIST_V1, os.path.join(LOSS_DIR, 'loss_v1_ep200.png'),
          'ComplexUNet v1 — SignalMSE  (200 epochs, DONE)')
plot_loss(HIST_V5, os.path.join(LOSS_DIR, 'loss_v5_ep81.png'),
          'ComplexUNet v5 — SignalMSE + 3×AmpWeightedMSE (from scratch, in progress)')

# ---------------------------------------------------------------------------
# 2. Load models
# ---------------------------------------------------------------------------
def load_model(path):
    m = ComplexUNet(DIMENSION, in_channels=IN_CHANNELS)
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval()
    return m

def best_epoch(hist_path):
    h = json.load(open(hist_path))
    vals = h.get('val_loss', [])
    return int(np.argmin(vals)) + 1 if vals else '?'

model_v1 = load_model(MODEL_V1)
model_v5 = load_model(MODEL_V5)
ep_v1 = best_epoch(HIST_V1)
ep_v5 = best_epoch(HIST_V5)
print(f'v1 loaded (ep {ep_v1}), v5 loaded (ep {ep_v5})')

# ---------------------------------------------------------------------------
# iSTFT helper
# ---------------------------------------------------------------------------
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
# 3-panel inference plot helper
# ---------------------------------------------------------------------------
def save_inference_plot(mix_norm, pred_time, fecg_norm, stem, model_name, epoch, out_path):
    t = np.linspace(START / FS, (START + WINDOW) / FS, WINDOW)
    n_ch = len(CHANNELS)
    fig, axes = plt.subplots(n_ch, 3, figsize=(18, 3 * n_ch), sharex=True)

    COLS   = ['Mixture (input X)', 'Predicted fECG', 'Ground truth fECG']
    COLORS = ['steelblue', 'darkorange', 'seagreen']
    SIGS   = [mix_norm, pred_time, fecg_norm]

    for i, ch in enumerate(CHANNELS):
        for j, (sig, col, title) in enumerate(zip(SIGS, COLORS, COLS)):
            ax = axes[i, j]
            ax.plot(t, sig[ch], linewidth=0.7, color=col)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            if i == 0:
                ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel(f'Channel {ch + 1}', fontsize=9)
            ax.set_xlim(t[0], t[-1])

    for j in range(3):
        axes[-1, j].set_xlabel('Time (s)', fontsize=9)

    fig.suptitle(
        f'fECG Extraction ({model_name}, ep {epoch}) — {stem}\n'
        f'Window: {START // FS}–{(START + WINDOW) // FS} s  |  normalised by mixture std per channel',
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out_path}')

# ---------------------------------------------------------------------------
# 3. Process each Test_DB file
# ---------------------------------------------------------------------------
files = sorted(f for f in os.listdir(TEST_DIR) if f.endswith('.mat'))
print(f'\nFound {len(files)} test files')

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
        pred_v1 = model_v1(x)
        pred_v5 = model_v5(x)

    pred_v1_time = spec_to_time(pred_v1)
    pred_v5_time = spec_to_time(pred_v5)

    save_inference_plot(mix_norm, pred_v1_time, fecg_norm, stem,
                        'v1 SignalMSE', ep_v1,
                        os.path.join(OUT_DIR, f'{stem}_v1_plot.png'))
    save_inference_plot(mix_norm, pred_v5_time, fecg_norm, stem,
                        'v5 SignalMSE+3×AmpW', ep_v5,
                        os.path.join(OUT_DIR, f'{stem}_v5_plot.png'))

print('\nDone.')
