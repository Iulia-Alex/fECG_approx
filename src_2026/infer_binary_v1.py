"""
Inferență binary_v1 pe Short_time_intervals (test set).
Plotează: semnalul fECG, masca prezisă, masca reală.
"""

import os, sys, glob
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_binary_v1 import (
    PrecisionResUNet, IN_CH, FS, WINDOW, Q_WIN, S_WIN, BL_WIN,
    _compute_a_qrs, _compute_baseline,
)

MODEL_PATH = '/shared_storage/iulia.orvas/paper/fECG_approx/models/binary_v1_best.pth'
TEST_DIR   = '/shared_storage/stan.edward/fECG_mvm_DB/DB_1/Short_time_intervals'
OUT_DIR    = '/shared_storage/iulia.orvas/paper/fECG_approx/plots'
STRIDE_INF = 500    # stride mic → mască netedă
THRESH     = 0.5
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUT_DIR, exist_ok=True)


def load_and_preprocess(fpath):
    """Aceeași preprocesare ca în train: z-norm + A_QRS + baseline."""
    mat  = sio.loadmat(fpath)
    out  = mat['out']
    fecg = out['fecg'][0][0]
    if fecg.dtype == object:
        fecg = fecg.flat[0]
    fecg = fecg.astype(np.float32)

    fqrs = out['fqrs'][0][0]
    if fqrs.dtype == object:
        fqrs = fqrs.flat[0]
    fqrs = fqrs.ravel().astype(np.int32)

    mask = out['movement_mask'][0][0].ravel().astype(np.uint8)

    N     = fecg.shape[1]
    feats = np.empty((IN_CH, N), dtype=np.float32)

    for ch in range(6):
        mu  = fecg[ch].mean()
        std = fecg[ch].std() + 1e-8
        feats[ch] = (fecg[ch] - mu) / std

    ch0      = feats[0]
    feats[6] = _compute_a_qrs(ch0, fqrs)
    bl       = _compute_baseline(ch0)
    bl       = (bl - bl.mean()) / (bl.std() + 1e-8)
    feats[7] = bl

    return feats, mask


@torch.no_grad()
def infer_full(model, feats):
    """Inferență sliding-window cu agregare prin sumă (overlap-add)."""
    N     = feats.shape[1]
    prob_sum = np.zeros(N, dtype=np.float64)
    count    = np.zeros(N, dtype=np.float64)

    model.eval()
    for start in range(0, N - WINDOW + 1, STRIDE_INF):
        end = start + WINDOW
        x   = torch.from_numpy(feats[:, start:end]).unsqueeze(0).to(DEVICE)
        logit = model(x).squeeze().cpu().numpy()       # (WINDOW,)
        prob  = 1 / (1 + np.exp(-logit))
        prob_sum[start:end] += prob
        count[start:end]    += 1

    count = np.maximum(count, 1)
    return (prob_sum / count).astype(np.float32)


def plot_file(feats, mask, prob, fname, out_path):
    t = np.arange(len(mask)) / FS

    # Găsim segmentele de mișcare (bande colorate)
    def segments(binary):
        segs = []
        in_seg = False
        for i, v in enumerate(binary):
            if v and not in_seg:
                start_i = i; in_seg = True
            elif not v and in_seg:
                segs.append((start_i / FS, i / FS)); in_seg = False
        if in_seg:
            segs.append((start_i / FS, len(binary) / FS))
        return segs

    gt_segs   = segments(mask > 0)
    pred_segs = segments(prob > THRESH)

    fig, axes = plt.subplots(3, 1, figsize=(18, 8), sharex=True)
    fig.suptitle(f'{fname}  |  F1={_f1(prob > THRESH, mask):.4f}', fontsize=12)

    # Panel 1: semnal fECG
    ax = axes[0]
    ax.plot(t, feats[0], lw=0.4, color='#1f77b4')
    for s, e in gt_segs:
        ax.axvspan(s, e, alpha=0.18, color='red', label='GT mișcare')
    ax.set_ylabel('fECG ch0 (z)')
    ax.set_title('Semnal fECG + GT mișcare (roșu)')
    ax.legend(loc='upper right', fontsize=8)

    # Panel 2: probabilitate prezisă
    ax = axes[1]
    ax.plot(t, prob, lw=0.6, color='darkorange', label='P(mișcare)')
    ax.axhline(THRESH, color='gray', lw=0.8, ls='--', label=f'prag {THRESH}')
    for s, e in gt_segs:
        ax.axvspan(s, e, alpha=0.12, color='red')
    ax.set_ylabel('Probabilitate')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Probabilitate prezisă')
    ax.legend(loc='upper right', fontsize=8)

    # Panel 3: comparație binară
    ax = axes[2]
    ax.fill_between(t, mask.astype(float), alpha=0.5, color='red',    label='GT')
    ax.fill_between(t, (prob > THRESH).astype(float), alpha=0.5,
                    color='green', label='Predicție')
    ax.set_ylabel('Clasă (0/1)')
    ax.set_xlabel('Timp (s)')
    ax.set_title('GT (roșu) vs Predicție (verde)')
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Salvat: {out_path}')


def _f1(pred_bin, gt):
    pred_bin = pred_bin.astype(bool)
    gt       = gt.astype(bool)
    tp = (pred_bin & gt).sum()
    fp = (pred_bin & ~gt).sum()
    fn = (~pred_bin & gt).sum()
    return float(2 * tp / (2 * tp + fp + fn + 1e-8))


def main():
    print(f'Device: {DEVICE}')
    model = PrecisionResUNet(in_ch=IN_CH, use_transformer=True).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print('Model loaded.')

    # Alege câteva fișiere reprezentative (SNR diferit)
    files = sorted(glob.glob(os.path.join(TEST_DIR, '*.mat')))
    # Câte un fișier per variantă SNR
    pick  = files[:6]

    all_f1 = []
    for fpath in pick:
        fname = os.path.splitext(os.path.basename(fpath))[0]
        print(f'\nProcesez: {fname}')
        feats, mask = load_and_preprocess(fpath)
        prob        = infer_full(model, feats)
        f1          = _f1(prob > THRESH, mask)
        all_f1.append(f1)
        print(f'  mov%={100*mask.mean():.1f}%  F1={f1:.4f}')

        out_path = os.path.join(OUT_DIR, f'binary_v1_{fname}.png')
        plot_file(feats, mask, prob, fname, out_path)

    print(f'\nMean F1 pe {len(pick)} fișiere test: {np.mean(all_f1):.4f}')


if __name__ == '__main__':
    main()
