"""
Inferență binary_v1 pe fișierele Sem din Test_DB.
Plotează semnal complet + zoom pe 2 minute pentru Sem1, Sem2, Sem3.
"""

import os, sys
import numpy as np
import scipy.io as sio
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_binary_v1 import (
    PrecisionResUNet, IN_CH, FS, WINDOW,
    _compute_a_qrs, _compute_baseline,
)

MODEL_PATH = '/shared_storage/iulia.orvas/paper/fECG_approx/models/binary_v1_best.pth'
TEST_DB    = '/shared_storage/stan.edward/fECG_mvm_DB/Test_DB'
OUT_DIR    = '/shared_storage/iulia.orvas/paper/fECG_approx/plots'
STRIDE_INF = 500
THRESH     = 0.5
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEM_FILES  = ['Sem1', 'Sem2', 'Sem3']


def load_sem(fname):
    fpath = os.path.join(TEST_DB, fname + '.mat')
    mat   = sio.loadmat(fpath)
    out   = mat['out']

    fecg = out['fecg'][0][0]
    if fecg.dtype == object: fecg = fecg.flat[0]
    fecg = fecg.astype(np.float32)

    fqrs = out['fqrs'][0][0]
    if fqrs.dtype == object: fqrs = fqrs.flat[0]
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

    return feats, mask, fecg


@torch.no_grad()
def infer_full(model, feats):
    N        = feats.shape[1]
    prob_sum = np.zeros(N, dtype=np.float64)
    count    = np.zeros(N, dtype=np.float64)
    model.eval()
    for start in range(0, N - WINDOW + 1, STRIDE_INF):
        end   = start + WINDOW
        x     = torch.from_numpy(feats[:, start:end]).unsqueeze(0).to(DEVICE)
        logit = model(x).squeeze().cpu().numpy()
        prob  = 1 / (1 + np.exp(-logit))
        prob_sum[start:end] += prob
        count[start:end]    += 1
    count = np.maximum(count, 1)
    return (prob_sum / count).astype(np.float32)


def metrics(prob, mask):
    pred = (prob > THRESH).astype(bool)
    gt   = mask.astype(bool)
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    tn = (~pred & ~gt).sum()
    f1   = float(2 * tp / (2 * tp + fp + fn + 1e-8))
    prec = float(tp / (tp + fp + 1e-8))
    rec  = float(tp / (tp + fn + 1e-8))
    acc  = float((tp + tn) / (tp + fp + fn + tn + 1e-8))
    return f1, prec, rec, acc


def shade_segs(ax, binary, t, color, alpha=0.20):
    in_seg = False
    for i, v in enumerate(binary):
        if v and not in_seg:
            s = t[i]; in_seg = True
        elif not v and in_seg:
            ax.axvspan(s, t[i], color=color, alpha=alpha, lw=0)
            in_seg = False
    if in_seg:
        ax.axvspan(s, t[-1], color=color, alpha=alpha, lw=0)


def plot_sem(fname, feats, mask, prob, out_path):
    f1, prec, rec, acc = metrics(prob, mask)
    t = np.arange(len(mask)) / FS

    # --- Figură cu 2 secțiuni: semnal complet + zoom 120s ---
    ZOOM_S, ZOOM_E = 0, 120   # primele 120s pentru zoom

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(
        f'{fname}  |  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  Acc={acc:.4f}'
        f'  |  mov%={100*mask.mean():.1f}%',
        fontsize=13, fontweight='bold'
    )

    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.08,
                          left=0.06, right=0.98, top=0.91, bottom=0.06)

    pred_bin = prob > THRESH

    # ---- coloana stângă: semnal complet ----
    for row, (ylabel, signal, color, label) in enumerate([
        ('fECG ch0 (z)', feats[0], '#1f77b4', None),
        ('P(mișcare)',   prob,      'darkorange', None),
        ('GT / Predicție', None,   None, None),
    ]):
        ax = fig.add_subplot(gs[row, 0])
        shade_segs(ax, mask,     t, 'red',   alpha=0.18)
        shade_segs(ax, pred_bin, t, 'green', alpha=0.12)

        if row == 0:
            ax.plot(t, feats[0], lw=0.3, color='#1f77b4', rasterized=True)
            ax.set_ylabel('fECG ch0 (z)', fontsize=9)
            ax.set_title('Semnal complet', fontsize=10)
        elif row == 1:
            ax.plot(t, prob, lw=0.5, color='darkorange', rasterized=True)
            ax.axhline(THRESH, color='gray', lw=0.8, ls='--')
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('P(mișcare)', fontsize=9)
        else:
            ax.fill_between(t, mask.astype(float), alpha=0.55, color='red',   step='pre')
            ax.fill_between(t, pred_bin.astype(float), alpha=0.45, color='green', step='pre')
            ax.set_ylabel('Clasă (0/1)', fontsize=9)
            ax.set_xlabel('Timp (s)', fontsize=9)

        ax.set_xlim(t[0], t[-1])
        ax.tick_params(labelsize=8)
        if row < 2:
            ax.set_xticklabels([])

    # ---- coloana dreaptă: zoom 120s ----
    i0, i1 = int(ZOOM_S * FS), int(ZOOM_E * FS)
    tz = t[i0:i1]
    fz = feats[0, i0:i1]
    pz = prob[i0:i1]
    mz = mask[i0:i1]
    pbz = pred_bin[i0:i1]

    for row, _ in enumerate(range(3)):
        ax = fig.add_subplot(gs[row, 1])
        shade_segs(ax, mz,  tz, 'red',   alpha=0.22)
        shade_segs(ax, pbz, tz, 'green', alpha=0.15)

        if row == 0:
            ax.plot(tz, fz, lw=0.5, color='#1f77b4')
            ax.set_ylabel('fECG ch0 (z)', fontsize=9)
            ax.set_title(f'Zoom: primele {ZOOM_E}s', fontsize=10)
        elif row == 1:
            ax.plot(tz, pz, lw=0.7, color='darkorange')
            ax.axhline(THRESH, color='gray', lw=0.8, ls='--')
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('P(mișcare)', fontsize=9)
        else:
            ax.fill_between(tz, mz.astype(float),  alpha=0.55, color='red',   step='pre', label='GT')
            ax.fill_between(tz, pbz.astype(float), alpha=0.45, color='green', step='pre', label='Predicție')
            ax.set_ylabel('Clasă (0/1)', fontsize=9)
            ax.set_xlabel('Timp (s)', fontsize=9)
            ax.legend(fontsize=8, loc='upper right')

        ax.set_xlim(tz[0], tz[-1])
        ax.tick_params(labelsize=8)
        if row < 2:
            ax.set_xticklabels([])

    # Legendă globală
    handles = [
        mpatches.Patch(color='red',   alpha=0.5, label='GT mișcare'),
        mpatches.Patch(color='green', alpha=0.5, label='Predicție mișcare'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 0.00))

    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Salvat: {out_path}')
    return f1, prec, rec, acc


def main():
    print(f'Device: {DEVICE}')
    model = PrecisionResUNet(in_ch=IN_CH, use_transformer=True).to(DEVICE)
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.eval()
    print('Model încărcat.')

    rows = []
    for fname in SEM_FILES:
        print(f'\n--- {fname} ---')
        feats, mask, fecg = load_sem(fname)
        print(f'  Lungime: {len(mask)/FS:.0f}s  |  mov%: {100*mask.mean():.1f}%')
        prob = infer_full(model, feats)
        print('  Inferență completă.')
        out_path = os.path.join(OUT_DIR, f'binary_v1_{fname}_full.png')
        f1, prec, rec, acc = plot_sem(fname, feats, mask, prob, out_path)
        print(f'  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  Acc={acc:.4f}')
        rows.append((fname, round(100*mask.mean(), 1), f1, prec, rec, acc))

    print('\n' + '='*60)
    print(f'{"Fișier":<8} {"mov%":>5} {"F1":>7} {"Prec":>7} {"Rec":>7} {"Acc":>7}')
    for r in rows:
        print(f'{r[0]:<8} {r[1]:>4.1f}% {r[2]:>7.4f} {r[3]:>7.4f} {r[4]:>7.4f} {r[5]:>7.4f}')
    means = [sum(r[i] for r in rows)/len(rows) for i in range(2, 6)]
    print(f'{"medie":<8} {"":>5} {means[0]:>7.4f} {means[1]:>7.4f} {means[2]:>7.4f} {means[3]:>7.4f}')


if __name__ == '__main__':
    main()
