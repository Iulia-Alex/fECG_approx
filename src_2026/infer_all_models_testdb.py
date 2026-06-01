"""
Inferenta completa pe Test_DB (Sem1, Sem2, Sem3) pentru toate modelele v1-v15.
Calculeaza metrici: PRD, SNR_dB, QRS_F1, QRS_Prec, QRS_Rec.
Genereaza plots individuale + INFERENCE_RESULTS.md in root.

Rulare: conda run -n ecg python3 src_2026/infer_all_models_testdb.py
(din directorul root al proiectului)
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
import scipy.signal
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Network architectures
from complex_network import ComplexUNet as ComplexUNetOrig
from complex_network_v7  import ComplexUNetV7
from complex_network_v9  import ComplexUNetV9
from complex_network_v10 import ComplexUNetV10
from complex_network_v11 import ComplexUNetV11
from complex_network_v12 import ComplexUNetV12
from complex_network_v13 import ComplexUNetV13
from complex_network_v15 import ComplexUNet as ComplexUNetV15

# Dataset utils — 1 kHz pipeline
from movement_dataset import (
    _stft_multichannel, _to_resized_complex_tensor, _extract_fecg,
    NFFT as NFFT1k, HOP_LENGTH as HOP1k, WIN_LENGTH as WIN1k,
    TARGET_SIZE_F, TARGET_SIZE_T, FS as FS1k,
)
# Dataset utils — 500 Hz pipeline (v15)
from movement_dataset_v15 import (
    NFFT as NFFT5, HOP_LENGTH as HOP5, WIN_LENGTH as WIN5,
    FS as FS5, WINDOW as WINDOW5,
)

# ---------------------------------------------------------------------------
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
PLOTS_DIR  = os.path.join(ROOT_DIR, 'plots_2026', 'testdb')
TEST_DIR   = '/shared_storage/stan.edward/fECG_mvm_DB/Test_DB'
TARGET_FILES = ['Sem1.mat', 'Sem2.mat', 'Sem3.mat']
CHANNELS   = [0, 1, 2]

WINDOW_1k   = 4 * FS1k          # 4000 samples
START_1k    = 10 * FS1k         # start at 10 s
DIMENSION   = TARGET_SIZE_F * TARGET_SIZE_T   # 128×400 = 51200
DIM_V15     = 128 * 128                        # 16384
IN_CHANNELS = 6
ORIG_F_1k   = NFFT1k // 2 + 1  # 129
ORIG_T_1k   = 1 + WINDOW_1k // HOP1k  # 401

os.makedirs(PLOTS_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

_hann_v15 = torch.hann_window(WIN5)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_CONFIGS = [
    dict(name='v1_resume', label='v1_resume',  color='royalblue',
         pth='movement_CUNet_128x400_composed.pth',
         best_ep=458, best_val=0.013701, loss='SignalMSE',
         arch='orig', pipeline='1k'),
    dict(name='v5',        label='v5',          color='darkorange',
         pth='movement_CUNet_128x400_ampw_scratch.pth',
         best_ep=200, best_val=0.022473, loss='SignalMSE+3×AmpW',
         arch='orig', pipeline='1k'),
    dict(name='v6',        label='v6',          color='goldenrod',
         pth='movement_CUNet_128x400_v6_baseline.pth',
         best_ep=68,  best_val=0.052941, loss='MSE+AmpW+Baseline',
         arch='orig', pipeline='1k'),
    dict(name='v7',        label='v7',          color='sienna',
         pth='movement_CUNet_v7_paper_direct.pth',
         best_ep=30,  best_val=0.081819, loss='SignalMAE(L1)',
         arch='v7',   pipeline='1k'),
    dict(name='v8',        label='v8',          color='peru',
         pth='movement_CUNet_v8_mse.pth',
         best_ep=41,  best_val=0.046987, loss='SignalMSE',
         arch='v7',   pipeline='1k'),
    dict(name='v9',        label='v9',          color='teal',
         pth='movement_CUNet_v9_mask.pth',
         best_ep=138, best_val=0.925510, loss='Sig+Cpl',
         arch='v9',   pipeline='1k'),
    dict(name='v10',       label='v10',         color='mediumorchid',
         pth='movement_CUNet_v10_fqrs.pth',
         best_ep=35,  best_val=2.378023, loss='Sig+Cpl+3×Peak',
         arch='v10',  pipeline='1k'),
    dict(name='v11',       label='v11',         color='crimson',
         pth='movement_CUNet_v11_direct_peak.pth',
         best_ep=73,  best_val=0.432913, loss='Sig+3×Peak(±30ms)',
         arch='v11',  pipeline='1k'),
    dict(name='v12',       label='v12',         color='chocolate',
         pth='movement_CUNet_v12_gainmask.pth',
         best_ep=49,  best_val=0.220880, loss='Sig+3×QRS(±100ms)',
         arch='v12',  pipeline='1k'),
    dict(name='v13',       label='v13',         color='steelblue',
         pth='movement_CUNet_v13_instanorm.pth',
         best_ep=39,  best_val=0.060508, loss='SignalMSE',
         arch='v13',  pipeline='1k'),
    dict(name='v15',       label='v15',         color='mediumpurple',
         pth='movement_CUNet_v15_paper.pth',
         best_ep=44,  best_val=0.032720, loss='SignalMSE',
         arch='v15',  pipeline='500'),
]

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

def load_model(cfg):
    pth = os.path.join(MODELS_DIR, cfg['pth'])
    arch = cfg['arch']
    if arch == 'orig':
        m = ComplexUNetOrig(DIMENSION, in_channels=IN_CHANNELS)
    elif arch == 'v7':
        m = ComplexUNetV7(DIMENSION, in_channels=IN_CHANNELS)
    elif arch == 'v9':
        m = ComplexUNetV9(DIMENSION, in_channels=IN_CHANNELS)
    elif arch == 'v10':
        m = ComplexUNetV10(DIMENSION, in_channels=IN_CHANNELS)
    elif arch == 'v11':
        m = ComplexUNetV11(DIMENSION, in_channels=IN_CHANNELS)
    elif arch == 'v12':
        m = ComplexUNetV12(DIMENSION, in_channels=IN_CHANNELS)
    elif arch == 'v13':
        m = ComplexUNetV13(DIMENSION, in_channels=IN_CHANNELS)
    elif arch == 'v15':
        m = ComplexUNetV15(DIM_V15, in_channels=IN_CHANNELS)
    else:
        raise ValueError(f'Unknown arch: {arch}')
    m.load_state_dict(torch.load(pth, map_location=device))
    return m.to(device).eval()

# ---------------------------------------------------------------------------
# Preprocessing & postprocessing
# ---------------------------------------------------------------------------

def preprocess_1k(mix_norm):
    spec = _stft_multichannel(mix_norm, NFFT1k, HOP1k, WIN1k)
    x = _to_resized_complex_tensor(spec, TARGET_SIZE_F, TARGET_SIZE_T)
    return x.unsqueeze(0).to(device)


def postprocess_1k(tensor):
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


def preprocess_500(mixture_1k_full):
    mix_500 = scipy.signal.decimate(mixture_1k_full, 2, axis=1).astype(np.float32)
    start_500 = START_1k // 2
    mix_win = mix_500[:, start_500:start_500 + WINDOW5]
    stds = mix_win.std(axis=1, keepdims=True)
    stds = np.where(stds < 1e-8, 1.0, stds)
    mix_win = mix_win / stds
    specs = []
    for ch in range(mix_win.shape[0]):
        S = librosa.stft(mix_win[ch], n_fft=NFFT5, hop_length=HOP5,
                         win_length=WIN5, center=True)
        specs.append(S)
    spec = np.stack(specs, axis=0)[:, :-1, :]  # (6,128,128)
    x = torch.from_numpy(spec.real) + 1j * torch.from_numpy(spec.imag)
    return x.unsqueeze(0).to(device), stds


def postprocess_500(tensor):
    t = tensor.squeeze(0).cpu()
    zeros = torch.zeros(t.shape[0], 1, t.shape[2], dtype=t.dtype)
    t_full = torch.cat([t, zeros], dim=1)
    out = []
    for ch in range(t_full.shape[0]):
        sig = torch.istft(t_full[ch], n_fft=NFFT5, hop_length=HOP5,
                          win_length=WIN5, window=_hann_v15,
                          center=True, length=WINDOW5).numpy()
        out.append(sig)
    return np.stack(out, axis=0)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_prd(pred, gt):
    return 100.0 * np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-12)


def compute_snr_db(pred, gt):
    noise = pred - gt
    return 20.0 * np.log10(np.linalg.norm(gt) / (np.linalg.norm(noise) + 1e-12))


def detect_qrs_peaks(signal, fs):
    """
    Detect R-peaks using prominence-based approach.
    Works for both positive and negative QRS complexes.
    """
    min_dist = int(0.25 * fs)    # 250 ms = max HR ~240 bpm
    sig_range = np.max(signal) - np.min(signal)
    prom_thresh = 0.20 * sig_range   # prominence ≥ 20% of signal range

    peaks_pos, props_pos = scipy.signal.find_peaks(
        signal, distance=min_dist, prominence=prom_thresh)
    peaks_neg, props_neg = scipy.signal.find_peaks(
        -signal, distance=min_dist, prominence=prom_thresh)

    # Pick whichever polarity gives more peaks (or higher total prominence)
    prom_pos = props_pos['prominences'].sum() if len(peaks_pos) > 0 else 0.0
    prom_neg = props_neg['prominences'].sum() if len(peaks_neg) > 0 else 0.0
    peaks = peaks_pos if prom_pos >= prom_neg else peaks_neg
    return peaks


def compute_qrs_f1(pred, gt, fs, tol_ms=100, amp_ratio_min=0.20):
    """
    QRS F1 on a single channel.
    Strategy:
      1. Detect GT peaks reliably.
      2. For each GT peak, search for a candidate peak in pred within ±tol_ms.
         A candidate is valid only if its amplitude ≥ amp_ratio_min × |GT peak|.
      3. Any pred peak not matched to a GT peak → FP.
    amp_ratio_min: minimum amplitude ratio pred_peak / gt_peak for a TP.
    """
    tol = int(tol_ms / 1000.0 * fs)
    gt_peaks = detect_qrs_peaks(gt, fs)
    if len(gt_peaks) == 0:
        return dict(f1=np.nan, prec=np.nan, rec=np.nan,
                    tp=0, fp=0, fn=0, n_gt=0, n_pred=0)

    # Determine polarity used for GT
    pk_pos, _ = scipy.signal.find_peaks(gt, distance=int(0.25*fs),
                                         prominence=0.20*(gt.max()-gt.min()))
    pk_neg, _ = scipy.signal.find_peaks(-gt, distance=int(0.25*fs),
                                         prominence=0.20*(gt.max()-gt.min()))
    use_neg = len(pk_neg) > len(pk_pos)
    gt_sign = -1.0 if use_neg else 1.0
    signed_pred = gt_sign * pred

    n_gt = len(gt_peaks)
    matched_gt = set()
    tp = 0

    for gp in gt_peaks:
        lo = max(0, gp - tol)
        hi = min(len(pred) - 1, gp + tol)
        window = signed_pred[lo:hi+1]
        if len(window) == 0:
            continue
        best_idx = np.argmax(window) + lo
        gt_amp = abs(float((gt_sign * gt)[gp]))
        pred_amp = float(signed_pred[best_idx])
        if pred_amp >= amp_ratio_min * gt_amp and gp not in matched_gt:
            tp += 1
            matched_gt.add(gp)

    # FP: pred peaks not matched (detect in pred and count unmatched)
    pred_peaks = detect_qrs_peaks(pred, fs)
    matched_pred = set()
    for gp in gt_peaks:
        lo = max(0, gp - tol)
        hi = min(len(pred) - 1, gp + tol)
        window = signed_pred[lo:hi+1]
        if len(window) == 0:
            continue
        best_local = np.argmax(window) + lo
        # find nearest pred peak
        if len(pred_peaks) > 0:
            dists = np.abs(pred_peaks - best_local)
            nearest = pred_peaks[np.argmin(dists)]
            if dists.min() <= tol:
                matched_pred.add(nearest)
    fp = len(pred_peaks) - len(matched_pred)
    fp = max(0, fp)

    fn = n_gt - tp
    prec = tp / (tp + fp + 1e-12) if (tp + fp) > 0 else 0.0
    rec  = tp / (n_gt + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12) if (prec + rec) > 0 else 0.0
    return dict(f1=f1, prec=prec, rec=rec, tp=tp, fp=fp, fn=fn,
                n_gt=n_gt, n_pred=len(pred_peaks))


def compute_qrs_f1_annot(pred, gt, gt_peaks_annot, fs, tol_ms=80, gt_ratio_min=0.30):
    """
    QRS F1 using ground-truth FQRS annotation.
    TP: annotated beat gp for which max|pred| in ±tol_ms ≥ gt_ratio_min × |gt[gp]|.
        This checks that the prediction has a peak of at least 30% the GT amplitude.
    FP: peaks detected in pred NOT near any GT annotation.
    Discriminates between:
      - Good reconstruction (pred peak ≈ GT amplitude) → F1 high
      - Smooth/zero output (pred ≈ 0 at beat location) → F1 low
    """
    tol = int(tol_ms / 1000.0 * fs)
    n_gt = len(gt_peaks_annot)
    if n_gt == 0:
        return dict(f1=np.nan, prec=np.nan, rec=np.nan,
                    tp=0, fp=0, fn=0, n_gt=0, n_pred=0)

    tp = 0
    for gp in gt_peaks_annot:
        if gp >= len(gt) or gp >= len(pred):
            continue
        lo = max(0, gp - tol)
        hi = min(len(pred) - 1, gp + tol)
        local_max_pred = float(np.max(np.abs(pred[lo:hi+1])))
        # Reference: peak GT amplitude in the same window (handles onset vs R-peak offset)
        gt_amp = float(np.max(np.abs(gt[lo:hi+1])))
        if gt_amp < 1e-8:
            continue
        if local_max_pred >= gt_ratio_min * gt_amp:
            tp += 1

    # FP: pred peaks not within tol of any GT annotation
    pred_peaks = detect_qrs_peaks(pred, fs)
    fp = 0
    for pp in pred_peaks:
        dists = np.abs(gt_peaks_annot - pp)
        if len(dists) == 0 or dists.min() > tol:
            fp += 1

    fn = n_gt - tp
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (n_gt + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12) if (prec + rec) > 0 else 0.0
    return dict(f1=f1, prec=prec, rec=rec, tp=tp, fp=fp, fn=fn,
                n_gt=n_gt, n_pred=len(pred_peaks))


def metrics_for_pred(pred, gt, fs, gt_peaks_annot):
    """Average metrics across channels. gt_peaks_annot: sample indices from fqrs annotation."""
    prd_list, snr_list, f1_list, prec_list, rec_list = [], [], [], [], []
    for ch in CHANNELS:
        prd_list.append(compute_prd(pred[ch], gt[ch]))
        snr_list.append(compute_snr_db(pred[ch], gt[ch]))
        qrs = compute_qrs_f1_annot(pred[ch], gt[ch], gt_peaks_annot, fs)
        f1_list.append(qrs['f1'])
        prec_list.append(qrs['prec'])
        rec_list.append(qrs['rec'])
    return dict(
        prd_mean=float(np.mean(prd_list)),
        snr_mean=float(np.mean(snr_list)),
        f1_mean=float(np.nanmean(f1_list)),
        prec_mean=float(np.nanmean(prec_list)),
        rec_mean=float(np.nanmean(rec_list)),
        prd_ch=prd_list, snr_ch=snr_list,
        f1_ch=f1_list, prec_ch=prec_list, rec_ch=rec_list,
    )

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

print('Loading models...')
loaded_models = {}
for cfg in MODEL_CONFIGS:
    pth = os.path.join(MODELS_DIR, cfg['pth'])
    if not os.path.exists(pth):
        print(f'  SKIP {cfg["name"]}: {pth} not found')
        continue
    try:
        loaded_models[cfg['name']] = load_model(cfg)
        print(f'  OK   {cfg["name"]:12s} ({cfg["arch"]}, {cfg["pipeline"]}Hz pipeline)')
    except Exception as e:
        print(f'  ERR  {cfg["name"]}: {e}')

all_results = {}   # {model_name: {sem_name: metrics_dict}}
all_plots   = {}   # {model_name: {sem_name: plot_path}}

for fname in TARGET_FILES:
    fpath = os.path.join(TEST_DIR, fname)
    if not os.path.exists(fpath):
        print(f'\nSKIP: {fpath}')
        continue
    stem = os.path.splitext(fname)[0]
    print(f'\n=== {fname} ===')

    mat     = sio.loadmat(fpath)
    mixture = mat['out']['mixture'][0][0].astype(np.float32)
    fecg    = _extract_fecg(mat['out'])
    # Ground-truth fQRS annotation (sample indices at 1000 Hz)
    _fq = mat['out']['fqrs'][0][0]
    fqrs_all = np.sort(np.concatenate(
        [_fq[0, i].flatten() for i in range(_fq.shape[1])]
    )).astype(int)
    del mat

    if fecg.shape[0] < mixture.shape[0]:
        fecg = np.repeat(fecg, mixture.shape[0] // fecg.shape[0], axis=0)

    # --- 1 kHz window ---
    mix_win_1k  = mixture[:, START_1k:START_1k + WINDOW_1k].copy()
    fecg_win_1k = fecg[:,   START_1k:START_1k + WINDOW_1k].copy()
    stds_1k = mix_win_1k.std(axis=1, keepdims=True)
    stds_1k = np.where(stds_1k < 1e-8, 1.0, stds_1k)
    mix_norm_1k  = mix_win_1k  / stds_1k
    gt_1k        = fecg_win_1k / stds_1k

    # --- 500 Hz window ---
    mix_full_norm = mixture / stds_1k
    fecg_full_norm = fecg  / stds_1k
    fecg_500_full = scipy.signal.decimate(fecg_full_norm, 2, axis=1).astype(np.float32)
    start_500 = START_1k // 2
    gt_500 = fecg_500_full[:, start_500:start_500 + WINDOW5]

    x_1k  = preprocess_1k(mix_norm_1k)
    x_500, _ = preprocess_500(mix_full_norm)

    # FQRS peaks within the 4-second window (relative to window start, at 1kHz)
    fqrs_in_win_1k = fqrs_all[(fqrs_all >= START_1k) & (fqrs_all < START_1k + WINDOW_1k)] - START_1k
    # Same peaks resampled to 500 Hz indices
    fqrs_in_win_500 = (fqrs_in_win_1k // 2).astype(int)
    # Keep unique and within bounds
    fqrs_in_win_500 = np.unique(fqrs_in_win_500[fqrs_in_win_500 < WINDOW5])
    print(f'  fQRS beats in window: {len(fqrs_in_win_1k)} (1kHz), {len(fqrs_in_win_500)} (500Hz)')

    t_1k  = np.linspace(START_1k / FS1k, (START_1k + WINDOW_1k) / FS1k, WINDOW_1k)
    t_500 = np.linspace(START_1k / FS1k, (START_1k + WINDOW_1k) / FS1k, WINDOW5)

    for cfg in MODEL_CONFIGS:
        mname = cfg['name']
        if mname not in loaded_models:
            continue
        model = loaded_models[mname]
        pipeline = cfg['pipeline']

        with torch.no_grad():
            if pipeline == '1k':
                pred = postprocess_1k(model(x_1k).cpu())
                gt   = gt_1k
                t_ax = t_1k
                fs   = FS1k
            else:
                pred = postprocess_500(model(x_500).cpu())
                gt   = gt_500
                t_ax = t_500
                fs   = FS5

        # Metrics
        gt_peaks_annot = fqrs_in_win_1k if pipeline == '1k' else fqrs_in_win_500
        m = metrics_for_pred(pred, gt, fs, gt_peaks_annot)
        if mname not in all_results:
            all_results[mname] = {}
        all_results[mname][stem] = m

        print(f'  {mname:12s} | PRD={m["prd_mean"]:6.2f}%  SNR={m["snr_mean"]:6.2f}dB  '
              f'F1={m["f1_mean"]:.3f}  P={m["prec_mean"]:.3f}  R={m["rec_mean"]:.3f}')

        # Plot: 2 rows (GT / pred), 3 channels
        fig, axes = plt.subplots(2, len(CHANNELS),
                                 figsize=(5 * len(CHANNELS), 4), sharex=True)
        for col, ch in enumerate(CHANNELS):
            ax_gt = axes[0][col]
            ax_pr = axes[1][col]
            ax_gt.plot(t_ax, gt[ch],   color='seagreen',      lw=0.8)
            ax_pr.plot(t_ax, pred[ch], color=cfg['color'],     lw=0.9)
            ax_gt.plot(t_ax, gt[ch],   color='seagreen',  lw=0.4, alpha=0.3)
            ax_pr.plot(t_ax, gt[ch],   color='seagreen',  lw=0.4, alpha=0.3)
            for ax in (ax_gt, ax_pr):
                ax.grid(True, alpha=0.25, lw=0.4)
            if col == 0:
                ax_gt.set_ylabel('GT',              fontsize=8, color='seagreen')
                ax_pr.set_ylabel(f'{mname}\nep{cfg["best_ep"]}', fontsize=7,
                                 color=cfg['color'], fontweight='bold')
            if col == 0 and axes[0][0] is ax_gt:
                pass
            ax_gt.set_title(f'Ch {ch+1}', fontsize=9)
            if True:
                ax_pr.set_xlabel('Time (s)', fontsize=8)

        prd_str = '  |  '.join([f'Ch{ch+1}: PRD={m["prd_ch"][i]:.1f}%  '
                                  f'SNR={m["snr_ch"][i]:.1f}dB  '
                                  f'F1={m["f1_ch"][i]:.3f}'
                                  for i, ch in enumerate(CHANNELS)])
        fig.suptitle(
            f'{stem}  —  {mname}  (ep{cfg["best_ep"]}, val={cfg["best_val"]:.4f}, '
            f'loss: {cfg["loss"]})\n{prd_str}',
            fontsize=7.5,
        )
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f'{stem}_{mname}.png')
        plt.savefig(plot_path, dpi=140, bbox_inches='tight')
        plt.close()

        if mname not in all_plots:
            all_plots[mname] = {}
        all_plots[mname][stem] = plot_path
        print(f'    -> {plot_path}')

# ---------------------------------------------------------------------------
# Save metrics JSON
# ---------------------------------------------------------------------------
metrics_path = os.path.join(ROOT_DIR, 'inference_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'\nMetrics saved -> {metrics_path}')

# ---------------------------------------------------------------------------
# Generate INFERENCE_RESULTS.md
# ---------------------------------------------------------------------------

# Compute per-model average across all 3 Sem files
def avg_across_sems(mname):
    vals = {'prd':[], 'snr':[], 'f1':[], 'prec':[], 'rec':[]}
    for sem in ['Sem1','Sem2','Sem3']:
        if sem in all_results.get(mname, {}):
            m = all_results[mname][sem]
            vals['prd'].append(m['prd_mean'])
            vals['snr'].append(m['snr_mean'])
            vals['f1'].append(m['f1_mean'])
            vals['prec'].append(m['prec_mean'])
            vals['rec'].append(m['rec_mean'])
    return {k: float(np.mean(v)) if v else float('nan') for k, v in vals.items()}

PLOTS_REL = 'plots_2026/testdb'  # relative path from root for .md

md_lines = [
    '# Inference Results — Test_DB (Sem1, Sem2, Sem3)',
    '',
    f'Generat automat. Toate modelele antrenate v1→v15, fereastră 10–14 s.',
    f'Metrici mediate pe 3 canale (Ch1, Ch2, Ch3). F1-score QRS cu toleranță ±100 ms.',
    '',
    '---',
    '',
    '## Tabel sumar (mediat pe Sem1+Sem2+Sem3)',
    '',
    '| Model | Best Ep | Val Loss | Loss fn | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |',
    '|-------|---------|----------|---------|-----------|-----------|---------|----------|---------|',
]

for cfg in MODEL_CONFIGS:
    mname = cfg['name']
    if mname not in all_results:
        continue
    avg = avg_across_sems(mname)
    md_lines.append(
        f'| **{mname}** | ep{cfg["best_ep"]} | {cfg["best_val"]:.4f} | {cfg["loss"]} '
        f'| {avg["prd"]:.2f} | {avg["snr"]:.2f} | {avg["f1"]:.3f} '
        f'| {avg["prec"]:.3f} | {avg["rec"]:.3f} |'
    )

md_lines += ['', '---', '']

for cfg in MODEL_CONFIGS:
    mname = cfg['name']
    if mname not in all_results:
        continue
    md_lines += [
        f'## {mname}',
        f'',
        f'**Arhitectură:** {cfg["arch"]}  |  '
        f'**Pipeline:** {cfg["pipeline"]} Hz  |  '
        f'**Loss:** {cfg["loss"]}  |  '
        f'**Best epoch:** {cfg["best_ep"]}  |  '
        f'**Best val loss:** {cfg["best_val"]:.6f}',
        '',
    ]

    for sem in ['Sem1', 'Sem2', 'Sem3']:
        if sem not in all_results.get(mname, {}):
            continue
        m = all_results[mname][sem]
        plot_rel = f'{PLOTS_REL}/{sem}_{mname}.png'

        ch_rows = []
        for i, ch in enumerate(CHANNELS):
            ch_rows.append(
                f'| Ch{ch+1} | {m["prd_ch"][i]:.2f} | {m["snr_ch"][i]:.2f} '
                f'| {m["f1_ch"][i]:.3f} | {m["prec_ch"][i]:.3f} | {m["rec_ch"][i]:.3f} |'
            )

        md_lines += [
            f'### {sem}',
            '',
            f'![{mname} {sem}]({plot_rel})',
            '',
            f'| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |',
            f'|-------|-----------|-----------|---------|----------|---------|',
        ] + ch_rows + [
            f'| **avg** | **{m["prd_mean"]:.2f}** | **{m["snr_mean"]:.2f}** '
            f'| **{m["f1_mean"]:.3f}** | **{m["prec_mean"]:.3f}** | **{m["rec_mean"]:.3f}** |',
            '',
        ]

    md_lines.append('---')
    md_lines.append('')

md_path = os.path.join(ROOT_DIR, 'INFERENCE_RESULTS.md')
with open(md_path, 'w') as f:
    f.write('\n'.join(md_lines))
print(f'Markdown saved -> {md_path}')
print('\nDone.')
