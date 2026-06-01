"""
infer_clf_testdb.py

Movement classification on Test_DB (Sem1-Sem11) WITHOUT ground-truth phase
boundaries.  The only GT used at inference time is fqrs (R-peak positions) —
which in a real system would come from a QRS detector.

Pipeline
--------
1. Train RF (500 trees, 36 spectral features) on PhaseDataset training split —
   same split as clf_v9/v14 (seed=42, val_frac=0.15).

2. Sliding-window inference (no category_mask):
   - Window: WINDOW_BEATS beats, stride: STRIDE_BEATS beats
   - Per window: compute 36 features, normalize with training stats, get
     RF class-probability vector (4,)
   - Smooth: boxcar average over K_SMOOTH consecutive prob vectors → argmax

3. Baseline check on val files (sliding window vs phase-based 93.07%).

4. Plot per file (PNG):
   - Panel 1: fECG channel 1 waveform + sliding-window prediction background
   - Panel 2: Ground Truth (category_mask) — for comparison only
   - Panel 3: Predicted sliding-window (smoothed)
   - Accuracy vs GT reported in title.
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.io as sio
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from phase_dataset import PhaseDataset, compute_phase_features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR      = '../data/movement_ecg'
TEST_DIR      = '/shared_storage/stan.edward/fECG_mvm_DB/Test_DB'
OUT_DIR       = '../plots_2026/clf_testdb'

FS            = 1000
WINDOW_BEATS  = 30    # sliding window width in beats (~20-25s at fetal HR 80bpm)
STRIDE_BEATS  = 5     # step in beats (~3-4s per prediction)
K_SMOOTH      = 7     # boxcar average over this many consecutive prob vectors
VAL_FRAC      = 0.15
SEED          = 42
N_TREES       = 500
N_CLASSES     = 4

CLASS_NAMES = ['Stationary', 'Linear', 'Helical', 'Screw']
COLORS_HEX  = ['#5b9bd5', '#ed7d31', '#70ad47', '#ffc000']
COLORS_RGB  = np.array([
    [91,  155, 213],
    [237, 125,  49],
    [112, 173,  71],
    [255, 192,   0],
], dtype=np.uint8)

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def extract_fecg(out):
    f = out['fecg'][0][0]
    if f.dtype == object:
        f = f.flat[0]
    return f.astype(np.float32)

def extract_fqrs(out):
    f = out['fqrs'][0][0]
    if f.dtype == object:
        f = f.flat[0]
    return f.ravel().astype(np.int32)

def extract_cat(out):
    return out['category_mask'][0][0].ravel().astype(np.uint8)


# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------
def infer_sliding(fecg, fqrs, feat_mean, feat_std, rf,
                  window_beats=WINDOW_BEATS, stride_beats=STRIDE_BEATS):
    """
    Returns
    -------
    probs      : (M, 4)  float — raw RF class probabilities per window
    pred_times : (M,)    float — center time (s) of each window
    gt_labels  : (M,)    int   — majority GT label in window (needs cat arg)
    """
    N = len(fqrs)
    probs      = []
    pred_times = []

    for start in range(0, N - window_beats + 1, stride_beats):
        end   = start + window_beats
        peaks = fqrs[start:end]
        amps  = fecg[:, peaks].T          # (window_beats, 6)

        feats      = compute_phase_features(amps)
        feats_norm = (feats - feat_mean) / feat_std

        p = rf.predict_proba(feats_norm.reshape(1, -1))[0]  # (4,)
        probs.append(p)

        center_sample = int(peaks[window_beats // 2])
        pred_times.append(center_sample / FS)

    return np.array(probs, dtype=np.float32), np.array(pred_times, dtype=np.float32)


def smooth_and_predict(probs, k=K_SMOOTH):
    """Boxcar average over k consecutive probability vectors, then argmax."""
    # uniform_filter1d along axis 0 (time), separately per class
    smoothed = uniform_filter1d(probs, size=k, axis=0, mode='nearest')
    return smoothed.argmax(axis=1).astype(np.int32), smoothed


def gt_labels_per_window(cat, fqrs, window_beats=WINDOW_BEATS,
                         stride_beats=STRIDE_BEATS):
    """Majority-class label in each sliding window (uses category_mask)."""
    N = len(fqrs)
    labels = []
    for start in range(0, N - window_beats + 1, stride_beats):
        end    = start + window_beats
        s_smp  = int(fqrs[start])
        e_smp  = int(fqrs[end - 1])
        seg    = cat[s_smp:e_smp + 1]
        label  = int(np.bincount(seg.astype(np.int32),
                                 minlength=N_CLASSES).argmax())
        labels.append(label)
    return np.array(labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_label_image(labels, times, duration, n_px=1800):
    """Convert sparse (times, labels) to dense RGB image (1, n_px, 3)."""
    t_dense   = np.linspace(0, duration, n_px)
    idx       = np.searchsorted(times, t_dense, side='right')
    idx       = np.clip(idx, 0, len(labels) - 1)
    dense_lab = labels[idx]
    rgb       = COLORS_RGB[dense_lab]          # (n_px, 3)
    return rgb[np.newaxis, :, :]               # (1, n_px, 3)


def make_cat_image(cat, duration, n_px=1800):
    """Convert dense category_mask to RGB image (1, n_px, 3)."""
    t_dense  = np.linspace(0, duration, n_px)
    smp_idx  = (t_dense * FS).astype(np.int64).clip(0, len(cat) - 1)
    rgb      = COLORS_RGB[cat[smp_idx]]
    return rgb[np.newaxis, :, :]


def plot_result(fecg, fqrs, cat, pred_labels, pred_times,
                fname, acc, val_ref_acc, out_path, duration=600.0):

    t      = np.arange(fecg.shape[1]) / FS
    ch1    = fecg[0]
    ch1_n  = (ch1 - ch1.mean()) / (ch1.std() + 1e-8)

    fig, axes = plt.subplots(3, 1, figsize=(18, 7),
                             gridspec_kw={'height_ratios': [4, 0.7, 0.7]},
                             sharex=True)

    # --- Panel 1: waveform + prediction background ---
    ax0 = axes[0]
    bg  = make_label_image(pred_labels, pred_times, duration)
    ax0.imshow(bg, extent=[0, duration, ch1_n.min(), ch1_n.max()],
               aspect='auto', interpolation='nearest', alpha=0.30, zorder=0)
    ax0.plot(t, ch1_n, color='k', lw=0.35, alpha=0.80, zorder=1)
    ax0.set_ylabel('fECG ch 1 (norm.)', fontsize=9)
    ax0.set_xlim(0, duration)
    title = (f'Sliding-window clf  |  {fname}\n'
             f'Acc vs GT: {acc:.1f}%   '
             f'(val phase-based: 93.07%  |  val sliding-window: {val_ref_acc:.1f}%)')
    ax0.set_title(title, fontsize=10)

    # --- Panel 2: Ground Truth ---
    ax1 = axes[1]
    gt_img = make_cat_image(cat, duration)
    ax1.imshow(gt_img, extent=[0, duration, 0, 1],
               aspect='auto', interpolation='nearest')
    ax1.set_yticks([0.5])
    ax1.set_yticklabels(['GT'], fontsize=8)
    ax1.set_ylim(0, 1)

    # --- Panel 3: Predicted ---
    ax2 = axes[2]
    pred_img = make_label_image(pred_labels, pred_times, duration)
    ax2.imshow(pred_img, extent=[0, duration, 0, 1],
               aspect='auto', interpolation='nearest')
    ax2.set_yticks([0.5])
    ax2.set_yticklabels(['Pred'], fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Time (s)', fontsize=9)

    # Legend
    patches = [mpatches.Patch(color=np.array(COLORS_RGB[c])/255,
                               label=CLASS_NAMES[c])
               for c in range(N_CLASSES)]
    fig.legend(handles=patches, loc='upper right', fontsize=8,
               ncol=4, bbox_to_anchor=(0.99, 0.99))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → {out_path}')


# ---------------------------------------------------------------------------
# Step 1 — Train RF
# ---------------------------------------------------------------------------
print('=' * 65)
print('Step 1: Loading PhaseDataset and training RF ...')
t0 = time.time()

ds = PhaseDataset(DATA_DIR)
train_idx, val_idx = ds.file_split(val_frac=VAL_FRAC, seed=SEED)

feat_mean = ds.feat_mean
feat_std  = ds.feat_std

X_tr = ds.X_norm[train_idx]
y_tr = ds.y[train_idx]
X_va = ds.X_norm[val_idx]
y_va = ds.y[val_idx]

rf = RandomForestClassifier(n_estimators=N_TREES, class_weight='balanced',
                             n_jobs=-1, random_state=SEED)
rf.fit(X_tr, y_tr)

phase_val_acc = accuracy_score(y_va, rf.predict(X_va)) * 100
print(f'RF phase-based val acc : {phase_val_acc:.2f}%  (trained in {time.time()-t0:.1f}s)')


# ---------------------------------------------------------------------------
# Step 2 — Sliding-window baseline on val files
# ---------------------------------------------------------------------------
print('\nStep 2: Sliding-window baseline on validation files ...')

val_file_ids  = sorted(set(ds.file_ids[val_idx].tolist()))
val_file_paths = [ds.files[fi] for fi in val_file_ids]

all_gt_sw, all_pred_sw = [], []

for fpath in val_file_paths:
    try:
        mat = sio.loadmat(fpath)
        out = mat['out']
        fecg_v = extract_fecg(out)
        fqrs_v = extract_fqrs(out)
        cat_v  = extract_cat(out)
        del mat
    except Exception as e:
        print(f'  SKIP {os.path.basename(fpath)}: {e}')
        continue

    probs_v, times_v = infer_sliding(fecg_v, fqrs_v, feat_mean, feat_std, rf)
    if len(probs_v) == 0:
        continue
    preds_v, _ = smooth_and_predict(probs_v)
    gt_v       = gt_labels_per_window(cat_v, fqrs_v)
    min_len    = min(len(preds_v), len(gt_v))
    all_gt_sw.extend(gt_v[:min_len].tolist())
    all_pred_sw.extend(preds_v[:min_len].tolist())

sw_val_acc = accuracy_score(all_gt_sw, all_pred_sw) * 100
print(f'RF sliding-window val acc: {sw_val_acc:.2f}%  '
      f'(Δ = {sw_val_acc - phase_val_acc:.1f}% vs phase-based)')
print(f'Confusion matrix (val, sliding-window):')
cm_val = confusion_matrix(all_gt_sw, all_pred_sw)
header = f'{"":14}' + ''.join(f'{n:>12}' for n in CLASS_NAMES)
print(f'  {header}')
for i, row in enumerate(cm_val):
    print(f'  {CLASS_NAMES[i]:14}' + ''.join(f'{v:12d}' for v in row))


# ---------------------------------------------------------------------------
# Step 3 — Inference on Test_DB
# ---------------------------------------------------------------------------
print('\nStep 3: Inference on Test_DB ...')

test_files = sorted(f for f in os.listdir(TEST_DIR) if f.endswith('.mat'))
print(f'Found {len(test_files)} files: {test_files}')

for fname in test_files:
    fpath = os.path.join(TEST_DIR, fname)
    stem  = os.path.splitext(fname)[0]
    print(f'\n  Processing {fname} ...')

    try:
        mat = sio.loadmat(fpath)
        out = mat['out']
        fecg_t = extract_fecg(out)
        fqrs_t = extract_fqrs(out)
        cat_t  = extract_cat(out)
        del mat
    except Exception as e:
        print(f'  ERROR: {e}')
        continue

    duration = fecg_t.shape[1] / FS

    probs_t, times_t = infer_sliding(fecg_t, fqrs_t, feat_mean, feat_std, rf)
    if len(probs_t) == 0:
        print(f'  SKIP — no windows extracted')
        continue

    pred_t, _ = smooth_and_predict(probs_t)
    gt_t      = gt_labels_per_window(cat_t, fqrs_t)
    min_len   = min(len(pred_t), len(gt_t))
    acc_t     = accuracy_score(gt_t[:min_len], pred_t[:min_len]) * 100

    cm_t = confusion_matrix(gt_t[:min_len], pred_t[:min_len],
                             labels=list(range(N_CLASSES)))
    print(f'  Acc: {acc_t:.1f}%')
    print(f'  CM: {cm_t.tolist()}')

    out_path = os.path.join(OUT_DIR, f'{stem}_clf_sliding.png')
    plot_result(fecg_t, fqrs_t, cat_t, pred_t, times_t,
                fname, acc_t, sw_val_acc, out_path, duration)

print('\nDone.')
