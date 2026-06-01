"""
PhaseDatasetV2 — features extinse per faza completa.

Fata de PhaseDataset (36 features):
  + RR intervals per faza: mean, std, cv, slope, min, max          -> 6 features
  + Morfologie QRS (PCA pe forma batailor): primele 3 PC per canal -> 18 features
  + Anvelopa amplitudine: std, dom_freq, slope anvelopa            -> 18 features
  Total: 36 + 6 + 18 + 18 = 78 features

QRS window: ±25 sample-uri in jurul fiecarui R-peak (50 samples total).
"""

import os
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
import torch

FS          = 1000
N_CLASSES   = 4
MIN_PHASE_S = 10
MIN_BEATS   = 8
QRS_HALF    = 25   # ±25 sample-uri in jurul R-peak


def _extract_fecg(out):
    f = out['fecg'][0][0]
    if f.dtype == object: f = f.flat[0]
    return f.astype(np.float32)

def _extract_fqrs(out):
    f = out['fqrs'][0][0]
    if f.dtype == object: f = f.flat[0]
    return f.ravel().astype(np.int32)

def _extract_cat(out):
    return out['category_mask'][0][0].ravel().astype(np.uint8)


def compute_extended_features(amps, peaks, fecg, N_sig):
    """
    amps:   (N_beats, 6) — amplitudini R-peak
    peaks:  (N_beats,)   — pozitii R-peak in semnal
    fecg:   (6, N_sig)   — semnal complet
    N_sig:  lungime semnal

    Returneaza vector de 78 features float32.
    """
    N_beats, C = amps.shape
    t = np.arange(N_beats, dtype=np.float64)
    feats = []

    # -----------------------------------------------------------------------
    # 1. Features spectrale originale (36) — identice cu PhaseDataset
    # -----------------------------------------------------------------------
    for ch in range(C):
        a    = amps[:, ch].astype(np.float64)
        a_dm = a - a.mean()
        feats.append(float(a_dm.std()))                               # std

        p     = np.polyfit(t, a_dm, 1)
        slope = p[0]
        feats.append(float(slope))                                    # slope

        pred   = slope * t + p[1]
        ss_res = ((a_dm - pred)**2).sum()
        ss_tot = ((a_dm - a_dm.mean())**2).sum()
        r2     = 1.0 - ss_res/ss_tot if ss_tot > 1e-12 else 0.0
        feats.append(float(np.clip(r2, -1.0, 1.0)))                  # R²

        F = np.abs(np.fft.rfft(a_dm))
        F[0] = 0.0
        feats.append(float(F.max()))                                  # FFT max amp
        feats.append(float(F.argmax()) / max(N_beats, 1))            # FFT dom freq

        if N_beats > 5:
            c = np.corrcoef(a_dm[:-5], a_dm[5:])[0, 1]
            feats.append(0.0 if np.isnan(c) else float(c))           # lag5 autocorr
        else:
            feats.append(0.0)

    # -----------------------------------------------------------------------
    # 2. RR intervals (6 features) — global, nu per canal
    # -----------------------------------------------------------------------
    if N_beats > 1:
        rr = np.diff(peaks.astype(np.float64))
        rr_mean  = float(rr.mean())
        rr_std   = float(rr.std())
        rr_cv    = float(rr_std / rr_mean) if rr_mean > 0 else 0.0
        rr_slope = float(np.polyfit(np.arange(len(rr)), rr, 1)[0]) if len(rr) > 1 else 0.0
        rr_min   = float(rr.min())
        rr_max   = float(rr.max())
    else:
        rr_mean = rr_std = rr_cv = rr_slope = rr_min = rr_max = 0.0
    feats += [rr_mean, rr_std, rr_cv, rr_slope, rr_min, rr_max]

    # -----------------------------------------------------------------------
    # 3. Morfologie QRS — PCA pe forma batailor, 3 PC per canal (18 features)
    # -----------------------------------------------------------------------
    valid_peaks = peaks[(peaks >= QRS_HALF) & (peaks < N_sig - QRS_HALF)]
    for ch in range(C):
        if len(valid_peaks) >= 3:
            # (N_valid, 2*QRS_HALF) — forma QRS per bataie
            morphs = np.stack([
                fecg[ch, p - QRS_HALF: p + QRS_HALF].astype(np.float64)
                for p in valid_peaks
            ])
            # zero-mean per bataie
            morphs -= morphs.mean(axis=1, keepdims=True)
            # SVD pentru PCA
            try:
                _, s, _ = np.linalg.svd(morphs, full_matrices=False)
                total = s.sum() + 1e-12
                feats.append(float(s[0] / total))   # var explicata PC1
                feats.append(float(s[1] / total) if len(s) > 1 else 0.0)
                feats.append(float(s[2] / total) if len(s) > 2 else 0.0)
            except Exception:
                feats += [0.0, 0.0, 0.0]
        else:
            feats += [0.0, 0.0, 0.0]

    # -----------------------------------------------------------------------
    # 4. Anvelopa amplitudine per canal (18 features = 3 per canal)
    # -----------------------------------------------------------------------
    for ch in range(C):
        a       = np.abs(amps[:, ch].astype(np.float64))  # anvelopa = |amplitudine|
        a_dm    = a - a.mean()
        # std anvelopa
        feats.append(float(a_dm.std()))
        # frecventa dominanta anvelopa
        if N_beats > 2:
            F = np.abs(np.fft.rfft(a_dm))
            F[0] = 0.0
            feats.append(float(F.argmax()) / max(N_beats, 1))
        else:
            feats.append(0.0)
        # slope anvelopa (creste / scade monoton?)
        if N_beats > 1:
            feats.append(float(np.polyfit(t, a_dm, 1)[0]))
        else:
            feats.append(0.0)

    return np.array(feats, dtype=np.float32)   # 78 features


def extract_extended_phases(mat_path, min_sec=MIN_PHASE_S):
    try:
        npz = os.path.splitext(mat_path)[0] + '_beat.npz'
        if os.path.exists(npz):
            data = np.load(npz)
            fecg = data['fecg']
            fqrs = data['fqrs']
            cat  = data['category_mask']
        else:
            mat  = sio.loadmat(mat_path)
            out  = mat['out']
            fecg = _extract_fecg(out)
            fqrs = _extract_fqrs(out)
            cat  = _extract_cat(out)
            del mat
    except Exception:
        return []

    N_sig       = fecg.shape[1]
    min_samples = int(min_sec * FS)
    transitions = np.where(np.diff(cat.astype(int)) != 0)[0]
    starts      = np.concatenate([[0], transitions + 1])
    ends        = np.concatenate([transitions + 1, [len(cat)]])

    phases = []
    for s, e in zip(starts, ends):
        if e - s < min_samples:
            continue
        label = int(cat[s])
        mask  = (fqrs >= s) & (fqrs < e)
        peaks = fqrs[mask]
        if len(peaks) < MIN_BEATS:
            continue
        amps  = fecg[:, peaks].T.astype(np.float64)   # (N_beats, 6)
        feats = compute_extended_features(amps, peaks, fecg, N_sig)
        phases.append((feats, label))

    return phases


class PhaseDatasetV2(Dataset):
    """
    Dataset cu 78 features per faza completa.
    Compatibil cu PhaseDataset (acelasi file_split, compute_class_weights).
    """

    N_FEATURES = 78

    def __init__(self, data_dir, min_phase_sec=MIN_PHASE_S):
        self.data_dir = data_dir
        self.files    = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith('.mat')
        )

        all_feats, all_labels, all_file_ids = [], [], []

        print(f'Loading PhaseDatasetV2 from {len(self.files)} files ...')
        for fi, fpath in enumerate(self.files):
            if (fi + 1) % 100 == 0 or (fi + 1) == len(self.files):
                print(f'  {fi+1}/{len(self.files)} ...', flush=True)
            for feats, label in extract_extended_phases(fpath, min_sec=min_phase_sec):
                all_feats.append(feats)
                all_labels.append(label)
                all_file_ids.append(fi)

        self.X        = np.stack(all_feats, axis=0)
        self.y        = np.array(all_labels, dtype=np.int64)
        self.file_ids = np.array(all_file_ids, dtype=np.int32)

        # standardizare
        self.feat_mean = self.X.mean(axis=0)
        self.feat_std  = self.X.std(axis=0)
        self.feat_std  = np.where(self.feat_std < 1e-8, 1.0, self.feat_std)
        self.X_norm    = (self.X - self.feat_mean) / self.feat_std

        counts = np.bincount(self.y, minlength=N_CLASSES)
        print(f'Total phases: {len(self.y)}  features: {self.N_FEATURES}')
        print(f'Class dist: {counts.tolist()}')

    def file_split(self, val_frac=0.15, seed=42):
        rng      = np.random.default_rng(seed)
        n_files  = len(self.files)
        n_val    = max(1, int(n_files * val_frac))
        val_set  = set(rng.permutation(n_files)[:n_val].tolist())
        train_idx = [i for i, fi in enumerate(self.file_ids) if fi not in val_set]
        val_idx   = [i for i, fi in enumerate(self.file_ids) if fi in val_set]
        return train_idx, val_idx

    def compute_class_weights(self, indices):
        labels  = self.y[indices]
        counts  = np.bincount(labels, minlength=N_CLASSES).astype(np.float64)
        counts  = np.where(counts == 0, 1, counts)
        w       = 1.0 / counts
        w       = w / w.sum() * N_CLASSES
        return torch.from_numpy(w).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.X_norm[idx]),
                torch.tensor(self.y[idx], dtype=torch.long))
