"""
PhaseDataset — extrage features pe faze COMPLETE din category_mask.

De ce faze complete si nu ferestre fixe:
  - Fazele dureaza 12-83s (median 24s)
  - O fereastra de 15s vede doar un fragment: helical in 15s arata ca linear
  - Pe faza completa, pattern-ul e complet vizibil si clasabil

Features per-canal (6 canale x 6 = 36 total):
  std         — magnitudine variatie (stationar≈0, miscare>0)
  slope       — trend linear (linear/screw au slope mare)
  linear_r2   — cat de liniar e trendrul (linear→1, helical→0)
  fft_max_amp — amplitudine FFT dominanta (helical/screw au peak clar)
  fft_dom_freq— frecventa dominanta normalizata (separare helical vs screw)
  lag5_autocorr—autocorelatia la lag 5 beats (detectie oscilatie)

Incarca totul in memorie la init (12K faze, fast).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio

FS          = 1000
N_CLASSES   = 4
MIN_PHASE_S = 10        # ignora fazele sub 10s
MIN_BEATS   = 8         # ignora fazele cu prea putine beats


# ---------------------------------------------------------------------------
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


def compute_phase_features(amps):
    """
    amps: (N_beats, 6) — beat amplitudini pe intreaga faza

    Returns: (36,) float32 feature vector
    """
    N, C = amps.shape
    t    = np.arange(N, dtype=np.float64)
    feats = []

    for ch in range(C):
        a    = amps[:, ch].astype(np.float64)
        a_dm = a - a.mean()

        # 1. std
        feats.append(a_dm.std())

        # 2. slope (beats per beat)
        p    = np.polyfit(t, a_dm, 1)
        slope = p[0]
        feats.append(slope)

        # 3. linear R²
        pred   = slope * t + p[1]
        ss_res = ((a_dm - pred) ** 2).sum()
        ss_tot = ((a_dm - a_dm.mean()) ** 2).sum()
        r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        feats.append(np.clip(r2, -1.0, 1.0))

        # 4. FFT dominant amplitude (excludem DC)
        F      = np.abs(np.fft.rfft(a_dm))
        F[0]   = 0.0
        feats.append(float(F.max()))

        # 5. FFT dominant freq (normalizat 0-1)
        feats.append(float(F.argmax()) / max(N, 1))

        # 6. Lag-5 autocorrelatie
        if N > 5:
            c = np.corrcoef(a_dm[:-5], a_dm[5:])[0, 1]
            feats.append(0.0 if np.isnan(c) else float(c))
        else:
            feats.append(0.0)

    return np.array(feats, dtype=np.float32)


def extract_phases_from_file(mat_path, min_sec=MIN_PHASE_S):
    """
    Returns list of (features_36, label) per faza din fisier.
    Foloseste cache _beat.npz daca exista (creat de BeatDataset).
    """
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
        amps = fecg[:, peaks].T   # (N_beats, 6)
        feats = compute_phase_features(amps)
        phases.append((feats, label))

    return phases


# ---------------------------------------------------------------------------
class PhaseDataset(Dataset):
    """
    Incarca toate fazele din toate fisierele in memorie.
    Returneaza (features_36, label).
    """

    def __init__(self, data_dir, min_phase_sec=MIN_PHASE_S):
        self.data_dir = data_dir
        self.files    = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith('.mat')
        )

        all_feats, all_labels, all_file_ids = [], [], []

        print(f'Loading phases from {len(self.files)} files ...')
        for fi, fpath in enumerate(self.files):
            if (fi + 1) % 100 == 0 or (fi + 1) == len(self.files):
                print(f'  {fi+1}/{len(self.files)} ...', flush=True)
            phases = extract_phases_from_file(fpath, min_sec=min_phase_sec)
            for feats, label in phases:
                all_feats.append(feats)
                all_labels.append(label)
                all_file_ids.append(fi)

        self.X         = np.stack(all_feats,  axis=0)   # (N, 36)
        self.y         = np.array(all_labels, dtype=np.int64)
        self.file_ids  = np.array(all_file_ids, dtype=np.int32)
        n_feat         = self.X.shape[1]

        # Standardizeaza features (media si std din tot dataset-ul)
        self.feat_mean = self.X.mean(0)
        self.feat_std  = self.X.std(0)
        self.feat_std  = np.where(self.feat_std < 1e-8, 1.0, self.feat_std)
        self.X_norm    = ((self.X - self.feat_mean) / self.feat_std).astype(np.float32)

        counts = np.bincount(self.y, minlength=N_CLASSES)
        print(f'Total phases: {len(self.y)}  features: {n_feat}')
        print(f'Class dist: {counts.tolist()}')

    # ------------------------------------------------------------------
    def file_split(self, val_frac=0.15, seed=42):
        rng       = np.random.default_rng(seed)
        n_files   = len(self.files)
        n_val     = max(1, int(n_files * val_frac))
        val_files = set(rng.permutation(n_files)[:n_val].tolist())

        train_idx = [i for i, fi in enumerate(self.file_ids) if fi not in val_files]
        val_idx   = [i for i, fi in enumerate(self.file_ids) if fi in val_files]
        return train_idx, val_idx

    def compute_class_weights(self, indices):
        labels  = self.y[indices]
        counts  = np.bincount(labels, minlength=N_CLASSES).astype(np.float64)
        counts  = np.where(counts == 0, 1, counts)
        weights = 1.0 / counts
        weights = weights / weights.sum() * N_CLASSES
        return torch.from_numpy(weights).float()

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X_norm[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/movement_ecg'
    ds = PhaseDataset(data_dir)
    tr, va = ds.file_split()
    print(f'Train: {len(tr)}  Val: {len(va)}')
    x, y = ds[0]
    print(f'  x: {x.shape}  y: {y.item()}')
