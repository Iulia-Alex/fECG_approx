"""
ClfDataset v15 — features spectrale + faza inter-canal + spectral peakedness.

Motivatie: RF 92.2% blocat pe Linear<->Helical (128 erori din ~180 total).
fft_freq captureaza CA oscileaza, dar nu CUM oscileaza intre canale.

Linear:  fatui se deplaseaza intr-o directie -> canale IN FAZA sau ANTI-FAZA
Helical: rotatia fetala -> canale cu OFFSET DE FAZA sistematic (val rotitor)

Features noi fata de 36 originale:
  Phase diff inter-canal la freq dominanta: 15 perechi  -> 15 features
  Coerenta spectrala inter-canal:           15 perechi  -> 15 features
  Spectral peakedness per canal:             6 canale   ->  6 features

Total: 36 + 15 + 15 + 6 = 72 features
"""

import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

FS          = 1000
N_CLASSES   = 4
MIN_PHASE_S = 10
MIN_BEATS   = 8


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


def compute_v15_features(amps):
    """
    amps: (N_beats, 6) amplitudini R-peak

    Returneaza vector de 72 features float32.
    """
    N, C = amps.shape
    t    = np.arange(N, dtype=np.float64)
    feats = []

    # -----------------------------------------------------------------------
    # 1. Features spectrale originale (36)
    # -----------------------------------------------------------------------
    ffts = []   # pastram FFT-urile pentru features de faza
    for ch in range(C):
        a    = amps[:, ch].astype(np.float64)
        a_dm = a - a.mean()

        feats.append(float(a_dm.std()))

        p     = np.polyfit(t, a_dm, 1)
        slope = p[0]
        feats.append(float(slope))

        pred   = slope * t + p[1]
        ss_res = ((a_dm - pred)**2).sum()
        ss_tot = ((a_dm - a_dm.mean())**2).sum()
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        feats.append(float(np.clip(r2, -1.0, 1.0)))

        F = np.fft.rfft(a_dm)
        Fabs = np.abs(F)
        Fabs[0] = 0.0
        feats.append(float(Fabs.max()))
        feats.append(float(Fabs.argmax()) / max(N, 1))

        if N > 5:
            c = np.corrcoef(a_dm[:-5], a_dm[5:])[0, 1]
            feats.append(0.0 if np.isnan(c) else float(c))
        else:
            feats.append(0.0)

        ffts.append(F)   # (N//2+1,) complex

    # -----------------------------------------------------------------------
    # 2. Phase difference inter-canal la frecventa dominanta (15 perechi)
    #
    # Pentru fiecare pereche (i,j): gasim frecventa dominanta comuna
    # (media argmax-urilor), extragem faza la acea frecventa, calculam diff.
    # Linear:  phase_diff ~ 0 sau ~ pi (in faza / anti-faza)
    # Helical: phase_diff sistematic, diferit de 0/pi
    # -----------------------------------------------------------------------
    dom_freqs = []
    for ch in range(C):
        Fabs = np.abs(ffts[ch]).copy()
        Fabs[0] = 0.0
        dom_freqs.append(int(Fabs.argmax()))

    for i in range(C):
        for j in range(i + 1, C):
            if N > 4:
                # frecventa dominanta comuna = cea mai puternica din cele doua
                Fabs_i = np.abs(ffts[i]).copy(); Fabs_i[0] = 0.0
                Fabs_j = np.abs(ffts[j]).copy(); Fabs_j[0] = 0.0
                # alege freq cu amplitudine maxima combinata
                k = int(np.argmax(Fabs_i + Fabs_j))
                if k == 0:
                    feats.append(0.0)
                else:
                    phase_diff = float(np.angle(ffts[i][k] * np.conj(ffts[j][k])))
                    # normalizeaza la [-1, 1]
                    feats.append(float(phase_diff / np.pi))
            else:
                feats.append(0.0)

    # -----------------------------------------------------------------------
    # 3. Coerenta spectrala inter-canal la frecventa dominanta (15 perechi)
    #
    # Coherence = |Sxy|^2 / (Sxx * Syy) la frecventa dominanta combinata
    # Linear:  coerenta mare (canale oscileaza sincron)
    # Helical: coerenta mai mica sau la frecventa diferita
    # -----------------------------------------------------------------------
    for i in range(C):
        for j in range(i + 1, C):
            if N > 4:
                Fabs_i = np.abs(ffts[i]).copy(); Fabs_i[0] = 0.0
                Fabs_j = np.abs(ffts[j]).copy(); Fabs_j[0] = 0.0
                k      = int(np.argmax(Fabs_i + Fabs_j))
                if k == 0:
                    feats.append(0.0)
                else:
                    sxy = abs(ffts[i][k] * np.conj(ffts[j][k]))
                    sxx = abs(ffts[i][k])**2
                    syy = abs(ffts[j][k])**2
                    coh = float(sxy**2 / (sxx * syy + 1e-12))
                    feats.append(float(np.clip(coh, 0.0, 1.0)))
            else:
                feats.append(0.0)

    # -----------------------------------------------------------------------
    # 4. Spectral peakedness per canal (6 features)
    #
    # peak / mean — cat de "ascutit" e peak-ul FFT
    # Helical: peak ascutit, energy concentrata la o singura frecventa
    # Linear:  energie distribuita, peak mai plat
    # -----------------------------------------------------------------------
    for ch in range(C):
        Fabs = np.abs(ffts[ch]).copy()
        Fabs[0] = 0.0
        mean_f = Fabs.mean()
        if mean_f > 1e-10:
            peakedness = float(np.clip(Fabs.max() / mean_f, 0.0, 50.0)) / 50.0
        else:
            peakedness = 0.0
        feats.append(peakedness)

    return np.array(feats, dtype=np.float32)   # 72 features


def extract_v15_phases(mat_path, min_sec=MIN_PHASE_S):
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
        amps  = fecg[:, peaks].T.astype(np.float64)
        feats = compute_v15_features(amps)
        phases.append((feats, label))

    return phases


class ClfDatasetV15(Dataset):
    N_FEATURES = 72

    def __init__(self, data_dir, min_phase_sec=MIN_PHASE_S):
        self.data_dir = data_dir
        self.files    = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith('.mat')
        )

        all_feats, all_labels, all_file_ids = [], [], []

        print(f'Loading ClfDatasetV15 from {len(self.files)} files ...')
        for fi, fpath in enumerate(self.files):
            if (fi + 1) % 100 == 0 or (fi + 1) == len(self.files):
                print(f'  {fi+1}/{len(self.files)} ...', flush=True)
            for feats, label in extract_v15_phases(fpath, min_sec=min_phase_sec):
                all_feats.append(feats)
                all_labels.append(label)
                all_file_ids.append(fi)

        self.X        = np.stack(all_feats, axis=0)
        self.y        = np.array(all_labels, dtype=np.int64)
        self.file_ids = np.array(all_file_ids, dtype=np.int32)

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
        labels = self.y[indices]
        counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float64)
        counts = np.where(counts == 0, 1, counts)
        w      = 1.0 / counts
        w      = w / w.sum() * N_CLASSES
        return torch.from_numpy(w).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.X_norm[idx]),
                torch.tensor(self.y[idx], dtype=torch.long))
