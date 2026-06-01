"""
EnvelopeDataset — anvelopa Hilbert a semnalului fECG per faza completa.

In loc de amplitudini R-peak (depind de detectia fqrs) sau statistici,
extragem anvelopa amplitudine a semnalului brut per faza si o downsample
la lungime fixa N_OUT=200 puncte.

Avantaje:
- Forma reala a modulatiei vizibila in plot
- Nu depinde de detectia fqrs
- Fara padding / masca — input consistent (6, 200)
- Linear: anvelopa monotona; Helical: anvelopa oscilatorie

Shape per sample: (6, N_OUT) float32
"""

import os
import numpy as np
import scipy.io as sio
from scipy.signal import hilbert, resample
import torch
from torch.utils.data import Dataset

FS          = 1000
N_CLASSES   = 4
MIN_PHASE_S = 10
N_OUT       = 200   # lungime fixa dupa downsample


def _extract_fecg(out):
    f = out['fecg'][0][0]
    if f.dtype == object: f = f.flat[0]
    return f.astype(np.float32)

def _extract_cat(out):
    return out['category_mask'][0][0].ravel().astype(np.uint8)


def extract_envelope_phases(mat_path, min_sec=MIN_PHASE_S, n_out=N_OUT):
    try:
        mat  = sio.loadmat(mat_path)
        out  = mat['out']
        fecg = _extract_fecg(out)
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

        env = np.zeros((6, n_out), dtype=np.float32)
        for ch in range(6):
            seg = fecg[ch, s:e].astype(np.float64)
            # anvelopa Hilbert
            amp = np.abs(hilbert(seg))
            # downsample la n_out puncte
            amp_ds = resample(amp, n_out).astype(np.float32)
            # zero-mean, unit-std
            mu  = amp_ds.mean()
            std = amp_ds.std()
            if std < 1e-8:
                std = 1.0
            env[ch] = (amp_ds - mu) / std

        phases.append((env, label))

    return phases


class EnvelopeDataset(Dataset):
    """
    Incarca anvelopele Hilbert per faza din toate fisierele.
    Returneaza (env, label).
    env:   (6, N_OUT) float32
    label: long
    """

    def __init__(self, data_dir, min_phase_sec=MIN_PHASE_S, n_out=N_OUT):
        self.data_dir = data_dir
        self.n_out    = n_out
        self.files    = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith('.mat')
        )

        all_envs, all_labels, all_file_ids = [], [], []

        print(f'Loading EnvelopeDataset from {len(self.files)} files ...')
        for fi, fpath in enumerate(self.files):
            if (fi + 1) % 100 == 0 or (fi + 1) == len(self.files):
                print(f'  {fi+1}/{len(self.files)} ...', flush=True)
            for env, label in extract_envelope_phases(fpath, min_sec=min_phase_sec,
                                                       n_out=n_out):
                all_envs.append(env)
                all_labels.append(label)
                all_file_ids.append(fi)

        self.X        = np.stack(all_envs,  axis=0)   # (N, 6, N_OUT)
        self.y        = np.array(all_labels, dtype=np.int64)
        self.file_ids = np.array(all_file_ids, dtype=np.int32)

        counts = np.bincount(self.y, minlength=N_CLASSES)
        mem_mb = self.X.nbytes / 1e6
        print(f'Total phases: {len(self.y)}  shape: {self.X.shape}  mem: {mem_mb:.0f}MB')
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
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
