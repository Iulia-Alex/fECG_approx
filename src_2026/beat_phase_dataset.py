"""
BeatPhaseDataset — faze complete cu beat amplitudini paddate la lungime fixa.

In loc de 36 features handcrafted (PhaseDataset), returnam secventa raw
de beat amplitudini per faza completa, paddata la MAX_BEATS.

Shape per sample: (6, MAX_BEATS) — 6 canale, MAX_BEATS timesteps
Label: clasa fazei (0-3)

Avantaj vs PhaseDataset (36 features):
  - ResNet1D poate invata pattern-uri temporale pe care RF nu le vede
  - Mai multa informatie bruta, mai putine presupuneri despre features
  - 200 beats = 200 timesteps → ResNet downsamplare OK (200→100→50→25→12→GAP)

Padding: zeros la sfarsit, mask pentru a nu penaliza padding in loss.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio

FS          = 1000
N_CLASSES   = 4
MIN_PHASE_S = 10
MIN_BEATS   = 8
MAX_BEATS   = 200   # paddam/truncam la 200 beats (~2 min la 160bpm = suficient)


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


def extract_beat_phases(mat_path, min_sec=MIN_PHASE_S, max_beats=MAX_BEATS):
    """
    Returneaza lista de (amps_padded, mask, label) per faza.
    amps_padded: (6, max_beats) float32
    mask:        (max_beats,)  bool — True unde e beat real
    label:       int
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

        # (N_beats, 6) -> (6, N_beats)
        amps = fecg[:, peaks]   # (6, N_beats)

        # zero-mean per canal
        amps = amps - amps.mean(axis=1, keepdims=True)

        # normalizeaza per canal cu std
        stds = amps.std(axis=1, keepdims=True)
        stds = np.where(stds < 1e-8, 1.0, stds)
        amps = amps / stds

        n = amps.shape[1]
        # pad sau truncheaza la max_beats
        amps_pad = np.zeros((6, max_beats), dtype=np.float32)
        beat_mask = np.zeros(max_beats, dtype=bool)
        n_use = min(n, max_beats)
        amps_pad[:, :n_use] = amps[:, :n_use]
        beat_mask[:n_use]   = True

        phases.append((amps_pad, beat_mask, label))

    return phases


class BeatPhaseDataset(Dataset):
    """
    Incarca toate fazele din toate fisierele in memorie.
    Returneaza (amps_padded, mask, label).
    amps_padded: (6, MAX_BEATS) float32
    mask:        (MAX_BEATS,)   bool
    label:       long
    """

    def __init__(self, data_dir, min_phase_sec=MIN_PHASE_S, max_beats=MAX_BEATS):
        self.data_dir  = data_dir
        self.max_beats = max_beats
        self.files     = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith('.mat')
        )

        all_amps, all_masks, all_labels, all_file_ids = [], [], [], []

        print(f'Loading BeatPhaseDataset from {len(self.files)} files ...')
        for fi, fpath in enumerate(self.files):
            if (fi + 1) % 100 == 0 or (fi + 1) == len(self.files):
                print(f'  {fi+1}/{len(self.files)} ...', flush=True)
            phases = extract_beat_phases(fpath, min_sec=min_phase_sec,
                                         max_beats=max_beats)
            for amps_pad, mask, label in phases:
                all_amps.append(amps_pad)
                all_masks.append(mask)
                all_labels.append(label)
                all_file_ids.append(fi)

        self.X         = np.stack(all_amps,  axis=0)   # (N, 6, MAX_BEATS)
        self.M         = np.stack(all_masks, axis=0)   # (N, MAX_BEATS)
        self.y         = np.array(all_labels, dtype=np.int64)
        self.file_ids  = np.array(all_file_ids, dtype=np.int32)

        counts = np.bincount(self.y, minlength=N_CLASSES)
        print(f'Total phases: {len(self.y)}  shape: {self.X.shape}')
        print(f'Class dist: {counts.tolist()}')

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

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])          # (6, MAX_BEATS)
        m = torch.from_numpy(self.M[idx])          # (MAX_BEATS,) bool
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, m, y
