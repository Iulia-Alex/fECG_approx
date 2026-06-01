"""
QRSPhaseDataset — morfologie completa QRS per bataie, per faza.

In loc de amplitudinea la R-peak (scalara), extragem fereastra completa
QRS: ±QRS_HALF sample-uri in jurul fiecarui R-peak.

Shape per sample: (N_beats, 6, 2*QRS_HALF) paddat la MAX_BEATS
  -> (MAX_BEATS, 6, QRS_WIN) pentru CNN per beat

Label: clasa fazei (0-3)
Mask:  (MAX_BEATS,) bool — True = beat real

Motivatie: morfologia QRS contine informatia despre distanta electrod-fat,
axele electrice, care se schimba cu miscarea. Un beat izolat are mai multa
informatie decat amplitudinea sa maxima.
"""

import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

FS        = 1000
N_CLASSES = 4
MIN_PHASE_S = 10
MIN_BEATS   = 8
MAX_BEATS   = 150    # 150 beats per faza e suficient (la 160bpm = ~56s)
QRS_HALF    = 25     # ±25 sample = 50ms window in jurul R-peak
QRS_WIN     = 2 * QRS_HALF   # 50 samples


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


def extract_qrs_phases(mat_path, min_sec=MIN_PHASE_S,
                        max_beats=MAX_BEATS, qrs_half=QRS_HALF):
    """
    Returneaza lista de (qrs_pad, mask, label) per faza.
    qrs_pad: (max_beats, 6, 2*qrs_half) float32
    mask:    (max_beats,) bool
    label:   int
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

    N_sig       = fecg.shape[1]
    qrs_win     = 2 * qrs_half
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
        # filtrare peaks prea aproape de margine
        peaks = peaks[(peaks >= qrs_half) & (peaks < N_sig - qrs_half)]
        if len(peaks) < MIN_BEATS:
            continue

        qrs_pad  = np.zeros((max_beats, 6, qrs_win), dtype=np.float32)
        beat_mask = np.zeros(max_beats, dtype=bool)
        n_use    = min(len(peaks), max_beats)

        for bi, p in enumerate(peaks[:n_use]):
            seg = fecg[:, p - qrs_half: p + qrs_half].copy()   # (6, qrs_win)
            # zero-mean per canal
            seg = seg - seg.mean(axis=1, keepdims=True)
            # normalizeaza per canal
            stds = seg.std(axis=1, keepdims=True)
            stds = np.where(stds < 1e-8, 1.0, stds)
            seg  = seg / stds
            qrs_pad[bi]  = seg.T if False else seg   # (6, qrs_win)
            beat_mask[bi] = True

        phases.append((qrs_pad, beat_mask, label))

    return phases


class QRSPhaseDataset(Dataset):
    """
    Dataset cu morfologie QRS completa per bataie per faza.
    Returneaza (qrs, mask, label).
    qrs:   (MAX_BEATS, 6, QRS_WIN) float32
    mask:  (MAX_BEATS,) bool
    label: long
    """

    def __init__(self, data_dir, min_phase_sec=MIN_PHASE_S,
                 max_beats=MAX_BEATS, qrs_half=QRS_HALF):
        self.data_dir  = data_dir
        self.max_beats = max_beats
        self.qrs_win   = 2 * qrs_half
        self.files     = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith('.mat')
        )

        all_qrs, all_masks, all_labels, all_file_ids = [], [], [], []

        print(f'Loading QRSPhaseDataset from {len(self.files)} files ...')
        for fi, fpath in enumerate(self.files):
            if (fi + 1) % 100 == 0 or (fi + 1) == len(self.files):
                print(f'  {fi+1}/{len(self.files)} ...', flush=True)
            for qrs_pad, beat_mask, label in extract_qrs_phases(
                    fpath, min_sec=min_phase_sec,
                    max_beats=max_beats, qrs_half=qrs_half):
                all_qrs.append(qrs_pad)
                all_masks.append(beat_mask)
                all_labels.append(label)
                all_file_ids.append(fi)

        self.X        = np.stack(all_qrs,   axis=0)   # (N, MAX_BEATS, 6, QRS_WIN)
        self.M        = np.stack(all_masks, axis=0)   # (N, MAX_BEATS)
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
        x = torch.from_numpy(self.X[idx])   # (MAX_BEATS, 6, QRS_WIN)
        m = torch.from_numpy(self.M[idx])   # (MAX_BEATS,)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, m, y
