"""
Beat-level dataset for fetal movement classification.

R-peak AMPLITUDE per channel per heartbeat → direct signature of fetal movement.

Key design decisions (v7+):
  - window_sec=5, stride_sec=2.5  — faze dureaza 12-83s (median 24s);
    ferestrele scurte au 85% puritate vs 31% la 30s
  - purity_threshold=0.99         — filtreaza ferestrele cu label noise
    (ferestre care span tranzitii intre clase)
  - normalize_std=False           — scadem doar media (zero-mean), NU impartim
    la std; std-ul rezidual ESTE semnalul: stationar≈0, miscare>0

  Stationary  → beat_amps ≈ 0 (fara variatie)
  Linear      → trend monoton clar
  Helical     → oscilatie sinusoidala
  Screw       → pattern complex (mix linear + rotatie)

Cache: <name>_beat.npz stores (fecg, fqrs, category_mask) per file.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio


FS              = 1000
WINDOW_SEC      = 5         # faze dureaza 12-83s; 5s ferestre → 85% pure
STRIDE_SEC      = 2.5       # 50% overlap
MAX_BEATS       = 20        # la 160bpm, 5s ≈ 13 beats; 20 e marja
N_CLASSES       = 4
N_SAMPLES_TOTAL = 600_000
PURITY_THRESH   = 0.99      # min fractie din fereastra care trebuie sa fie o singura clasa


# ---------------------------------------------------------------------------
def _extract_fecg(out):
    field = out['fecg'][0][0]
    if field.dtype == object:
        return field.flat[0].astype(np.float32)
    return field.astype(np.float32)


def _extract_fqrs(out):
    field = out['fqrs'][0][0]
    if field.dtype == object:
        inner = field.flat[0]
        return inner.ravel().astype(np.int32)
    return field.ravel().astype(np.int32)


def _load_beat_signals(mat_path):
    """Load fecg, fqrs, category_mask. Uses <name>_beat.npz cache."""
    npz = os.path.splitext(mat_path)[0] + '_beat.npz'
    if os.path.exists(npz):
        data = np.load(npz)
        return data['fecg'], data['fqrs'], data['category_mask']

    mat = sio.loadmat(mat_path)
    out = mat['out']
    fecg     = _extract_fecg(out)
    fqrs     = _extract_fqrs(out)
    cat_mask = out['category_mask'][0][0].ravel().astype(np.uint8)
    del mat

    try:
        np.savez_compressed(npz, fecg=fecg, fqrs=fqrs, category_mask=cat_mask)
    except Exception as e:
        print(f'[WARN] Could not write cache {npz}: {e}')
    return fecg, fqrs, cat_mask


# ---------------------------------------------------------------------------
class BeatDataset(Dataset):
    """
    Returns (beat_amps, mask, label) for movement classification.

    beat_amps : float32 tensor (MAX_BEATS, 6) — zero-mean R-peak amplitudes
    mask      : bool tensor (MAX_BEATS,) — True where padded
    label     : int64 scalar — majority class (0-3)
    """

    def __init__(self, data_dir,
                 window_sec=WINDOW_SEC,
                 stride_sec=STRIDE_SEC,
                 fs=FS,
                 max_beats=MAX_BEATS,
                 purity_threshold=PURITY_THRESH,
                 normalize_std=False):
        self.data_dir        = data_dir
        self.window          = int(window_sec * fs)
        self.stride          = int(stride_sec * fs)
        self.fs              = fs
        self.max_beats       = max_beats
        self.purity_threshold = purity_threshold
        self.normalize_std   = normalize_std

        self.files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith('.mat')
        )

        n = N_SAMPLES_TOTAL
        all_windows = []
        for file_idx in range(len(self.files)):
            for start in range(0, n - self.window + 1, self.stride):
                all_windows.append((file_idx, start))

        self._cache: dict    = {}
        self._bad_files: set = set()
        self.norm_stats      = None

        # Filtreaza ferestrele impure daca e setat threshold
        if purity_threshold is not None and purity_threshold > 0:
            print(f'Filtering windows by purity >= {purity_threshold:.0%} ...')
            self.windows = self._filter_pure(all_windows, purity_threshold)
            print(f'  {len(all_windows)} total → {len(self.windows)} pure windows '
                  f'({100*len(self.windows)/max(1,len(all_windows)):.0f}%)')
        else:
            self.windows = all_windows

        self._wins_per_file = max(1, len(self.windows) // max(1, len(self.files)))

    # ------------------------------------------------------------------
    def _filter_pure(self, all_windows, threshold):
        """Pastreaza doar ferestrele unde >= threshold din samples e o singura clasa."""
        clean = []
        prev_fi = -1
        cat = None
        for fi, start in all_windows:
            if fi != prev_fi:
                try:
                    _, _, cat = self._get_signals(fi)
                    prev_fi = fi
                except Exception:
                    cat = None
                    continue
            if cat is None:
                continue
            seg    = cat[start:start + self.window]
            counts = np.bincount(seg.astype(np.int32), minlength=N_CLASSES)
            if counts.max() / len(seg) >= threshold:
                clean.append((fi, start))
        return clean

    # ------------------------------------------------------------------
    def file_split(self, val_frac=0.15, seed=42):
        rng       = np.random.default_rng(seed)
        n_files   = len(self.files)
        n_val     = max(1, int(n_files * val_frac))
        order     = rng.permutation(n_files)
        val_files = set(order[:n_val].tolist())

        train_idx, val_idx = [], []
        for i, (fi, _) in enumerate(self.windows):
            (val_idx if fi in val_files else train_idx).append(i)
        return train_idx, val_idx

    # ------------------------------------------------------------------
    def compute_class_weights(self, indices=None, cache_path=None):
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                counts = np.array(json.load(f), dtype=np.float64)
            print(f'  [weights] loaded from cache: {counts.tolist()}')
        else:
            idx_set      = set(indices) if indices is not None else set(range(len(self)))
            file_windows = {}
            for wi, (fi, start) in enumerate(self.windows):
                if wi in idx_set:
                    file_windows.setdefault(fi, []).append((wi, start))

            counts  = np.zeros(N_CLASSES, dtype=np.float64)
            n_files = len(file_windows)
            for done, (fi, wins) in enumerate(file_windows.items()):
                if (done + 1) % 100 == 0 or (done + 1) == n_files:
                    print(f'  [weights] {done+1}/{n_files} files ...', flush=True)
                try:
                    _, _, cat = self._get_signals(fi)
                except Exception:
                    continue
                for _, start in wins:
                    seg = cat[start:start + self.window]
                    counts[np.bincount(seg.astype(np.int32),
                                       minlength=N_CLASSES).argmax()] += 1

            if cache_path:
                os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
                with open(cache_path, 'w') as f:
                    json.dump(counts.tolist(), f)

        counts  = np.where(counts == 0, 1, counts)
        weights = 1.0 / counts
        weights = weights / weights.sum() * N_CLASSES
        return torch.from_numpy(weights).float()

    # ------------------------------------------------------------------
    def _get_signals(self, file_idx):
        if file_idx in self._bad_files:
            raise ValueError(f'Bad file: {self.files[file_idx]}')
        if file_idx not in self._cache:
            try:
                fecg, fqrs, cat = _load_beat_signals(self.files[file_idx])
                self._cache[file_idx] = (fecg, fqrs, cat)
            except Exception as e:
                print(f'\n[WARN] Skipping {os.path.basename(self.files[file_idx])}: {e}')
                self._bad_files.add(file_idx)
                raise
        return self._cache[file_idx]

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.windows)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        for attempt in range(20):
            try:
                candidate = (idx + attempt * self._wins_per_file) % len(self.windows)
                file_idx, start = self.windows[candidate]
                fecg, fqrs, cat = self._get_signals(file_idx)
                break
            except Exception:
                continue
        else:
            raise RuntimeError(f'Could not load valid data after 20 attempts (idx={idx})')

        end = start + self.window

        mask_peaks = (fqrs >= start) & (fqrs < end)
        peak_locs  = fqrs[mask_peaks] - start

        if len(peak_locs) > 0:
            amps = fecg[:, peak_locs].T.copy()   # (N_beats, 6)
        else:
            amps = np.zeros((0, 6), dtype=np.float32)

        # Zero-mean per canal; impartim la std DOAR daca normalize_std=True
        if amps.shape[0] > 1:
            mu  = amps.mean(axis=0, keepdims=True)
            amps = amps - mu
            if self.normalize_std:
                std = amps.std(axis=0, keepdims=True)
                std = np.where(std < 1e-8, 1.0, std)
                amps = amps / std

        n_beats   = min(len(peak_locs), self.max_beats)
        beat_amps = np.zeros((self.max_beats, 6), dtype=np.float32)
        pad_mask  = np.ones(self.max_beats, dtype=bool)
        if n_beats > 0:
            beat_amps[:n_beats] = amps[:n_beats]
            pad_mask[:n_beats]  = False

        cat_win = cat[start:end]
        label   = int(np.bincount(cat_win.astype(np.int32),
                                  minlength=N_CLASSES).argmax())

        x    = torch.from_numpy(beat_amps).float()
        mask = torch.from_numpy(pad_mask)
        y    = torch.tensor(label, dtype=torch.long)
        return x, mask, y


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/movement_ecg'
    dset = BeatDataset(data_dir)
    print(f'Total pure windows: {len(dset)}')
    train_idx, val_idx = dset.file_split()
    print(f'Train: {len(train_idx)}  Val: {len(val_idx)}')
    x, mask, y = dset[0]
    n_real = (~mask).sum().item()
    print(f'  x: {x.shape}  real beats: {n_real}  label: {y.item()}')
    print(f'  amp range: [{x[~mask].min():.4f}, {x[~mask].max():.4f}]')
