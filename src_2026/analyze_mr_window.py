"""
Analiză M_R pe fereastră lungă.

M_R_beat   = 1 - corr(QRS_i, QRS_{i+1})       (paper original, beat-to-beat)
M_R_window = 1 - corr(QRS_first, QRS_last)     (prima vs ultima bătaie din fereastră)
M_R_spread = std( corr(QRS_i, QRS_ref) )       (dispersia față de template mediu)

Comparăm distribițiile pe clasele movement / no-movement.
"""

import os, glob, random
import numpy as np
import scipy.io as sio
from scipy.signal import find_peaks

MAT_DIR  = '/shared_storage/stan.edward/fECG_mvm_DB/DB_1/Long_time_intervals'
WINDOW   = 8000   # 8 s la 1000 Hz
STRIDE   = 4000
QRS_HALF = 40     # ±40 ms template
MIN_DIST = 300    # distanță minimă între vârfuri (300 ms)
N_FILES  = 60     # câte fișiere analizăm (random sample)
SEED     = 42

random.seed(SEED)
np.random.seed(SEED)

def load_mat(fpath):
    mat = sio.loadmat(fpath)
    out = mat['out']
    fecg = out['fecg'][0][0]
    if fecg.dtype == object:
        fecg = fecg.flat[0]
    fecg = fecg.astype(np.float32)
    mask = out['movement_mask'][0][0].ravel().astype(np.uint8)
    return fecg, mask

def detect_qrs(sig):
    """Detectare vârfuri QRS pe semnalul z-normat."""
    s = (sig - sig.mean()) / (sig.std() + 1e-8)
    peaks, _ = find_peaks(s, distance=MIN_DIST, height=0.5)
    return peaks

def qrs_template(sig, peak):
    lo = peak - QRS_HALF
    hi = peak + QRS_HALF
    if lo < 0 or hi >= len(sig):
        return None
    t = sig[lo:hi].copy()
    n = np.linalg.norm(t)
    return t / (n + 1e-8)

def corr_templates(a, b):
    return float(np.dot(a, b))   # ambele sunt normalizate

def compute_mr_window(sig, peaks_in_win):
    """M_R variante per fereastră."""
    templates = []
    for p in peaks_in_win:
        t = qrs_template(sig, p)
        if t is not None:
            templates.append(t)
    if len(templates) < 2:
        return None, None, None

    # M_R_beat: medie beat-to-beat consecutive
    beat_mr = [1 - corr_templates(templates[i], templates[i+1])
               for i in range(len(templates) - 1)]
    mr_beat = float(np.mean(beat_mr))

    # M_R_window: prima vs ultima bătaie
    mr_win = 1 - corr_templates(templates[0], templates[-1])

    # M_R_spread: std al corelației față de template mediu
    ref = np.mean(templates, axis=0)
    ref /= (np.linalg.norm(ref) + 1e-8)
    corrs = [corr_templates(t, ref) for t in templates]
    mr_spread = float(np.std(corrs))

    return mr_beat, mr_win, mr_spread


def main():
    files = sorted(glob.glob(os.path.join(MAT_DIR, '*.mat')))
    sample = random.sample(files, min(N_FILES, len(files)))
    print(f'Analizăm {len(sample)} fișiere ...')

    results = {'mov': [], 'nomov': []}  # fiecare entry: (mr_beat, mr_win, mr_spread)

    for i, fpath in enumerate(sample):
        try:
            fecg, mask = load_mat(fpath)
        except Exception as e:
            print(f'  SKIP {os.path.basename(fpath)}: {e}')
            continue

        sig  = fecg[0]  # canal 0
        sig  = (sig - sig.mean()) / (sig.std() + 1e-8)
        all_peaks = detect_qrs(sig)
        N = len(sig)

        for start in range(0, N - WINDOW + 1, STRIDE):
            end   = start + WINDOW
            label = int(mask[start:end].mean() > 0.5)
            peaks_in_win = all_peaks[(all_peaks >= start) & (all_peaks < end)]

            mr_b, mr_w, mr_s = compute_mr_window(sig, peaks_in_win)
            if mr_b is None:
                continue

            key = 'mov' if label else 'nomov'
            results[key].append((mr_b, mr_w, mr_s))

        if (i + 1) % 10 == 0:
            print(f'  {i+1}/{len(sample)} fișiere procesate')

    # --- Statistici ---
    print('\n' + '='*60)
    print(f'Ferestre movement   : {len(results["mov"])}')
    print(f'Ferestre no-movement: {len(results["nomov"])}')
    print()

    for feat_idx, feat_name in enumerate(['M_R_beat (paper)', 'M_R_window (first vs last)', 'M_R_spread (std vs ref)']):
        mov_vals   = [r[feat_idx] for r in results['mov']]
        nomov_vals = [r[feat_idx] for r in results['nomov']]
        print(f'{feat_name}')
        print(f'  movement   : mean={np.mean(mov_vals):.5f}  std={np.std(mov_vals):.5f}  median={np.median(mov_vals):.5f}')
        print(f'  no-movement: mean={np.mean(nomov_vals):.5f}  std={np.std(nomov_vals):.5f}  median={np.median(nomov_vals):.5f}')

        # effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(mov_vals)**2 + np.std(nomov_vals)**2) / 2)
        d = (np.mean(mov_vals) - np.mean(nomov_vals)) / (pooled_std + 1e-8)
        print(f'  Cohen d    : {d:.4f}')
        print()

if __name__ == '__main__':
    main()
