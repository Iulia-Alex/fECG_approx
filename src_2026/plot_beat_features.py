"""
Vizualizare beat-level amplitude features pentru clasificarea mișcării fetale.

Scop: verifică dacă pattern-urile de amplitudine sunt vizual distincte
      între clase (staționar/linear/helical/screw) și dacă se generalizează
      între fișiere diferite.

Generează:
  plots/beat_features_per_class.png   — amplitudini absolute + diff, per clasă
  plots/beat_features_cross_file.png  — aceeași clasă din fișiere diferite
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from beat_dataset import BeatDataset, N_CLASSES

DATA_DIR  = '/shared_storage/iulia.orvas/paper/fECG_approx/data/movement_ecg'
OUT_DIR   = '/shared_storage/iulia.orvas/paper/fECG_approx/plots'
N_EXAMPLES = 4   # ferestre per clasă
SEED       = 42

CLASS_NAMES = ['Stationary (0)', 'Linear (1)', 'Helical (2)', 'Screw (3)']
CH_COLORS   = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628']

os.makedirs(OUT_DIR, exist_ok=True)


def get_raw_window(ds, idx):
    """Returnează amplitudinile brute (ne-normalizate) și label pentru fereastra idx."""
    file_idx, start = ds.windows[idx]
    try:
        fecg, fqrs, cat = ds._get_signals(file_idx)
    except Exception:
        return None
    end = start + ds.window
    mask_peaks = (fqrs >= start) & (fqrs < end)
    peak_locs  = fqrs[mask_peaks] - start
    if len(peak_locs) < 5:
        return None
    amps = fecg[:, peak_locs].T.copy()   # (N_beats, 6)
    label = int(np.bincount(cat[start:end].astype(np.int32), minlength=4).argmax())
    fname = os.path.basename(ds.files[file_idx])
    return amps, label, fname, file_idx


def normalize(amps):
    mu  = amps.mean(axis=0, keepdims=True)
    std = amps.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (amps - mu) / std


# ------------------------------------------------------------------
# Colectează N_EXAMPLES ferestre per clasă, din fișiere diferite
# ------------------------------------------------------------------
print('Loading dataset...')
ds = BeatDataset(DATA_DIR)
rng = np.random.default_rng(SEED)
order = rng.permutation(len(ds.windows)).tolist()

# per clasă: list of (amps_raw, fname, file_idx)
examples = {c: [] for c in range(N_CLASSES)}
seen_files = {c: set() for c in range(N_CLASSES)}

print('Searching for examples...')
for wi in order:
    res = get_raw_window(ds, wi)
    if res is None:
        continue
    amps, label, fname, fid = res
    if label not in examples:
        continue
    if fid in seen_files[label]:
        continue   # un singur exemplu per fișier
    if len(examples[label]) >= N_EXAMPLES:
        continue
    examples[label].append((amps, fname))
    seen_files[label].add(fid)
    if all(len(v) >= N_EXAMPLES for v in examples.values()):
        break

for c in range(N_CLASSES):
    print(f'  Class {c}: {len(examples[c])} examples collected')


# ------------------------------------------------------------------
# Plot 1: amplitudini absolute (norm) + diff, per clasă
# Figure: N_CLASSES linii × (N_EXAMPLES * 2) coloane
# ------------------------------------------------------------------
print('Plotting beat_features_per_class.png ...')
fig, axes = plt.subplots(N_CLASSES, N_EXAMPLES * 2,
                          figsize=(N_EXAMPLES * 2 * 4, N_CLASSES * 3),
                          squeeze=False)
fig.suptitle('Beat amplitude features per class\n'
             'Left columns: normalized amplitudes  |  Right columns: consecutive differences',
             fontsize=13, y=1.01)

for row, cls in enumerate(range(N_CLASSES)):
    for col_ex, (amps_raw, fname) in enumerate(examples[cls]):
        amps_norm = normalize(amps_raw)
        amps_diff = np.diff(amps_norm, axis=0)

        # Amplitudini normaliz
        ax_amp = axes[row][col_ex]
        for ch in range(6):
            ax_amp.plot(amps_norm[:, ch], color=CH_COLORS[ch],
                        lw=1.2, alpha=0.85, label=f'ch{ch+1}')
        ax_amp.axhline(0, color='k', lw=0.5, ls='--')
        ax_amp.set_title(f'{CLASS_NAMES[cls]}\n{fname[:28]}', fontsize=7)
        ax_amp.set_xlabel('Beat #', fontsize=7)
        if col_ex == 0:
            ax_amp.set_ylabel('Norm. amplitude', fontsize=7)
        ax_amp.tick_params(labelsize=6)

        # Diferențe
        ax_diff = axes[row][N_EXAMPLES + col_ex]
        for ch in range(6):
            ax_diff.plot(amps_diff[:, ch], color=CH_COLORS[ch],
                         lw=1.2, alpha=0.85)
        ax_diff.axhline(0, color='k', lw=0.5, ls='--')
        ax_diff.set_title(f'{CLASS_NAMES[cls]} [DIFF]\n{fname[:28]}', fontsize=7)
        ax_diff.set_xlabel('Beat #', fontsize=7)
        if col_ex == 0:
            ax_diff.set_ylabel('Amplitude diff', fontsize=7)
        ax_diff.tick_params(labelsize=6)

# Legendă o singură dată
handles = [plt.Line2D([0],[0], color=CH_COLORS[i], lw=1.5, label=f'ch{i+1}')
           for i in range(6)]
fig.legend(handles=handles, loc='lower center', ncol=6,
           bbox_to_anchor=(0.5, -0.02), fontsize=8)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, 'beat_features_per_class.png')
fig.savefig(out1, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'  Saved: {out1}')


# ------------------------------------------------------------------
# Plot 2: aceeași clasă din fișiere diferite (cross-file consistency)
# N_CLASSES linii × N_EXAMPLES coloane, doar amplitudini norm
# ------------------------------------------------------------------
print('Plotting beat_features_cross_file.png ...')
fig2, axes2 = plt.subplots(N_CLASSES, N_EXAMPLES,
                            figsize=(N_EXAMPLES * 4, N_CLASSES * 3),
                            squeeze=False)
fig2.suptitle('Cross-file consistency: same class, different files\n'
              '(normalized amplitudes — if pattern generalizes, curves should look similar)',
              fontsize=12, y=1.01)

for row, cls in enumerate(range(N_CLASSES)):
    for col, (amps_raw, fname) in enumerate(examples[cls]):
        amps_norm = normalize(amps_raw)
        ax = axes2[row][col]
        for ch in range(6):
            ax.plot(amps_norm[:, ch], color=CH_COLORS[ch], lw=1.3, alpha=0.85)
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.set_title(f'{CLASS_NAMES[cls]}\n{fname[:30]}', fontsize=7)
        ax.set_xlabel('Beat #', fontsize=7)
        if col == 0:
            ax.set_ylabel('Norm. amplitude', fontsize=7)
        ax.tick_params(labelsize=6)

fig2.legend(handles=handles, loc='lower center', ncol=6,
            bbox_to_anchor=(0.5, -0.02), fontsize=8)
plt.tight_layout()
out2 = os.path.join(OUT_DIR, 'beat_features_cross_file.png')
fig2.savefig(out2, dpi=120, bbox_inches='tight')
plt.close(fig2)
print(f'  Saved: {out2}')


# ------------------------------------------------------------------
# Plot 3: statistici per clasă — varianta DIFF
# Boxplot al magnitudinii |diff| per canal, per clasă
# ------------------------------------------------------------------
print('Plotting beat_features_diff_magnitude.png ...')
fig3, axes3 = plt.subplots(1, N_CLASSES, figsize=(N_CLASSES * 4, 4), squeeze=False)
fig3.suptitle('|Amplitude diff| magnitude per channel — should differ across classes',
              fontsize=11)

for cls in range(N_CLASSES):
    ax = axes3[0][cls]
    data_per_ch = [[] for _ in range(6)]
    for amps_raw, _ in examples[cls]:
        amps_norm = normalize(amps_raw)
        d = np.abs(np.diff(amps_norm, axis=0))   # (N_beats-1, 6)
        for ch in range(6):
            data_per_ch[ch].extend(d[:, ch].tolist())
    ax.boxplot(data_per_ch, patch_artist=True,
               boxprops=dict(facecolor='lightblue'),
               medianprops=dict(color='red', lw=2))
    ax.set_xticklabels([f'ch{i+1}' for i in range(6)], fontsize=8)
    ax.set_title(CLASS_NAMES[cls], fontsize=9)
    ax.set_ylabel('|diff|', fontsize=8)
    ax.tick_params(labelsize=7)

plt.tight_layout()
out3 = os.path.join(OUT_DIR, 'beat_features_diff_magnitude.png')
fig3.savefig(out3, dpi=120, bbox_inches='tight')
plt.close(fig3)
print(f'  Saved: {out3}')

print('Done.')
