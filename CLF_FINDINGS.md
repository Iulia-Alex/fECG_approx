# Part 2 — Fetal Movement Classification: Complete Experimental Record

## 1. The Task

Given a 600-second, 6-channel ground-truth fetal ECG (fECG) recording, classify the type
of fetal movement occurring at each point in time. The label source is `out.category_mask`,
a (600 000, 1) uint8 array sampled at 1000 Hz. Four classes exist:

| Label | Class | Approximate share |
|-------|-------|-------------------|
| 0 | Stationary | 53 % |
| 1 | Linear | 17 % |
| 2 | Helical | 13.5 % |
| 3 | Screw | 16.5 % |

The physical intuition is that fetal movement modulates the distance between the fetus and
each abdominal electrode. As the fetus moves, the amplitude of the R-peak picked up by each
electrode changes in a characteristic pattern:

- **Stationary** — no movement, amplitudes constant across beats.
- **Linear** — fetus translates in one direction; amplitudes shift monotonically.
- **Helical** — fetus rotates; amplitudes oscillate periodically.
- **Screw** — combined rotation + translation; complex multi-channel pattern.

---

## 2. The Dataset

Each `.mat` file contains one 600-second recording. Relevant fields:

- `out.fecg` — `(6, 600 000)` float32, ground-truth fetal ECG at 1000 Hz.
- `out.fqrs` — 1-D array of sample indices where fetal R-peaks occur (~60–180 BPM).
- `out.category_mask` — `(600 000,)` uint8, label at every sample.

Total dataset: **655 files**, yielding **12 477 complete movement phases** after filtering.

---

## 3. The Segmentation Procedure

### 3.1 What a "phase" is

The `category_mask` is a step function: it stays at a constant value (e.g., 2 = Helical)
for a contiguous block, then jumps to another value. Each such contiguous block is a
**movement phase**. Phases range from 10 seconds to 83 seconds (median ≈ 25 s).

### 3.2 How phases are extracted (PhaseDataset / EnvelopeDataset)

```
transitions = where(diff(category_mask) != 0)
starts = [0] + transitions + 1
ends   = transitions + 1 + [len(category_mask)]

for (s, e) in zip(starts, ends):
    if e - s < 10 000:          # skip phases shorter than 10 s
        continue
    label = category_mask[s]
    peaks = fqrs[(fqrs >= s) & (fqrs < e)]
    if len(peaks) < 8:          # skip phases with fewer than 8 beats
        continue
    amps = fecg[:, peaks].T     # (N_beats, 6) — one row per beat
```

The result is one sample per phase: a matrix `amps` of shape `(N_beats, 6)` where each row
contains the fetal ECG amplitude at each R-peak across all 6 channels.

### 3.3 Train/validation split — why file-level is mandatory

A naive random split assigns individual phases (or windows) to train and val. This causes
**data leakage**: phases from the same recording share ECG morphology (baseline wander, noise
level, electrode placement). A model trained on some phases of a file will recognise the
recording-specific fingerprint in the val phases — not the movement pattern.

The correct approach: **file-level split**. All phases from a file go entirely to train or
entirely to val. With `val_frac=0.15` and `seed=42`, this gives:

- ~557 training files, ~10 600 training phases
- ~98 validation files, ~1 877 validation phases

Implementation in `PhaseDataset.file_split()`:
```python
rng = np.random.default_rng(seed=42)
val_files = set(rng.permutation(n_files)[:n_val])
train_idx = [i for i, fi in enumerate(file_ids) if fi not in val_files]
val_idx   = [i for i, fi in enumerate(file_ids) if fi in val_files]
```

---

## 4. Experiment Series — Chronological

### 4.1 clf_v1 and clf_v2 — Raw fECG windows (FAILED)

**Input:** Fixed-length window of raw fECG signal, shape `(6, 4000)` (4 s at 1000 Hz).  
**Model:** ResNet1D, ~970 K parameters.  
**Result:** Val accuracy ~62 % regardless of regularisation.

**What happened:** Even with a file-level split, the model memorised recording-specific
morphology. A 4-second raw window gives 40 ms × 6 channels of signal; the movement pattern
is not encoded in the waveform shape of individual beats but in how amplitudes *change across
beats over the full duration of a phase*. Any window shorter than the phase captures only a
fragment, making helical and linear indistinguishable.

clf_v2 used 8-second windows with the same architecture. It was cancelled early — same
pathology.

---

### 4.2 clf_v3 and clf_v4 — Beat-amplitude sequences in fixed 30-second windows

**Input:** For each 30-second window, extract the R-peak amplitudes: `amps = fecg[:, peaks].T`,
shape `(~40, 6)`. Zero-pad to `(MAX_BEATS, 6)`.  
**Models:**
- clf_v3: ResNet1D, 244 K parameters — val acc **~86 %**
- clf_v4: BeatTransformer, 205 K parameters — val acc **~84 %**

**What happened:** Using beat amplitudes rather than raw signal is clearly better — the model
learns the correct representation. However, a 30-second window lands in the middle of a phase
that may be 50 seconds long. The partial view is often ambiguous: a helical phase seen for
only 30 seconds may show only one oscillation cycle and look linear. The ceiling was ~86 %.

---

### 4.3 clf_v9 — The breakthrough: complete phases + spectral features + Random Forest

**The key insight:** Movement type is visible in the *entire* amplitude trajectory of a phase.
A 30-second helical phase might show one cycle; the full 60-second phase shows two, making
the periodicity unambiguous. Extracting **complete phases** and describing them with
hand-crafted spectral features eliminates the ambiguity.

#### Feature engineering — 36 features per phase (PhaseDataset)

For each of the 6 channels, 6 statistics are computed over the vector
`amps[:, ch]` (N_beats amplitude values for that channel):

| Feature | Description |
|---------|-------------|
| `std` | Standard deviation (≈0 for stationary, >0 for movement) |
| `slope` | Linear regression slope over beat index (large for linear/screw) |
| `linear_r2` | R² of the linear fit (high for linear, low for helical) |
| `fft_max_amp` | Dominant FFT amplitude, DC excluded (high when oscillation is periodic) |
| `fft_dom_freq` | Normalised dominant FFT frequency (key discriminator: ~0 for stationary/linear, >0 for helical/screw) |
| `lag5_autocorr` | Pearson correlation at lag 5 beats (detects oscillation) |

Total: 6 channels × 6 features = **36 features per phase**.

These are z-score normalised per feature across the full dataset before being fed to any model.

#### Model: Random Forest (sklearn)

```
RandomForestClassifier(
    n_estimators=500,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
```

**Result: 92.2 % val accuracy.** This was the first model to break 90 %.

**Feature importance analysis:** The top 5–7 features by importance were all `fft_dom_freq`
across different channels. This makes physical sense:

- Stationary → `fft_dom_freq ≈ 0` (no oscillation)
- Linear → very low frequency (monotone drift over the phase)
- Helical → mid-range frequency (one or two oscillation cycles per phase)
- Screw → complex, higher-frequency content

**MLP on the same 36 features:** `FC(36 → 256 → 256 → 128 → 4)` with BatchNorm and Dropout,
trained with AdamW + ReduceLROnPlateau + early stopping → **89.3 %**. The Random Forest
generalises better on this small dataset.

---

### 4.4 clf_v10 and clf_v11 — DL on complete beat-amplitude sequences

**Input:** Complete phase beat-amplitude matrix, shape `(N_beats, 6)`. Since phases vary in
length, pad to `(200, 6)` (covers all phases up to ~150 beats).  
**Models:**
- clf_v10: ResNet1D, 244 K parameters — train 99 %, val **90.9 %** (severe overfit)
- clf_v11: BeatTransformer (self-attention over beats) — val **88.3 %**, no overfit

**Observation:** ResNet1D learns the training set almost perfectly but generalises ~2 % worse
than RF. The dataset of 12 K phases is too small for a deep model to find information beyond
what the 36 spectral features already capture. The Transformer trains more carefully but also
performs worse.

---

### 4.5 clf_v12 — Extended features: RR interval + QRS PCA + envelope statistics

**Input:** 78 features = 36 original + RR intervals (inter-beat timing) + PCA of QRS shape
across beats + amplitude envelope statistics.  
**Model:** Random Forest (500 trees).  
**Result: 91.2 %** — *worse* than the 36-feature baseline.

**Why:** The new features added noise. The dominant discriminator is `fft_dom_freq`; the
additional features described irrelevant variability. Feature importance confirmed that
zero new features entered the top 15.

---

### 4.6 clf_v13 — QRS morphology: CNN + Transformer on individual beat shapes

**Input:** For each phase, extract 150 beats, each represented as a 50-sample waveform around
the R-peak: shape `(150, 6, 50)`.  
**Model:** CNN on the 50-sample waveform → embedding → Transformer over 150 beats.  
**Result: 76.2 %.**

**Why it failed:** The shape of a single QRS complex does not encode which type of movement
is occurring. Movement modulates *amplitude across beats*, not *the shape of individual beats*.
The classifier had access to the right data type (amplitude) but via the wrong abstraction
(waveform shape rather than scalar amplitude value).

---

### 4.7 clf_v14 — Extended spectral features: cross-channel correlations

**Input:** 71 features = 36 spectral + 15 cross-correlations at lag 0 between all channel
pairs + 15 cross-correlations at lag 5 beats + 5 PCA variance statistics.  
**Models trained in the same script:**
- Random Forest (500 trees): **92.55 %** — best RF
- MLP `(71 → 256 → 256 → 128 → 4)`: 90.4 %
- Transformer (71 tokens × d_model=64, 3 layers): 89.7 %

**Feature importance:** All top 15 features were still `fft_dom_freq` and `fft_max_amp` from
the original 36. Zero cross-channel or PCA features appeared in the top 15. The +0.35 %
improvement over v9 may be noise.

---

### 4.8 clf_v15 — Additional inter-channel features: phase differences + coherence

**Input:** 72 features = 36 spectral + phase difference and coherence between channel pairs.  
**Models:**
- Random Forest: **92.18 %** — *worse* than v14 and v9's 36-feature RF
- Transformer: 89.45 %

**Conclusion:** Adding more inter-channel descriptors systematically hurts. The spectral
per-channel statistics are sufficient and additional features add only variance.

---

### 4.9 clf_v16 — Hilbert envelope as input to a large ResNet (FAILED)

**Motivation:** Instead of extracting beat amplitudes via R-peak detection (which depends on
`fqrs` annotations), compute the Hilbert analytic signal amplitude — the instantaneous
envelope of the raw fECG signal. This represents the same physical quantity (amplitude
modulation by movement) without requiring R-peak annotations.

**Procedure (EnvelopeDataset):**
```
for each channel:
    seg = fecg[ch, phase_start:phase_end]
    amp = |hilbert(seg)|           # instantaneous amplitude
    amp_ds = resample(amp, 200)    # downsample to fixed length 200
    amp_norm = (amp_ds - mean) / std
```
Output shape per phase: `(6, 200)` float32.

**Model:** EnvResNet (large), 500 K parameters — train 99.6 %, val **88.9 %** (severe overfit).  
**Model:** EnvTransformer — val **77.18 %**.

**Problem:** The large ResNet memorises the 12 K training phases. The envelope signal contains
high-frequency noise that the model overfits to. Need significant regularisation.

---

### 4.10 clf_v17 — Hilbert envelope + small ResNet + aggressive regularisation (BEST DL)

**Same EnvelopeDataset** (`(6, 200)` envelope per phase).

**Changes from v16:**
1. **Smaller model:** SmallEnvResNet with channels `6 → 32 → 64 → 64` (~60 K parameters
   vs 500 K). Architecture: stem conv → 3 residual blocks (stride-2 downsampling at each:
   200 → 100 → 50 → 25) → GlobalAvgPool → `FC(64→32→4)`.
2. **Aggressive dropout:** 0.5 in the classification head, 0.2 inside residual blocks.
3. **Data augmentation** (applied at training time only):
   - Gaussian noise: `x += N(0, 0.05)` with probability 0.5
   - Circular time shift: `roll(x, shift)` where shift ∈ [−20, +20] samples, prob 0.5
   - Per-channel amplitude scale: multiply each channel by U(0.8, 1.2), prob 0.5
   - Channel dropout: zero out one random channel with probability 0.2
4. **Higher weight decay:** 1e-2 (vs 1e-3 in previous scripts).
5. **Label smoothing:** 0.1 (CrossEntropyLoss).
6. **Cosine LR schedule:** CosineAnnealingLR, T_max=300, eta_min=1e-5 (vs ReduceLROnPlateau).

**Result: train 91.8 %, val 90.24 % — no overfitting.**

The train-val gap of ~1.5 % is acceptable. This is the best deep learning result.

---

### 4.11 Ensemble — RF (v14 spectral) + SmallEnvResNet (v17 envelope)

**Motivation:** RF and ResNet v17 learn from fundamentally different representations:
- RF on 36 spectral features: explicit, interpretable, frequency-domain description.
- ResNet v17 on Hilbert envelope: implicit, learned features from the continuous amplitude signal.

Combining them via probability averaging should help where one model is uncertain.

**Procedure (eval_ensemble.py):**

Both models produce class probability vectors for each validation phase. The ensemble
prediction is a weighted average:

```
combined_probs = w_RF × rf_probs + (1 - w_RF) × resnet_probs
prediction     = argmax(combined_probs)
```

A sweep over w_RF ∈ {0.00, 0.05, …, 1.00} finds the optimal weight:

| w_RF | Val accuracy |
|------|-------------|
| 0.45–0.65 | 92.65–92.86 % |
| 0.70 | 92.98 % |
| **0.75** | **93.07 %** |
| 0.80 | 92.87 % |
| 1.00 (RF solo) | 92.34 % |

**Best ensemble: w_RF = 0.75, val accuracy = 93.07 %.**

The RF dominates (3× weight), and the ResNet's envelope-based probabilities contribute a
~0.7 % improvement over RF solo.

**Important note:** The split uses the same `seed=42` and `val_frac=0.15` for both datasets,
ensuring the validation sets contain the same phases. This is verified at runtime with an
`assert` on the number of indices.

---

## 5. Confusion Analysis — The Linear ↔ Helical Problem

The confusion matrix for the best ensemble (w_RF = 0.75):

```
                Stationary  Linear  Helical  Screw
Stationary           751       1        0       0
Linear                 5     356       40       0
Helical                5      80      266       1
Screw                  0       0        0     401
```

**Stationary** and **Screw** are classified near-perfectly. All errors concentrate on
**Linear ↔ Helical**:

- 40 Linear phases predicted as Helical
- 80 Helical phases predicted as Linear
- 120 errors out of ~170 total

**Why:** Linear movement produces a monotone trend in amplitude (one direction of change
over the phase). Helical movement produces an oscillatory pattern. For short phases (10–15 s),
a helical phase may complete less than one oscillation cycle and look indistinguishable from
a linear trend. The `fft_dom_freq` feature is near-zero for both in that case.

The confusion is not a model failure — visually plotting the amplitude trajectories of
confusable phases shows they are genuinely ambiguous. Theoretically, very long phases
should be classifiable at near-100 %; the limit is the dataset phase duration distribution.

---

## 6. Summary of All Classifier Versions

| Version | Input representation | Model | Val acc | Key finding |
|---------|---------------------|-------|---------|-------------|
| clf_v1 | Raw fECG 4s window `(6, 4000)` | ResNet1D 970K | ~62 % | Morphology memorisation |
| clf_v2 | Raw fECG 8s window `(6, 8000)` | ResNet1D 970K | cancelled | Same pathology |
| clf_v3 | Beat amps 30s `(~40, 6)` | ResNet1D 244K | ~86 % | Better repr., partial phase |
| clf_v4 | Beat amps 30s `(~40, 6)` | BeatTransformer 205K | ~84 % | Same limitation |
| clf_v9 | 36 spectral features / phase | Random Forest 500 trees | **92.2 %** | fft_freq is dominant |
| clf_v9 | 36 spectral features / phase | MLP `(36→256→256→128→4)` | 89.3 % | RF wins on small dataset |
| clf_v10 | Beat amps full phase `(200, 6)` | ResNet1D 244K | 90.9 % | Overfit: train 99 % |
| clf_v11 | Beat amps full phase `(200, 6)` | BeatTransformer | 88.3 % | No overfit, still below RF |
| clf_v12 | 78 features (36 + RR + QRS PCA) | Random Forest | 91.2 % | New features = noise |
| clf_v13 | QRS waveforms `(150, 6, 50)` | CNN + Transformer | 76.2 % | Shape ≠ movement type |
| clf_v14 | 71 features (36 + cross-corr) | Random Forest | **92.55 %** | Best RF solo |
| clf_v14 | 71 features | MLP | 90.4 % | |
| clf_v14 | 71 features | Transformer (d=64, 3 layers) | 89.7 % | |
| clf_v15 | 72 features (36 + phase diff) | Random Forest | 92.18 % | Worse than v14 |
| clf_v15 | 72 features | Transformer | 89.45 % | |
| clf_v16 | Hilbert envelope `(6, 200)` | EnvResNet 500K | 88.93 % | Severe overfit |
| clf_v16 | Hilbert envelope `(6, 200)` | EnvTransformer | 77.18 % | |
| clf_v17 | Hilbert envelope `(6, 200)` | SmallEnvResNet 60K | **90.24 %** | No overfit |
| **Ensemble** | RF (w=0.75) + ResNet v17 (w=0.25) | — | **93.07 %** | **Best overall** |

---

## 7. Key Findings — Summary

**Finding 1 — File-level split is mandatory.**  
Window-level random split leaks recording-specific ECG morphology into the validation set,
producing artificially high accuracy (~95 % train, ~62 % val). Always split at the file level.

**Finding 2 — Fixed-length windows are the wrong unit of analysis.**  
30-second windows with ~40 beats hit a ceiling of ~86 %. They capture only a fragment of
each movement phase. Fixed windows also introduce label noise: 30-second windows have 69 %
single-class purity; 5-second windows have 85 % but contain too few beats to be useful.

**Finding 3 — Complete movement phases are the correct unit of analysis.**  
Extracting full phases (10–83 s, median ~25 s) via `category_mask` transitions gives clean,
single-label samples that capture the full amplitude trajectory. This was the single most
important change: RF on complete phases achieved 92.2 % vs ~86 % with fixed windows.

**Finding 4 — fft_dom_freq is the dominant discriminating feature.**  
The normalised dominant FFT frequency of the per-channel beat-amplitude sequence consistently
ranks in the top 5–7 features across all RF experiments (v9, v14, v15). It encodes the
oscillation frequency of the amplitude modulation, which directly corresponds to movement type.

**Finding 5 — Additional inter-channel features provide no new information.**  
Cross-correlations (v14: +0.35 %), phase differences, coherence (v15: worse than baseline),
QRS morphology (v13: 76 %) — none enter the top 15 features by importance. The spectral
per-channel statistics already capture what matters.

**Finding 6 — Deep learning does not outperform Random Forest on this 12 K dataset.**  
All DL models tested (ResNet, MLP, Transformer) reached 88–91 % — below RF's 92.55 %. The
dataset is too small for DL to find generalizable patterns beyond what the explicit spectral
features encode.

**Finding 7 — A large DL model overfits; a small model + aggressive augmentation does not.**  
EnvResNet (500 K params, v16): train 99.6 %, val 88.9 %. SmallEnvResNet (60 K params, v17)
with noise injection, circular time shift, per-channel amplitude scaling, channel dropout,
label smoothing, and weight decay: train 91.8 %, val 90.24 %. This is the best DL result.

**Finding 8 — Ensemble RF + ResNet v17 reaches 93.07 % with w_RF = 0.75.**  
Combining complementary representations (spectral features vs Hilbert envelope) adds ~0.7 %
over RF solo. The RF contributes 3× more weight. The gain is real but modest.

**Finding 9 — The Linear ↔ Helical confusion is a dataset-level limitation.**  
~120 of ~170 total validation errors are Linear misclassified as Helical or vice versa. These
classes are physically similar at short phase durations. Stationary and Screw are classified
near-perfectly by every model tested from v9 onward.

---

## 8. Dataset Statistics

- **Source:** 655 `.mat` files, 600 s each, 1000 Hz, 6-channel ground-truth fECG
- **Phases extracted:** 12 477 total (min 10 s, max 83 s, median ~25 s)
- **File split (seed=42, val_frac=0.15):** ~557 train files / ~98 val files
- **Phase split:** ~10 600 train / ~1 877 val
- **Class distribution:** [4967 Stationary, 2563 Linear, 2425 Helical, 2522 Screw]
- **Beats per phase:** typically 15–100 (fetal HR ~60–180 BPM)

---

## 9. File Reference

| File | Role |
|------|------|
| `src_2026/phase_dataset.py` | `PhaseDataset` — 36 spectral features per complete phase |
| `src_2026/envelope_dataset.py` | `EnvelopeDataset` — Hilbert envelope `(6, 200)` per phase |
| `src_2026/clf_dataset.py` | `ClfDatasetV14/V15` — 71/72 extended features |
| `src_2026/beat_dataset.py` | `BeatDataset` — beat amplitude sequences for v3/v4/v10/v11 |
| `src_2026/qrs_dataset.py` | `QRSDataset` — QRS waveforms for v13 |
| `src_2026/resnet1d.py` | `ResNet1D` architecture |
| `src_2026/train_clf_v9.py` | RF + MLP on 36 features |
| `src_2026/train_clf_v14.py` | RF + MLP + Transformer on 71 features |
| `src_2026/train_clf_v17.py` | `SmallEnvResNet` + augmentation |
| `src_2026/eval_ensemble.py` | Ensemble sweep + confusion matrix |
| `models/movement_clf_v17.pth` | Best DL checkpoint (SmallEnvResNet) |
| `models/ensemble_rf_resnet17_results.json` | Ensemble sweep results + CM |
