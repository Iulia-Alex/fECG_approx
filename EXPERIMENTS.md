# fECG Extraction from Movement ECG — Experiment Log

**Last updated:** 2026-05-18
**Active jobs (Part 1):** 1381 (v15, Lenovo6, ep95+) | 1389 (v16, Lenovo2, ep17+) | 1391 (v17, Lenovo6, ep4+)
**Active jobs (Part 2 — 4-class):** — (toate terminate, best ensemble 93.07%)
**Active jobs (Part 2b — binary):** — (binary_v1 terminat, F1=0.945 pe Short_time_intervals)

---

## 1. Problem Statement

Extract the **fetal ECG (fECG)** from a 6-channel abdominal mixture containing:
- **mECG** (maternal ECG) — dominant interferer, 8 dB stronger than fECG
- **Movement noise** — weaker than fECG

**Goal for downstream task:** amplitude of R-peaks must be predicted accurately (1:1 with ground truth) — these amplitudes are used to estimate fetal movement/position.

### SNR Summary (measured across dataset)

| Signal pair | SNR |
|---|---|
| fECG vs mECG | −8.1 dB |
| fECG vs mixture | −9.4 dB |
| fECG vs movement noise | +4.0 dB |
| mECG vs movement noise | +12.1 dB |

**Key insight:** mECG is the dominant interferer — the model's main job is mECG suppression, not denoising.

---

## 2. Model Architecture — ComplexUNet (v1 baseline)

The network operates on **complex-valued spectrograms** (both real and imaginary parts).
Input: 6-channel mixture spectrogram → Output: 6-channel predicted fECG spectrogram.

### Building Blocks

| Block | Description |
|---|---|
| `ComplexConvLayer` | Shared weights applied to real and imaginary parts with cross-mixing |
| `ComplexReLU` | LeakyReLU(0.2) applied independently to real and imaginary parts |
| `ComplexDownBlock` | ComplexConvLayer + average pooling (encoder) |
| `ComplexUpBlock` | Bilinear upsampling + ComplexConvLayer (decoder) |
| `Diag` | Learnable diagonal (element-wise) scaling, zero-initialised — skip gate at input/output. Scales magnitude, preserves phase. |

Skip connections via **addition** (not concatenation).

### Channel Configuration

| Block | Channels |
|---|---|
| Diag (input) | 6 → 6 |
| down1 | 6 → 32 |
| down2 | 32 → 64 |
| down3 | 64 → 128 |
| bottleneck | 128 → 128 |
| up1 | 128 → 128 |
| up2 | 128 → 64 |
| up3 | 64 → 32 |
| up4 | 32 → 6 |
| Diag (output) | 6 → 6 |

**Total parameters: 0.59 M**

---

## 3. Key Design Decisions (v1 baseline)

### 3.1 Spectrogram Size: 128×128 → 128×400

| | 128×128 | **128×400** |
|---|---|---|
| Time resolution | 31 ms/frame | **10 ms/frame** |
| QRS complex spans | ~2 frames | **6–8 frames** |
| QRS recoverable? | No | Yes |

STFT parameters: `n_fft=256, hop=10, win_len=128, FS=1000 Hz` → shape `(129, 401)` → resized to `(128, 400)`.

---

### 3.2 Loss Function: ComplexMSE → SignalMSE (most impactful)

| | ComplexMSE | **SignalMSE** |
|---|---|---|
| What it penalises | Spectrogram distance (real + imag) | Time-domain MSE after iSTFT |
| Model behaviour | Learns average spectral shape → amplitude suppression | Penalised for wrong peak height → sharp QRS |
| Inference (ep 6) | Peaks at ±0.2–0.4 (GT: ±1.5–3.5) | Peaks at ~90–95% of GT amplitude |

**Why ComplexMSE suppresses amplitude:** it is symmetric with respect to phase — a model can minimise it by predicting low-amplitude smooth outputs that average out phase errors. SignalMSE directly penalises wrong peak heights in the time domain.

**Implementation:**
1. Resize predicted spectrogram `(128, 400)` → `(129, 401)` via bilinear interpolation
2. Apply `torch.istft` on GPU (Hann window, cached per device)
3. `F.mse_loss` vs ground-truth time-domain fECG (dataset 3rd return value)

---

### 3.3 GPU iSTFT (7× speedup)

`torch.istft` on CUDA: 0.073 s/batch vs 0.502 s on CPU (B=192, M4000). Implemented with a cached `_DEVICE_HANN` dict.

### 3.4 Channel Width: halved → 0.59 M params

Halving all channel counts reduces parameters 3.4× and epoch time from ~4 h to ~2.3 h with no observed underfitting (98,250 training windows).

### 3.5 DataLoader Workers: 0 → 2

Overlapping CPU data loading (STFT computation) with GPU training halved epoch time (~4.5 h → ~2.3 h).

---

## 4. Experiments Comparison

| | **v1** | **v2** | **v3** | **v4** | **v5** | **v6** | **v7** | **v8** | **v9** | **v10** | **v11** | **v12** | **v13** | **v14** | **v15** | **v16** | **v17** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Script** | `train_movement.py` | `archive/train_v2.py` | `archive/train_v3.py` | `archive/train_v4.py` | `train_movement_v5.py` | `train_movement_v6.py` | `train_movement_v7.py` | `train_movement_v8.py` | `train_movement_v9.py` | `train_movement_v10.py` | `train_movement_v11.py` | `train_movement_v12.py` | `train_movement_v13.py` | `train_movement_v14.py` | `train_movement_v15.py` | **`train_movement_v16.py`** | **`train_movement_v17.py`** |
| **Architecture** | ComplexUNet | Paper ComplexUNet | ComplexUNet | ComplexUNet | ComplexUNet | ComplexUNet | ComplexUNetV7 | ComplexUNetV7 | ComplexUNetV9 | ComplexUNetV10 | ComplexUNetV11 | ComplexUNetV12 | ComplexUNetV13 | ComplexUNetV14 | ComplexUNetV15 | **ComplexAttentionUNet** | **ComplexAttentionUNet** |
| **Params** | 0.59 M | 7.13 M | 0.59 M | 0.59 M | 0.59 M | 0.59 M | 1.87 M | 1.87 M | 1.87 M | 0.59 M | 1.87 M | 1.87 M | 1.87 M | 7.09 M | 7.13 M | **7.13 M** | **7.13 M** |
| **Output** | direct | direct | direct | direct | direct | direct | direct | direct | soft mask 1× | soft mask 1× | direct | gain mask 1.5× | soft mask 1× | soft mask 1× | soft mask 1× | **soft mask 1×** | **soft mask 1×** |
| **Conv** | Shared Re/Im | Sep. cross-mix | Shared | Shared | Shared | Shared | Split Re/Im | Split Re/Im | Split Re/Im | Shared Re/Im | Split Re/Im | Split Re/Im | Split Re/Im | Split Re/Im | Sep. cross-mix | **Sep. cross-mix** | **Sep. cross-mix** |
| **Activation** | LeakyReLU(0.2) | RoActivation | LeakyReLU | LeakyReLU | LeakyReLU | LeakyReLU | RoActivation | RoActivation | RoActivation | LeakyReLU(0.2) | RoActivation | RoActivation | RoActivation | RoActivation | RoActivation | **RoActivation** | **RoActivation** |
| **Skip conn.** | Addition | Concat | Addition | Addition | Addition | Addition | Concat | Concat | Concat | Addition | Concat | Concat | Concat | Concat | Concat | **Concat** | **Concat** |
| **Diagonal** | DiagMag exp(β) | Phase rot. e^{iβ} | DiagMag | DiagMag | DiagMag | DiagMag | Phase rot. | Phase rot. | Phase rot. | DiagMag exp(β) | Phase rot. | Phase rot. | Phase rot. | Phase rot. | exp(β) sep R/I | **exp(β) sep R/I** | **exp(β) sep R/I** |
| **Norm** | — | mean+std | — | — | — | — | — | — | — | — | — | — | mag scale | mag scale | mean+std | **mean+std** | **mean+std** |
| **Attention** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **AG1(64×64)+AG2(32×32)** | **AG1+AG2 + L_att** |
| **FS / Spec** | 1000/128×400 | 500/128×128 | 1000/128×400 | 1000/128×400 | 1000/128×400 | 1000/128×400 | 1000/128×400 | 1000/128×400 | 1000/128×400 | 1000/128×400 | 1000/128×400 | 1000/128×400 | 1000/128×400 | 500/128×128 | 500/128×128 | **500/128×128** | **500/128×128** |
| **Loss** | SignalMSE | SignalMSE | MSE+5×Peak+0.1×Cpl | MSE+3×AmpW | MSE+3×AmpW | MSE+AmpW+BL | SignalMAE | SignalMSE | Sig+Cpl | Sig+Cpl+3×Peak | Sig+3×Peak(±30ms) | Sig+3×Peak(±100ms) | SignalMSE | SignalMSE | SignalMSE | **SignalMSE** | **Sig+L_att** |
| **Optimizer** | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW | Adam | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW | **AdamW** | **AdamW** |
| **Scheduler** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | ReduceLROnPlateau | **ReduceLROnPlateau** | **ReduceLROnPlateau** |
| **LR** | 1e-4 | 1e-4 | 1e-4 | 1e-5 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | **1e-4** | **1e-4** |
| **BS** | 32 | 32 | 32 | 32 | 32 | 32 | 16 | 16 | 16 | 32 | 16 | 16 | 16 | 8 | 8 | **8** | **8** |
| **Job** | 1276 | 1284 | 1287 | 1292 | 1294 | 1311 | 1310 | 1319 | 1346 | 1349 | 1359 | 1371 | 1378 | 1382 | 1381 | **1389** | **1391** |
| **Node** | Lenovo2 | Lenovo6 | Lenovo2 | Lenovo2 | Lenovo2 | Lenovo2 | Lenovo6 | Lenovo6 | Lenovo6 | Lenovo2 | Lenovo6 | Lenovo2 | Lenovo6 | Lenovo2 | Lenovo6 | **Lenovo2** | **Lenovo6** |
| **Best val loss** | 0.014745 @ ep199 | 0.03217 @ ep99 | — | 0.02319 @ ep36 | 0.022473 @ ep200 | 0.052952 @ ep67 | 0.081819 @ ep30 | 0.046987 @ ep41 | 0.925510 @ ep138 | 2.378023 @ ep35 | **0.432913 @ ep73** | **0.220880 @ ep49** | **0.060014 @ ep84** | **PD** | **0.031271 @ ep92** | **0.032987 @ ep15** | **0.046913 total @ ep3** |
| **Status** | DONE | DONE | CANCELLED | CANCELLED | DONE | stopped | stopped (L1) | DONE (poor) | DONE (ep138) | DONE (ep55) | **DONE** (ep93) | **DONE** (ep51) | **STOPPED** (ep84) | **PD** | **RUNNING** (ep95+) | **RUNNING** (ep17+) | **RUNNING** (ep4+) |

---

## 5. Loss Function Evolution

### v1 — SignalMSE only
```
loss = F.mse_loss(pred_time, fecg_time)
```
Result: good peak positions and amplitudes (~90–95% of GT), some baseline noise.

### v2 — SignalMSE only (paper architecture)
```
loss = F.mse_loss(pred_time, fecg_time)
```
Result: peak amplitudes comparable to v1 but more baseline noise; paper architecture (7.13M params) doesn't outperform v1 (0.59M).

### v3 — SignalMSE + PeakMSE + ComplexMSE (cancelled ep 101)
```
loss = signal_mse + 5.0 * peak_mse + 0.1 * complex_mse
```
Where `peak_mse` uses a hard threshold mask (|signal| > 3×std) dilated ±40 samples.

**Problem:** even 0.1×ComplexMSE caused amplitude suppression (~50–60% GT). "Clean baseline" was actually the model predicting near-zero everywhere.

**Lesson learned:** ComplexMSE must never be combined with a loss that requires correct amplitudes — even at small weight.

### v4 — SignalMSE + AmpWeightedMSE, warm start from v1 (cancelled ep 37)
```
w = (|target| / max|target|)²
loss = signal_mse + 3.0 * mean((pred - target)² * w)
```
**Problem:** fine-tuning from v1 checkpoint with a *different* loss caused immediate train/val gap. V1's internal representations were optimised for pure SignalMSE; switching loss mid-training destabilised generalisation.

**Lesson learned:** changing loss at fine-tuning doesn't work — model needs to learn representations for the new objective from scratch.

### v5 — SignalMSE + AmpWeightedMSE, from scratch (DONE)
```
w = (|target| / max|target|)²          # soft: 0 on baseline, 1 on R-peaks
loss = signal_mse + 3.0 * mean((pred - target)² * w)
```
**Why from scratch (not warm start):** model will learn representations optimised for amplitude accuracy from the beginning, avoiding the landscape mismatch of v4.

**Result:** peaks ~95% GT, baseline noisier than v1. AmpWeightedMSE ponderează toate amplitudinile mari (inclusiv artefacte), nu doar R-peaks fetale.

### v9 — SignalMSE + ComplexMSE (soft mask, 1.87M arch)
```
loss = signal_mse + complex_mse
```
**ComplexMSE dominates at convergence (~10× SignalMSE).** Model optimizează structura spectrală, nu amplitudinile → amplitudini R-peak suprimate față de GT.

### v10 — SignalMSE + ComplexMSE + 3×PeakMSE(fqrs) (soft mask, 0.59M)
```
loss = signal_mse + complex_mse + 3.0 * peak_mse
# la ep55: sig=0.079, cpl=1.626, pk=0.233 — ComplexMSE domina la fel
```
**ComplexMSE domină indiferent de PeakMSE.** PeakMSE nu poate compensa efectul de suprimare.

### v11 — SignalMSE + 3×PeakMSE(fqrs), direct prediction (1.87M arch)
```
loss = signal_mse + 3.0 * peak_mse   # fara ComplexMSE
```
**De ce fără ComplexMSE:** din v9 și v10 s-a constatat că ComplexMSE domină și suprimă amplitudinile indiferent de configurație. Paper-ul original folosea și el SignalMSE pur. **Direct prediction** (nu soft mask) pentru că sigma mask limitează output ≤ input per bin.

---

## 6. Training Results

### v1 — Job 1276 — **DONE** (ep 200/200, finished 2026-03-23)

**Hardware:** Lenovo2, Quadro M4000 8 GB
**Config:** BS=32, LR=1e-4, WD=1e-5, patience=15, max 200 epochs
**Best val loss: 0.014745 @ ep 199**

| Epoch | Val loss | Notes |
|---|---|---|
| 1 | 0.0833 | — |
| 30 | 0.0228 | — |
| 60 | 0.0192 | — |
| 100 | 0.0169 | — |
| 152 | 0.01547 | — |
| 179 | 0.01500 | checkpoint saved |
| 199 | **0.014745** | **BEST checkpoint saved** |
| 200 | 0.014818 | final epoch (no improvement) |

### v2 — Job 1284 — FINISHED (ep 100)

**Hardware:** Lenovo6, Quadro M4000 8 GB
**Config:** BS=32, LR=1e-4, patience=15, max 100 epochs
**Best val loss:** 0.03217 @ ep 99
**Note:** val loss unstable (spikes ±0.01) — no weight decay, no grad clipping.

### v3 — Job 1287 — CANCELLED at epoch 101

Cancelled: amplitude suppression from 0.1×ComplexMSE. R-peaks at ~50–60% GT.

### v4 — Job 1292 — CANCELLED at epoch 37

Cancelled: immediate train/val gap from ep 1 (fine-tuning with different loss). Best val 0.02319.

### v5 — Job 1294 — DONE (ep 200/200)

**Hardware:** Lenovo2, Quadro M4000 8 GB
**Config:** BS=32, LR=1e-4, WD=1e-5, patience=15, max 200 epochs
**Init:** random (no warm start)
**Loss:** SignalMSE + 3×AmpWeightedMSE
**Best val loss: 0.022473 @ ep200**

| Epoch | Val loss | Notes |
|---|---|---|
| 1 | ~0.25 | high due to AmpWeightedMSE×3 scale |
| 35 | 0.03546 | — |
| 81 | 0.027981 | — |
| 154 | 0.023846 | — |
| 200 | **0.022473** | **BEST — final epoch** |

Inference (ep200): peaks ~95% GT, baseline noisier than v1. Sem3 near-perfect.

### v6 — Job 1311 — STOPPED (ep 67, plateau)

**Hardware:** Lenovo2, Quadro M4000 8 GB
**Config:** BS=32, LR=1e-4, WD=1e-5, patience=15
**Init:** warm start from v1 best checkpoint
**Loss:** SignalMSE + AmpWeightedMSE + BaselinePenalty
**Best val loss: 0.052952 @ ep67**

Stopped manually — val loss flat at 0.052–0.053 for 25+ epochs with no improvement trend.

### v7 — Job 1310 — STOPPED (ep 30, L1 suppresses peaks)

**Hardware:** Lenovo6, Quadro M4000 8 GB
**Config:** BS=16, LR=1e-4, patience=15 (BS=16 — RoActivation 3× memory overhead)
**Init:** random
**Loss:** SignalMAE (L1)
**Architecture:** ComplexUNetV7 (paper arch, corrected)
**Best val loss: 0.081819 @ ep30**

Stopped: L1 loss fundamentally suppresses R-peaks. Peaks are ~8% of signal — L1 minimises median error so the model ignores them. At ep30 peaks were still suppressed vs GT with no improvement trend. Replaced by v8 (same arch + MSE).

### v8 — Job 1319 — DONE (ep 41, 2026-04-07)

**Hardware:** Lenovo6, Quadro M4000 8 GB
**Config:** BS=16, LR=1e-4, WD=1e-5, patience=15, max 200 epochs
**Init:** random (no warm start)
**Loss:** SignalMSE — same as v1, paper architecture
**Architecture:** ComplexUNetV7 (1.87M) — paper arch + MSE
**Best val loss: 0.046987 @ ep41**

| Epoch | Val loss | Notes |
|---|---|---|
| 1 | 0.093784 | — |
| 16 | 0.049408 | — |
| 41 | **0.046987** | **BEST** |
| 56 | 0.050296 | early stop (patience=15) |

**Inference Test_DB (ep41):** slab vizual — rețeaua nu reconstruiește bine amplitudinile R-peak. Plots: `plots_2026/testdb/Sem1-11_v8_ep41.png`.

**Root cause:** predictie directă (fără mască) + loss singur (SignalMSE fără ComplexMSE) → model nu captează structura spectrala.

---

### v9 — Job 1346 — **DONE** (ep 138, 2026-04-26)

**Hardware:** Lenovo6, Quadro M4000 8 GB
**Config:** BS=16, LR=1e-4, WD=1e-5, patience=20, max 300 epochs
**Init:** random
**Loss:** SignalMSE + ComplexMSE
**Architecture:** ComplexUNetV9 (1.87M) = v7 + soft mask `sigmoid(logits) × mixture_spec`
**Finished:** 2026-04-26 22:59:03

| Epocă | Train | Val | Note |
|-------|-------|-----|------|
| 1 | 2.2227 | 1.8970 | start |
| 20 | 0.9669 | 0.9664 | |
| 97 | — | 0.931426 | first plateau |
| 138 | — | **0.925510** | **BEST — early stop** |

**Verdict:** ComplexMSE domină ~10-20× SignalMSE la convergență → amplitudini R-peak suprimate vizual față de GT. Val loss 0.9255 nu este comparabil cu v1/v11 (scală diferită — include ComplexMSE).

### v1 resume — Job 1357 — **DONE** (ep 478, 2026-04-27)

**Hardware:** Lenovo2, Quadro M4000 8 GB
**Config:** BS=32, LR=1e-4 → 5e-5 (ReduceLROnPlateau), WD=1e-5, patience=20, max 478 epochs
**Init:** checkpoint v1 ep199
**Loss:** SignalMSE only
**Finished:** 2026-04-27 04:33:59

| Epocă | Val | Note |
|-------|-----|------|
| 199 (original) | 0.014745 | v1 checkpoint |
| 287 | 0.014058 | îmbunătățire după resume |
| 389 | 0.013844 | — |
| 458 | **0.013701** | **BEST checkpoint** |
| 478 | — | DONE (early stop) |

**Verdict:** cel mai bun model Part 1 la 1 kHz. Amplitudini ~90-95% GT, formă QRS curată. Test_DB: PRD avg 35.5%, QRS F1=1.000 pe Sem1/2/3.

### v10 — Job 1349 — DONE (ep 55, 2026-04-15)

**Hardware:** Lenovo2, Quadro M4000 8 GB
**Config:** BS=32, LR=1e-4, WD=1e-5, patience=20, max 300 epochs
**Init:** random
**Loss:** SignalMSE + ComplexMSE + **3×PeakMSE(fqrs)**
**Architecture:** ComplexUNetV10 (0.59M) = v1-style (shared weights, LeakyReLU, add-skip, DiagMag, soft mask)
**Dataset:** `MovementECGDatasetV10` — returnează `(mix_spec, fecg_spec, fecg_time, peak_mask)`
**Best val loss: 2.378023 @ ep35**

**PeakMSE:**
```python
def peak_mse(pred_time, fecg_time, peak_mask):
    # peak_mask: (B, T) float32 — 1 la fqrs ±30 sample, 0 în rest
    mask = peak_mask.unsqueeze(1)                          # (B, 1, T)
    n    = mask.sum() * pred_time.shape[1] + 1e-8
    return ((pred_time - fecg_time) ** 2 * mask).sum() / n

loss = signal_mse + complex_mse + 3.0 * peak_mse
```

**Progress:**

| Epocă | Train (sig/cpl/pk) | Val | Note |
|-------|---------------------|-----|------|
| 35 | — | **2.378023** | **BEST** |
| 55 | 1.52 (0.059/0.927/0.180) | 2.403 (0.079/1.626/0.233) | early stop |

**Lecție:** ComplexMSE=1.63 >> SignalMSE=0.08 la convergență — ComplexMSE domină identic cu v9. PeakMSE(fqrs) nu poate compensa. Soluție: v11 elimină ComplexMSE complet.

### v11 — Job 1359 — **DONE** (ep 93, 2026-05-03)

**Hardware:** Lenovo6, Quadro M4000 8 GB
**Config:** BS=16, LR=1e-4 → 5e-5 → 2.5e-5, WD=1e-5, patience=20, max 300 epochs
**Init:** random
**Loss:** SignalMSE + **3×PeakMSE(fqrs ±30 samples)** — fără ComplexMSE
**Architecture:** ComplexUNetV11 (1.87M) = v9 arch + direct prediction (fără sigmoid mask)
**Finished:** 2026-05-03 04:58:31

```python
loss = signal_mse + 3.0 * peak_mse   # fara ComplexMSE
```

| Epocă | Train (sig/pk) | Val (sig/pk) | Note |
|-------|----------------|--------------|------|
| 1 | — | ~0.99 | start |
| 49 | — | 0.446454 | prim best checkpoint |
| 73 | — | **0.432913** | **BEST checkpoint** |
| 93 | 0.466 | 0.4607 | early stop 20/20 |

**Verdict:** direct prediction + SignalMSE+PeakMSE convergă dar val loss ~0.43 rămâne ridicat față de v1 (0.013). Cauza: loss compus pe scală diferită față de v1. Inferența vizuală pe Test_DB inclusă în Sem*_all_models_grid.

### v12 — Job 1371 — **DONE** (ep 51, 2026-05-08)

**Hardware:** Lenovo2, Quadro M4000 8 GB
**Config:** BS=16, LR=1e-4, WD=1e-5, patience=20, max 300 epochs
**Init:** random
**Loss:** SignalMSE + **3×QRSwideMSE(fqrs ±100 samples)** — fără ComplexMSE
**Architecture:** ComplexUNetV12 (1.87M) = v9 arch + gain mask `1.5 × sigmoid(logits) × mixture_spec`
**Dataset:** `MovementECGDatasetV12` — peak_mask cu DILATION=100 (vs 30 în v10/v11)

```python
# Forward:
mask = 1.5 * sigmoid(logits)   # ∈ [0, 1.5] — poate depasi mixture magnitude
return mask * mixture_spec

# Loss:
loss = signal_mse + 3.0 * qrs_wide_mse   # fara ComplexMSE
# qrs_wide_mse: peak_mask ±100ms acoperă PQRS complet (50.2% din semnal)
```

**Motivație față de v11:**
- Gain mask 1.5× recuperează energia la bin-urile STFT cu interferență destructivă fECG/mECG (mask > 1 necesar, sigmoid pur nu poate ajunge acolo)
- QRSwideMSE ±100ms (vs ±30ms) supervizează complexul PQRS complet, nu doar R-peak-ul

**Progress:**

| Epocă | Train | Val | Note |
|-------|-------|-----|------|
| 1 | — | — | start |
| 49 | — | **0.220880** | **BEST checkpoint** |
| 51 | — | 0.221809 | — |
| 52+ | — | — | early stop (patience=20) |

**Verdict:** DONE ep51. Gain mask 1.5× ajută recuperarea energiei la bin-uri cu interferență destructivă, dar gain mask oversmooth-ează unele canale. PRD~68% pe Test_DB, QRS F1=0.933.

---

### v13 — Job 1378 — **STOPPED** (ep 84, 2026-05-10)

**Hardware:** Lenovo6, Quadro M4000 8 GB
**Config:** BS=16, LR=1e-4, WD=1e-5, patience=20, max 300 epochs
**Init:** random
**Loss:** SignalMSE only (fără PeakMSE)
**Architecture:** ComplexUNetV13 (1.87M) = v9 arch + **instance normalization** pe magnitudine
**Dataset:** `MovementECGDataset` — 1000 Hz, 128×400

```python
# Instance norm în forward (față de v9 care nu are):
scale = (x.real.pow(2) + x.imag.pow(2)).mean(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-8)
x = x / scale
# ... rețea ...
mask = sigmoid(logits_real) + j * sigmoid(logits_imag)
return mask * x_in   # fara denormalizare
```

**Motivație:** v13 = v9 + instance norm — testează dacă normalizarea per-sample ajută convergența față de v9 (care are val loss pe scală 0.93 datorită ComplexMSE). Față de paper (v15): fără cross-mixing, fără denormalizare, FS=1000Hz.

**Progress:**

| Epocă | Train | Val | Note |
|-------|-------|-----|------|
| 1 | — | — | start |
| 2 | 0.073683 | 0.072829 | — |
| 20 | — | ~0.064 | convergență lentă |
| 84 | — | **0.060014** | **BEST — STOPPED** |

**Verdict:** STOPPED ep84. Platou cauzat de (1) instance norm pe magnitudine aplicată înainte de mask → mask scale-dependent; (2) bottleneck 128ch vs 512ch în v15. Val loss 0.060 vs v15 0.031 — vădita inferioritate față de arhitectura 500 Hz paper-fidel.

---

### v14 — Job 1382 — PD (niciodată rulat)

**Hardware:** Lenovo2, Quadro M4000 8 GB (în așteptare — Resources)
**Config:** BS=8, LR=1e-4, WD=1e-5, patience=20, max 300 epochs
**Init:** random
**Loss:** SignalMSE only
**Architecture:** ComplexUNetV14 (7.09M) = v9-style MAI MARE: 6→32→64→128→256, bottleneck 512
**Dataset:** `MovementECGDatasetV14` — 500 Hz (subsampling simplu [::2]), 128×128

**Diferențe față de paper (v15):**
- `ComplexConvLayer`: `r + j*i` (fără cross-mixing)
- `Diag`: rotație de fază `e^{iβ}` (nu scalare exp separată Re/Im)
- Normalizare: magnitudine scale (nu mean+std cu denormalizare)
- Dataset: `[::2]` (nu `scipy.signal.decimate` cu anti-aliasing)
- Head: `conv_out(32→6, k=1)` (fără layer intermediar 32→16)

**Scop:** experiment controlat — v14 vs v15 izolează efectul cross-mixing + normalizare exactă din paper.

---

### v15 — Job 1381 — **RUNNING** (ep 95+, 2026-05-18)

**Hardware:** Lenovo6, Quadro M4000 8 GB
**Config:** BS=8, LR=1e-4, WD=1e-5, patience=20, max 300 epochs, ReduceLROnPlateau(factor=0.5, patience=8)
**Init:** random
**Loss:** SignalMSE only
**Architecture:** ComplexUNetV15 (7.13M) = **paper fidel** (copiat din `archive/complex_network_paper.py`)
**Dataset:** `MovementECGDatasetV15` — 500 Hz (`scipy.signal.decimate`), 128×128 (natural STFT, drop last freq bin)

**Fidel față de paper:**
- `ComplexConvLayer` cu cross-mixing: `(r - i) + j*(r + i)`
- `Diag` cu `exp(β)` separat pe real/imag
- Normalizare mean+std cu denormalizare completă în forward
- Head: conv2(32→16) + conv3(16→6, k=1)
- `scipy.signal.decimate` cu anti-aliasing

**Progress:**

| Epocă | Train | Val | Note |
|-------|-------|-----|------|
| 1 | — | — | start |
| 2 | 0.053391 | 0.049102 | — |
| 20 | — | ~0.038 | convergență stabilă |
| 92 | — | **0.031271** | **BEST checkpoint** |
| 95 | — | în curs | RUNNING |

**Verdict curent:** cel mai bun model overall. Test_DB: PRD avg 59.6%, QRS F1=1.000 pe Sem1/2/3 (ep92 checkpoint). Continuă antrenarea.

---

### v16 — Job 1389 — **RUNNING** (ep 17+, 2026-05-18)

**Hardware:** Lenovo2, Quadro M4000 8 GB
**Config:** BS=8, LR=1e-4, WD=1e-5, patience=20, max 300 epochs, ReduceLROnPlateau(factor=0.5, patience=8)
**Init:** random
**Loss:** SignalMSE only
**Architecture:** ComplexAttentionUNet (7.13M) = v15 + 2 Attention Gates (AG1 la 64×64, AG2 la 32×32)
**Dataset:** `MovementECGDatasetV15` — 500 Hz, 128×128

**Față de v15:**
- AG1 între decoder 64×64 și encoder skip 64×64
- AG2 între decoder 32×32 și encoder skip 32×32
- Fiecare AG: g (gating signal from decoder) + x (encoder) → sigmoid → reweight x

**Progress:**

| Epocă | Train | Val | Note |
|-------|-------|-----|------|
| 1 | — | — | start |
| 15 | — | **0.032987** | **BEST checkpoint** |
| 17 | — | în curs | RUNNING |

**Verdict curent:** val loss apropiată de v15 la ep15 (0.033 vs 0.031 la ep92 v15). Modelul continuă să antreneze — attention gates pot ajuta la SNR scăzut.

---

### v17 — Job 1391 — **RUNNING** (ep 4+, 2026-05-18)

**Hardware:** Lenovo6, Quadro M4000 8 GB
**Config:** BS=8, LR=1e-4, WD=1e-5, patience=20, max 300 epochs, ReduceLROnPlateau(factor=0.5, patience=8)
**Init:** random
**Loss:** SignalMSE + **L_att** (attention supervision)
**Architecture:** ComplexAttentionUNet (7.13M) = v16 (identic)
**Dataset:** `MovementECGDatasetV17` — 500 Hz, 128×128 + fqrs mask downsampled 128→64

**L_att:**
```python
# fqrs_target_mask: (B, 1, 64, 64) — 1 la frame-uri ±3T (±80ms @ 500Hz) în jurul fQRS
L_att = F.mse_loss(alpha_AG1, fqrs_target_mask)
loss = signal_mse + L_att
```

**Motivație:** AG1 fără supervizare poate converige la pattern-uri greșite în primele epoci. L_att forțează attention map-ul să activeze pe complexele QRS fetale de la primele epoci → reprezentări mai utile din encoder.

**Progress:**

| Epocă | Train (sig/L_att/total) | Val total | Note |
|-------|------------------------|-----------|------|
| 1 | — | — | start |
| 2 | — | — | — |
| 3 | — | **0.046913** | **BEST total** |
| 4 | — | în curs | RUNNING |

**Verdict curent:** prea puține epoci pentru concluzii. Val loss total ~0.047 la ep3 (include L_att ~0.046 + sig_mse ~0.001). Se așteaptă L_att să scadă odată cu antrenamentul.

---

## 7. Inference Results (overlays)

Test file: `fecgsyn_Long_time_segment_01_snr6dB.mat`, window 10–14 s, channels 1–3.
Plots saved in `plots_2026/`.

| Model | Epoch | Peak amplitude | Baseline | Plot |
|---|---|---|---|---|
| v1 | 199 | ~100% of GT | clean | `inference_overlay_ep199.png` |
| v2 | 99 | ~85–90% of GT | more noise than v1 | `inference_overlay_v2_ep99.png` |
| v3 | 101 | ~50–60% of GT | clean (suppression artefact) | `inference_overlay_v3_ep101.png` |
| v4 | 36 | — | — | cancelled |
| v5 | 81 | ~95% of GT | noisier baseline than v1 (still converging) | `inference_overlay_v5_ep81_val42.png` |

### Test_DB Inference (2026-03-24)

v1 (ep199) and v5 (ep81) run on all 11 files from `/shared_storage/stan.edward/fECG_mvm_DB/Test_DB/`.
Plots saved to `plots_2026/testdb/` as `{Sem1..Sem11}_{v1,v5}.png`.
Window: 10–14 s, channels 1–3, normalised amplitude.

**Observations:**
- Sem1, Sem2, Sem4, Sem9: v1 excellent — near-perfect peak tracking
- Sem5: harder case — low amplitude fECG, model recovers rough shape but less precise

---

## 8. File Reference

### Active scripts (src_2026/)

| File | Role |
|---|---|
| `complex_network.py` | ComplexUNet 0.59M — v1/v5/v6 architecture |
| `complex_network_v7.py` | ComplexUNetV7 1.87M — paper architecture (v7) |
| `complex_network_v9.py` | ComplexUNetV9 1.87M — v7 + soft mask output (v9) |
| `complex_network_v10.py` | ComplexUNetV10 0.59M — v1-style + soft mask (v10) |
| `complex_network_v11.py` | ComplexUNetV11 1.87M — v9 arch + direct prediction (v11) |
| `complex_network_v12.py` | ComplexUNetV12 1.87M — v9 arch + gain mask 1.5×sigmoid (v12) |
| `complex_network_v13.py` | ComplexUNetV13 1.87M — v9 arch + instance norm magnitudine (v13) |
| `complex_network_v14.py` | ComplexUNetV14 7.09M — v9-style larger, 500Hz, fără cross-mixing (v14) |
| `complex_network_v15.py` | ComplexUNetV15 7.13M — paper fidel: cross-mixing, exp Diag, mean+std norm (v15) |
| `complex_network_v16.py` | ComplexAttentionUNet 7.13M — v15 + AG1(64×64) + AG2(32×32) attention gates (v16) |
| `complex_network_v17.py` | ComplexAttentionUNet 7.13M — identic v16 (arhitectura), supervizare externă (v17) |
| `movement_dataset.py` | MovementECGDataset — `(mix_spec, fecg_spec, fecg_time)` |
| `movement_dataset_v10.py` | MovementECGDatasetV10 — `(mix_spec, fecg_spec, fecg_time, peak_mask)` DILATION=30 |
| `movement_dataset_v12.py` | MovementECGDatasetV12 — idem V10 dar DILATION=100 (±100ms, 50.2% coverage) |
| `movement_dataset_v14.py` | MovementECGDatasetV14 — 500Hz subsampling [::2], 128×128, fără anti-aliasing |
| `movement_dataset_v15.py` | MovementECGDatasetV15 — 500Hz decimate, 128×128 natural STFT, drop last freq bin (v15/v16) |
| `movement_dataset_v17.py` | MovementECGDatasetV17 — idem v15 + fqrs_target_mask downsampled 128→64 pentru L_att (v17) |
| `train_movement.py` | v1 training (SignalMSE, AdamW) |
| `train_movement_v5.py` | v5 training (SignalMSE+3×AmpW, from scratch) |
| `train_movement_v6.py` | v6 training (SignalMSE+AmpW+Baseline, warm start v1) |
| `train_movement_v7.py` | v7 training (SignalMAE/L1, Adam, paper arch) |
| `train_movement_v9.py` | v9 training (SignalMSE+ComplexMSE, soft mask, 1.87M) |
| `train_movement_v10.py` | v10 training (SignalMSE+ComplexMSE+3×PeakMSE(fqrs), 0.59M) |
| `train_movement_v11.py` | v11 training (SignalMSE+3×PeakMSE ±30ms, direct, 1.87M) |
| `train_movement_v12.py` | v12 training (SignalMSE+3×QRSwideMSE ±100ms, gain mask 1.5×, 1.87M) |
| `train_movement_v13.py` | v13 training (SignalMSE only, instance norm, 1.87M, 1000Hz) |
| `train_movement_v14.py` | v14 training (SignalMSE only, 7.09M, 500Hz, v9-style no cross-mix) |
| `train_movement_v15.py` | v15 training (SignalMSE only, 7.13M, 500Hz, paper fidel) |
| `train_movement_v16.py` | v16 training (SignalMSE only, ComplexAttentionUNet, AG1+AG2) |
| `train_movement_v17.py` | v17 training (SignalMSE + L_att, attention supervision pe fQRS mask) |
| `infer_save_results.py` | Full inference Sem1-11 × 5 modele → results_fECG_extraction/ (.mat, fs=1000) |
| `run_infer_save_results.sh` | SLURM launcher pentru infer_save_results.py |
| `train_movement_v1_resume.py` | v1 resume de la ep200 → ep400 |
| `run_training.sh` | SLURM launcher v1 |
| `run_training_v5.sh` | SLURM launcher v5 |
| `run_training_v6.sh` | SLURM launcher v6 |
| `run_training_v7.sh` | SLURM launcher v7 |
| `run_training_v9.sh` | SLURM launcher v9 (Lenovo6) |
| `run_training_v10.sh` | SLURM launcher v10 (Lenovo2) |
| `run_training_v11.sh` | SLURM launcher v11 (Lenovo6) |
| `run_training_v12.sh` | SLURM launcher v12 (Lenovo2) |
| `run_training_v13.sh` | SLURM launcher v13 (Lenovo6) |
| `run_training_v14.sh` | SLURM launcher v14 (Lenovo2, PD) |
| `run_training_v15.sh` | SLURM launcher v15 (Lenovo6) |
| `run_training_v16.sh` | SLURM launcher v16 (Lenovo2, job 1389) |
| `run_training_v17.sh` | SLURM launcher v17 (Lenovo6, job 1391) |
| `run_training_v1_resume.sh` | SLURM launcher v1 resume (Lenovo2) |
| `infer_overlay.py` | Overlay inference v1 (training data) |
| `infer_overlay_v5.py` | Overlay inference v5 (val set) |
| `infer_testdb.py` | Test_DB inference v1+v5 |
| `infer_testdb_v5v6.py` | Test_DB inference v5+v6 |
| `infer_testdb_v7.py` | Test_DB inference v7 |
| `infer_v6v7_sem125.py` | Quick inference v6+v7 on Sem1/Sem2/Sem5 |
| `generate_model_report.py` | Generate model summary PDF |
| `compute_baseline.py` | Baseline (no-model) signal metrics |
| `snr_check.py` | SNR analysis across dataset |
| `infer_snr_comparison.py` | Inference SNR comparison across models |
| `inspect_movement_data.py` | Data structure inspection utility |
| `infer_real_signals.py` | Inference on real (non-synthetic) signals |

### Archived scripts (src_2026/archive/)

| File | Why archived |
|---|---|
| `train_movement_v2.py` | 500 Hz / paper arch experiment — superseded |
| `train_movement_v3.py` | PeakMSE+ComplexMSE — caused amplitude suppression |
| `train_movement_v4.py` | Warm start different loss — caused train/val gap |
| `train_movement_v1_resume.py` | One-time resume script for v1 — job complete |
| `complex_network_paper.py` | First attempt at paper arch — superseded by v7 |
| `movement_dataset_paper.py` | 500 Hz dataset for v2 — superseded |
| `infer_overlay_v2/v3/v4.py` | Overlay inference for archived models |
| `run_training_v2/v3/v4.sh` | SLURM launchers for archived models |
| `run_training_v1_resume.sh` | One-time v1 resume launcher |
| `test_gpu_istft.sh` | One-time GPU iSTFT benchmark |

### Models & Logs (gitignored — local only)

| Path | Contents |
|---|---|
| `models/movement_CUNet_128x400_composed.pth` | v1 best checkpoint (ep199) |
| `models/movement_CUNet_128x400_ampw_scratch.pth` | v5 best checkpoint |
| `models/movement_CUNet_128x400_v6_baseline.pth` | v6 best checkpoint |
| `models/movement_CUNet_v7_paper_direct.pth` | v7 best checkpoint |
| `logs/movement_CUNet_128x400_composed.log` | v1 training log |
| `logs/movement_CUNet_128x400_ampw_scratch.log` | v5 training log |
| `logs/movement_CUNet_128x400_v6_baseline.log` | v6 training log |
| `logs/movement_CUNet_v7_paper_direct.log` | v7 training log |
| `src/loss.py` | Paper's loss functions (reference) |
| `src/train.py` | Paper's original training (reference) |

---

## 9. Part 2 — Movement Classification

### 9.1 Task

Given ground-truth fECG (6-channel, 600s recording), classify the type of fetal movement at each time step.

Label source: `out.category_mask` — (600000, 1) uint8, values 0-3:
- 0 = stationary
- 1 = linear movement
- 2 = helical movement
- 3 = screw movement

Class distribution (across dataset): 0=53%, 1=17%, 2=13.5%, 3=16.5%
Windows homogeneity (4s windows): 87.4% are single-class.

### 9.2 Key Findings

**Finding 1 — File-level split is mandatory.**
Window-level random split causes data leakage: windows from the same recording share ECG morphology, so the model "recognises" recordings it saw at training. Val loss is artificially optimistic. Fix: all windows from a file go entirely to train or val.

clf_v1 with window-level split: train 95%, val 62% at ep19 → massive overfitting.
clf_v1 with file-level split: train 93%, val 62% at ep4 (best), then train→95%, val→62% → still overfits.

**Finding 2 — Raw signal is the wrong input for classification.**
Even with file-level split, the ResNet on raw fECG (4s/8s windows) overfits. The model memorises file-specific ECG morphology instead of learning movement-discriminative patterns. There is insufficient generalizable information in a single raw window.

**Finding 3 — Beat-level amplitude features are the correct representation.**
Movement type manifests as a pattern of change in R-peak amplitudes across channels over successive heartbeats:
- Linear: amplitudes shift monotonically across beats
- Helical: amplitudes oscillate
- Screw: amplitudes rotate across channels
- Stationary: amplitudes constant

Extracting the 6-channel amplitude at each R-peak (from `fqrs` in the .mat file) gives a compact sequence that directly encodes movement type. A 30s window yields ~40 beats → (40, 6) matrix.

### 9.3 Classification Experiments — rezultate complete

| Model | Input | Architecture | Val acc | Note |
|---|---|---|---|---|
| clf_v1 | raw fECG 4s | ResNet1D (0.97M) | — | stopped ep19, severe overfit |
| clf_v2 | raw fECG 8s | ResNet1D (0.97M) | — | cancelled |
| clf_v3 | beat amps 30s (~40×6) | ResNet1D (244K) | ~86% | |
| clf_v4 | beat amps 30s (~40×6) | BeatTransformer (205K) | ~84% | |
| **clf_v9** | **36 spectral features per faza** | **RF** | **92.2%** | **fft_freq dominant** |
| clf_v9 | 36 spectral features | MLP | 89.3% | |
| clf_v10 | beat amps faze complete (6,200) | ResNet1D | 90.9% | overfit train 99% |
| clf_v11 | beat amps faze complete | Transformer | 88.3% | fara overfit |
| clf_v12 | 78 features (36 + RR + QRS PCA + envelope) | RF | 91.2% | mai slab — features noi = zgomot |
| clf_v13 | QRS morphology (150,6,50) | CNN+Transformer | 76.2% | forma beat izolat nu e discriminativa |
| clf_v14 | 71 features (36 + cross-corr inter-canal) | RF | **92.55%** | **best RF solo** |
| clf_v14 | 71 features | MLP / Transformer | 90.4% / 89.7% | |
| clf_v15 | 72 features (36 + phase_diff + coherence) | RF | 92.18% | mai slab decat v14 |
| clf_v15 | 72 features | Transformer | 89.45% | |
| clf_v16 | Hilbert envelope (6,200) | EnvResNet (500K) | 88.93% | overfit sever train 99.6% |
| clf_v16 | Hilbert envelope (6,200) | EnvTransformer | 77.18% | |
| clf_v17 | Hilbert envelope (6,200) | SmallEnvResNet (60K) | 90.24% | **no overfit** (augmentare agresiva) |
| **Ensemble** | RF(w=0.75) + ResNet v17(w=0.25) | — | **93.07%** | **BEST OVERALL** |

### 9.4 Key Findings Part 2

**Finding 1 — File-level split obligatoriu** (window-level → data leakage)

**Finding 2 — fft_freq este feature-ul dominant** (top 5-7 importanță în toate RF experiments). Frecvența de oscilație a amplitudinii beat-urilor de-a lungul fazei discriminează tipul mișcării.

**Finding 3 — Faze complete > ferestre fixe.** Mișcarea se vede pe toată durata fazei (12-83s). Ferestre fixe de 5-30s → label noise sau info insuficientă.

**Finding 4 — Features noi nu aduc informație peste fft_freq.** Cross-correlații, phase differences inter-canal, coherence, QRS morphology — zero features noi în top 15 importanță RF.

**Finding 5 — Singura confuzie reală: Linear ↔ Helical** (120/~170 erori totale). Stationary și Screw: aproape perfecte. Limitare în dataset, nu în model.

**Finding 6 — Soft mask + model mic elimină overfit-ul (v17).** SmallEnvResNet 60K + augmentare (noise σ=0.05, shift ±20, scale [0.8,1.2], channel dropout p=0.2) → train 91.8% ≈ val 90.24%.

**Finding 7 — Ensemble RF + ResNet17 → +0.5% față de RF solo.** Combinarea reprezentărilor diferite (spectral features vs Hilbert envelope) ajută marginal. Nu rezolvă Linear↔Helical complet.

### Confusion matrix Ensemble (w_RF=0.75, val 2026-04-08)

```
             Stationary  Linear  Helical  Screw
Stationary        751       1       0       0
Linear              5     356      40       0
Helical             5      80     266       1
Screw               0       0       0     401
```

### 9.5 Dataset stats (phase-level, PhaseDataset)

- 12,477 faze complete din 655 fișiere, durate 10-83s, median ~25s
- Clase: [4967 Stat, 2563 Lin, 2425 Hel, 2522 Screw]
- File split: ~557 train / ~98 val (VAL_FRAC=0.15)

### 9.6 Classification File Reference

| File | Role |
|---|---|
| `phase_dataset.py` | PhaseDataset — 36 spectral features per faza (BEST pentru RF) |
| `beat_phase_dataset.py` | BeatPhaseDataset — (6, 200) beat amps paddate |
| `qrs_dataset.py` | QRSDataset — (150, 6, 50) morfologie QRS completa |
| `clf_dataset.py` | ClfDatasetV15 — 72 features (spectral + phase_diff + coherence) |
| `envelope_dataset.py` | EnvelopeDataset — Hilbert envelope (6, 200) per faza |
| `resnet1d.py` | ResNet1D architecture |
| `train_clf_v9.py` → `train_clf_v17.py` | Training scripts v9-v17 |
| `eval_ensemble.py` | Ensemble RF + SmallEnvResNet sweep + confusion matrix |
| `run_clf_v9.sh` → `run_clf_v17.sh` | SLURM launchers |
| `models/ensemble_rf_resnet17_results.json` | Rezultate ensemble (sweep + CM) |

---

## 10. Part 2b — Binary Movement Detection

### 10.1 Motivație

Detecție binară mișcare/fără mișcare din semnal fECG continuu, **fără informație despre granițele segmentelor GT**. Modelul prezice per-sample pe semnal brut → evită segmentation tax din abordarea RF pe ferestre aliniate la GT.

### 10.2 Arhitectură — PrecisionResUNet (adaptată după Edward Bîndilă)

- **Input:** 8 canale × 8000 samples (8s @ 1000Hz)
  - 0–5: fECG z-normat per fișier
  - 6: A_QRS — amplitudine QRS per bătaie interpolată (Rooijakkers et al. 2016)
  - 7: Baseline wander (moving avg 1s)
- **Encoder:** 4× ResidualBlock (Conv1d k=15, GroupNorm, skip connection)
- **Bottleneck:** ASPP (dilații 1,2,4,8 + global avg pool) + TransformerBottleneck (2 layers, 4 heads)
- **Decoder:** 4× upsample + ResidualBlock cu skip connections
- **Ieșire:** (B, 1, T) logits → sigmoid → mască binară per sample
- **Parametri:** 5.99M
- **Loss:** BCE + Dice (pos_weight=1.5, ~60% no-mov / ~40% mov)

### 10.3 Date și preprocesare

- **Train:** `Long_time_intervals/` — 654 fișiere (1 corupt), file-level split 85/15
  - 557 train / 98 val, stride=4s → 82.844 ferestre train, 7.350 val
- **Test:** `Short_time_intervals/` — 25 fișiere (fișiere separate, nevăzute la antrenament)
- **A_QRS:** R_val − mean(Q_val, S_val) per bătaie, z-normat, interpolat la semnal dens
- **Cache:** `.npy` per fișier în `data/binary_npy/` (preprocesare one-time ~35min)

### 10.4 Rezultate binary_v1 (@ epoch 12/100)

| Fișier test | SNR | mov% | F1 | Accuracy |
|---|---|---|---|---|
| var1_01 | 3dB | 32% | 0.920 | 0.944 |
| var1_01 | 6dB | 41% | 0.922 | 0.933 |
| var1_02 | 3dB | 32% | 0.859 | 0.897 |
| var1_02 | 6dB | 63% | 0.937 | 0.916 |
| var1_03 | 3dB | 53% | 0.933 | 0.924 |
| var1_03 | 6dB | 35% | 0.953 | 0.965 |
| **Medie** | | | **0.921** | **0.930** |

Plot inferență: `plots/binary_v1_inference_summary.png`

### 10.5 File Reference

| File | Rol |
|---|---|
| `train_binary_v1.py` | Script antrenament + arhitectură PrecisionResUNet |
| `run_binary_v1.sh` | SLURM launcher (job 1386, Lenovo2) |
| `infer_binary_v1.py` | Inferență + plot pe Short_time_intervals |
| `analyze_mr_window.py` | Analiză feature M_R (beat-to-beat vs fereastră) |
| `models/binary_v1_best.pth` | Best model (val_loss=0.2770 @ ep12) |
| `models/binary_v1_history.json` | Istoric antrenament |
| `plots/binary_v1_inference_summary.png` | Plot inferență test set |
