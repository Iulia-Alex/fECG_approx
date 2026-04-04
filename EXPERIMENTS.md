# fECG Extraction from Movement ECG — Experiment Log

**Last updated:** 2026-03-29
**Active jobs:** 1294 (v5 ep154), 1311 (v6 ep42), 1310 (v7 ep15) — v1 DONE

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

### 3.2 Loss Function: ComplexMSE → SignalMSE ⭐ most impactful

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

| | **v1** | **v2** | **v3** ❌ | **v4** ❌ | **v5** | **v6** | **v7** |
|---|---|---|---|---|---|---|---|
| **Script** | `train_movement.py` | `archive/train_movement_v2.py` | `archive/train_movement_v3.py` | `archive/train_movement_v4.py` | `train_movement_v5.py` | `train_movement_v6.py` | `train_movement_v7.py` |
| **Architecture** | ComplexUNet | Paper-style ComplexUNet | ComplexUNet (=v1) | ComplexUNet (=v1) | ComplexUNet (=v1) | ComplexUNet (=v1) | **ComplexUNetV7 (paper)** |
| **Params** | 0.59 M | 7.13 M | 0.59 M | 0.59 M | 0.59 M | 0.59 M | **1.87 M** |
| **Conv** | Shared Re/Im | Separate (cross-mix) | Shared | Shared | Shared | Shared | **Split Re/Im (no cross-mix)** |
| **Activation** | LeakyReLU(0.2) | RoActivation | LeakyReLU | LeakyReLU | LeakyReLU | LeakyReLU | **RoActivation** |
| **Skip conn.** | Addition | Concat | Addition | Addition | Addition | Addition | **Concatenation** |
| **Diagonal** | Mag. scaling exp(β) | Phase rot. e^{iβ} | Mag. scaling | Mag. scaling | Mag. scaling | Mag. scaling | **Phase rot. e^{iβ}** |
| **FS / Spec** | 1000 Hz / 128×400 | 500 Hz / 128×128 | 1000 / 128×400 | 1000 / 128×400 | 1000 / 128×400 | 1000 / 128×400 | 1000 / 128×400 |
| **Loss** | SignalMSE | SignalMSE | MSE+5×PeakMSE+0.1×CplxMSE | MSE+3×AmpW | MSE+3×AmpW | MSE+AmpW+Baseline | **SignalMAE (L1)** |
| **Optimizer** | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW | **Adam** |
| **Warm start** | No | No | No | v1 best | No | **v1 best** | No |
| **LR** | 1e-4 | 1e-4 | 1e-4 | 1e-5 | 1e-4 | 1e-4 | 1e-4 |
| **BS** | 32 | 32 | 32 | 32 | 32 | 32 | **16** |
| **Job** | 1276 | 1284 ✅ | 1287 ❌ | 1292 ❌ | 1294 | 1311 | 1310 |
| **Node** | Lenovo2 | Lenovo6 | Lenovo2 | Lenovo2 | Lenovo2 | Lenovo2 | Lenovo6 |
| **Best val loss** | **0.014745** @ ep199 ✅ | 0.03217 @ ep99 | — | 0.02319 @ ep36 | 0.023846 @ ep154 | 0.053158 @ ep42 | 0.085450 @ ep15 |
| **Status** | DONE | DONE | CANCELLED | CANCELLED | Running ep154 | Running ep42 | Running ep15 |

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

### v3 — SignalMSE + PeakMSE + ComplexMSE ❌ (cancelled ep 101)
```
loss = signal_mse + 5.0 * peak_mse + 0.1 * complex_mse
```
Where `peak_mse` uses a hard threshold mask (|signal| > 3×std) dilated ±40 samples.

**Problem:** even 0.1×ComplexMSE caused amplitude suppression (~50–60% GT). "Clean baseline" was actually the model predicting near-zero everywhere.

**Lesson learned:** ComplexMSE must never be combined with a loss that requires correct amplitudes — even at small weight.

### v4 — SignalMSE + AmpWeightedMSE, warm start from v1 ❌ (cancelled ep 37)
```
w = (|target| / max|target|)²
loss = signal_mse + 3.0 * mean((pred - target)² * w)
```
**Problem:** fine-tuning from v1 checkpoint with a *different* loss caused immediate train/val gap. V1's internal representations were optimised for pure SignalMSE; switching loss mid-training destabilised generalisation.

**Lesson learned:** changing loss at fine-tuning doesn't work — model needs to learn representations for the new objective from scratch.

### v5 — SignalMSE + AmpWeightedMSE, from scratch ← current
```
w = (|target| / max|target|)²          # soft: 0 on baseline, 1 on R-peaks
loss = signal_mse + 3.0 * mean((pred - target)² * w)
```
**Why from scratch (not warm start):** model will learn representations optimised for amplitude accuracy from the beginning, avoiding the landscape mismatch of v4.

**Expected result:** baseline as clean as v3 (model not penalised for baseline), peak amplitudes 1:1 with GT.

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

### v5 — Job 1294 — running (ep 154/200)

**Hardware:** Lenovo2, Quadro M4000 8 GB
**Config:** BS=32, LR=1e-4, WD=1e-5, patience=15, max 200 epochs
**Init:** random (no warm start)
**Loss:** SignalMSE + 3×AmpWeightedMSE

| Epoch | Val loss | Notes |
|---|---|---|
| 1 | ~0.25 | high due to AmpWeightedMSE×3 scale |
| 5 | 0.09359 | — |
| 20 | ~0.050 | — |
| 35 | 0.03546 | — |
| 75 | 0.028499 | — |
| 81 | 0.027981 | — |
| 154 | **0.023846** | best so far (as of 2026-03-29) |

### v6 — Job 1311 — running (ep 42/227)

**Hardware:** Lenovo2, Quadro M4000 8 GB
**Config:** BS=32, LR=1e-4, WD=1e-5, patience=15, max 200 epochs
**Init:** warm start from v1 best checkpoint
**Loss:** SignalMSE + AmpWeightedMSE + BaselinePenalty

| Epoch | Val loss | Notes |
|---|---|---|
| 1 | ~0.046 | warm start; loss landscape changes from v1's SignalMSE |
| 26 | 0.053158 | best so far |
| 42 | **0.053158** | best (as of 2026-03-29) — possibly plateauing |

### v7 — Job 1310 — running (ep 15/204)

**Hardware:** Lenovo6, Quadro M4000 8 GB
**Config:** BS=16, LR=1e-4, patience=15, max 200 epochs (BS=16 due to RoActivation 3× memory)
**Init:** random (no warm start)
**Loss:** SignalMAE (L1, paper's loss function)
**Architecture:** ComplexUNetV7 — paper architecture with corrected split-conv and phase-rotation diagonal

| Epoch | Val loss | Notes |
|---|---|---|
| 1 | ~0.095 | — |
| 8 | 0.087x | peaks visible, amplitudes still calibrating |
| 14 | 0.086466 | peaks better, Ch3 excellent |
| 15 | **0.085450** | best so far (as of 2026-03-29) |

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
| `movement_dataset.py` | MovementECGDataset — `(mix_spec, fecg_spec, fecg_time)` |
| `train_movement.py` | v1 training (SignalMSE, AdamW) |
| `train_movement_v5.py` | v5 training (SignalMSE+3×AmpW, from scratch) |
| `train_movement_v6.py` | v6 training (SignalMSE+AmpW+Baseline, warm start v1) |
| `train_movement_v7.py` | v7 training (SignalMAE/L1, Adam, paper arch) |
| `run_training.sh` | SLURM launcher v1 |
| `run_training_v5.sh` | SLURM launcher v5 |
| `run_training_v6.sh` | SLURM launcher v6 |
| `run_training_v7.sh` | SLURM launcher v7 |
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
