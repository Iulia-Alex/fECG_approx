# Fetal ECG Extraction and Movement Classification

Two-part pipeline: (1) extract fetal ECG from abdominal mixture, (2) classify fetal movement type from extracted fECG.

## Part 1 — fECG Extraction

- Input: 6-channel mixture (fECG + mECG + movement noise), spectrogram 128×400
- Output: 6-channel predicted fECG spectrogram
- Main challenge: mECG is 8 dB stronger than fECG; model must suppress it while preserving fetal R-peak amplitudes

### Extraction Models

| Model | Architecture | Output | Loss | Best val loss | Status |
|---|---|---|---|---|---|
| v1 | ComplexUNet (0.59M) | direct | SignalMSE | 0.014745 @ ep199 | DONE |
| v1 resume | ComplexUNet (0.59M) | direct | SignalMSE | **0.013701 @ ep458** | **DONE** (ep478) — best signal quality overall |
| v5 | ComplexUNet (0.59M) | direct | SignalMSE + 3×AmpWeightedMSE | 0.022473 @ ep200 | DONE |
| v6 | ComplexUNet (0.59M) | direct | SignalMSE + AmpW + BaselinePenalty | 0.052952 @ ep67 | DONE (stopped, plateau) |
| v7 | ComplexUNetV7 (1.87M) | direct | SignalMAE (L1) | 0.081819 @ ep30 | DONE (stopped, L1 suppresses peaks) |
| v8 | ComplexUNetV7 (1.87M) | direct | SignalMSE | 0.046987 @ ep41 | DONE — poor visually |
| v9 | ComplexUNetV9 (1.87M) | soft mask 1× | SignalMSE + ComplexMSE | 0.925510 @ ep138 | DONE — ComplexMSE suppresses amplitudes |
| v10 | ComplexUNetV10 (0.59M) | soft mask 1× | SignalMSE + ComplexMSE + 3×PeakMSE(fqrs) | 2.378023 @ ep35 | DONE (ep55, early stop) |
| v11 | ComplexUNetV11 (1.87M) | direct | SignalMSE + 3×PeakMSE(fqrs ±30ms) | 0.432913 @ ep73 | DONE (ep93, plateaued) |
| v12 | ComplexUNetV12 (1.87M) | gain mask 1.5× | SignalMSE + 3×QRSwideMSE(fqrs ±100ms) | 0.220880 @ ep49 | DONE (ep51) |
| v13 | ComplexUNetV13 (1.87M) | soft mask 1× | SignalMSE | 0.060014 @ ep84 | STOPPED (ep84, InstNorm plateau) |
| **v15** | **ComplexUNetV15 (7.13M)** | **soft mask** | **SignalMSE** | **0.031271 @ ep92** | **RUNNING ep95+, job 1381, Lenovo6** |
| **v16** | **ComplexAttentionUNet (7.13M)** | **soft mask** | **SignalMSE** | **0.032987 @ ep15** | **RUNNING ep17+, job 1389, Lenovo2** |
| **v17** | **ComplexAttentionUNet+att.sup (7.13M)** | **soft mask** | **SignalMSE + L_att** | **0.046913 total @ ep3** | **RUNNING ep4+, job 1391, Lenovo6** |

### Key Findings Part 1

**Soft mask > direct prediction for stability.** v8 (direct, 1.87M) fails to reconstruct R-peak amplitudes well. v1 was good because it uses `sigmoid(logits) × mixture_spec` — learns what to *keep*, not what to *generate*. But the mask constrains per-bin values to [0, 1] × mixture, which is a real limitation at STFT bins where fECG/mECG are in antiphase (mixture magnitude small → required mask > 1).

**ComplexMSE suppresses amplitudes.** At convergence, ComplexMSE is 10–20× larger than SignalMSE and dominates training. Model optimises spectral structure, not peak heights. Paper's original code used SignalMSE only — consistent with this finding. Never combine ComplexMSE with amplitude-accuracy objectives.

**L1 suppresses R-peaks, L2 forces them.** Peaks are sparse (~8% of signal) — L1 minimises median error and ignores peaks. L2 penalises large errors quadratically, forcing correct peak amplitudes.

**500 Hz (paper-faithful) pipeline outperforms 1 kHz.** v15 (7.13M, 500 Hz, paper arch) reaches best val loss 0.031271 @ ep92 vs v1 resume best 0.013701 (different loss scale) — and visually produces sharper QRS on Test_DB. Key: 128×128 natural STFT at 500 Hz vs resized 128×400 at 1 kHz.

**v13 (1 kHz + InstNorm) plateaus at ep84.** Instance normalisation on magnitude before the network + soft mask applied to unnormalized input creates a scale-dependent mask. Also, bottleneck 128ch vs 512ch in v15 limits capacity. Best val 0.060014.

**Attention supervision (v17) guides AG gates to fQRS regions.** L_att = MSE(alpha_AG1, fqrs_target_mask) forces the attention map at 64×64 resolution to activate around each fetal QRS complex from epoch 1. Expected to improve QRS shape recovery vs v16 (same arch, no supervision).

---

## Part 2 — Movement Classification

- Input: ground-truth fECG (6 channels) from .mat files
- Output: 4-class movement label (0=stationary, 1=linear, 2=helical, 3=screw)
- Label source: `category_mask` in .mat files
- Best result: **Ensemble RF(w=0.75) + SmallEnvResNet(w=0.25) → 93.07%**

### Classification Models

| Model | Input | Architecture | Val acc |
|---|---|---|---|
| clf_v9 | 36 spectral features per complete phase | RF | 92.2% |
| clf_v10 | beat amplitudes per phase (6, 200) | ResNet1D | 90.9% |
| clf_v11 | beat amplitudes per phase | Transformer | 88.3% |
| clf_v12 | 78 features (36 + RR + QRS PCA + envelope) | RF | 91.2% |
| clf_v13 | QRS morphology (150, 6, 50) | CNN+Transformer | 76.2% |
| clf_v14 | 71 features (36 + cross-corr inter-canal) | RF | **92.55%** |
| clf_v15 | 72 features (36 + phase_diff + coherence) | RF | 92.18% |
| clf_v16 | Hilbert envelope (6, 200) | EnvResNet (500K) | 88.93% |
| clf_v17 | Hilbert envelope (6, 200) | SmallEnvResNet (60K) | 90.24% |
| **Ensemble** | RF(w=0.75) + ResNet v17(w=0.25) | — | **93.07%** |

### Key Findings Part 2

**Complete phases are the right unit** (not fixed windows). Movement manifests over the full duration of a phase (12–83s, median ~25s). PhaseDataset extracts complete phases from `category_mask`.

**fft_freq is the dominant feature** in all RF experiments — top 5-7 features are always `fft_freq` per channel. The frequency of R-peak amplitude oscillation directly encodes movement type (stationary ≈ 0 Hz, linear = low freq monotone, helical = mid freq oscillation).

**New features consistently fail** to improve over fft_freq (cross-correlations, phase differences, inter-channel coherence, QRS morphology — zero new features in top 15 RF importance).

**Only remaining confusion: Linear ↔ Helical** (~120/170 total errors). Stationary and Screw are nearly perfect. The distinction is visually clear — 99% accuracy should theoretically be achievable.

**File-level split is mandatory.** Window-level split causes data leakage across the 12,477 phases from 655 files.

---

## Folder Structure

```
src_2026/                  active training and analysis scripts
src_2026/archive/          intermediate/obsolete experiments (kept for reference)
src/                       original paper reference code
data/                      training data (gitignored)
models/                    saved checkpoints (gitignored)
logs/                      training logs (gitignored)
plots_2026/                inference plots and PDF reports
plots_2026/testdb/         Test_DB overlays Sem1-11 per model version
results_fECG_extraction/   .mat files — full inference on Sem1-11 × 5 models (55 files, all fs=1000)
```

## Key Design Decisions

- **Soft mask (v9/v12/v15/v16/v17)**: `fECG_pred = gain × sigmoid(logits) × mixture_spec` — learns what to keep; v12 uses gain=1.5
- **500 Hz + 128×128 STFT (v15/v16/v17)**: paper-faithful pipeline; natural STFT resolution, no resizing needed
- **Attention gates (v16/v17)**: AG1 at 64×64, AG2 at 32×32 — focus decoder on fECG-relevant regions
- **Attention supervision (v17)**: L_att forces AG1 to activate on fQRS locations from the start of training
- **L2 over L1**: MSE forces correct peak amplitudes; MAE suppresses sparse peaks
- **128×400 spectrogram (1 kHz)**: 10 ms/frame → QRS spans 6–8 frames (recoverable)
- **Complete phases for classification**: fft_freq per phase captures movement type directly

## Environment

```bash
conda activate ecg        # /shared_storage/iulia.orvas/miniconda3/envs/ecg
# PyTorch + CUDA, numpy 1.26.4 (pinned — numpy 2.0 incompatible)
```

## Running

```bash
# Submit training job (from repo root)
sbatch src_2026/run_training_v17.sh

# Check training logs
tail -f logs/movement_CUNet_v15_paper.log         # v15 ep95+, Lenovo6  (job 1381)
tail -f logs/movement_CUNet_v16_attention.log     # v16 ep17+, Lenovo2  (job 1389)
tail -f logs/movement_CUNet_v17_attsup.log        # v17 ep4+,  Lenovo6  (job 1391)

# Check SLURM queue
squeue -u iulia.orvas

# Full inference on Test_DB (all Sem1-11, 5 models) → results_fECG_extraction/
sbatch src_2026/run_infer_save_results.sh
```

## Contributors

- [Iulia Orvas](https://github.com/Iulia-Alex)
