# Fetal ECG Extraction from Movement ECG

Complex-valued U-Net for extracting fetal ECG (fECG) from 6-channel abdominal recordings containing maternal ECG (mECG) and movement noise.

## Problem

- Input: 6-channel mixture (fECG + mECG + movement noise), spectrogram 128×400
- Output: 6-channel predicted fECG spectrogram
- Main challenge: mECG is 8 dB stronger than fECG; model must suppress it while preserving fetal R-peak amplitudes

## Models

| Model | Architecture | Params | Loss | Best val loss | Status |
|---|---|---|---|---|---|
| v1 | ComplexUNet | 0.59 M | SignalMSE | **0.014745** @ ep199 | DONE |
| v5 | ComplexUNet | 0.59 M | SignalMSE + 3×AmpWeightedMSE | 0.022473 @ ep200 | DONE |
| v6 | ComplexUNet | 0.59 M | SignalMSE + AmpW + BaselinePenalty | 0.052952 @ ep67 | stopped (plateau) |
| v7 | ComplexUNetV7 (paper arch) | 1.87 M | SignalMAE (L1) | 0.081819 @ ep30 | stopped (L1 suppresses peaks) |
| **v8** | ComplexUNetV7 (paper arch) | 1.87 M | **SignalMSE** | 0.049408 @ ep16 | running |

Full experiment log: [EXPERIMENTS.md](EXPERIMENTS.md)

## Key Finding: L1 vs L2 Loss

**L1 (MAE) suppresses R-peaks.** Peaks are sparse events (~8% of signal) — L1 minimises median error so the model can ignore peaks and still converge. L2 (MSE) penalises large errors quadratically, forcing the model to learn correct peak amplitudes.

v7 (L1, ep30) had suppressed peaks. v8 (MSE, same architecture, ep16) already shows clearer peaks than v7 at ep30.

## Folder Structure

```
src_2026/         active training and analysis scripts
src_2026/archive/ intermediate/obsolete experiments (kept for reference)
src/              original paper reference code
data/             training data (gitignored)
models/           saved checkpoints (gitignored)
logs/             training logs (gitignored)
plots_2026/       inference plots and PDF reports (gitignored)
```

## Key Design Decisions

- **SignalMSE over ComplexMSE**: time-domain MSE via GPU iSTFT → correct peak heights (7× faster than CPU iSTFT)
- **L2 over L1**: MSE forces correct peak amplitudes; MAE suppresses sparse peaks
- **128×400 spectrogram**: 10 ms/frame → QRS spans 6–8 frames (recoverable)
- **0.59M ComplexUNet (v1–v6)**: halved channels → fits in 8 GB with BS=32
- **1.87M ComplexUNetV7 (v7–v8)**: paper architecture — max BS=16 on M4000 8GB due to RoActivation 3× memory overhead

## Environment

```bash
conda activate ecg        # /home/iulia.orvas/miniconda3/envs/ecg
# PyTorch 1.12.1+cu113, numpy 1.26.4 (pinned — numpy 2.0 incompatible)
```

## Running

```bash
# Submit training job (from repo root)
sbatch src_2026/run_training_v8.sh

# Resume automatically continues from checkpoint + history.json
```

## Contributors

- [Iulia Orvas](https://github.com/Iulia-Alex)