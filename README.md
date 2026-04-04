# Fetal ECG Extraction from Movement ECG

Complex-valued U-Net for extracting fetal ECG (fECG) from 6-channel abdominal recordings containing maternal ECG (mECG) and movement noise.

## Problem

- Input: 6-channel mixture (fECG + mECG + movement noise), spectrogram 128×400
- Output: 6-channel predicted fECG spectrogram
- Main challenge: mECG is 8 dB stronger than fECG; model must suppress it while preserving fetal R-peak amplitudes

## Models

| Model | Architecture | Params | Loss | Best val loss |
|---|---|---|---|---|
| v1 | ComplexUNet | 0.59 M | SignalMSE | **0.014745** @ ep199 ✅ |
| v5 | ComplexUNet | 0.59 M | SignalMSE + 3×AmpWeightedMSE | 0.023846 @ ep154 (running) |
| v6 | ComplexUNet | 0.59 M | SignalMSE + AmpW + BaselinePenalty | 0.053158 @ ep42 (running) |
| v7 | ComplexUNetV7 (paper arch) | 1.87 M | SignalMAE (L1) | 0.085450 @ ep15 (running) |

Full experiment log: [EXPERIMENTS.md](EXPERIMENTS.md)

## Folder Structure

```
src_2026/         active training, inference, and analysis scripts
src_2026/archive/ intermediate/obsolete experiments (kept for reference)
src/              original paper reference code
data/             training data (gitignored)
models/           saved checkpoints (gitignored)
logs/             training logs (gitignored)
plots_2026/       inference plots and PDF reports (gitignored)
```

## Key Design Decisions

- **SignalMSE/MAE over ComplexMSE**: time-domain loss via GPU iSTFT → correct peak heights
- **128×400 spectrogram**: 10 ms/frame → QRS spans 6–8 frames (recoverable)
- **Halved channels (v1–v6)**: 0.59M params, 2.3 h/epoch on M4000
- **v7 paper architecture**: split complex conv, RoActivation, concatenation skips, phase-rotation diagonal

## Environment

```bash
conda activate ecg        # /home/iulia.orvas/miniconda3/envs/ecg
# PyTorch 1.12.1+cu113, numpy 1.26.4 (pinned — numpy 2.0 incompatible)
```

## Running

```bash
# Submit training job
sbatch src_2026/run_training_v7.sh

# Run inference on Test_DB (Sem1/Sem2/Sem5)
cd src_2026
python infer_v6v7_sem125.py

# Generate model report PDF
python generate_model_report.py
```

## Contributors

- [Iulia Orvas](https://github.com/Iulia-Alex)
- [Andrei Radu](https://github.com/andrei-radu)
