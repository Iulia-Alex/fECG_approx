# Inference Results — Test_DB (Sem1, Sem2, Sem3)

**Metrici:**
- **PRD (%)** ↓ — `100 × ‖pred−gt‖ / ‖gt‖`
- **SNR (dB)** ↑ — `20·log10(‖gt‖ / ‖pred−gt‖)`
- **QRS F1 / Prec / Rec** — bazat pe adnotările fqrs din mat-file (GT exact). TP = predicție ≥30% din amplitudinea GT la locația beat-ului ±80ms.
---

## Tabel sumar (mediat pe Sem1+Sem2+Sem3)

| Model | Best Ep | Val Loss | Loss fn | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|---------|----------|---------|-----------|-----------|---------|----------|---------|
| **v1_resume** | ep458 | 0.0137 | SignalMSE | 35.52 | 9.33 | 1.000 | 1.000 | 1.000 |
| **v5** | ep200 | 0.0225 | SignalMSE+3×AmpW | 39.15 | 8.58 | 1.000 | 1.000 | 1.000 |
| **v6** | ep68 | 0.0529 | MSE+AmpW+Baseline | 33.38 | 9.70 | 1.000 | 1.000 | 1.000 |
| **v7** | ep30 | 0.0818 | SignalMAE(L1) | 79.45 | 2.12 | 0.274 | 0.444 | 0.253 |
| **v8** | ep41 | 0.0470 | SignalMSE | 57.79 | 4.82 | 0.954 | 1.000 | 0.926 |
| **v9** | ep138 | 0.9255 | Sig+Cpl | 75.70 | 2.53 | 0.526 | 1.000 | 0.426 |
| **v10** | ep35 | 2.3780 | Sig+Cpl+3×Peak | 83.63 | 1.62 | 0.450 | 0.778 | 0.383 |
| **v11** | ep73 | 0.4329 | Sig+3×Peak(±30ms) | 54.59 | 5.32 | 1.000 | 1.000 | 1.000 |
| **v12** | ep49 | 0.2209 | Sig+3×QRS(±100ms) | 68.33 | 3.56 | 0.933 | 1.000 | 0.889 |
| **v13** | ep84 | 0.0600 | SignalMSE | 76.44 | 2.43 | 0.848 | 0.986 | 0.772 |
| **v15** | ep92 | 0.0313 | SignalMSE | 59.60 | 4.65 | 1.000 | 1.000 | 1.000 |
| **v16** | ep15 | 0.0330 | SignalMSE | — | — | — | — | — |
| **v17** | ep3 | 0.0469 | SignalMSE+L_att | — | — | — | — | — |

---

## v1_resume

**Arhitectură:** orig  |  **Pipeline:** 1k Hz  |  **Loss:** SignalMSE  |  **Best epoch:** 458  |  **Best val loss:** 0.013701

**Loss curve:**

![loss_v1_resume](plots_2026/loss_plots_per_model/loss_v1_resume.png)

### Sem1

![v1_resume Sem1](plots_2026/testdb/Sem1_v1_resume.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 31.70 | 9.98 | 1.000 | 1.000 | 1.000 |
| Ch2 | 32.33 | 9.81 | 1.000 | 1.000 | 1.000 |
| Ch3 | 19.93 | 14.01 | 1.000 | 1.000 | 1.000 |
| **avg** | **27.98** | **11.27** | **1.000** | **1.000** | **1.000** |

### Sem2

![v1_resume Sem2](plots_2026/testdb/Sem2_v1_resume.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 52.25 | 5.64 | 1.000 | 1.000 | 1.000 |
| Ch2 | 45.18 | 6.90 | 1.000 | 1.000 | 1.000 |
| Ch3 | 26.85 | 11.42 | 1.000 | 1.000 | 1.000 |
| **avg** | **41.43** | **7.99** | **1.000** | **1.000** | **1.000** |

### Sem3

![v1_resume Sem3](plots_2026/testdb/Sem3_v1_resume.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 45.21 | 6.90 | 1.000 | 1.000 | 1.000 |
| Ch2 | 37.23 | 8.58 | 1.000 | 1.000 | 1.000 |
| Ch3 | 28.98 | 10.76 | 1.000 | 1.000 | 1.000 |
| **avg** | **37.14** | **8.75** | **1.000** | **1.000** | **1.000** |

---

## v5

**Arhitectură:** orig  |  **Pipeline:** 1k Hz  |  **Loss:** SignalMSE+3×AmpW  |  **Best epoch:** 200  |  **Best val loss:** 0.022473

**Loss curve:**

![loss_v5](plots_2026/loss_plots_per_model/loss_v5.png)

### Sem1

![v5 Sem1](plots_2026/testdb/Sem1_v5.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 37.88 | 8.43 | 1.000 | 1.000 | 1.000 |
| Ch2 | 36.68 | 8.71 | 1.000 | 1.000 | 1.000 |
| Ch3 | 21.13 | 13.50 | 1.000 | 1.000 | 1.000 |
| **avg** | **31.90** | **10.21** | **1.000** | **1.000** | **1.000** |

### Sem2

![v5 Sem2](plots_2026/testdb/Sem2_v5.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 62.01 | 4.15 | 1.000 | 1.000 | 1.000 |
| Ch2 | 54.72 | 5.24 | 1.000 | 1.000 | 1.000 |
| Ch3 | 29.88 | 10.49 | 1.000 | 1.000 | 1.000 |
| **avg** | **48.87** | **6.63** | **1.000** | **1.000** | **1.000** |

### Sem3

![v5 Sem3](plots_2026/testdb/Sem3_v5.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 45.76 | 6.79 | 1.000 | 1.000 | 1.000 |
| Ch2 | 36.80 | 8.68 | 1.000 | 1.000 | 1.000 |
| Ch3 | 27.44 | 11.23 | 1.000 | 1.000 | 1.000 |
| **avg** | **36.67** | **8.90** | **1.000** | **1.000** | **1.000** |

---

## v6

**Arhitectură:** orig  |  **Pipeline:** 1k Hz  |  **Loss:** MSE+AmpW+Baseline  |  **Best epoch:** 68  |  **Best val loss:** 0.052941

**Loss curve:**

![loss_v6](plots_2026/loss_plots_per_model/loss_v6.png)

### Sem1

![v6 Sem1](plots_2026/testdb/Sem1_v6.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 35.77 | 8.93 | 1.000 | 1.000 | 1.000 |
| Ch2 | 30.61 | 10.28 | 1.000 | 1.000 | 1.000 |
| Ch3 | 31.47 | 10.04 | 1.000 | 1.000 | 1.000 |
| **avg** | **32.62** | **9.75** | **1.000** | **1.000** | **1.000** |

### Sem2

![v6 Sem2](plots_2026/testdb/Sem2_v6.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 48.35 | 6.31 | 1.000 | 1.000 | 1.000 |
| Ch2 | 38.58 | 8.27 | 1.000 | 1.000 | 1.000 |
| Ch3 | 36.21 | 8.82 | 1.000 | 1.000 | 1.000 |
| **avg** | **41.05** | **7.80** | **1.000** | **1.000** | **1.000** |

### Sem3

![v6 Sem3](plots_2026/testdb/Sem3_v6.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 26.91 | 11.40 | 1.000 | 1.000 | 1.000 |
| Ch2 | 25.86 | 11.75 | 1.000 | 1.000 | 1.000 |
| Ch3 | 26.62 | 11.50 | 1.000 | 1.000 | 1.000 |
| **avg** | **26.46** | **11.55** | **1.000** | **1.000** | **1.000** |

---

## v7

**Arhitectură:** v7  |  **Pipeline:** 1k Hz  |  **Loss:** SignalMAE(L1)  |  **Best epoch:** 30  |  **Best val loss:** 0.081819

**Loss curve:**

![loss_v7](plots_2026/loss_plots_per_model/loss_v7.png)

### Sem1

![v7 Sem1](plots_2026/testdb/Sem1_v7.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 83.46 | 1.57 | 0.000 | 0.000 | 0.000 |
| Ch2 | 84.68 | 1.44 | 0.000 | 0.000 | 0.000 |
| Ch3 | 88.69 | 1.04 | 0.000 | 0.000 | 0.000 |
| **avg** | **85.61** | **1.35** | **0.000** | **0.000** | **0.000** |

### Sem2

![v7 Sem2](plots_2026/testdb/Sem2_v7.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 90.73 | 0.84 | 0.105 | 1.000 | 0.056 |
| Ch2 | 89.80 | 0.93 | 0.000 | 0.000 | 0.000 |
| Ch3 | 90.85 | 0.83 | 0.000 | 0.000 | 0.000 |
| **avg** | **90.46** | **0.87** | **0.035** | **0.333** | **0.019** |

### Sem3

![v7 Sem3](plots_2026/testdb/Sem3_v7.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 56.91 | 4.90 | 1.000 | 1.000 | 1.000 |
| Ch2 | 62.54 | 4.08 | 1.000 | 1.000 | 1.000 |
| Ch3 | 67.34 | 3.43 | 0.364 | 1.000 | 0.222 |
| **avg** | **62.27** | **4.14** | **0.788** | **1.000** | **0.741** |

---

## v8

**Arhitectură:** v7  |  **Pipeline:** 1k Hz  |  **Loss:** SignalMSE  |  **Best epoch:** 41  |  **Best val loss:** 0.046987

**Loss curve:**

![loss_v8](plots_2026/loss_plots_per_model/loss_v8.png)

### Sem1

![v8 Sem1](plots_2026/testdb/Sem1_v8.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 54.47 | 5.28 | 1.000 | 1.000 | 1.000 |
| Ch2 | 56.63 | 4.94 | 1.000 | 1.000 | 1.000 |
| Ch3 | 70.57 | 3.03 | 0.714 | 1.000 | 0.556 |
| **avg** | **60.56** | **4.41** | **0.905** | **1.000** | **0.852** |

### Sem2

![v8 Sem2](plots_2026/testdb/Sem2_v8.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 51.48 | 5.77 | 1.000 | 1.000 | 1.000 |
| Ch2 | 53.61 | 5.42 | 1.000 | 1.000 | 1.000 |
| Ch3 | 69.83 | 3.12 | 0.875 | 1.000 | 0.778 |
| **avg** | **58.31** | **4.77** | **0.958** | **1.000** | **0.926** |

### Sem3

![v8 Sem3](plots_2026/testdb/Sem3_v8.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 50.08 | 6.01 | 1.000 | 1.000 | 1.000 |
| Ch2 | 54.51 | 5.27 | 1.000 | 1.000 | 1.000 |
| Ch3 | 58.91 | 4.60 | 1.000 | 1.000 | 1.000 |
| **avg** | **54.50** | **5.29** | **1.000** | **1.000** | **1.000** |

---

## v9

**Arhitectură:** v9  |  **Pipeline:** 1k Hz  |  **Loss:** Sig+Cpl  |  **Best epoch:** 138  |  **Best val loss:** 0.925510

**Loss curve:**

![loss_v9](plots_2026/loss_plots_per_model/loss_v9.png)

### Sem1

![v9 Sem1](plots_2026/testdb/Sem1_v9.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 86.50 | 1.26 | 0.200 | 1.000 | 0.111 |
| Ch2 | 85.42 | 1.37 | 0.200 | 1.000 | 0.111 |
| Ch3 | 77.28 | 2.24 | 0.500 | 1.000 | 0.333 |
| **avg** | **83.06** | **1.62** | **0.300** | **1.000** | **0.185** |

### Sem2

![v9 Sem2](plots_2026/testdb/Sem2_v9.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 87.12 | 1.20 | 0.364 | 1.000 | 0.222 |
| Ch2 | 85.90 | 1.32 | 0.286 | 1.000 | 0.167 |
| Ch3 | 80.16 | 1.92 | 0.364 | 1.000 | 0.222 |
| **avg** | **84.39** | **1.48** | **0.338** | **1.000** | **0.204** |

### Sem3

![v9 Sem3](plots_2026/testdb/Sem3_v9.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 61.74 | 4.19 | 0.941 | 1.000 | 0.889 |
| Ch2 | 59.94 | 4.45 | 0.941 | 1.000 | 0.889 |
| Ch3 | 57.29 | 4.84 | 0.941 | 1.000 | 0.889 |
| **avg** | **59.66** | **4.49** | **0.941** | **1.000** | **0.889** |

---

## v10

**Arhitectură:** v10  |  **Pipeline:** 1k Hz  |  **Loss:** Sig+Cpl+3×Peak  |  **Best epoch:** 35  |  **Best val loss:** 2.378023

**Loss curve:**

![loss_v10](plots_2026/loss_plots_per_model/loss_v10.png)

### Sem1

![v10 Sem1](plots_2026/testdb/Sem1_v10.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 92.45 | 0.68 | 0.105 | 1.000 | 0.056 |
| Ch2 | 91.28 | 0.79 | 0.200 | 1.000 | 0.111 |
| Ch3 | 83.16 | 1.60 | 0.560 | 1.000 | 0.389 |
| **avg** | **88.96** | **1.03** | **0.288** | **1.000** | **0.185** |

### Sem2

![v10 Sem2](plots_2026/testdb/Sem2_v10.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 93.56 | 0.58 | 0.000 | 0.000 | 0.000 |
| Ch2 | 92.78 | 0.65 | 0.000 | 0.000 | 0.000 |
| Ch3 | 88.08 | 1.10 | 0.364 | 1.000 | 0.222 |
| **avg** | **91.47** | **0.78** | **0.121** | **0.333** | **0.074** |

### Sem3

![v10 Sem3](plots_2026/testdb/Sem3_v10.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 71.87 | 2.87 | 0.941 | 1.000 | 0.889 |
| Ch2 | 70.78 | 3.00 | 0.941 | 1.000 | 0.889 |
| Ch3 | 68.73 | 3.26 | 0.941 | 1.000 | 0.889 |
| **avg** | **70.46** | **3.04** | **0.941** | **1.000** | **0.889** |

---

## v11

**Arhitectură:** v11  |  **Pipeline:** 1k Hz  |  **Loss:** Sig+3×Peak(±30ms)  |  **Best epoch:** 73  |  **Best val loss:** 0.432913

**Loss curve:**

![loss_v11](plots_2026/loss_plots_per_model/loss_v11.png)

### Sem1

![v11 Sem1](plots_2026/testdb/Sem1_v11.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 51.73 | 5.73 | 1.000 | 1.000 | 1.000 |
| Ch2 | 52.53 | 5.59 | 1.000 | 1.000 | 1.000 |
| Ch3 | 66.24 | 3.58 | 1.000 | 1.000 | 1.000 |
| **avg** | **56.83** | **4.97** | **1.000** | **1.000** | **1.000** |

### Sem2

![v11 Sem2](plots_2026/testdb/Sem2_v11.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 53.83 | 5.38 | 1.000 | 1.000 | 1.000 |
| Ch2 | 52.61 | 5.58 | 1.000 | 1.000 | 1.000 |
| Ch3 | 65.00 | 3.74 | 1.000 | 1.000 | 1.000 |
| **avg** | **57.15** | **4.90** | **1.000** | **1.000** | **1.000** |

### Sem3

![v11 Sem3](plots_2026/testdb/Sem3_v11.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 44.82 | 6.97 | 1.000 | 1.000 | 1.000 |
| Ch2 | 49.00 | 6.20 | 1.000 | 1.000 | 1.000 |
| Ch3 | 55.58 | 5.10 | 1.000 | 1.000 | 1.000 |
| **avg** | **49.80** | **6.09** | **1.000** | **1.000** | **1.000** |

---

## v12

**Arhitectură:** v12  |  **Pipeline:** 1k Hz  |  **Loss:** Sig+3×QRS(±100ms)  |  **Best epoch:** 49  |  **Best val loss:** 0.220880

**Loss curve:**

![loss_v12](plots_2026/loss_plots_per_model/loss_v12.png)

### Sem1

![v12 Sem1](plots_2026/testdb/Sem1_v12.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 84.10 | 1.50 | 0.759 | 1.000 | 0.611 |
| Ch2 | 81.63 | 1.76 | 0.759 | 1.000 | 0.611 |
| Ch3 | 69.75 | 3.13 | 0.971 | 1.000 | 0.944 |
| **avg** | **78.50** | **2.13** | **0.830** | **1.000** | **0.722** |

### Sem2

![v12 Sem2](plots_2026/testdb/Sem2_v12.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 86.11 | 1.30 | 0.971 | 1.000 | 0.944 |
| Ch2 | 80.89 | 1.84 | 0.941 | 1.000 | 0.889 |
| Ch3 | 69.56 | 3.15 | 1.000 | 1.000 | 1.000 |
| **avg** | **78.85** | **2.10** | **0.971** | **1.000** | **0.944** |

### Sem3

![v12 Sem3](plots_2026/testdb/Sem3_v12.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 50.62 | 5.91 | 1.000 | 1.000 | 1.000 |
| Ch2 | 47.34 | 6.50 | 1.000 | 1.000 | 1.000 |
| Ch3 | 44.94 | 6.95 | 1.000 | 1.000 | 1.000 |
| **avg** | **47.63** | **6.45** | **1.000** | **1.000** | **1.000** |

---

## v13

**Arhitectură:** v13  |  **Pipeline:** 1k Hz  |  **Loss:** SignalMSE  |  **Best epoch:** 84  |  **Best val loss:** 0.060014

**Loss curve:**

![loss_v13](plots_2026/loss_plots_per_model/loss_v13.png)

### Sem1

![v13 Sem1](plots_2026/testdb/Sem1_v13.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 87.67 | 1.14 | 0.538 | 0.875 | 0.389 |
| Ch2 | 86.22 | 1.29 | 0.615 | 1.000 | 0.444 |
| Ch3 | 78.81 | 2.07 | 0.714 | 1.000 | 0.556 |
| **avg** | **84.24** | **1.50** | **0.623** | **0.958** | **0.463** |

### Sem2

![v13 Sem2](plots_2026/testdb/Sem2_v13.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 88.47 | 1.06 | 0.909 | 1.000 | 0.833 |
| Ch2 | 84.27 | 1.49 | 0.909 | 1.000 | 0.833 |
| Ch3 | 76.87 | 2.29 | 0.941 | 1.000 | 0.889 |
| **avg** | **83.20** | **1.61** | **0.920** | **1.000** | **0.852** |

### Sem3

![v13 Sem3](plots_2026/testdb/Sem3_v13.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 63.30 | 3.97 | 1.000 | 1.000 | 1.000 |
| Ch2 | 61.94 | 4.16 | 1.000 | 1.000 | 1.000 |
| Ch3 | 60.42 | 4.38 | 1.000 | 1.000 | 1.000 |
| **avg** | **61.88** | **4.17** | **1.000** | **1.000** | **1.000** |

---

## v15

**Arhitectură:** v15  |  **Pipeline:** 500 Hz  |  **Loss:** SignalMSE  |  **Best epoch:** 92  |  **Best val loss:** 0.031271

**Loss curve:**

![loss_v15](plots_2026/loss_plots_per_model/loss_v15.png)

### Sem1

![v15 Sem1](plots_2026/testdb/Sem1_v15.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 72.62 | 2.78 | 1.000 | 1.000 | 1.000 |
| Ch2 | 67.30 | 3.44 | 1.000 | 1.000 | 1.000 |
| Ch3 | 49.59 | 6.09 | 1.000 | 1.000 | 1.000 |
| **avg** | **63.17** | **4.10** | **1.000** | **1.000** | **1.000** |

### Sem2

![v15 Sem2](plots_2026/testdb/Sem2_v15.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 77.95 | 2.16 | 1.000 | 1.000 | 1.000 |
| Ch2 | 69.34 | 3.18 | 1.000 | 1.000 | 1.000 |
| Ch3 | 48.98 | 6.20 | 1.000 | 1.000 | 1.000 |
| **avg** | **65.42** | **3.85** | **1.000** | **1.000** | **1.000** |

### Sem3

![v15 Sem3](plots_2026/testdb/Sem3_v15.png)

| Canal | PRD (%) ↓ | SNR (dB) ↑ | QRS F1 ↑ | QRS Prec | QRS Rec |
|-------|-----------|-----------|---------|----------|---------|
| Ch1 | 54.81 | 5.22 | 1.000 | 1.000 | 1.000 |
| Ch2 | 49.06 | 6.19 | 1.000 | 1.000 | 1.000 |
| Ch3 | 46.76 | 6.60 | 1.000 | 1.000 | 1.000 |
| **avg** | **50.21** | **6.00** | **1.000** | **1.000** | **1.000** |

---

## v16

**Arhitectură:** v16 (ComplexAttentionUNet)  |  **Pipeline:** 500 Hz  |  **Loss:** SignalMSE  |  **Best epoch:** 15  |  **Best val loss:** 0.032987

**Status:** RUNNING (ep17+ la 2026-05-18) — metrici complete nu au fost calculate încă (checkpoint prea timpuriu la momentul inferenței).

**Notă:** Arhitectură = v15 + 2 attention gates (AG1 la 64×64, AG2 la 32×32). Val loss apropiată de v15 la ep15 — modelul continuă să antreneze.

---

## v17

**Arhitectură:** v17 (ComplexAttentionUNet + att. supervision)  |  **Pipeline:** 500 Hz  |  **Loss:** SignalMSE + L_att  |  **Best epoch:** 3  |  **Best val loss (total):** 0.046913

**Status:** RUNNING (ep4+ la 2026-05-18) — metrici complete nu au fost calculate (prea puține epoci).

**Notă:** v17 = v16 + supervizare attention gates cu mască GT fQRS (±80ms, downsampled 128→64). L_att = MSE(alpha_AG1, fqrs_target_mask). Scopul: attention gates să se concentreze pe complexele QRS fetale de la primele epoci.

---

## binary_v1 — Detecție binară mișcare fetală

**Arhitectură:** PrecisionResUNet (5.99M)  |  **Pipeline:** 1k Hz  |  **Loss:** BCE + Dice  |  **Best epoch:** 29  |  **Best val loss:** 0.2031

**Input:** 8 canale (fECG ×6 z-norm + A_QRS + baseline wander)  |  **Fereastră:** 8s  |  **Ieșire:** mască binară per sample

**Notă:** Modelul prezice mișcare/fără mișcare per sample, fără informație despre granițele segmentelor GT. Metricile sunt calculate la nivel de sample pe semnalul continuu.

**Metrici:**
- **F1** ↑ — 2·TP / (2·TP + FP + FN) la prag 0.5
- **Prec** ↑ — TP / (TP + FP)
- **Rec** ↑ — TP / (TP + FN)
- **Acc** ↑ — (TP + TN) / total

### Tabel sumar (Sem1 + Sem2 + Sem3)

| Fișier | mov% | F1 ↑ | Prec ↑ | Rec ↑ | Acc ↑ |
|--------|------|------|--------|-------|-------|
| Sem1 | 23.2% | 0.9814 | 0.9955 | 0.9678 | 0.9915 |
| Sem2 | 39.2% | 0.9701 | 0.9825 | 0.9580 | 0.9768 |
| Sem3 | 38.2% | 0.8846 | 0.9933 | 0.7974 | 0.9206 |
| **medie** | | **0.9454** | **0.9904** | **0.9077** | **0.9630** |

**Observație Sem3:** Recall scăzut (0.797) — modelul ratează unele segmente de mișcare. Precizia rămâne foarte ridicată (0.993), deci predicțiile pozitive sunt corecte, dar există false negative.

### Sem1

![binary_v1 Sem1](plots/binary_v1_Sem1.png)

| Metrică | Valoare |
|---------|---------|
| F1 | 0.9814 |
| Prec | 0.9955 |
| Rec | 0.9678 |
| Acc | 0.9915 |

### Sem2

![binary_v1 Sem2](plots/binary_v1_Sem2.png)

| Metrică | Valoare |
|---------|---------|
| F1 | 0.9701 |
| Prec | 0.9825 |
| Rec | 0.9580 |
| Acc | 0.9768 |

### Sem3

![binary_v1 Sem3](plots/binary_v1_Sem3.png)

| Metrică | Valoare |
|---------|---------|
| F1 | 0.8846 |
| Prec | 0.9933 |
| Rec | 0.7974 |
| Acc | 0.9206 |

---
