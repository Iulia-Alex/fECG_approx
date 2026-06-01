# fECG Extraction — Cum funcționează, ce am încercat, ce am constatat

*Ultima actualizare: 2026-04-27*

---

## 1. Ce vrem să facem (problema)

Avem înregistrări ECG abdominale cu **6 canale**. Semnalul înregistrat este un amestec:

```
mixture = mECG + fECG + noise_mișcare
```

- **mECG** (ECG matern) — dominant, ~8 dB mai puternic decât fECG
- **fECG** (ECG fetal) — slab, îngropat sub mECG
- **noise_mișcare** — relativ mic (+4 dB față de fECG)

**Scopul** extracției: obținem un semnal fECG curat cu amplitudinile R-peak fetale corecte. Amplitudinile R-peak sunt folosite downstream pentru clasificarea tipului de mișcare fetală (Part 2).

---

## 2. Pipeline: de la date brute la predicție

### 2.1 Datele

- **Fișiere .mat** din `/data/movement_ecg/` — câte unul per înregistrare
- Fiecare fișier conține: `out.mixture` (6 canale, 1000 Hz), `out.fecg` (ground truth), `out.fqrs` (pozițiile R-peak fetale în eșantioane)
- ~655 fișiere, ~98,250 ferestre de 4s total

### 2.2 Pre-procesare (în `MovementECGDataset`)

Fiecare fereastră de 4 secunde (4000 eșantioane per canal) parcurge:

```
1. Extrage fereastra: mixture[:, start:start+4000]  → (6, 4000)
2. Normalizare per canal: x / std(x)  → amplitudini comparabile
3. STFT per canal (pe CPU, în DataLoader workers):
     n_fft=256, hop_length=10, win_length=128
     → spectrogramă complexă (129, 401) per canal
4. Resize bilinear: (129, 401) → (128, 400)
5. Stack 6 canale → tensor complex (6, 128, 400)
```

Returnează: `(mixture_spec, fecg_spec, fecg_time)` unde `fecg_time` e semnalul fECG în domeniu timp (folosit pentru loss).

`MovementECGDatasetV10` returnează în plus `peak_mask` — un vector (4000,) cu 1.0 la pozițiile fqrs ±30 eșantioane și 0 în rest.

### 2.3 Modelul (ComplexUNet)

Rețeaua operează **exclusiv pe spectrograme complexe**. Primește `(B, 6, 128, 400)` complex și produce tot `(B, 6, 128, 400)` complex.

Arhitectura de bază este un **U-Net cu convoluții complexe**:

```
Input (6, 128, 400) complex
    ↓ DiagMag / Diag (scalare element-wise)
  Encoder:
    conv1 → down1 → down2 → down3   (dimensiune spațială /2 la fiecare down)
  Bottleneck:
    convoluții la cea mai mică rezoluție
  Decoder:
    up1 → up2 → up3  (upsample + skip connection din encoder)
    ↓ conv_out (1×1)
    ↓ DiagMag / Diag (scalare)
Output raw

    ↓ [opțional] Soft mask: sigmoid(real) + j·sigmoid(imag) → [0,1] per bin
                             output = mask × mixture_input
```

### 2.4 Calcul loss (în training)

**SignalMSE** — cel mai important:
```python
1. Resize predicted_spec: (128,400) → (129,401) bilinear
2. iSTFT pe GPU (Hann window, cached): → pred_time (B, 6, 4000)
3. MSE(pred_time, fecg_time)
```
Penalizează direct erorile de amplitudine în domeniu timp.

**ComplexMSE** (folosit în v9):
```python
MSE(pred.real, fecg.real) + MSE(pred.imag, fecg.imag)
```
Penalizează distanța în domeniu spectral (real + imaginar).

**PeakMSE** (folosit în v10, v11):
```python
((pred_time - fecg_time)² * peak_mask).sum() / (n_peaks * C)
```
MSE calculat DOAR la pozițiile R-peak fetale (fqrs ±30 eșantioane).

### 2.5 De ce durează ~2h per epocă?

**Volumul de date:**
- 83,513 ferestre de training → cu BS=16: **5,220 batch-uri** per epocă
- Cu BS=32: 2,610 batch-uri

**Gâtuielile principale:**

| Operație | Unde | Cost |
|----------|------|------|
| STFT per fereastră (6 canale) | CPU, DataLoader workers | ~mare (scipy.stft) |
| Forward pass cu convoluții complexe | GPU | moderat |
| iSTFT per batch pentru loss (B×6 spectrograme) | GPU | semnificativ |
| Backward pass | GPU | moderat |

**Arhitectura v9/v11 (1.87M)** e de ~3× mai lentă decât v1 (0.59M) din cauza:
- Concat skip connections (activări mai mari)
- RoActivation (mai costisitor decât LeakyReLU)
- Convoluții separate Re/Im (nu shared)

**Viteze observate:**
- v1/v1-resume (0.59M, BS=32): ~65–90 min/epocă
- v9/v11 (1.87M, BS=16): ~2.2h/epocă (singur pe GPU), ~3.2h (împărțit)

---

## 3. Arhitecturi folosite

### ComplexUNet (v1-style, 0.59M)

```
conv1: 6 → 32   (full res)
down1: 32 → 64  (/2)
down2: 64 → 128 (/4)
down3: 128 → 128 → bottleneck (/8)
up1: 128 → 64 (/4), skip via ADUNARE
up2: 64 → 32 (/2), skip via ADUNARE
up3: 32 → 6 (full res), skip via ADUNARE
```

- **Shared weights Re/Im**: același Conv2d aplicat separat pe real și imaginar
- **LeakyReLU(0.2)**: simplu, converge ușor
- **DiagMag**: scalare de magnitudine `exp(β)`, zero-init → identitate la start
- **Skip via adunare**: nu concatenare → decoder primește aceeași dimensiune ca skip

### ComplexUNetV7/V9 (paper-style, 1.87M)

```
conv1: 6 → 16   (full res)
down1: 16 → 32  (/2)   — res1 (32 ch)
down2: 32 → 64  (/4)   — res2 (64 ch)
down3: 64 → 128 (/8)   — res3 (128 ch)
bottleneck: 128 → 256 → 128
up1: cat(128,128)=256 → 64 (/4)
up2: cat(64,64)=128   → 32 (/2)
up3: cat(32,32)=64    → 16 (full)
conv_out: 16 → 6 (1×1)
```

- **Separate weights Re/Im**: Conv_real și Conv_imag independenți → mai multă capacitate
- **RoActivation**: amestec learnable între `x/(1+|x|)` și `x·sigmoid(x)` — mai expresiv
- **Diag cu rotație de fază**: `x × exp(j·β)` — permite ajustări de fază
- **Skip via concatenare (UNet-style)**: decodorul vede atât features noi cât și skip-ul

---

## 4. Toate modelele — ce am încercat și ce am constatat

### v1 — DONE → resumat (ep285+)

| | |
|---|---|
| **Arch** | ComplexUNet (0.59M), shared Re/Im, LeakyReLU, add-skip |
| **Output** | Direct prediction |
| **Loss** | SignalMSE only |
| **Best val** | 0.014061 @ ep285 (ep199 original: 0.014745) |
| **Job** | 1357 (Lenovo2, resume) |

**Ce face bine:** amplitudinile R-peak sunt ~90-95% din GT, forma QRS curată, fără overfit.
**Problema:** uneori supraestimează amplitudinea (poate prezice orice valoare, nu e constrâns). SignalMSE medie pe toți cei 4000 eșantioane → peak-urile (~8% din semnal) primesc mai puțin gradient decât baseline-ul (92%).
**Lecție:** arhitectura simplă cu obiectiv direct funcționează surprinzător de bine.

---

### v5 — DONE

| | |
|---|---|
| **Arch** | ComplexUNet (0.59M) — identic v1 |
| **Output** | Direct prediction |
| **Loss** | SignalMSE + 3×AmpWeightedMSE |
| **Best val** | 0.022473 @ ep200 |

**AmpWeightedMSE**: MSE ponderat cu `(|target| / max|target|)²` — mai multă penalizare la peak-uri.
**Constatare:** mai slab decât v1. Ponderea extra pe peak-uri forțează baseline-ul să fie mai zgomotos.

---

### v6 — STOPPED (plateau)

| | |
|---|---|
| **Arch** | ComplexUNet (0.59M) — warm start din v1 |
| **Loss** | SignalMSE + AmpW + BaselinePenalty |
| **Best val** | 0.052952 @ ep67 |

**Lecție:** schimbarea loss-ului la fine-tuning nu funcționează — reprezentările interne din v1 (optimizate pentru SignalMSE) sunt incompatibile cu noul obiectiv.

---

### v7 — STOPPED (L1 suprimă peaks)

| | |
|---|---|
| **Arch** | ComplexUNetV7 (1.87M) |
| **Loss** | SignalMAE (L1) |
| **Best val** | 0.081819 @ ep30 |

**De ce L1 e rău pentru peaks:** L1 minimizează eroarea mediană. Peak-urile sunt rare (~8% din semnal) → modelul ignoră outlier-ii și prezice zero. L2 (MSE) penalizează pătratic erorile mari → forțează predicția corectă a peak-urilor înalte.

---

### v8 — DONE (slab vizual)

| | |
|---|---|
| **Arch** | ComplexUNetV7 (1.87M) |
| **Output** | Direct prediction |
| **Loss** | SignalMSE |
| **Best val** | 0.046987 @ ep41 |

**Surpriză negativă:** aceeași arhitectură ca v9 dar 3× mai slab val loss decât v1. Convergat rapid dar reconstruit prost vizual.
**Root cause:** direct prediction fără mască + SignalMSE singur. Rețeaua "inventează" fECG în loc să îl extragă din mixture.

---

### v9 — DONE (ep138, 2026-04-26)

| | |
|---|---|
| **Arch** | ComplexUNetV9 (1.87M) — v7 + soft mask `sigmoid(logits) × mixture_spec` |
| **Output** | Soft mask 1× |
| **Loss** | SignalMSE + ComplexMSE |
| **Best val** | **0.925510 @ ep138** |
| **Job** | 1346 (Lenovo6) — FINALIZAT |

**Verdict:** ComplexMSE domină ~10-20× SignalMSE → amplitudini R-peak suprimate față de GT. Nu este util pentru Part 2 (amplitudinile nu sunt suficient de corecte). Val loss 0.9255 nu este comparabil cu v1/v11 — scala diferită din cauza ComplexMSE.

---

### v10 — DONE (early stop ep55)

| | |
|---|---|
| **Arch** | ComplexUNetV10 (0.59M) — v1-style + soft mask |
| **Output** | Soft mask |
| **Loss** | SignalMSE + ComplexMSE + 3×PeakMSE(fqrs) |
| **Best val** | 2.378023 @ ep35 |

**Ideea:** v1 arch (simplu, converge rapid) + supervizare directă pe R-peaks via PeakMSE.
**Constatare:** **gap mare train/val** vizibil în loss curves. Arhitectura de 0.59M e insuficientă pentru 3 obiective simultan. ComplexMSE interferează și cu amplitudinile (v. v9).
**Lecție confirmată:** arhitectura contează, și ComplexMSE e problematic pentru amplitudini.

---

### v11 — RUNNING (ep59+, 2026-04-27)

| | |
|---|---|
| **Arch** | ComplexUNetV11 (1.87M) — identic v9 arhitectural |
| **Output** | Direct prediction (fără mască) |
| **Loss** | SignalMSE + 3×PeakMSE(fqrs ±30ms) — fără ComplexMSE |
| **Best val** | **0.446454 @ ep49** |
| **Early stop** | 9/20 la ep58 (LR acum 2.5e-5) |
| **Job** | 1359 (Lenovo6) |

**Progres:** train≈0.470, val oscilează 0.44-0.49. Se va opri în ~38h dacă nu găsește new best.

### v12 — RUNNING (ep1+, 2026-04-27)

| | |
|---|---|
| **Arch** | ComplexUNetV12 (1.87M) — v9 arch + gain mask 1.5×sigmoid |
| **Output** | Gain mask: `1.5 × sigmoid(logits) × mixture_spec` ∈ [0, 1.5]×mixture |
| **Loss** | SignalMSE + 3×QRSwideMSE(fqrs ±100ms) — fără ComplexMSE |
| **Dataset** | MovementECGDatasetV12 — DILATION=100 (50.2% coverage vs 15% în v11) |
| **Job** | 1371 (Lenovo2) — lansat 2026-04-27 |

**Motivație față de v11:**
- Gain 1.5× → mask poate depăși 1.0 la bin-urile cu interferență destructivă fECG/mECG
- ±100ms acoperă complexul PQRS complet (Q, R, S wave), nu doar R-peak ±30ms

---

## 5. Tabel comparativ rapid

| Model | Arch | Params | Output | Loss | Best val | Status |
|-------|------|--------|--------|------|----------|--------|
| v1 | ComplexUNet | 0.59M | direct | SignalMSE | 0.014745 @ ep199 | DONE |
| **v1 resume** | ComplexUNet | 0.59M | direct | SignalMSE | **0.013776 @ ep399** | **DONE** (best overall) |
| v5 | ComplexUNet | 0.59M | direct | Sig+AmpW | 0.022473 @ ep200 | DONE |
| v6 | ComplexUNet | 0.59M | direct | Sig+AmpW+Base | 0.052952 @ ep67 | STOPPED |
| v7 | ComplexUNetV7 | 1.87M | direct | SignalMAE | 0.081819 @ ep30 | STOPPED |
| v8 | ComplexUNetV7 | 1.87M | direct | SignalMSE | 0.046987 @ ep41 | DONE (slab) |
| v9 | ComplexUNetV9 | 1.87M | soft mask 1× | Sig+Cpl | 0.925510 @ ep138 | **DONE** (ComplexMSE suprimă amp.) |
| v10 | ComplexUNetV10 | 0.59M | soft mask 1× | Sig+Cpl+Peak | 2.378023 @ ep35 | DONE (early stop) |
| **v11** | ComplexUNetV11 | 1.87M | direct | Sig+3×Peak(±30ms) | **0.446454 @ ep49** | **RUNNING** ep59+, Lenovo6 |
| **v12** | ComplexUNetV12 | 1.87M | gain mask 1.5× | Sig+3×Peak(±100ms) | — | **RUNNING** ep1+, Lenovo2 |

*Notă: v9/v10 au loss scale diferit (includ ComplexMSE ~1.5–2.0 la convergență). v11/v12 sunt comparabile cu v1.*

---

## 6. Lecții învățate

1. **Soft mask > direct prediction** pentru stabilitatea antrenamentului — dar nu neapărat pentru amplitudini finale.

2. **ComplexMSE interferează cu amplitudinile.** Chiar și la greutate 1.0 (v9) sau 0.1 (v3) suprimă amplitudinile. Modelul minimizează distanța spectrală prezicând valori mici.

3. **L1 suprimă peaks, L2 le forțează.** Peak-urile sunt rare (~8% din semnal) → L1 le ignoră (soluție constantă minimizează L1), L2 le penalizează pătratic.

4. **Arhitectura mică (0.59M) + loss complex (3 termeni) = overfit/underfitting.** V10 arată gap mare train/val — capacitatea insuficientă.

5. **Warm start cu loss diferit nu funcționează** (v6) — reprezentările interne sunt incompatibile cu noul obiectiv.

6. **SignalMSE singur (v1) e baseline puternic** — simplu și direct. Optimizează exact ce contează (amplitudinile în domeniu timp).

7. **Scheduler ReduceLROnPlateau** cu patience=8 ajută — v1 resume a îmbunătățit val loss de la 0.014745 → 0.014061 după reducerea la 5e-5.

---

## 7. Stare curentă cluster (2026-04-27)

| Nod | Job | Model | Status |
|-----|-----|-------|--------|
| Lenovo6 | 1359 | v11 | RUNNING ep59+, early stop 9/20, best=0.446454 |
| Lenovo2 | 1371 | v12 | RUNNING ep1+, gain mask 1.5×, nou lansat |
| Lenovo6 | 1346 | v9 | **DONE** 2026-04-26 — best=0.925510 @ ep138 |
| Lenovo2 | 1357 | v1 resume | **DONE** 2026-04-27 — best=0.013776 @ ep399 |
| Lenovo1, 3, 7 | — | — | DOWN |

**Prioritate de monitorizat:** v12 ep1-10 — converge normal? v11 ep60-70 — new best sau early stop?

## 8. Lecții v9 → v12

8. **Gain mask 1.5× (v12):** `sigmoid ≤ 1` per bin STFT. La bin-urile cu interferență destructivă fECG/mECG, magnitudinea amestecului e mică → mask necesară > 1 → sigmoid nu poate ajunge → energie pierdută. Gain 1.5× extinde range-ul la [0, 1.5] și recuperează aceste bin-uri. (Notă: în domeniu timp, iSTFT overlap-add poate produce amplitudini > orice bin individual — deci limita nu e în domeniu timp, ci în domeniu STFT per bin.)

9. **QRSwideMSE ±100ms (v12):** PeakMSE standard ±30ms supervizează doar R-peak-ul. Complexul PQRS complet la 1000Hz: P(~80ms înainte), Q(~20ms înainte), R(0), S(~20ms după), T(~150ms după). ±100ms acoperă Q, R, S complet și parțial P/T. Coverage: 50.2% din semnal (vs ~15% cu ±30ms).
