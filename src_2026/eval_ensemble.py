"""
Ensemble RF (36 spectral features) + SmallEnvResNet v17 (Hilbert envelope).
Sweep w_RF de la 0.0 la 1.0 si raporteaza best accuracy.
"""

import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clf_dataset import ClfDatasetV15
from envelope_dataset import EnvelopeDataset
from train_clf_v17 import SmallEnvResNet

DATA_DIR         = '../data/movement_ecg'
MODEL_V17_PATH   = '../models/movement_clf_v17.pth'
VAL_SPLIT        = 0.15
SEED             = 42
BATCH_SIZE       = 256
N_CLASSES        = 4
LABELS           = ['Stationary', 'Linear', 'Helical', 'Screw']

# ---------------------------------------------------------------------------
print('Loading PhaseDataset (36 features)...')
clf_ds = ClfDatasetV15(DATA_DIR)
# iau doar primele 36 features (spectral), nu inter-canal
print(f'Total phases: {len(clf_ds)}  features: {clf_ds.X.shape[1]}')
print(f'Class dist: {[int((clf_ds.y == c).sum()) for c in range(N_CLASSES)]}')

print('Loading EnvelopeDataset...')
env_ds = EnvelopeDataset(DATA_DIR)

# Acelasi split (seed=42) -> aceleasi indici
train_idx_clf, val_idx_clf = clf_ds.file_split(val_frac=VAL_SPLIT, seed=SEED)
train_idx_env, val_idx_env = env_ds.file_split(val_frac=VAL_SPLIT, seed=SEED)

print(f'CLF split: train={len(train_idx_clf)}  val={len(val_idx_clf)}')
print(f'ENV split: train={len(train_idx_env)}  val={len(val_idx_env)}')

# Verificare ca ordinea e aceeasi
assert len(train_idx_clf) == len(train_idx_env), "Split-urile au marimi diferite!"
assert len(val_idx_clf)   == len(val_idx_env),   "Split-urile au marimi diferite!"

# ---------------------------------------------------------------------------
# RF pe primele 36 features spectrale
X_train = clf_ds.X[train_idx_clf, :36]
y_train = clf_ds.y[train_idx_clf]
X_val   = clf_ds.X[val_idx_clf,   :36]
y_val   = clf_ds.y[val_idx_clf]

print(f'\nTraining RF pe {X_train.shape[1]} features spectrale...')
rf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1,
                             class_weight='balanced')
rf.fit(X_train, y_train)
rf_val_acc = accuracy_score(y_val, rf.predict(X_val)) * 100
print(f'RF val acc: {rf_val_acc:.2f}%')

# Probabilitati RF pe val
rf_probs = rf.predict_proba(X_val)   # (N, 4)
print(f'RF probs shape: {rf_probs.shape}')

# ---------------------------------------------------------------------------
# ResNet v17 — extrage softmax probabilities pe val
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'\nDevice: {device}')

model = SmallEnvResNet(in_ch=6, n_classes=N_CLASSES, dropout=0.5).to(device)
model.load_state_dict(torch.load(MODEL_V17_PATH, map_location=device))
model.eval()

val_env_loader = DataLoader(Subset(env_ds, val_idx_env),
                            batch_size=BATCH_SIZE, shuffle=False)

all_probs = []
all_labels = []
with torch.no_grad():
    for x, y in val_env_loader:
        x = x.to(device)
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())

resnet_probs  = np.concatenate(all_probs,  axis=0)   # (N, 4)
resnet_labels = np.concatenate(all_labels, axis=0)   # (N,)

resnet_acc = accuracy_score(resnet_labels, resnet_probs.argmax(1)) * 100
print(f'ResNet v17 val acc: {resnet_acc:.2f}%')

# Verificare ca labelele sunt aceleasi
if not np.array_equal(y_val, resnet_labels):
    print('ATENTIE: label-urile nu se potrivesc intre CLF si ENV dataset!')
    print(f'CLF labels[:10]: {y_val[:10]}')
    print(f'ENV labels[:10]: {resnet_labels[:10]}')
else:
    print('Labels match OK.')

# ---------------------------------------------------------------------------
# Sweep w_RF
print('\n--- Ensemble sweep ---')
print(f'{"w_RF":>6}  {"w_RN":>6}  {"acc":>8}')
best_acc  = 0.0
best_w_rf = 0.5
results   = []

for w_rf in np.arange(0.0, 1.01, 0.05):
    w_rn = 1.0 - w_rf
    combined = w_rf * rf_probs + w_rn * resnet_probs
    preds    = combined.argmax(1)
    acc      = accuracy_score(y_val, preds) * 100
    results.append((w_rf, acc))
    print(f'{w_rf:6.2f}  {w_rn:6.2f}  {acc:8.2f}%')
    if acc > best_acc:
        best_acc  = acc
        best_w_rf = w_rf

print(f'\nBest ensemble: w_RF={best_w_rf:.2f}  acc={best_acc:.2f}%')
print(f'RF solo:    {rf_val_acc:.2f}%')
print(f'ResNet solo:{resnet_acc:.2f}%')
print(f'RF v14 baseline: 92.55%')

# Confusion matrix la best weight
w_rn = 1.0 - best_w_rf
combined_best = best_w_rf * rf_probs + w_rn * resnet_probs
preds_best    = combined_best.argmax(1)
cm = confusion_matrix(y_val, preds_best)
print(f'\nConfusion matrix (best w_RF={best_w_rf:.2f}):')
print(f'  {"":12}' + ''.join(f'{l:>12}' for l in LABELS))
for i, row in enumerate(cm):
    print(f'  {LABELS[i]:12}' + ''.join(f'{v:12d}' for v in row))

# Salveaza rezultatele
out = {
    'rf_acc': rf_val_acc,
    'resnet_acc': resnet_acc,
    'best_ensemble_acc': best_acc,
    'best_w_rf': float(best_w_rf),
    'sweep': [(float(w), float(a)) for w, a in results],
    'confusion_matrix': cm.tolist(),
}
out_path = '../models/ensemble_rf_resnet17_results.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f'\nSalvat: {out_path}')
