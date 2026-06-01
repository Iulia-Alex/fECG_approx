"""
Training script v12 — RF + MLP pe 78 features extinse per faza.

Features (78 total):
  36 spectrale originale (std, slope, R², FFT amp, FFT freq, lag5) x 6 canale
  6  RR intervals (mean, std, cv, slope, min, max)
  18 morfologie QRS — varianta explicata primele 3 PC per canal
  18 anvelopa amplitudine (std, dom_freq, slope) x 6 canale

Modele antrenate:
  1. RandomForest (sklearn) — baseline rapid
  2. MLP (PyTorch)          — cu weighted CE loss

Motivatie: RF la 92.2% cu 36 features; RR intervals + morfologie QRS
ar trebui sa dea informatia necesara sa depasim 95%.
"""

import os, sys, json, time, datetime
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase_dataset_v2 import PhaseDatasetV2, N_CLASSES

DATA_DIR         = '../data/movement_ecg'
MODEL_MLP_PATH   = '../models/movement_clf_v12_mlp.pth'
MODEL_RF_PATH    = '../models/movement_clf_v12_rf.json'
HISTORY_PATH     = '../models/movement_clf_v12_history.json'
LOG_DIR          = '../logs'

# MLP hyperparams
LEARNING_RATE    = 3e-4
WEIGHT_DECAY     = 1e-3
BATCH_SIZE       = 256
MAX_EPOCHS       = 200
PATIENCE         = 25
VAL_SPLIT        = 0.15
NUM_WORKERS      = 0
PRINT_EVERY      = 10
SEED             = 42
DROPOUT          = 0.3
N_FEATURES       = 78


# ---------------------------------------------------------------------------
class PhaseMLP(nn.Module):
    def __init__(self, n_features=N_FEATURES, n_classes=4, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),        nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )
    def forward(self, x): return self.net(x)


class EarlyStopping:
    def __init__(self, patience, save_path):
        self.patience   = patience
        self.save_path  = save_path
        self.best_loss  = float('inf')
        self.counter    = 0
        self.best_epoch = 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path)
            print(f'    [checkpoint] val_loss={val_loss:.6f} -> saved')
        else:
            self.counter += 1
            print(f'    [early stop] {self.counter}/{self.patience} '
                  f'(best={self.best_loss:.6f} @ ep {self.best_epoch + 1})')
            if self.counter >= self.patience:
                return True
        return False


def run_epoch(model, loader, criterion, optimizer, device, train, ep, total):
    model.train() if train else model.eval()
    phase = 'train' if train else 'val'
    total_loss, correct, n_total = 0.0, 0, 0
    n_batches = len(loader)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for bi, (x, y) in enumerate(loader):
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            if train:
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            correct    += (logits.argmax(1) == y).sum().item()
            n_total    += y.size(0)
            if (bi + 1) % PRINT_EVERY == 0 or (bi + 1) == n_batches:
                print(f'  [{phase}] ep {ep}/{total} | batch {bi+1}/{n_batches} '
                      f'| loss={loss.item():.4f} | avg={total_loss/(bi+1):.4f} '
                      f'| acc={100*correct/n_total:.1f}%')
                sys.stdout.flush()
    return total_loss / n_batches, 100.0 * correct / n_total


# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    os.makedirs(os.path.dirname(MODEL_MLP_PATH), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU   : {torch.cuda.get_device_name(0)}')

    print(f'\nLoading PhaseDatasetV2 from {DATA_DIR} ...')
    full_ds = PhaseDatasetV2(DATA_DIR)
    train_idx, val_idx = full_ds.file_split(val_frac=VAL_SPLIT, seed=SEED)
    print(f'Train: {len(train_idx)}  Val: {len(val_idx)}')

    # -----------------------------------------------------------------------
    # 1. Random Forest
    # -----------------------------------------------------------------------
    print('\n' + '='*60)
    print('Training Random Forest (300 trees) on 78 features ...')
    X_train = full_ds.X_norm[train_idx]
    y_train = full_ds.y[train_idx]
    X_val   = full_ds.X_norm[val_idx]
    y_val   = full_ds.y[val_idx]

    rf = RandomForestClassifier(n_estimators=300, max_depth=20,
                                 min_samples_leaf=2, n_jobs=-1,
                                 class_weight='balanced', random_state=SEED)
    t0 = time.time()
    rf.fit(X_train, y_train)
    print(f'RF trained in {time.time()-t0:.1f}s')

    rf_train_acc = 100 * accuracy_score(y_train, rf.predict(X_train))
    rf_val_acc   = 100 * accuracy_score(y_val,   rf.predict(X_val))
    print(f'RF train acc: {rf_train_acc:.2f}%')
    print(f'RF val   acc: {rf_val_acc:.2f}%')
    cm = confusion_matrix(y_val, rf.predict(X_val))
    print(f'Confusion matrix:\n{cm}')

    # feature importance top 10
    imp = rf.feature_importances_
    top10 = np.argsort(imp)[::-1][:10]
    feature_names = (
        [f'spec_ch{c}_{n}' for c in range(6) for n in ['std','slope','r2','fft_amp','fft_freq','lag5']] +
        ['rr_mean','rr_std','rr_cv','rr_slope','rr_min','rr_max'] +
        [f'pca_ch{c}_pc{p}' for c in range(6) for p in [1,2,3]] +
        [f'env_ch{c}_{n}' for c in range(6) for n in ['std','freq','slope']]
    )
    print('Top 10 features:')
    for i in top10:
        print(f'  [{i:2d}] {feature_names[i]}: {imp[i]:.4f}')

    # -----------------------------------------------------------------------
    # 2. MLP
    # -----------------------------------------------------------------------
    print('\n' + '='*60)
    print('Training MLP on 78 features ...')

    weights     = full_ds.compute_class_weights(train_idx)
    train_set   = Subset(full_ds, train_idx)
    val_set     = Subset(full_ds, val_idx)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    model      = PhaseMLP(n_features=N_FEATURES, n_classes=N_CLASSES,
                          dropout=DROPOUT).to(device)
    n_params   = sum(p.numel() for p in model.parameters())
    print(f'MLP params: {n_params/1e3:.1f}K')

    criterion  = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer  = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, verbose=True)
    early_stop = EarlyStopping(patience=PATIENCE, save_path=MODEL_MLP_PATH)
    history    = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
                  'lr': [], 'rf_val_acc': rf_val_acc}

    t_start = time.time()
    print(f'Started: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f'LR={LEARNING_RATE}, BS={BATCH_SIZE}, patience={PATIENCE}')

    for epoch in range(MAX_EPOCHS):
        ep_start   = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch+1}/{MAX_EPOCHS}  |  lr={current_lr:.2e}')
        print('-'*50)

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer,
                                    device, True,  epoch+1, MAX_EPOCHS)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer,
                                    device, False, epoch+1, MAX_EPOCHS)

        elapsed = time.time() - ep_start
        print(f'\n  >> Epoch {epoch+1}: train={tr_loss:.4f} acc={tr_acc:.1f}%  '
              f'val={va_loss:.4f} acc={va_acc:.1f}%  '
              f't={elapsed:.0f}s  total={(time.time()-t_start)/60:.1f}min')

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        history['lr'].append(current_lr)

        with open(HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)

        scheduler.step(va_loss)
        if early_stop.step(va_loss, model, epoch):
            print(f'\nEarly stopping after {epoch+1} epochs.')
            break

    print(f'\n{"="*60}')
    print(f'RF  val acc: {rf_val_acc:.2f}%')
    print(f'MLP val acc: {max(history["val_acc"]):.2f}% @ ep{history["val_acc"].index(max(history["val_acc"]))+1}')
    print(f'Best val_loss: {early_stop.best_loss:.6f} @ ep{early_stop.best_epoch+1}')


if __name__ == '__main__':
    main()
