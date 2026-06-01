"""
Training script v15 — RF + MLP + Transformer pe 72 features (spectrale + faza inter-canal).

Features v15 (72 total):
  36 spectrale originale
  15 phase difference inter-canal la frecventa dominanta
  15 coerenta spectrala inter-canal
   6 spectral peakedness per canal

Tinta: discriminare mai buna Linear<->Helical (128 erori in v14).
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
from clf_dataset import ClfDatasetV15, N_CLASSES

DATA_DIR         = '../data/movement_ecg'
MODEL_MLP_PATH   = '../models/movement_clf_v15_mlp.pth'
MODEL_TRANS_PATH = '../models/movement_clf_v15_transformer.pth'
HISTORY_PATH     = '../models/movement_clf_v15_history.json'
LOG_DIR          = '../logs'

LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-3
BATCH_SIZE    = 256
MAX_EPOCHS    = 200
PATIENCE      = 25
VAL_SPLIT     = 0.15
NUM_WORKERS   = 0
PRINT_EVERY   = 10
SEED          = 42
DROPOUT       = 0.3
N_FEATURES    = 72
D_MODEL       = 64


# ---------------------------------------------------------------------------
class PhaseMLP(nn.Module):
    def __init__(self, n_in=N_FEATURES, n_classes=4, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )
    def forward(self, x): return self.net(x)


class FeatureTransformer(nn.Module):
    def __init__(self, n_features=N_FEATURES, d_model=D_MODEL, nhead=4,
                 num_layers=3, n_classes=4, dropout=0.3):
        super().__init__()
        self.proj    = nn.Linear(1, d_model)
        self.pos_emb = nn.Embedding(n_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )
        self.n_features = n_features

    def forward(self, x):
        x   = x.unsqueeze(-1)
        x   = self.proj(x)
        pos = torch.arange(self.n_features, device=x.device)
        x   = x + self.pos_emb(pos)
        x   = self.encoder(x)
        x   = x.mean(dim=1)
        return self.head(x)


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
                  f'(best={self.best_loss:.6f} @ ep {self.best_epoch+1})')
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
            if (bi+1) % PRINT_EVERY == 0 or (bi+1) == n_batches:
                print(f'  [{phase}] ep {ep}/{total} | batch {bi+1}/{n_batches} '
                      f'| loss={loss.item():.4f} | avg={total_loss/(bi+1):.4f} '
                      f'| acc={100*correct/n_total:.1f}%')
                sys.stdout.flush()
    return total_loss / n_batches, 100.0 * correct / n_total


def train_model(model, name, save_path, train_loader, val_loader,
                criterion, device, history):
    optimizer  = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, verbose=True)
    early_stop = EarlyStopping(patience=PATIENCE, save_path=save_path)
    n_params   = sum(p.numel() for p in model.parameters())
    print(f'\n{"="*60}\nTraining {name}  params={n_params/1e3:.1f}K')
    print(f'Started: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    t_start = time.time()

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
        history[f'{name}_train_loss'].append(tr_loss)
        history[f'{name}_val_loss'].append(va_loss)
        history[f'{name}_train_acc'].append(tr_acc)
        history[f'{name}_val_acc'].append(va_acc)
        with open(HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)
        scheduler.step(va_loss)
        if early_stop.step(va_loss, model, epoch):
            print(f'\nEarly stopping after {epoch+1} epochs.')
            break

    best_acc = max(history[f'{name}_val_acc'])
    best_ep  = history[f'{name}_val_acc'].index(best_acc) + 1
    print(f'\n{name} best val acc: {best_acc:.2f}% @ ep{best_ep}')
    return best_acc


def main():
    torch.manual_seed(SEED)
    os.makedirs(os.path.dirname(MODEL_MLP_PATH), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU   : {torch.cuda.get_device_name(0)}')

    print(f'\nLoading ClfDatasetV15 from {DATA_DIR} ...')
    full_ds = ClfDatasetV15(DATA_DIR)
    train_idx, val_idx = full_ds.file_split(val_frac=VAL_SPLIT, seed=SEED)
    print(f'Train: {len(train_idx)}  Val: {len(val_idx)}')

    weights      = full_ds.compute_class_weights(train_idx)
    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(Subset(full_ds, val_idx),   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)
    history      = {}

    # -----------------------------------------------------------------------
    # 1. Random Forest
    # -----------------------------------------------------------------------
    print('\n' + '='*60)
    print('Training Random Forest (500 trees) on 72 features ...')
    X_tr = full_ds.X_norm[train_idx]; y_tr = full_ds.y[train_idx]
    X_va = full_ds.X_norm[val_idx];   y_va = full_ds.y[val_idx]

    rf = RandomForestClassifier(n_estimators=500, max_depth=None,
                                 min_samples_leaf=1, n_jobs=-1,
                                 class_weight='balanced', random_state=SEED)
    t0 = time.time()
    rf.fit(X_tr, y_tr)
    print(f'RF trained in {time.time()-t0:.1f}s')

    rf_train_acc = 100 * accuracy_score(y_tr, rf.predict(X_tr))
    rf_val_acc   = 100 * accuracy_score(y_va, rf.predict(X_va))
    print(f'RF train acc: {rf_train_acc:.2f}%')
    print(f'RF val   acc: {rf_val_acc:.2f}%')
    print(f'Confusion matrix:\n{confusion_matrix(y_va, rf.predict(X_va))}')

    imp   = rf.feature_importances_
    top15 = np.argsort(imp)[::-1][:15]
    names = (
        [f'spec_ch{c}_{n}' for c in range(6)
         for n in ['std','slope','r2','fft_amp','fft_freq','lag5']] +
        [f'phase_ch{i}{j}' for i in range(6) for j in range(i+1, 6)] +
        [f'coh_ch{i}{j}'   for i in range(6) for j in range(i+1, 6)] +
        [f'peak_ch{c}'     for c in range(6)]
    )
    print('Top 15 features:')
    for i in top15:
        print(f'  [{i:2d}] {names[i]}: {imp[i]:.4f}')

    history['rf_val_acc'] = rf_val_acc

    # -----------------------------------------------------------------------
    # 2. MLP
    # -----------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    mlp = PhaseMLP(n_in=N_FEATURES, n_classes=N_CLASSES, dropout=DROPOUT).to(device)
    for k in ['mlp_train_loss','mlp_val_loss','mlp_train_acc','mlp_val_acc']:
        history[k] = []
    mlp_acc = train_model(mlp, 'mlp', MODEL_MLP_PATH,
                          train_loader, val_loader, criterion, device, history)

    # -----------------------------------------------------------------------
    # 3. Transformer
    # -----------------------------------------------------------------------
    transformer = FeatureTransformer(n_features=N_FEATURES, d_model=D_MODEL,
                                     nhead=4, num_layers=3, n_classes=N_CLASSES,
                                     dropout=DROPOUT).to(device)
    for k in ['transformer_train_loss','transformer_val_loss',
              'transformer_train_acc','transformer_val_acc']:
        history[k] = []
    trans_acc = train_model(transformer, 'transformer', MODEL_TRANS_PATH,
                            train_loader, val_loader, criterion, device, history)

    print(f'\n{"="*60}')
    print(f'REZULTATE FINALE v15 (72 features: spectrale + faza inter-canal):')
    print(f'  RF          val acc: {rf_val_acc:.2f}%')
    print(f'  MLP         val acc: {mlp_acc:.2f}%')
    print(f'  Transformer val acc: {trans_acc:.2f}%')
    print(f'  [v14 RF baseline:    92.55%]')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
