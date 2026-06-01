"""
Training script v6 — BeatTransformer mic cu normalizare globala.

Diferente fata de v4:
  - Normalizare globala (refoloseste cache-ul de la v5)
  - Transformer mai mic: n_layers=2, d_model=32 (era 4 layers, d_model=64)
    → mai putini parametri, mai greu de overfit, gradient mai stabil
  - LR=1e-4 (era 3e-4 — probabil prea mare pt ce a cauzat minimul plat)
  - GPU (--gres=gpu:1 in run script)
"""

import os, sys, json, time, datetime
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from beat_dataset    import BeatDataset
from transformer_clf import BeatTransformer

# ---------------------------------------------------------------------------
DATA_DIR         = '../data/movement_ecg'
MODEL_SAVE_PATH  = '../models/movement_clf_v6.pth'
HISTORY_PATH     = '../models/movement_clf_v6_history.json'
WEIGHTS_PATH     = '../models/movement_clf_v3_weights.json'   # refolosim cache
NORM_STATS_PATH  = '../models/movement_clf_v5_norm_stats.npz' # refolosim de la v5
LOG_DIR          = '../logs'

LEARNING_RATE    = 1e-4
WEIGHT_DECAY     = 1e-3
BATCH_SIZE       = 256
MAX_EPOCHS       = 150
PATIENCE         = 20
VAL_SPLIT        = 0.15
NUM_WORKERS      = 2
PRINT_EVERY      = 20
SEED             = 42

N_CLASSES        = 4
D_MODEL          = 32
N_LAYERS         = 2
N_HEADS          = 4
DROPOUT          = 0.3


# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, train,
              epoch_num, total_epochs):
    model.train() if train else model.eval()
    phase      = 'train' if train else 'val'
    total_loss = 0.0
    correct    = 0
    n_total    = 0
    n_batches  = len(loader)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for bi, (x, mask, y) in enumerate(loader):
            x    = x.to(device)       # (B, MAX_BEATS, 6)
            mask = mask.to(device)    # (B, MAX_BEATS) bool — True = padded
            y    = y.to(device)

            logits = model(x, key_padding_mask=mask)
            loss   = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == y).sum().item()
            n_total    += y.size(0)

            if (bi + 1) % PRINT_EVERY == 0 or (bi + 1) == n_batches:
                avg = total_loss / (bi + 1)
                acc = 100.0 * correct / n_total
                print(f'  [{phase}] ep {epoch_num}/{total_epochs} '
                      f'| batch {bi+1}/{n_batches} '
                      f'| loss={loss.item():.4f} | avg={avg:.4f} | acc={acc:.1f}%')
                sys.stdout.flush()

    return total_loss / n_batches, 100.0 * correct / n_total


# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU   : {torch.cuda.get_device_name(0)}')

    print(f'\nLoading dataset from {DATA_DIR} ...')
    full_dataset = BeatDataset(DATA_DIR)
    train_idx, val_idx = full_dataset.file_split(val_frac=VAL_SPLIT, seed=SEED)
    print(f'Total windows: {len(full_dataset)}  '
          f'(train={len(train_idx)}, val={len(val_idx)})')
    print(f'File-level split: ~{int(len(full_dataset.files)*(1-VAL_SPLIT))} train files, '
          f'~{int(len(full_dataset.files)*VAL_SPLIT)} val files\n')

    print('Loading global norm stats ...')
    t0 = time.time()
    full_dataset.compute_norm_stats(train_idx, cache_path=NORM_STATS_PATH)
    print(f'  Done ({time.time()-t0:.1f}s)')

    print('Computing class weights ...')
    t0      = time.time()
    weights = full_dataset.compute_class_weights(train_idx, cache_path=WEIGHTS_PATH)
    print(f'Class weights: {weights.tolist()}  ({time.time()-t0:.1f}s)')

    train_set = Subset(full_dataset, train_idx)
    val_set   = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=(device == 'cuda'))
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=(device == 'cuda'))

    model    = BeatTransformer(n_channels=6, d_model=D_MODEL, n_heads=N_HEADS,
                               n_layers=N_LAYERS, n_classes=N_CLASSES,
                               dropout=DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: BeatTransformer (d_model={D_MODEL}, n_layers={N_LAYERS}, '
          f'dropout={DROPOUT})  params={n_params/1e3:.1f}K')

    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f'Resumed from checkpoint')

    criterion  = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer  = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, verbose=True)
    early_stop = EarlyStopping(patience=PATIENCE, save_path=MODEL_SAVE_PATH)

    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            history = json.load(f)
        early_stop.best_loss  = min(history['val_loss'])
        early_stop.best_epoch = history['val_loss'].index(early_stop.best_loss)
        print(f'History: {len(history["val_loss"])} epochs, '
              f'best={early_stop.best_loss:.6f} @ ep{early_stop.best_epoch+1}')
    else:
        history = {'train_loss': [], 'val_loss': [],
                   'train_acc': [], 'val_acc': [], 'lr': []}

    epochs_done = len(history['val_loss'])
    t_start     = time.time()

    print('=' * 70)
    print(f'Started      : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f'From epoch   : {epochs_done + 1}')
    print(f'Loss         : CrossEntropyLoss (weighted)')
    print(f'LR={LEARNING_RATE}, WD={WEIGHT_DECAY}, BS={BATCH_SIZE}, '
          f'patience={PATIENCE}')
    print('=' * 70)

    for epoch in range(epochs_done, epochs_done + MAX_EPOCHS):
        ep_start   = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        total_ep   = epochs_done + MAX_EPOCHS
        print(f'\nEpoch {epoch+1}/{total_ep}  |  lr={current_lr:.2e}')
        print('-' * 50)

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer,
                                    device, True,  epoch+1, total_ep)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer,
                                    device, False, epoch+1, total_ep)

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

    print(f'\nBest val_loss: {early_stop.best_loss:.6f} @ ep{early_stop.best_epoch+1}')


if __name__ == '__main__':
    main()
