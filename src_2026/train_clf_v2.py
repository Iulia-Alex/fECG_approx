"""
Training script v2 — ResNet1D for fetal movement classification, 8s windows.

Same as v1 but WINDOW_SEC=8 (8000 samples) instead of 4s.

Motivation: larger windows show more complete movement trajectories — a helical
or screw movement may span 10-20s, so 4s captures only a fragment. 8s gives
the model 10 heartbeats to see how QRS morphology evolves across channels.

Tradeoff: ~49K windows (vs 98K for v1) — half the data, but richer per sample.
"""

import os, sys, json, time, datetime
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clf_dataset   import MovementClfDataset
from resnet1d      import ResNet1D

# ---------------------------------------------------------------------------
DATA_DIR        = '../data/movement_ecg'
MODEL_SAVE_PATH  = '../models/movement_clf_v2.pth'
HISTORY_PATH     = '../models/movement_clf_v2_history.json'
WEIGHTS_PATH     = '../models/movement_clf_v2_weights.json'
LOG_DIR          = '../logs'

WINDOW_SEC      = 8    # 8s windows instead of 4s

LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4
BATCH_SIZE      = 64
MAX_EPOCHS      = 100
PATIENCE        = 15
VAL_SPLIT       = 0.15
NUM_WORKERS     = 2
PRINT_EVERY     = 50
SEED            = 42

IN_CHANNELS     = 6
N_CLASSES       = 4


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
            print(f'    [early stop] no improvement {self.counter}/{self.patience} '
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
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss   = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            preds       = logits.argmax(dim=1)
            correct    += (preds == y).sum().item()
            n_total    += y.size(0)

            if (batch_idx + 1) % PRINT_EVERY == 0 or (batch_idx + 1) == n_batches:
                avg = total_loss / (batch_idx + 1)
                acc = 100.0 * correct / n_total
                print(f'  [{phase}] ep {epoch_num}/{total_epochs} '
                      f'| batch {batch_idx + 1}/{n_batches} '
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
    full_dataset = MovementClfDataset(DATA_DIR, window_sec=WINDOW_SEC, stride_sec=WINDOW_SEC)
    train_idx, val_idx = full_dataset.file_split(val_frac=VAL_SPLIT, seed=SEED)
    train_set = Subset(full_dataset, train_idx)
    val_set   = Subset(full_dataset, val_idx)
    print(f'Total windows: {len(full_dataset)}  (train={len(train_set)}, val={len(val_set)})')
    print(f'File-level split: ~{int(len(full_dataset.files)*(1-VAL_SPLIT))} train files, '
          f'~{int(len(full_dataset.files)*VAL_SPLIT)} val files\n')

    # Class weights — computed file-by-file on first run, cached to JSON after
    print('Computing class weights (file-by-file, cached after first run) ...')
    t0 = time.time()
    weights = full_dataset.compute_class_weights(
        train_idx, cache_path=WEIGHTS_PATH
    )
    print(f'Class weights: {weights.tolist()}  ({time.time()-t0:.1f}s)')

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=(device == 'cuda'))
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=(device == 'cuda'))

    model    = ResNet1D(in_channels=IN_CHANNELS, n_classes=N_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params / 1e6:.2f} M')

    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f'Resumed from checkpoint: {MODEL_SAVE_PATH}')

    criterion  = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer  = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True,
    )
    early_stop = EarlyStopping(patience=PATIENCE, save_path=MODEL_SAVE_PATH)

    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            history = json.load(f)
        early_stop.best_loss  = min(history['val_loss'])
        early_stop.best_epoch = history['val_loss'].index(early_stop.best_loss)
        print(f'History loaded: {len(history["val_loss"])} epochs, '
              f'best val_loss={early_stop.best_loss:.6f} @ ep {early_stop.best_epoch + 1}')
    else:
        history = {'train_loss': [], 'val_loss': [],
                   'train_acc': [], 'val_acc': [], 'lr': []}

    epochs_done = len(history['val_loss'])
    t_start     = time.time()

    print('=' * 70)
    print(f'Started          : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Continuing from ep {epochs_done + 1}')
    print(f'Loss             : CrossEntropyLoss (weighted)')
    print(f'LR={LEARNING_RATE}, WD={WEIGHT_DECAY}, BS={BATCH_SIZE}, patience={PATIENCE}')
    print('=' * 70)

    for epoch in range(epochs_done, epochs_done + MAX_EPOCHS):
        epoch_start = time.time()
        current_lr  = optimizer.param_groups[0]['lr']
        total_ep    = epochs_done + MAX_EPOCHS
        print(f'\nEpoch {epoch + 1}/{total_ep}  |  lr={current_lr:.2e}')
        print('-' * 50)

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer,
                                          device, train=True,
                                          epoch_num=epoch+1, total_epochs=total_ep)
        val_loss, val_acc     = run_epoch(model, val_loader, criterion, optimizer,
                                          device, train=False,
                                          epoch_num=epoch+1, total_epochs=total_ep)

        elapsed       = time.time() - epoch_start
        total_elapsed = time.time() - t_start
        print(f'\n  >> Epoch {epoch + 1}: '
              f'train_loss={train_loss:.4f} acc={train_acc:.1f}%  '
              f'val_loss={val_loss:.4f} acc={val_acc:.1f}%  '
              f'time={elapsed:.1f}s  total={total_elapsed/60:.1f}min')

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        with open(HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)

        scheduler.step(val_loss)

        if early_stop.step(val_loss, model, epoch):
            print(f'\nEarly stopping after {epoch + 1} epochs.')
            break

    print('\n' + '=' * 70)
    print(f'Training finished: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Best val_loss    : {early_stop.best_loss:.6f} at epoch {early_stop.best_epoch + 1}')
    print(f'Model saved to   : {MODEL_SAVE_PATH}')
    print('=' * 70)


if __name__ == '__main__':
    main()
