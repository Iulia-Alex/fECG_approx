"""
Resume training for v1 (ComplexUNet, SignalMSE) from the saved checkpoint.
Continues from epoch 201, appending to the existing history JSON.
"""

import os
import sys
import json
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from complex_network import ComplexUNet
from movement_dataset import (
    MovementECGDataset, TARGET_SIZE_F, TARGET_SIZE_T,
    NFFT, HOP_LENGTH, WIN_LENGTH, FS,
)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
DATA_DIR        = '../data/movement_ecg'
MODEL_SAVE_PATH = '../models/movement_CUNet_128x400_composed.pth'
HISTORY_PATH    = '../models/movement_CUNet_128x400_composed_history.json'
LOG_DIR         = '../logs'

LEARNING_RATE   = 1e-4      # same as original — LR never reduced during v1
WEIGHT_DECAY    = 1e-5
BATCH_SIZE      = 32
EXTRA_EPOCHS    = 100       # train 100 more epochs (201→300)
PATIENCE        = 20        # slightly more generous since we're near convergence
VAL_SPLIT       = 0.15
NUM_WORKERS     = 2
PRINT_EVERY     = 20
SEED            = 42

IN_CHANNELS     = 6
DIMENSION       = TARGET_SIZE_F * TARGET_SIZE_T
WINDOW_SAMPLES  = 4 * FS
ORIG_F          = NFFT // 2 + 1
ORIG_T          = 1 + WINDOW_SAMPLES // HOP_LENGTH

_DEVICE_HANN: dict = {}

def _get_hann(device):
    key = str(device)
    if key not in _DEVICE_HANN:
        _DEVICE_HANN[key] = torch.hann_window(WIN_LENGTH, device=device)
    return _DEVICE_HANN[key]


def signal_mse(pred_spec: torch.Tensor, fecg_time: torch.Tensor) -> torch.Tensor:
    B, C, H, W = pred_spec.shape
    real = F.interpolate(pred_spec.real.reshape(B * C, 1, H, W),
                         size=(ORIG_F, ORIG_T), mode='bilinear', align_corners=False).squeeze(1)
    imag = F.interpolate(pred_spec.imag.reshape(B * C, 1, H, W),
                         size=(ORIG_F, ORIG_T), mode='bilinear', align_corners=False).squeeze(1)
    spec = torch.complex(real, imag)
    window = _get_hann(spec.device)
    pred_time_bc = torch.istft(
        spec, n_fft=NFFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, window=window, length=WINDOW_SAMPLES,
    )
    pred_time = pred_time_bc.reshape(B, C, WINDOW_SAMPLES)
    return F.mse_loss(pred_time, fecg_time)


class EarlyStopping:
    def __init__(self, patience: int, save_path: str, best_loss: float, best_epoch: int):
        self.patience   = patience
        self.save_path  = save_path
        self.best_loss  = best_loss
        self.counter    = 0
        self.best_epoch = best_epoch

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        if val_loss < self.best_loss:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path)
            print(f'    [checkpoint] val_loss={val_loss:.6f} → saved')
        else:
            self.counter += 1
            print(f'    [early stop] no improvement {self.counter}/{self.patience} '
                  f'(best={self.best_loss:.6f} @ ep {self.best_epoch + 1})')
            if self.counter >= self.patience:
                return True
        return False


def run_epoch(model, loader, optimizer, device, train: bool,
              epoch_num: int, total_epochs: int):
    model.train() if train else model.eval()
    phase = 'train' if train else 'val'
    total_loss = 0.0
    n_batches  = len(loader)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_idx, (x, y, y_time) in enumerate(loader):
            x      = x.to(device)
            y      = y.to(device)
            y_time = y_time.to(device)
            pred   = model(x)
            loss   = signal_mse(pred, y_time)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % PRINT_EVERY == 0 or (batch_idx + 1) == n_batches:
                avg = total_loss / (batch_idx + 1)
                print(f'  [{phase}] epoch {epoch_num}/{total_epochs} '
                      f'| batch {batch_idx + 1}/{n_batches} '
                      f'| batch_loss={loss.item():.6f} '
                      f'| running_avg={avg:.6f}')
                sys.stdout.flush()
    return total_loss / n_batches


def main():
    torch.manual_seed(SEED)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU   : {torch.cuda.get_device_name(0)}')

    # Load existing history to find out how many epochs already done
    history = json.load(open(HISTORY_PATH))
    epochs_done = len(history['train_loss'])
    best_val    = min(history['val_loss'])
    best_ep_idx = history['val_loss'].index(best_val)
    print(f'\nResuming from epoch {epochs_done + 1}')
    print(f'Previous best val: {best_val:.6f} @ epoch {best_ep_idx + 1}')

    # Dataset (same split as original)
    print(f'\nLoading dataset from {DATA_DIR} ...')
    full_dataset = MovementECGDataset(DATA_DIR)
    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val
    print(f'Total windows: {n_total}  (train={n_train}, val={n_val})')

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(full_dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device == 'cuda'))
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(device == 'cuda'))

    # Load model from checkpoint
    model = ComplexUNet(DIMENSION, in_channels=IN_CHANNELS).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print(f'Checkpoint loaded from {MODEL_SAVE_PATH}')

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    early_stop = EarlyStopping(
        patience=PATIENCE,
        save_path=MODEL_SAVE_PATH,
        best_loss=best_val,
        best_epoch=best_ep_idx,
    )

    total_epochs = epochs_done + EXTRA_EPOCHS
    t_start = time.time()

    print('\n' + '=' * 70)
    print(f'Resume training started: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Epochs {epochs_done + 1} → {total_epochs}  |  patience={PATIENCE}  |  LR={LEARNING_RATE}')
    print('=' * 70)

    for i in range(EXTRA_EPOCHS):
        epoch = epochs_done + i        # 0-based absolute epoch index
        epoch_num = epoch + 1          # 1-based for display
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch_num}/{total_epochs}  |  lr={current_lr:.2e}')
        print('-' * 50)

        train_loss = run_epoch(model, train_loader, optimizer, device,
                               train=True, epoch_num=epoch_num, total_epochs=total_epochs)
        val_loss   = run_epoch(model, val_loader,   optimizer, device,
                               train=False, epoch_num=epoch_num, total_epochs=total_epochs)

        elapsed = time.time() - epoch_start
        total_elapsed = time.time() - t_start
        print(f'\n  >> Epoch {epoch_num} summary: '
              f'train={train_loss:.6f}  val={val_loss:.6f}  '
              f'time={elapsed:.1f}s  total={total_elapsed/60:.1f}min')

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        with open(HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)

        scheduler.step(val_loss)

        if early_stop.step(val_loss, model, epoch):
            print(f'\nEarly stopping triggered after {epoch_num} epochs.')
            break

    print('\n' + '=' * 70)
    print(f'Training finished: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Best val_loss   : {early_stop.best_loss:.6f} at epoch {early_stop.best_epoch + 1}')
    print(f'Model saved to  : {MODEL_SAVE_PATH}')
    print(f'History saved to: {HISTORY_PATH}')
    print('=' * 70)


if __name__ == '__main__':
    main()
