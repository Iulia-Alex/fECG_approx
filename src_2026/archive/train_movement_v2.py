"""
Training script v2 — Paper-faithful architecture on movement ECG data.

Key differences from v1:
  - 500 Hz (resampled from 1000 Hz), 3.83 s windows
  - Natural 128×128 spectrograms (no resize)
  - Paper architecture with halved channels: 3 conv/block, BN, concat skips,
    RoActivation, sigmoid mask, separate real/imag weights (~15M params)
  - SignalMSE loss (iSTFT → time-domain MSE)
  - Adam (no weight decay), lr=1e-4, weight clipping
  - 100 epochs
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

from complex_network_paper import ComplexUNet
from movement_dataset_paper import (
    MovementECGDatasetPaper, NFFT, HOP_LENGTH, WIN_LENGTH,
    FS, WINDOW, TARGET_SIZE_F, TARGET_SIZE_T,
)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
DATA_DIR        = '../data/movement_ecg'
MODEL_SAVE_PATH = '../models/movement_CUNet_128x128_paper.pth'
HISTORY_PATH    = '../models/movement_CUNet_128x128_paper_history.json'
LOG_DIR         = '../logs'

LEARNING_RATE   = 1e-4
BATCH_SIZE      = 32      # 1 conv/block + halved channels fits BS=32
MAX_EPOCHS      = 100
PATIENCE        = 15
VAL_SPLIT       = 0.15
NUM_WORKERS     = 2
PRINT_EVERY     = 20
SEED            = 42

IN_CHANNELS     = 6
DIMENSION       = TARGET_SIZE_F * TARGET_SIZE_T  # 128 × 128 = 16384

# iSTFT constants
ORIG_F          = NFFT // 2 + 1   # 129
ORIG_T          = 128              # natural STFT time frames (no resize)
WINDOW_SAMPLES  = WINDOW           # 1915 samples at 500 Hz

# Hann window cache for GPU iSTFT
_DEVICE_HANN: dict = {}

def _get_hann(device):
    key = str(device)
    if key not in _DEVICE_HANN:
        _DEVICE_HANN[key] = torch.hann_window(WIN_LENGTH, device=device)
    return _DEVICE_HANN[key]


# ---------------------------------------------------------------------------
# Loss: SignalMSE (iSTFT → time-domain MSE, as in paper)
# ---------------------------------------------------------------------------
def signal_mse(pred_spec: torch.Tensor, fecg_time: torch.Tensor) -> torch.Tensor:
    """
    pred_spec : (B, C, 128, 128) complex — model output (last freq bin was dropped)
    fecg_time : (B, C, 1915) float32     — ground-truth time-domain fECG at 500 Hz
    """
    B, C, Fq, T = pred_spec.shape

    # Add back the dropped 129th freq bin as zeros
    zeros = torch.zeros(B, C, 1, T, dtype=pred_spec.dtype, device=pred_spec.device)
    pred_full = torch.cat([pred_spec, zeros], dim=2)  # (B, C, 129, 128)

    # Reshape for batch iSTFT
    pred_flat = pred_full.reshape(B * C, ORIG_F, ORIG_T)

    # GPU iSTFT
    window = _get_hann(pred_flat.device)
    pred_time = torch.istft(
        pred_flat, n_fft=NFFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, window=window,
        center=True, length=WINDOW_SAMPLES,
    )  # (B*C, 1915)

    pred_time = pred_time.reshape(B, C, WINDOW_SAMPLES)
    return F.mse_loss(pred_time, fecg_time)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience, save_path):
        self.patience = patience
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
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
# One epoch
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, device, train, epoch_num, total_epochs):
    model.train() if train else model.eval()
    phase = 'train' if train else 'val'
    total_loss = 0.0
    n_batches = len(loader)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_idx, (x, y, y_time) in enumerate(loader):
            x      = x.to(device)
            y_time = y_time.to(device)

            pred = model(x)
            loss = signal_mse(pred, y_time)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Weight clipping (paper approach)
                model.apply(model.W_clipper)

            total_loss += loss.item()

            if (batch_idx + 1) % PRINT_EVERY == 0 or (batch_idx + 1) == n_batches:
                avg = total_loss / (batch_idx + 1)
                print(f'  [{phase}] ep {epoch_num}/{total_epochs} '
                      f'| batch {batch_idx + 1}/{n_batches} '
                      f'| loss={loss.item():.6f} '
                      f'| avg={avg:.6f}')
                sys.stdout.flush()

    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU   : {torch.cuda.get_device_name(0)}')
    print()

    # Dataset (500 Hz, 128×128 spectrograms)
    print(f'Loading paper-style dataset from {DATA_DIR} ...')
    full_dataset = MovementECGDatasetPaper(DATA_DIR)
    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val
    print(f'Total windows: {n_total}  (train={n_train}, val={n_val})')
    print()

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(full_dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=(device == 'cuda'))
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=(device == 'cuda'))

    # Model — paper architecture for 6 channels
    model = ComplexUNet(DIMENSION, in_channels=IN_CHANNELS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params / 1e6:.2f} M')
    print()

    # Adam without weight decay (paper approach)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    early_stop = EarlyStopping(patience=PATIENCE, save_path=MODEL_SAVE_PATH)

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    t_start = time.time()

    print('=' * 70)
    print(f'Training started: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Architecture : Paper ComplexUNet (6ch, 128x128, halved channels)')
    print(f'Loss         : SignalMSE (iSTFT -> time-domain MSE)')
    print(f'Activation   : RoActivation (CReLU + GK + GroupSort)')
    print(f'LR={LEARNING_RATE}, BS={BATCH_SIZE}, epochs={MAX_EPOCHS}')
    print(f'STFT: n_fft={NFFT}, win={WIN_LENGTH}, hop={HOP_LENGTH}, FS={FS}Hz')
    print('=' * 70)

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch + 1}/{MAX_EPOCHS}  |  lr={current_lr:.2e}')
        print('-' * 50)

        train_loss = run_epoch(model, train_loader, optimizer, device,
                               train=True, epoch_num=epoch + 1, total_epochs=MAX_EPOCHS)
        val_loss   = run_epoch(model, val_loader,   optimizer, device,
                               train=False, epoch_num=epoch + 1, total_epochs=MAX_EPOCHS)

        elapsed = time.time() - epoch_start
        total_elapsed = time.time() - t_start

        print(f'\n  >> Epoch {epoch + 1}: '
              f'train={train_loss:.6f}  val={val_loss:.6f}  '
              f'time={elapsed:.1f}s  total={total_elapsed/60:.1f}min')

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        with open(HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)

        if early_stop.step(val_loss, model, epoch):
            print(f'\nEarly stopping after {epoch + 1} epochs.')
            break

    print('\n' + '=' * 70)
    print(f'Training finished: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Best val_loss   : {early_stop.best_loss:.6f} at epoch {early_stop.best_epoch + 1}')
    print(f'Model saved to  : {MODEL_SAVE_PATH}')
    print('=' * 70)


if __name__ == '__main__':
    main()
