"""
Training script v3 — v1 architecture + peak-weighted loss.

Same as v1:
  - ComplexUNet 0.59M params, 128×400, 1000 Hz
  - AdamW, LR=1e-4, WD=1e-5, ReduceLROnPlateau, grad clipping

New loss:
  - SignalMSE + λ_peak * PeakMSE  (extra penalty on QRS regions)
  - PeakMSE: MSE computed only on samples near detected R-peaks
  - λ_complex * ComplexMSE  (spectral real+imag fidelity)
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
MODEL_SAVE_PATH = '../models/movement_CUNet_128x400_peakw.pth'
HISTORY_PATH    = '../models/movement_CUNet_128x400_peakw_history.json'
LOG_DIR         = '../logs'

LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-5
BATCH_SIZE      = 32
MAX_EPOCHS      = 200
PATIENCE        = 15
VAL_SPLIT       = 0.15
NUM_WORKERS     = 2
PRINT_EVERY     = 20
SEED            = 42

IN_CHANNELS     = 6
DIMENSION       = TARGET_SIZE_F * TARGET_SIZE_T  # 128 × 400 = 51200

# Time-domain loss constants
WINDOW_SAMPLES  = 4 * FS                        # 4000
ORIG_F          = NFFT // 2 + 1                  # 129
ORIG_T          = 1 + WINDOW_SAMPLES // HOP_LENGTH  # 401

# Loss weights
LAMBDA_PEAK     = 5.0    # extra weight on QRS peak regions
LAMBDA_COMPLEX  = 0.1    # spectral loss weight (small, to not suppress amplitude)

# Peak detection params
PEAK_HALF_WIN   = 40     # ±40 samples around R-peak = ±40ms at 1000Hz (covers QRS)

# Hann window cache
_DEVICE_HANN: dict = {}

def _get_hann(device):
    key = str(device)
    if key not in _DEVICE_HANN:
        _DEVICE_HANN[key] = torch.hann_window(WIN_LENGTH, device=device)
    return _DEVICE_HANN[key]


# ---------------------------------------------------------------------------
# Peak mask: detect R-peaks in ground-truth fECG and create binary mask
# ---------------------------------------------------------------------------
def _make_peak_mask(fecg_time: torch.Tensor, half_win: int = PEAK_HALF_WIN) -> torch.Tensor:
    """
    fecg_time: (B, C, T) float32 — ground-truth fECG in time domain
    Returns: (B, C, T) float32 mask — 1.0 near QRS peaks, 0.0 elsewhere

    Strategy: per-channel, find samples where |signal| > 3 * std,
    then dilate by ±half_win samples.
    """
    B, C, T = fecg_time.shape
    mask = torch.zeros_like(fecg_time)

    abs_sig = fecg_time.abs()
    # Per-channel threshold: 3× std (robust peak detection)
    std = abs_sig.std(dim=2, keepdim=True)
    threshold = 3.0 * std
    peaks = abs_sig > threshold  # (B, C, T) bool

    # Dilate: use max_pool1d with kernel=2*half_win+1
    # Reshape to (B*C, 1, T) for pooling
    peaks_float = peaks.float().reshape(B * C, 1, T)
    kernel = 2 * half_win + 1
    dilated = F.max_pool1d(peaks_float, kernel_size=kernel, stride=1,
                           padding=half_win)  # (B*C, 1, T)
    mask = dilated.reshape(B, C, T)
    return mask


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def _istft_pred(pred_spec: torch.Tensor) -> torch.Tensor:
    """Convert predicted spectrogram (B, C, 128, 400) → time domain (B, C, 4000)."""
    B, C, H, W = pred_spec.shape
    real = F.interpolate(
        pred_spec.real.reshape(B * C, 1, H, W),
        size=(ORIG_F, ORIG_T), mode='bilinear', align_corners=False,
    ).squeeze(1)
    imag = F.interpolate(
        pred_spec.imag.reshape(B * C, 1, H, W),
        size=(ORIG_F, ORIG_T), mode='bilinear', align_corners=False,
    ).squeeze(1)
    spec = torch.complex(real, imag)
    window = _get_hann(spec.device)
    pred_time = torch.istft(
        spec, n_fft=NFFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, window=window, length=WINDOW_SAMPLES,
    )
    return pred_time.reshape(B, C, WINDOW_SAMPLES)


def peak_weighted_loss(pred_spec: torch.Tensor, target_spec: torch.Tensor,
                       fecg_time: torch.Tensor) -> torch.Tensor:
    """
    Combined loss:
      1. SignalMSE: full time-domain MSE
      2. PeakMSE:   MSE only on QRS regions (weighted by LAMBDA_PEAK)
      3. ComplexMSE: spectral real+imag MSE (weighted by LAMBDA_COMPLEX)
    """
    # Time-domain reconstruction
    pred_time = _istft_pred(pred_spec)

    # 1. Full signal MSE
    sig_mse = F.mse_loss(pred_time, fecg_time)

    # 2. Peak-weighted MSE
    mask = _make_peak_mask(fecg_time)  # (B, C, T), 1 near peaks
    n_peak = mask.sum()
    if n_peak > 0:
        peak_mse = ((pred_time - fecg_time).pow(2) * mask).sum() / n_peak
    else:
        peak_mse = torch.tensor(0.0, device=pred_spec.device)

    # 3. Complex MSE (spectral)
    diff = pred_spec - target_spec
    complex_mse = diff.real.pow(2).mean() + diff.imag.pow(2).mean()

    total = sig_mse + LAMBDA_PEAK * peak_mse + LAMBDA_COMPLEX * complex_mse
    return total


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
            y      = y.to(device)
            y_time = y_time.to(device)

            pred = model(x)
            loss = peak_weighted_loss(pred, y, y_time)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

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

    print(f'Loading dataset from {DATA_DIR} ...')
    full_dataset = MovementECGDataset(DATA_DIR)
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

    model = ComplexUNet(DIMENSION, in_channels=IN_CHANNELS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params / 1e6:.2f} M')
    print()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    early_stop = EarlyStopping(patience=PATIENCE, save_path=MODEL_SAVE_PATH)

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    t_start = time.time()

    print('=' * 70)
    print(f'Training started: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Architecture : ComplexUNet v1 (6ch, 128x400, 0.59M params)')
    print(f'Loss         : SignalMSE + {LAMBDA_PEAK}*PeakMSE + {LAMBDA_COMPLEX}*ComplexMSE')
    print(f'Peak window  : +/-{PEAK_HALF_WIN} samples ({PEAK_HALF_WIN}ms at 1000Hz)')
    print(f'LR={LEARNING_RATE}, WD={WEIGHT_DECAY}, BS={BATCH_SIZE}, epochs={MAX_EPOCHS}')
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

        scheduler.step(val_loss)

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
