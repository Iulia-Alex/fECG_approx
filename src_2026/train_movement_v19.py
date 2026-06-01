"""
Training script v19 — ComplexAttentionUNet cu SI-SDR loss.

Față de v16 (SignalMSE):
  - Loss: SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) in loc de MSE.
  - SI-SDR e scale-invariant: modelul nu mai poate "trisa" suprimand amplitudinea
    fECG-ului — e fortat sa maximizeze raportul semnal/distorsiune indiferent de scala.
  - Rezolva problema de amplitudine suprimata observata la v1/v16/v17.
  - Arhitectura si dataset identice cu v16 (500 Hz, 128x128 STFT).

SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
  s_target = (dot(pred, gt) / ||gt||^2) * gt
  e_noise  = pred - s_target
Loss = -mean(SI-SDR)  [maximizam SI-SDR = minimizam loss negativ]
"""

import os, sys, json, time, datetime
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from complex_network_v16 import ComplexAttentionUNet
from movement_dataset_v15 import (
    MovementECGDatasetPaper,
    NFFT, HOP_LENGTH, WIN_LENGTH,
    FS, WINDOW, TARGET_SIZE_F, TARGET_SIZE_T,
)

# ---------------------------------------------------------------------------
DATA_DIR        = '../data/movement_ecg'
MODEL_SAVE_PATH = '../models/movement_CUNet_v19_sisdr.pth'
HISTORY_PATH    = '../models/movement_CUNet_v19_sisdr_history.json'
LOG_DIR         = '../logs'

LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-5
BATCH_SIZE      = 8
MAX_EPOCHS      = 400
PATIENCE        = 20
VAL_SPLIT       = 0.15
NUM_WORKERS     = 0  # 0 = single process, _bad_files se acumuleaza corect
PRINT_EVERY     = 20
SEED            = 42

IN_CHANNELS     = 6
DIMENSION       = TARGET_SIZE_F * TARGET_SIZE_T   # 128*128 = 16384
ORIG_F          = NFFT // 2 + 1                   # 129
WINDOW_SAMPLES  = WINDOW                          # 1915

_DEVICE_HANN: dict = {}


def _get_hann(device):
    if str(device) not in _DEVICE_HANN:
        _DEVICE_HANN[str(device)] = torch.hann_window(WIN_LENGTH, device=device)
    return _DEVICE_HANN[str(device)]


def _to_time(pred_spec: torch.Tensor) -> torch.Tensor:
    """(B, C, 128, 128) complex → (B, C, 1915) float via iSTFT."""
    B, C, Fq, T = pred_spec.shape
    zeros     = torch.zeros(B, C, 1, T, dtype=pred_spec.dtype, device=pred_spec.device)
    pred_full = torch.cat([pred_spec, zeros], dim=2)
    pred_flat = pred_full.reshape(B * C, ORIG_F, T)
    window    = _get_hann(pred_flat.device)
    pred_time = torch.istft(
        pred_flat, n_fft=NFFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, window=window,
        center=True, length=WINDOW_SAMPLES,
    ).reshape(B, C, WINDOW_SAMPLES)
    return pred_time


def si_sdr_loss(pred_time: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    pred_time, target: (B, C, T) float32
    Returneaza -SI-SDR mediat pe toate canalele si batch-ul.
    """
    # zero-mean
    pred   = pred_time   - pred_time.mean(dim=-1, keepdim=True)
    target = target      - target.mean(dim=-1, keepdim=True)

    # proiectia pred pe target
    dot          = (pred * target).sum(dim=-1, keepdim=True)
    target_norm2 = (target * target).sum(dim=-1, keepdim=True) + eps
    s_target     = (dot / target_norm2) * target

    # componenta de zgomot
    e_noise = pred - s_target

    # SI-SDR per (batch, canal)
    si_sdr = (s_target * s_target).sum(dim=-1) / ((e_noise * e_noise).sum(dim=-1) + eps)
    si_sdr = 10 * torch.log10(si_sdr + eps)

    return -si_sdr.mean()


class EarlyStopping:
    def __init__(self, patience):
        self.patience   = patience
        self.counter    = 0
        self.best_loss  = float('inf')
        self.best_epoch = 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.counter    = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'    [checkpoint] val_loss={val_loss:.6f} -> saved')
        else:
            self.counter += 1
            print(f'    [early stop] no improvement '
                  f'{self.counter}/{self.patience} '
                  f'(best={self.best_loss:.6f} @ ep {self.best_epoch + 1})')
        return self.counter >= self.patience


_current_epoch = 0


def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches  = len(loader)
    ctx        = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch_idx, (x, _, y_time) in enumerate(loader):
            x      = x.to(device)
            y_time = y_time.to(device)

            pred      = model(x)
            pred_time = _to_time(pred)
            loss      = si_sdr_loss(pred_time, y_time)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.apply(model.W_clipper)

            total_loss += loss.item()

            if (batch_idx + 1) % PRINT_EVERY == 0 or (batch_idx + 1) == n_batches:
                avg = total_loss / (batch_idx + 1)
                tag = 'train' if train else 'val'
                print(f'  [{tag}] ep {_current_epoch}/{MAX_EPOCHS} '
                      f'| batch {batch_idx+1}/{n_batches} '
                      f'| SI-SDR_loss={avg:.4f} dB')

    return total_loss / n_batches


def main():
    global _current_epoch

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, 'movement_CUNet_v19_sisdr.log')
    sys.stdout = open(log_path, 'w', buffering=1)
    sys.stderr = sys.stdout

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Device: {device}')
    print(f'Start : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f'Loss  : SI-SDR (Scale-Invariant SDR) — scale-invariant, rezolva supresia de amplitudine')

    dataset = MovementECGDatasetPaper(DATA_DIR)

    # Pre-marcare fisiere fara cache .npz ca bad — evita epuizarea fallback-ului
    n_bad = 0
    for file_idx, fpath in enumerate(dataset.files):
        npz = fpath.replace('.mat', '_signals_500hz.npz')
        if not os.path.exists(npz):
            dataset._bad_files.add(file_idx)
            n_bad += 1
    if n_bad:
        print(f'Pre-scan: {n_bad} fisiere fara cache .npz marcate bad (vor fi sarite)')

    n_val   = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f'Train: {n_train}  Val: {n_val}')

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model  = ComplexAttentionUNet(DIMENSION, in_channels=IN_CHANNELS).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params / 1e6:.3f} M')

    optimizer  = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8,
    )
    early_stop = EarlyStopping(patience=PATIENCE)

    epochs_done = 0
    history     = {'train_loss': [], 'val_loss': [], 'lr': []}
    if os.path.exists(HISTORY_PATH):
        history = json.load(open(HISTORY_PATH))
        if os.path.exists(MODEL_SAVE_PATH) and history['val_loss']:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
            epochs_done = len(history['val_loss'])
            early_stop.best_loss  = min(history['val_loss'])
            early_stop.best_epoch = history['val_loss'].index(early_stop.best_loss)
            print(f'Resumed: {epochs_done} epochs, '
                  f'best val={early_stop.best_loss:.4f} dB @ ep {early_stop.best_epoch + 1}')

    t0 = time.time()
    for epoch in range(epochs_done, MAX_EPOCHS):
        _current_epoch = epoch + 1
        t_ep = time.time()

        lr_now = optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch + 1}/{MAX_EPOCHS}  |  lr={lr_now:.2e}')
        print('-' * 50)

        tr_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        va_loss = run_epoch(model, val_loader,   optimizer, device, train=False)

        elapsed = time.time() - t0
        ep_time = time.time() - t_ep
        print(f'\n  >> Epoch {epoch+1}: '
              f'train={tr_loss:.4f} dB  val={va_loss:.4f} dB  '
              f't={ep_time:.1f}s  total={elapsed/60:.1f}min')

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['lr'].append(lr_now)
        json.dump(history, open(HISTORY_PATH, 'w'), indent=2)

        scheduler.step(va_loss)

        if early_stop.step(va_loss, model, epoch):
            print(f'\nEarly stopping after {epoch + 1} epochs.')
            break

    print(f'\n{"=" * 70}')
    print(f'Training finished: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f'Best val SI-SDR  : {-early_stop.best_loss:.4f} dB at epoch {early_stop.best_epoch + 1}')
    print(f'Model saved to   : {MODEL_SAVE_PATH}')
    print('=' * 70)


if __name__ == '__main__':
    main()
