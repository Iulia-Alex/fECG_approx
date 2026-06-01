"""
Training script v17 — ResNet1D mic + regularizare agresiva pe anvelopa Hilbert.

Problema v16: ResNet mare (64->128->256->256) overfit masiv (train 99.6%, val 88.9%).

Solutii combinate:
  1. ResNet mai mic: 6->32->64->64, ~60K params (vs 500K in v16)
  2. Dropout agresiv: 0.5 in head, 0.2 in blocks
  3. Data augmentation pe anvelopa:
     - Gaussian noise (sigma=0.05)
     - Random time shift circular (±20 pts)
     - Random amplitude scale per canal (0.8-1.2)
     - Random channel dropout (zero out 1 canal cu p=0.2)
  4. Weight decay 1e-2 (mai mare)
  5. Label smoothing 0.1
"""

import os, sys, json, time, datetime
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envelope_dataset import EnvelopeDataset, N_OUT

DATA_DIR          = '../data/movement_ecg'
MODEL_SAVE_PATH   = '../models/movement_clf_v17.pth'
HISTORY_PATH      = '../models/movement_clf_v17_history.json'
LOG_DIR           = '../logs'

LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-2
BATCH_SIZE    = 256
MAX_EPOCHS    = 300
PATIENCE      = 30
VAL_SPLIT     = 0.15
NUM_WORKERS   = 0
PRINT_EVERY   = 10
SEED          = 42
N_CLASSES     = 4
IN_CH         = 6
DROPOUT       = 0.5
LABEL_SMOOTH  = 0.1


# ---------------------------------------------------------------------------
# Augmentare pe anvelopa
# ---------------------------------------------------------------------------
class AugmentedSubset(Dataset):
    """Wrapper care aplica augmentari random pe train set."""
    def __init__(self, dataset, indices, augment=True):
        self.dataset = dataset
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.dataset[self.indices[i]]
        if self.augment:
            x = self._augment(x)
        return x, y

    def _augment(self, x):
        # x: (6, 200) tensor
        x = x.clone()

        # 1. Gaussian noise
        if torch.rand(1) < 0.5:
            x = x + torch.randn_like(x) * 0.05

        # 2. Random time shift circular (±20 pts)
        if torch.rand(1) < 0.5:
            shift = torch.randint(-20, 21, (1,)).item()
            x = torch.roll(x, shift, dims=1)

        # 3. Random amplitude scale per canal
        if torch.rand(1) < 0.5:
            scale = 0.8 + torch.rand(6, 1) * 0.4   # [0.8, 1.2]
            x = x * scale

        # 4. Random channel dropout (zero out 1 canal cu p=0.2)
        if torch.rand(1) < 0.2:
            ch = torch.randint(0, 6, (1,)).item()
            x[ch] = 0.0

        return x


# ---------------------------------------------------------------------------
# ResNet1D mic
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 5, stride=stride, padding=2, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, 5, padding=2, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        ) if stride != 1 or in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class SmallEnvResNet(nn.Module):
    """
    ResNet1D mic pe anvelopa (6, 200).
    Canale: 6 -> 32 -> 64 -> 64 -> GAP -> FC
    200 -> 100 -> 50 -> 25 -> GAP(1) -> 64 -> 4
    ~60K params
    """
    def __init__(self, in_ch=6, n_classes=4, dropout=0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        )
        self.l1 = ResBlock(32, 32, stride=2, dropout=0.2)   # 200->100
        self.l2 = ResBlock(32, 64, stride=2, dropout=0.2)   # 100->50
        self.l3 = ResBlock(64, 64, stride=2, dropout=0.2)   # 50->25
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x); x = self.l2(x); x = self.l3(x)
        return self.head(self.pool(x))


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


def main():
    torch.manual_seed(SEED)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU   : {torch.cuda.get_device_name(0)}')

    print(f'\nLoading EnvelopeDataset from {DATA_DIR} ...')
    full_ds = EnvelopeDataset(DATA_DIR)
    train_idx, val_idx = full_ds.file_split(val_frac=VAL_SPLIT, seed=SEED)
    print(f'Train: {len(train_idx)}  Val: {len(val_idx)}')

    weights   = full_ds.compute_class_weights(train_idx)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device),
                                     label_smoothing=LABEL_SMOOTH)

    train_set    = AugmentedSubset(full_ds, train_idx, augment=True)
    val_set      = AugmentedSubset(full_ds, val_idx,   augment=False)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    model = SmallEnvResNet(in_ch=IN_CH, n_classes=N_CLASSES,
                           dropout=DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: SmallEnvResNet  params={n_params/1e3:.1f}K')

    optimizer  = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    early_stop = EarlyStopping(patience=PATIENCE, save_path=MODEL_SAVE_PATH)
    history    = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[], 'lr':[]}

    t_start = time.time()
    print('='*70)
    print(f'Started : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f'LR={LEARNING_RATE}, WD={WEIGHT_DECAY}, dropout={DROPOUT}, '
          f'label_smooth={LABEL_SMOOTH}')
    print(f'Augment: noise + time_shift + amp_scale + channel_dropout')
    print('='*70)

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
        scheduler.step()
        if early_stop.step(va_loss, model, epoch):
            print(f'\nEarly stopping after {epoch+1} epochs.')
            break

    best_acc = max(history['val_acc'])
    best_ep  = history['val_acc'].index(best_acc) + 1
    print(f'\nBest val acc: {best_acc:.2f}% @ ep{best_ep}')
    print(f'[RF v14 baseline: 92.55%]')


if __name__ == '__main__':
    main()
