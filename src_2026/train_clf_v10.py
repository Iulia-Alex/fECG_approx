"""
Training script v10 — ResNet1D pe faze complete (beat amplitude sequences).

Input: (6, 200) beat amplitudini paddate per faza completa
Arhitectura: ResNet1D cu 4 blocuri + AdaptiveAvgPool + FC→4
200 timesteps → 100→50→25→12 → GAP(1) → FC(256→4)

De ce bate RF/MLP (sper):
  - RF/MLP vad 36 features handcrafted — ResNet invata features direct din date
  - Acces la pattern temporale subtile in secventa de beats
  - Nu presupunem ce e relevant (std, slope, fft) — reteaua decide
"""

import os, sys, json, time, datetime
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beat_phase_dataset import BeatPhaseDataset, MAX_BEATS

DATA_DIR        = '../data/movement_ecg'
MODEL_SAVE_PATH = '../models/movement_clf_v10.pth'
HISTORY_PATH    = '../models/movement_clf_v10_history.json'
LOG_DIR         = '../logs'

LEARNING_RATE   = 3e-4
WEIGHT_DECAY    = 1e-3
BATCH_SIZE      = 256
MAX_EPOCHS      = 200
PATIENCE        = 25
VAL_SPLIT       = 0.15
NUM_WORKERS     = 0
PRINT_EVERY     = 10
SEED            = 42
N_CLASSES       = 4
DROPOUT         = 0.3
IN_CHANNELS     = 6


# ---------------------------------------------------------------------------
class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=5, stride=1, downsample=None, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, stride=stride,
                               padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(channels)
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class PhaseResNet1D(nn.Module):
    """
    ResNet1D pe secvente de beat amplitudini (6, 200).
    Arhitectura:
      stem: Conv(6→64) + BN + ReLU
      block1: 64ch, stride=2  → 200→100
      block2: 128ch, stride=2 → 100→50
      block3: 256ch, stride=2 → 50→25
      block4: 256ch, stride=2 → 25→12
      AdaptiveAvgPool → (256, 1)
      FC: 256→128→4
    """
    def __init__(self, in_channels=6, n_classes=4, dropout=0.3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64,  64,  stride=2, dropout=0.1)
        self.layer2 = self._make_layer(64,  128, stride=2, dropout=0.1)
        self.layer3 = self._make_layer(128, 256, stride=2, dropout=0.2)
        self.layer4 = self._make_layer(256, 256, stride=2, dropout=0.2)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def _make_layer(self, in_ch, out_ch, stride, dropout):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        layers = []
        # primul bloc face downsampling si schimba canale
        layers.append(nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=stride, padding=2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_ch),
        ))
        layers.append(nn.ModuleList([downsample]))
        return nn.ModuleList(layers)

    def _forward_layer(self, layer, x):
        conv_block, (downsample,) = layer
        identity = x
        out = conv_block(x)
        if downsample is not None:
            identity = downsample(x)
        return torch.relu(out + identity)

    def forward(self, x):
        x = self.stem(x)
        x = self._forward_layer(self.layer1, x)
        x = self._forward_layer(self.layer2, x)
        x = self._forward_layer(self.layer3, x)
        x = self._forward_layer(self.layer4, x)
        x = self.pool(x)
        return self.classifier(x)


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
    phase = 'train' if train else 'val'
    total_loss, correct, n_total = 0.0, 0, 0
    n_batches = len(loader)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for bi, (x, mask, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
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

    print(f'\nLoading BeatPhaseDataset from {DATA_DIR} ...')
    full_dataset = BeatPhaseDataset(DATA_DIR, max_beats=MAX_BEATS)
    train_idx, val_idx = full_dataset.file_split(val_frac=VAL_SPLIT, seed=SEED)
    train_set = Subset(full_dataset, train_idx)
    val_set   = Subset(full_dataset, val_idx)
    print(f'Train: {len(train_set)}  Val: {len(val_set)}\n')

    weights = full_dataset.compute_class_weights(train_idx)
    print(f'Class weights: {weights.tolist()}')

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    model   = PhaseResNet1D(in_channels=IN_CHANNELS, n_classes=N_CLASSES,
                            dropout=DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: PhaseResNet1D (6,200→64→128→256→256→GAP→FC)  params={n_params/1e3:.1f}K')

    criterion  = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer  = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, verbose=True)
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
    print(f'Started    : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f'From epoch : {epochs_done + 1}')
    print(f'LR={LEARNING_RATE}, WD={WEIGHT_DECAY}, BS={BATCH_SIZE}, patience={PATIENCE}')
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
