"""
Training script v16 — ResNet1D + Transformer pe anvelopa Hilbert a fECG.

Input: (6, 200) anvelopa amplitudine per faza completa
  - Hilbert transform pe semnalul brut per canal
  - Downsample la 200 puncte fixe (fara padding, fara masca)
  - Zero-mean, unit-std per canal per faza

Modele:
  1. ResNet1D — identic cu v10 dar input anvelopa, nu beat amplitudini
  2. Transformer — identic cu v11 dar input anvelopa

De ce va merge mai bine:
  - Linear:  anvelopa monotona (rampa) — usor de recunoscut
  - Helical: anvelopa oscilatorie (sinus) — usor de recunoscut
  - Exact ce se vede vizual in plots
  - Fara padding: 100% din cele 200 puncte sunt informatie reala
"""

import os, sys, json, time, datetime, math
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envelope_dataset import EnvelopeDataset, N_OUT

DATA_DIR           = '../data/movement_ecg'
MODEL_RESNET_PATH  = '../models/movement_clf_v16_resnet.pth'
MODEL_TRANS_PATH   = '../models/movement_clf_v16_transformer.pth'
HISTORY_PATH       = '../models/movement_clf_v16_history.json'
LOG_DIR            = '../logs'

LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-3
BATCH_SIZE    = 256
MAX_EPOCHS    = 200
PATIENCE      = 25
VAL_SPLIT     = 0.15
NUM_WORKERS   = 0
PRINT_EVERY   = 10
SEED          = 42
N_CLASSES     = 4
IN_CH         = 6
DROPOUT       = 0.3
D_MODEL       = 64


# ---------------------------------------------------------------------------
# ResNet1D
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.1):
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


class EnvResNet1D(nn.Module):
    """
    ResNet1D pe anvelopa (6, 200).
    200 -> 100 -> 50 -> 25 -> 12 -> GAP -> FC
    """
    def __init__(self, in_ch=6, n_classes=4, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )
        self.l1 = ResBlock(64,  64,  stride=2, dropout=0.1)
        self.l2 = ResBlock(64,  128, stride=2, dropout=0.1)
        self.l3 = ResBlock(128, 256, stride=2, dropout=0.2)
        self.l4 = ResBlock(256, 256, stride=2, dropout=0.2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x); x = self.l2(x)
        x = self.l3(x); x = self.l4(x)
        return self.head(self.pool(x))


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------
class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=N_OUT, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.drop(x + self.pe[:x.size(1)])


class EnvTransformer(nn.Module):
    """
    Transformer pe anvelopa (6, 200) — 200 timesteps, 6 canale ca features.
    """
    def __init__(self, in_ch=6, d_model=64, nhead=4, num_layers=3,
                 n_classes=4, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(in_ch, d_model)
        self.pe   = SinusoidalPE(d_model, dropout=dropout)
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

    def forward(self, x):
        # x: (B, 6, 200) -> (B, 200, 6) -> proj -> (B, 200, d_model)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = self.pe(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)


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
    os.makedirs(os.path.dirname(MODEL_RESNET_PATH), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU   : {torch.cuda.get_device_name(0)}')

    print(f'\nLoading EnvelopeDataset from {DATA_DIR} ...')
    full_ds = EnvelopeDataset(DATA_DIR)
    train_idx, val_idx = full_ds.file_split(val_frac=VAL_SPLIT, seed=SEED)
    print(f'Train: {len(train_idx)}  Val: {len(val_idx)}')

    weights      = full_ds.compute_class_weights(train_idx)
    criterion    = nn.CrossEntropyLoss(weight=weights.to(device))
    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(Subset(full_ds, val_idx),   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    history = {}

    # ResNet1D
    resnet = EnvResNet1D(in_ch=IN_CH, n_classes=N_CLASSES, dropout=DROPOUT).to(device)
    for k in ['resnet_train_loss','resnet_val_loss','resnet_train_acc','resnet_val_acc']:
        history[k] = []
    resnet_acc = train_model(resnet, 'resnet', MODEL_RESNET_PATH,
                             train_loader, val_loader, criterion, device, history)

    # Transformer
    transformer = EnvTransformer(in_ch=IN_CH, d_model=D_MODEL, nhead=4,
                                  num_layers=3, n_classes=N_CLASSES,
                                  dropout=DROPOUT).to(device)
    for k in ['transformer_train_loss','transformer_val_loss',
              'transformer_train_acc','transformer_val_acc']:
        history[k] = []
    trans_acc = train_model(transformer, 'transformer', MODEL_TRANS_PATH,
                            train_loader, val_loader, criterion, device, history)

    print(f'\n{"="*60}')
    print(f'REZULTATE FINALE v16 (anvelopa Hilbert 200pts):')
    print(f'  ResNet1D    val acc: {resnet_acc:.2f}%')
    print(f'  Transformer val acc: {trans_acc:.2f}%')
    print(f'  [RF v14 baseline:    92.55%]')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
