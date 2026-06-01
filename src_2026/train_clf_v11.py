"""
Training script v11 — Transformer pe faze complete (beat amplitude sequences).

Input: (200, 6) — 200 beats ca tokens, 6 canale ca features per token
Arhitectura:
  Linear(6 -> d_model) + positional encoding sinusoidal
  3x TransformerEncoderLayer (d_model=64, nhead=4, dropout=0.3)
  Masked mean pooling (ignora padding) -> FC(64->32->4)

Avantaj vs ResNet:
  - Self-attention = fiecare beat "vede" toate celelalte beats din faza
  - Mask-ul de padding e folosit nativ in atentie -> nu penalizeaza zeros
  - Nu presupune localitate (ca ResNet cu kernel-uri locale)
  - Poate prinde relatii globale: "beat-ul de la inceput vs sfarsit"

Refoloseste BeatPhaseDataset (acelasi input ca v10).
"""

import os, sys, json, time, datetime, math
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beat_phase_dataset import BeatPhaseDataset, MAX_BEATS

DATA_DIR        = '../data/movement_ecg'
MODEL_SAVE_PATH = '../models/movement_clf_v11.pth'
HISTORY_PATH    = '../models/movement_clf_v11_history.json'
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
IN_CHANNELS     = 6   # features per beat token
D_MODEL         = 64
N_HEAD          = 4
N_LAYERS        = 3
DROPOUT         = 0.3


# ---------------------------------------------------------------------------
class SinusoidalPE(nn.Module):
    """Positional encoding sinusoidal standard."""
    def __init__(self, d_model, max_len=MAX_BEATS, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)   # (max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:x.size(1)]
        return self.drop(x)


class BeatTransformer(nn.Module):
    """
    Transformer encoder pe secvente de beat amplitudini.

    Input:
      x:    (B, 6, T) beat amplitudini  -- reordonam la (B, T, 6)
      mask: (B, T)    True = beat real, False = padding

    Procesare:
      1. Linear(6 -> d_model)
      2. Positional encoding
      3. N x TransformerEncoderLayer
         src_key_padding_mask = ~mask (True = ignora in atentie)
      4. Masked mean pooling -> (B, d_model)
      5. FC -> 4 clase
    """
    def __init__(self, in_features=6, d_model=64, nhead=4, num_layers=3,
                 n_classes=4, dropout=0.3, max_len=MAX_BEATS):
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)
        self.pe   = SinusoidalPE(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # pre-norm = mai stabil
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                             enable_nested_tensor=False)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x, mask):
        # x: (B, 6, T) -> (B, T, 6)
        x = x.permute(0, 2, 1)
        x = self.proj(x)           # (B, T, d_model)
        x = self.pe(x)             # + positional encoding

        # TransformerEncoder: src_key_padding_mask=True => ignora pozitia
        pad_mask = ~mask           # (B, T): True = padding, ignora
        x = self.encoder(x, src_key_padding_mask=pad_mask)   # (B, T, d_model)

        # Masked mean pooling: media doar pe beats reale
        mask_f = mask.unsqueeze(-1).float()   # (B, T, 1)
        x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)  # (B, d_model)

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
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits = model(x, mask)
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

    model = BeatTransformer(
        in_features=IN_CHANNELS, d_model=D_MODEL, nhead=N_HEAD,
        num_layers=N_LAYERS, n_classes=N_CLASSES, dropout=DROPOUT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: BeatTransformer d={D_MODEL} h={N_HEAD} L={N_LAYERS}  params={n_params/1e3:.1f}K')

    criterion  = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer  = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, verbose=True)
    early_stop = EarlyStopping(patience=PATIENCE, save_path=MODEL_SAVE_PATH)

    history = {'train_loss': [], 'val_loss': [],
               'train_acc':  [], 'val_acc':  [], 'lr': []}

    t_start = time.time()
    print('=' * 70)
    print(f'Started : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f'LR={LEARNING_RATE}, WD={WEIGHT_DECAY}, BS={BATCH_SIZE}, patience={PATIENCE}')
    print(f'd_model={D_MODEL}, nhead={N_HEAD}, layers={N_LAYERS}, dropout={DROPOUT}')
    print('=' * 70)

    for epoch in range(MAX_EPOCHS):
        ep_start   = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch+1}/{MAX_EPOCHS}  |  lr={current_lr:.2e}')
        print('-' * 50)

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

        scheduler.step(va_loss)
        if early_stop.step(va_loss, model, epoch):
            print(f'\nEarly stopping after {epoch+1} epochs.')
            break

    print(f'\nBest val_loss: {early_stop.best_loss:.6f} @ ep{early_stop.best_epoch+1}')


if __name__ == '__main__':
    main()
