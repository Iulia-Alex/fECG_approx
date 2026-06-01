"""
Training script v13 — CNN per beat + Transformer pe morfologie QRS completa.

Input per faza: (MAX_BEATS, 6, 50) — 150 beats x 6 canale x 50 sample-uri QRS
Arhitectura:
  BeatCNN: Conv1d(6->32->64) pe cei 50 sample-uri -> embedding 64-dim per beat
  Transformer: 3 layere, d=64, 4 capete, masked mean pooling
  FC: 64 -> 4 clase

De ce e mai bun decat v11 (beat amplitudini):
  - Morfologia completa QRS (50 sample-uri) vs un singur scalar per canal
  - CNN per beat invata ce e relevant in forma QRS (P, QRS complex, T wave)
  - Transformer pe secventa de embeddings: pattern temporal intre batai
  - Informatia despre distanta electrod-fat e in forma QRS, nu doar amplitudine
"""

import os, sys, json, time, datetime, math
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qrs_dataset import QRSPhaseDataset, MAX_BEATS, QRS_WIN

DATA_DIR        = '../data/movement_ecg'
MODEL_SAVE_PATH = '../models/movement_clf_v13.pth'
HISTORY_PATH    = '../models/movement_clf_v13_history.json'
LOG_DIR         = '../logs'

LEARNING_RATE   = 3e-4
WEIGHT_DECAY    = 1e-3
BATCH_SIZE      = 128
MAX_EPOCHS      = 200
PATIENCE        = 25
VAL_SPLIT       = 0.15
NUM_WORKERS     = 0
PRINT_EVERY     = 10
SEED            = 42
N_CLASSES       = 4
IN_CHANNELS     = 6
D_MODEL         = 64
N_HEAD          = 4
N_LAYERS        = 3
DROPOUT         = 0.3


# ---------------------------------------------------------------------------
class BeatCNN(nn.Module):
    """
    CNN mic care proceseaza morfologia unui singur beat: (6, 50) -> embedding 64.
    Aplicat independent pe fiecare beat din faza (shared weights).
    """
    def __init__(self, in_ch=6, d_out=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            # (B*T, 6, 50)
            nn.Conv1d(in_ch, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                              # -> (B*T, 32, 25)
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, d_out, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_out), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),                      # -> (B*T, d_out, 1)
        )

    def forward(self, x):
        # x: (B*T, 6, 50)
        return self.net(x).squeeze(-1)   # (B*T, d_out)


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=MAX_BEATS, dropout=0.1):
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


class QRSTransformer(nn.Module):
    """
    CNN per beat + Transformer pe secventa de embeddings.

    Input:
      x:    (B, T, 6, 50) morfologie QRS per beat
      mask: (B, T) bool — True = beat real

    Pipeline:
      1. BeatCNN pe fiecare beat: (B*T, 6, 50) -> (B*T, 64) -> (B, T, 64)
      2. Positional encoding
      3. TransformerEncoder cu masked attention
      4. Masked mean pooling -> (B, 64)
      5. FC -> 4 clase
    """
    def __init__(self, in_ch=6, qrs_win=QRS_WIN, d_model=64, nhead=4,
                 num_layers=3, n_classes=4, dropout=0.3):
        super().__init__()
        self.beat_cnn = BeatCNN(in_ch=in_ch, d_out=d_model, dropout=0.1)
        self.pe       = SinusoidalPE(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers,
                                             enable_nested_tensor=False)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x, mask):
        B, T, C, W = x.shape
        # aplica BeatCNN pe fiecare beat (shared weights)
        x_flat = x.reshape(B * T, C, W)          # (B*T, 6, 50)
        emb    = self.beat_cnn(x_flat)            # (B*T, 64)
        emb    = emb.reshape(B, T, -1)            # (B, T, 64)

        emb = self.pe(emb)

        pad_mask = ~mask                          # True = ignora
        emb = self.encoder(emb, src_key_padding_mask=pad_mask)  # (B, T, 64)

        # masked mean pooling
        mask_f = mask.unsqueeze(-1).float()
        emb    = (emb * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)  # (B, 64)

        return self.head(emb)


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


def run_epoch(model, loader, criterion, optimizer, device, train, ep, total):
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
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            correct    += (logits.argmax(1) == y).sum().item()
            n_total    += y.size(0)
            if (bi + 1) % PRINT_EVERY == 0 or (bi + 1) == n_batches:
                print(f'  [{phase}] ep {ep}/{total} | batch {bi+1}/{n_batches} '
                      f'| loss={loss.item():.4f} | avg={total_loss/(bi+1):.4f} '
                      f'| acc={100*correct/n_total:.1f}%')
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

    print(f'\nLoading QRSPhaseDataset from {DATA_DIR} ...')
    full_ds = QRSPhaseDataset(DATA_DIR, max_beats=MAX_BEATS)
    train_idx, val_idx = full_ds.file_split(val_frac=VAL_SPLIT, seed=SEED)
    train_set = Subset(full_ds, train_idx)
    val_set   = Subset(full_ds, val_idx)
    print(f'Train: {len(train_set)}  Val: {len(val_set)}\n')

    weights = full_ds.compute_class_weights(train_idx)
    print(f'Class weights: {weights.tolist()}')

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device=='cuda'))
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(device=='cuda'))

    model = QRSTransformer(
        in_ch=IN_CHANNELS, qrs_win=QRS_WIN, d_model=D_MODEL,
        nhead=N_HEAD, num_layers=N_LAYERS, n_classes=N_CLASSES,
        dropout=DROPOUT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: QRSTransformer (BeatCNN+Transformer)  params={n_params/1e3:.1f}K')

    criterion  = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer  = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, verbose=True)
    early_stop = EarlyStopping(patience=PATIENCE, save_path=MODEL_SAVE_PATH)
    history    = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}

    t_start = time.time()
    print('='*70)
    print(f'Started : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f'LR={LEARNING_RATE}, WD={WEIGHT_DECAY}, BS={BATCH_SIZE}, patience={PATIENCE}')
    print(f'd_model={D_MODEL}, nhead={N_HEAD}, layers={N_LAYERS}, dropout={DROPOUT}')
    print(f'Input: ({MAX_BEATS}, {IN_CHANNELS}, {QRS_WIN}) per faza')
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

        scheduler.step(va_loss)
        if early_stop.step(va_loss, model, epoch):
            print(f'\nEarly stopping after {epoch+1} epochs.')
            break

    print(f'\nBest val_loss: {early_stop.best_loss:.6f} @ ep{early_stop.best_epoch+1}')


if __name__ == '__main__':
    main()
