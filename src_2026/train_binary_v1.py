"""
train_binary_v1.py

Binary fetal movement detection (movement / no-movement) using a
Precision Res-U-Net (adapted from Edward's architecture) with 8-channel input.

Input channels per window (1000 Hz, WINDOW samples):
  0-5 : fECG channels, z-normalised per file
  6   : A_QRS signal (Rooijakkers et al., amplitude per beat, interpolated)
  7   : Baseline wander (moving avg 1 s window) on ch0

Target : movement_mask (0 = stationary, 1 = any movement: linear/helical/screw)

Pipeline
--------
1. Pre-process step: .mat → .npy cache (one-time, ~5 min for 655 files)
2. BinaryMovementDataset: mmap-based lazy window loading
3. File-level train/val split (val_frac=0.15, seed=42)
4. PrecisionResUNet(in_ch=8): BCE + Dice, AdamW, CosineAnnealingLR
5. Val metrics: loss, F1, IoU, per-window accuracy
"""

import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAT_DIR   = '/shared_storage/stan.edward/fECG_mvm_DB/DB_1/Long_time_intervals'
CACHE_DIR = '/shared_storage/iulia.orvas/paper/fECG_approx/data/binary_npy'
OUT_DIR   = '/shared_storage/iulia.orvas/paper/fECG_approx/models'

FS         = 1000
WINDOW     = 8000   # 8 s
STRIDE_TR  = 4000   # 4 s stride → ~82k windows total train
STRIDE_VA  = 8000   # non-overlapping for validation
VAL_FRAC   = 0.15
SEED       = 42
BATCH_TR   = 16
BATCH_VA   = 16
EPOCHS     = 100
LR         = 1e-4
POS_WEIGHT = 1.5    # ~60% no-mov / ~40% mov at sample level
IN_CH      = 8
Q_WIN      = 25     # 25 ms Q-peak search window
S_WIN      = 25     # 25 ms S-peak search window
BL_WIN     = 1000   # 1 s baseline moving average
THRESH     = 0.5    # decision threshold for F1/IoU

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)

MODEL_PATH = os.path.join(OUT_DIR, 'binary_v1_best.pth')
HIST_PATH  = os.path.join(OUT_DIR, 'binary_v1_history.json')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if BATCH_TR > 4 and device.type == 'cpu':
    BATCH_TR = 4
    BATCH_VA = 4
    STRIDE_TR = 4000
    print('CPU mode: batch=4, stride_tr=4000')


# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------
def _load_mat(fpath):
    mat = sio.loadmat(fpath)
    out = mat['out']
    fecg = out['fecg'][0][0]
    if fecg.dtype == object:
        fecg = fecg.flat[0]
    fecg = fecg.astype(np.float32)          # (6, N)

    fqrs = out['fqrs'][0][0]
    if fqrs.dtype == object:
        fqrs = fqrs.flat[0]
    fqrs = fqrs.ravel().astype(np.int32)

    mask = out['movement_mask'][0][0].ravel().astype(np.uint8)
    return fecg, fqrs, mask


def _compute_a_qrs(fecg_ch, fqrs):
    """A_QRS per beat (z-normalised), interpolated to dense signal."""
    vals, peaks = [], []
    for r in fqrs:
        if r < Q_WIN or r + S_WIN >= len(fecg_ch):
            continue
        r_val = float(fecg_ch[r])
        q_val = float(fecg_ch[r - Q_WIN:r].min())
        s_val = float(fecg_ch[r:r + S_WIN].min())
        vals.append(r_val - (q_val + s_val) / 2.0)
        peaks.append(r)
    if len(peaks) < 2:
        return np.zeros(len(fecg_ch), dtype=np.float32)
    v = np.array(vals, dtype=np.float32)
    p = np.array(peaks)
    v = (v - v.mean()) / (v.std() + 1e-8)
    return np.interp(np.arange(len(fecg_ch)), p, v).astype(np.float32)


def _compute_baseline(sig, win=BL_WIN):
    """Moving average baseline wander (edge-padded)."""
    pad = win // 2
    padded = np.pad(sig, (pad, pad), mode='edge')
    bl = np.convolve(padded, np.ones(win) / win, mode='valid')[:len(sig)]
    return bl.astype(np.float32)


def preprocess_file(fpath, cache_dir):
    stem = os.path.splitext(os.path.basename(fpath))[0]
    feat_p = os.path.join(cache_dir, f'{stem}_feat.npy')
    mask_p = os.path.join(cache_dir, f'{stem}_mask.npy')
    if os.path.exists(feat_p) and os.path.exists(mask_p):
        return True

    try:
        fecg, fqrs, mask = _load_mat(fpath)
    except Exception as e:
        print(f'  SKIP {stem}: {e}')
        return False

    N = fecg.shape[1]
    feats = np.empty((IN_CH, N), dtype=np.float32)

    # Channels 0-5: z-normalise per channel over whole file
    for ch in range(6):
        mu  = fecg[ch].mean()
        std = fecg[ch].std() + 1e-8
        feats[ch] = (fecg[ch] - mu) / std

    # Use normalised channel 0 for derived features
    ch0 = feats[0]

    # Channel 6: A_QRS
    feats[6] = _compute_a_qrs(ch0, fqrs)

    # Channel 7: baseline wander
    bl = _compute_baseline(ch0)
    bl = (bl - bl.mean()) / (bl.std() + 1e-8)
    feats[7] = bl

    np.save(feat_p, feats)
    np.save(mask_p, mask)
    return True


def run_preprocessing(mat_dir, cache_dir):
    files = sorted(f for f in os.listdir(mat_dir) if f.endswith('.mat'))
    print(f'Pre-processing {len(files)} .mat files → {cache_dir}')
    t0 = time.time()
    done = 0
    for i, fname in enumerate(files):
        ok = preprocess_file(os.path.join(mat_dir, fname), cache_dir)
        if ok:
            done += 1
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(files)} ({time.time()-t0:.0f}s)')
    print(f'Pre-processing done: {done}/{len(files)} files in {time.time()-t0:.0f}s')


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BinaryMovementDataset(Dataset):
    def __init__(self, stems, cache_dir, window, stride):
        self.cache_dir = cache_dir
        self.window    = window
        self.windows   = []   # list of (stem, start_sample)

        for stem in stems:
            feat_p = os.path.join(cache_dir, f'{stem}_feat.npy')
            if not os.path.exists(feat_p):
                continue
            feat = np.load(feat_p, mmap_mode='r')
            N = feat.shape[1]
            for start in range(0, N - window + 1, stride):
                self.windows.append((stem, start))

        print(f'  Dataset: {len(stems)} files, {len(self.windows)} windows '
              f'(window={window}, stride={stride})')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        stem, start = self.windows[idx]
        end  = start + self.window
        feat = np.load(
            os.path.join(self.cache_dir, f'{stem}_feat.npy'),
            mmap_mode='r'
        )[:, start:end].copy()
        mask = np.load(
            os.path.join(self.cache_dir, f'{stem}_mask.npy'),
            mmap_mode='r'
        )[start:end].copy().astype(np.float32)
        return torch.from_numpy(feat), torch.from_numpy(mask)


# ---------------------------------------------------------------------------
# File split
# ---------------------------------------------------------------------------
def file_split(mat_dir, val_frac=VAL_FRAC, seed=SEED):
    stems = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(mat_dir) if f.endswith('.mat')
    )
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(stems))
    n_val = max(1, int(len(stems) * val_frac))
    val_stems   = [stems[i] for i in idx[:n_val]]
    train_stems = [stems[i] for i in idx[n_val:]]
    return train_stems, val_stems


# ---------------------------------------------------------------------------
# Model: PrecisionResUNet (Edward's architecture, in_ch=8)
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=15, padding=7),
            nn.GroupNorm(min(4, out_c), out_c),
            nn.ReLU(),
            nn.Conv1d(out_c, out_c, kernel_size=15, padding=7),
            nn.GroupNorm(min(4, out_c), out_c),
        )
        self.shortcut = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))


class ASPPBlock(nn.Module):
    def __init__(self, in_c, out_c, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_c, out_c, 3, padding=d, dilation=d),
                nn.GroupNorm(4, out_c), nn.ReLU()
            ) for d in dilations
        ])
        self.global_b = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_c, out_c, 1), nn.ReLU()
        )
        self.project = nn.Sequential(
            nn.Conv1d(out_c * (len(dilations) + 1), out_c, 1),
            nn.GroupNorm(4, out_c), nn.ReLU()
        )

    def forward(self, x):
        bs = [b(x) for b in self.branches]
        gp = self.global_b(x).expand(-1, -1, x.shape[2])
        return self.project(torch.cat(bs + [gp], dim=1))


class TransformerBottleneck(nn.Module):
    def __init__(self, dim, nhead=4, nlayers=2):
        super().__init__()
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, batch_first=True,
            dim_feedforward=dim * 2, dropout=0.1
        )
        self.tf = nn.TransformerEncoder(enc, num_layers=nlayers)

    def forward(self, x):
        return self.tf(x.permute(0, 2, 1)).permute(0, 2, 1)


class PrecisionResUNet(nn.Module):
    def __init__(self, in_ch=IN_CH, use_transformer=True):
        super().__init__()
        self.use_transformer = use_transformer
        self.pool = nn.MaxPool1d(2)

        self.enc1 = ResidualBlock(in_ch, 32)
        self.enc2 = ResidualBlock(32, 64)
        self.enc3 = ResidualBlock(64, 128)
        self.enc4 = ResidualBlock(128, 256)

        self.aspp = ASPPBlock(256, 256)
        if use_transformer:
            self.bottleneck = TransformerBottleneck(256)

        self.up4  = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec4 = ResidualBlock(512, 128)
        self.up3  = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec3 = ResidualBlock(256, 64)
        self.up2  = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec2 = ResidualBlock(128, 32)
        self.up1  = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec1 = ResidualBlock(64, 32)

        self.final = nn.Conv1d(32, 1, 1)

    def forward(self, x):
        s1 = self.enc1(x);          p1 = self.pool(s1)
        s2 = self.enc2(p1);         p2 = self.pool(s2)
        s3 = self.enc3(p2);         p3 = self.pool(s3)
        s4 = self.enc4(p3);         p4 = self.pool(s4)

        b = self.aspp(p4)
        if self.use_transformer:
            b = self.bottleneck(b)

        d4 = self.dec4(torch.cat([self.up4(b),  s4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], 1))
        return self.final(d1)   # (B, 1, T)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def bce_dice_loss(pred, target, pos_weight=POS_WEIGHT):
    pw = torch.tensor([pos_weight], device=pred.device)
    bce  = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pw)
    p    = torch.sigmoid(pred).view(pred.shape[0], -1)
    t    = target.view(pred.shape[0], -1)
    inter = (p * t).sum(1)
    dice = 1 - ((2 * inter + 1) / (p.sum(1) + t.sum(1) + 1)).mean()
    return bce + dice


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_pred, all_gt = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).unsqueeze(1)   # (B, 1, T)

        out  = model(x)
        loss = bce_dice_loss(out, y)
        total_loss += loss.item()

        pred_bin = (torch.sigmoid(out) >= THRESH).cpu().numpy().ravel()
        gt_bin   = (y >= 0.5).cpu().numpy().ravel()
        all_pred.append(pred_bin)
        all_gt.append(gt_bin)

    all_pred = np.concatenate(all_pred)
    all_gt   = np.concatenate(all_gt)

    f1  = f1_score(all_gt, all_pred, zero_division=0)
    tp  = np.logical_and(all_pred, all_gt).sum()
    fp  = np.logical_and(all_pred, ~all_gt).sum()
    fn  = np.logical_and(~all_pred, all_gt).sum()
    iou = tp / (tp + fp + fn + 1e-8)
    acc = (all_pred == all_gt).mean()

    return total_loss / len(loader), float(f1), float(iou), float(acc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Step 1: pre-process
    run_preprocessing(MAT_DIR, CACHE_DIR)

    # Step 2: file split
    train_stems, val_stems = file_split(MAT_DIR)
    print(f'Split: {len(train_stems)} train / {len(val_stems)} val files')

    # Step 3: datasets
    print('Building datasets ...')
    ds_tr = BinaryMovementDataset(train_stems, CACHE_DIR, WINDOW, STRIDE_TR)
    ds_va = BinaryMovementDataset(val_stems,   CACHE_DIR, WINDOW, STRIDE_VA)

    nw = 4 if device.type == 'cuda' else 2
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_TR, shuffle=True,
                       num_workers=nw, pin_memory=(device.type=='cuda'))
    dl_va = DataLoader(ds_va, batch_size=BATCH_VA, shuffle=False,
                       num_workers=nw, pin_memory=(device.type=='cuda'))

    # Step 4: model
    use_tf = device.type == 'cuda'   # skip Transformer on CPU (too slow)
    model  = PrecisionResUNet(in_ch=IN_CH, use_transformer=use_tf).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model: {n_params:.2f}M params  (transformer={use_tf})')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR/20)

    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_iou': [], 'val_acc': []}
    best_val_loss = float('inf')

    print(f'\n{"Epoch":>5} {"tr_loss":>9} {"va_loss":>9} {"F1":>7} {"IoU":>7} {"Acc":>7}  {"time":>6}')
    print('-' * 60)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0

        for x, y in dl_tr:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = bce_dice_loss(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()

        tr_loss /= len(dl_tr)
        scheduler.step()

        va_loss, f1, iou, acc = evaluate(model, dl_va)
        dt = time.time() - t0

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['val_f1'].append(f1)
        history['val_iou'].append(iou)
        history['val_acc'].append(acc)

        marker = ''
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), MODEL_PATH)
            marker = '  ← best'

        print(f'{epoch:5d} {tr_loss:9.4f} {va_loss:9.4f} {f1:7.4f} {iou:7.4f} {acc:7.4f}  {dt:5.0f}s{marker}')

        with open(HIST_PATH, 'w') as fh:
            json.dump(history, fh)

    print(f'\nDone. Best val loss: {best_val_loss:.4f}')
    print(f'Model saved to {MODEL_PATH}')
