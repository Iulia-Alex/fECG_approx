import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
import os


# --- 1. FEATURE EXTRACTION ---
def extract_perfect_features(sig, qrs_indices, baseline_window=750):
    sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

    # Baseline wander: moving average over ~1.5s at 500Hz
    pad = baseline_window // 2
    padded = np.pad(sig_norm, (pad, pad), mode='edge')
    baseline = np.convolve(padded, np.ones(baseline_window) / baseline_window, mode='valid')[:len(sig_norm)]

    if len(qrs_indices) == 0:
        return np.stack([sig_norm, np.zeros_like(sig_norm), baseline], axis=0)

    qrs_values = sig_norm[qrs_indices]
    x_all = np.arange(len(sig_norm))
    linear_outline = np.interp(x_all, qrs_indices, qrs_values)

    return np.stack([sig_norm, linear_outline, baseline], axis=0)


# --- 2. COMPONENTS ---
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=15, padding=7),
            nn.GroupNorm(4, out_c),
            nn.ReLU(),
            nn.Conv1d(out_c, out_c, kernel_size=15, padding=7),
            nn.GroupNorm(4, out_c)
        )
        self.shortcut = nn.Conv1d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))


class ASPPBlock(nn.Module):
    def __init__(self, in_c, out_c, dilations=[1, 2, 4, 8]):
        super().__init__()
        self.dilations = dilations
        self.branch_norms = [0.0] * len(dilations)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=3, padding=d, dilation=d),
                nn.GroupNorm(4, out_c),
                nn.ReLU()
            ) for d in dilations
        ])
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_c, out_c, kernel_size=1),
            nn.ReLU()
        )
        self.project = nn.Sequential(
            nn.Conv1d(out_c * (len(dilations) + 1), out_c, kernel_size=1),
            nn.GroupNorm(4, out_c),
            nn.ReLU()
        )

    def forward(self, x):
        branches = [b(x) for b in self.branches]
        self.branch_norms = [b.abs().mean().item() for b in branches]
        gp = self.global_branch(x).expand(-1, -1, x.shape[2])
        return self.project(torch.cat(branches + [gp], dim=1))


class TransformerBottleneck(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        return x.permute(0, 2, 1)


# --- 3. ARCHITECTURE ---
class PrecisionResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ResidualBlock(3, 32)
        self.enc2 = ResidualBlock(32, 64)
        self.enc3 = ResidualBlock(64, 128)
        self.enc4 = ResidualBlock(128, 256)
        self.pool = nn.MaxPool1d(2)
        self.aspp = ASPPBlock(256, 256)
        self.bottleneck = TransformerBottleneck(256)
        self.up4 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec4 = ResidualBlock(512, 128)  # 256 + 256
        self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec3 = ResidualBlock(256, 64)  # 128 + 128
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec2 = ResidualBlock(128, 32)  # 64 + 64
        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec1 = ResidualBlock(64, 32)  # 32 + 32
        self.final = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x);
        p1 = self.pool(s1)
        s2 = self.enc2(p1);
        p2 = self.pool(s2)
        s3 = self.enc3(p2);
        p3 = self.pool(s3)
        s4 = self.enc4(p3);
        p4 = self.pool(s4)
        b = self.bottleneck(self.aspp(p4))
        d4 = self.dec4(torch.cat([self.up4(b), s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        return self.final(d1)


# --- 4. ADJUSTABLE STRIDE DATA LOADER ---
class TripleFolderLoader:
    def __init__(self, sig_paths, mask_paths, qrs_paths):
        self.sig_paths, self.mask_paths, self.qrs_paths = sig_paths, mask_paths, qrs_paths

    def get_batches(self, window_size, stride, batch_size):
        indices = np.arange(len(self.sig_paths))
        np.random.shuffle(indices)

        x_batch, y_batch = [], []

        for idx in indices:
            sig = np.load(self.sig_paths[idx]).flatten()
            mask = np.load(self.mask_paths[idx]).flatten()
            qrs_locs = np.load(self.qrs_paths[idx]).flatten()

            # Sequential windowing based on STRIDE
            for start in range(0, sig.shape[0] - window_size, stride):
                end = start + window_size
                v_qrs = qrs_locs[(qrs_locs >= start) & (qrs_locs < end)] - start

                x_batch.append(extract_perfect_features(sig[start:end], v_qrs))
                y_batch.append(mask[start:end])

                if len(x_batch) == batch_size:
                    yield (torch.from_numpy(np.array(x_batch)).float(),
                           torch.from_numpy(np.array(y_batch)).float().unsqueeze(1))
                    x_batch, y_batch = [], []


# --- 5. MAIN ---
def main():
    # --- CONFIGURATION AREA ---
    WINDOW_SIZE = 3840
    STRIDE = 250  # <--- Change this to adjust overlap (250 = 0.5s)
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-5
    # --------------------------

    DATA_ROOT = "/home/20251020/ECG/Npy_DB"
    sig_p = sorted(glob.glob(os.path.join(DATA_ROOT, "signals", "*.npy")))
    mask_p = sorted(glob.glob(os.path.join(DATA_ROOT, "masks", "*.npy")))
    qrs_p = sorted(glob.glob(os.path.join(DATA_ROOT, "qrs_locs", "*.npy")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PrecisionResUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    def criterion(pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=torch.tensor([5.0]).to(pred.device))
        p = torch.sigmoid(pred).view(pred.shape[0], -1)
        t = target.view(target.shape[0], -1)
        intersection = (p * t).sum(dim=1)
        dice = 1 - ((2. * intersection + 1) / (p.sum(dim=1) + t.sum(dim=1) + 1)).mean()
        return bce + dice

    print(f"Phase 7: Precision Res-U-Net | Stride: {STRIDE} | LR: {LEARNING_RATE}")
    best_loss = float('inf')
    train_loader = TripleFolderLoader(sig_p[:int(0.8 * len(sig_p))], mask_p[:int(0.8 * len(sig_p))],
                                      qrs_p[:int(0.8 * len(sig_p))])
    val_loader = TripleFolderLoader(sig_p[int(0.8 * len(sig_p)):], mask_p[int(0.8 * len(sig_p)):],
                                    qrs_p[int(0.8 * len(sig_p)):])

    for epoch in range(100):
        model.train()
        t_loss, t_steps = 0, 0
        for x, y in train_loader.get_batches(WINDOW_SIZE, STRIDE, BATCH_SIZE):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            t_steps += 1

        model.eval()
        v_loss, v_steps = 0, 0
        branch_accum = [0.0] * len(model.aspp.dilations)
        with torch.no_grad():
            for x, y in val_loader.get_batches(WINDOW_SIZE, STRIDE, BATCH_SIZE):
                x, y = x.to(device), y.to(device)
                v_loss += criterion(model(x), y).item()
                for i, n in enumerate(model.aspp.branch_norms):
                    branch_accum[i] += n
                v_steps += 1

        avg_v = v_loss / v_steps
        avg_norms = [a / v_steps for a in branch_accum]
        dominant = model.aspp.dilations[avg_norms.index(max(avg_norms))]
        dilation_str = " | ".join(f"d={d}: {avg_norms[i]:.4f}" for i, d in enumerate(model.aspp.dilations))
        print(f"Epoch {epoch + 1:03d} | Train Loss: {t_loss / t_steps:.4f} | Val Loss: {avg_v:.4f} | ASPP [{dilation_str}] -> dominant d={dominant}")

        if avg_v < best_loss:
            best_loss = avg_v
            torch.save(model.state_dict(), "(9)ASPP_Trans_Res_UNET_hybrid_best.pth")
            print("--> Saved New Best Model")


if __name__ == "__main__":
    main()