"""
ResNet1D for fetal movement classification.

Input : (B, 6, 4000)  — 6-channel fECG, 4 s window at 1000 Hz
Output: (B, 4)         — logits for 4 movement classes

Architecture mirrors ResNet-18 but uses Conv1d.
Stem uses kernel=15 (15 ms) to capture local ECG morphology.
Total parameters: ~1.5 M.
"""

import torch
import torch.nn as nn


class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """
    ResNet-18 style 1D network for ECG-based movement classification.

    Parameters
    ----------
    in_channels : int   number of ECG channels (default 6)
    n_classes   : int   number of output classes (default 4)
    base_ch     : int   base channel width (default 32); doubles each layer
    """

    def __init__(self, in_channels=6, n_classes=4, base_ch=32, dropout=0.0):
        super().__init__()

        # Stem: large kernel to see full ECG beat at 1000 Hz
        # 4000 --(stride 2)--> 2000 --(maxpool stride 2)--> 1000
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, kernel_size=15,
                      stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages (each halves time length except layer1)
        # After stem: (B, base_ch, 1000)
        self.layer1 = self._make_layer(base_ch,      base_ch,      2, stride=1)  # 1000
        self.layer2 = self._make_layer(base_ch,      base_ch * 2,  2, stride=2)  # 500
        self.layer3 = self._make_layer(base_ch * 2,  base_ch * 4,  2, stride=2)  # 250
        self.layer4 = self._make_layer(base_ch * 4,  base_ch * 8,  2, stride=2)  # 125

        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc      = nn.Linear(base_ch * 8, n_classes)

        self._init_weights()

    def _make_layer(self, in_ch, out_ch, n_blocks, stride):
        layers = [BasicBlock1D(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(self.dropout(x))


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    model = ResNet1D(in_channels=6, n_classes=4, base_ch=32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'ResNet1D parameters: {n_params / 1e6:.2f} M')

    dummy = torch.randn(4, 6, 4000)
    out   = model(dummy)
    print(f'Input:  {dummy.shape}')
    print(f'Output: {out.shape}')
