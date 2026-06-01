"""
BeatTransformer — lightweight Transformer for fetal movement classification.

Input : sequence of R-peak amplitude vectors, shape (B, MAX_BEATS, 6)
        + padding mask (B, MAX_BEATS) — True where padded
Output: (B, 4) logits

Architecture:
  - Linear projection: 6 → d_model (default 64)
  - Learned positional embedding
  - CLS token classification
  - N_LAYERS × TransformerEncoderLayer (batch_first=True)
  - LayerNorm + dropout + FC → 4 classes

~120K parameters with defaults (d_model=64, n_layers=4, n_heads=4).
"""

import torch
import torch.nn as nn
import math


class BeatTransformer(nn.Module):
    def __init__(self,
                 n_channels  = 6,
                 d_model     = 64,
                 n_heads     = 4,
                 n_layers    = 4,
                 max_beats   = 60,
                 n_classes   = 4,
                 dropout     = 0.2):
        super().__init__()

        self.d_model = d_model

        # Project each beat vector (n_channels values) to d_model
        self.input_proj = nn.Linear(n_channels, d_model)

        # Learnable positional embedding — position 0 is reserved for CLS token
        self.pos_emb = nn.Embedding(max_beats + 1, d_model)

        # CLS token (learned)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            activation      = 'gelu',
            batch_first     = True,
            norm_first      = True,   # Pre-LN — more stable training
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(d_model)
        self.dropout     = nn.Dropout(dropout)
        self.head        = nn.Linear(d_model, n_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, key_padding_mask=None):
        """
        Parameters
        ----------
        x               : (B, N, 6) float — beat amplitude sequences
        key_padding_mask: (B, N) bool — True = padded position (ignored)

        Returns
        -------
        logits : (B, n_classes)
        """
        B, N, _ = x.shape

        # Project beats to d_model
        x = self.input_proj(x)   # (B, N, d_model)

        # Add positional embeddings (positions 1..N; 0 is CLS)
        positions = torch.arange(1, N + 1, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(positions)

        # Prepend CLS token with position 0
        cls = self.cls_token.expand(B, -1, -1)                # (B, 1, d_model)
        cls = cls + self.pos_emb(torch.zeros(1, 1, dtype=torch.long, device=x.device))
        x   = torch.cat([cls, x], dim=1)                      # (B, N+1, d_model)

        # Extend padding mask: CLS token is never masked
        if key_padding_mask is not None:
            cls_mask         = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        # Transformer
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # Classify from CLS token
        cls_out = self.norm(x[:, 0])          # (B, d_model)
        return self.head(self.dropout(cls_out))


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    model   = BeatTransformer()
    n_param = sum(p.numel() for p in model.parameters())
    print(f'BeatTransformer parameters: {n_param / 1e3:.1f} K')

    B, N = 8, 60
    x    = torch.randn(B, N, 6)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, 45:] = True   # last 15 positions padded

    out = model(x, mask)
    print(f'Input:  {x.shape}  mask: {mask.shape}')
    print(f'Output: {out.shape}')
