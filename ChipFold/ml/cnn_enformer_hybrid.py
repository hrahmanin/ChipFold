# ChipFold/ml/cnn_enformer_hybrid.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEnformerHybrid(nn.Module):
    def __init__(self, embed_dim=64, kernel_size=15, n_channels=16, n_classes=1):
        super().__init__()
        # CNN encoder for sequences (length-agnostic via adaptive pooling)
        self.conv1 = nn.Conv1d(4, n_channels, kernel_size, padding=kernel_size // 2)
        self.bn1   = nn.BatchNorm1d(n_channels)
        self.gap   = nn.AdaptiveAvgPool1d(1)   # -> (B, C, 1)
        self.proj  = nn.Linear(n_channels, embed_dim)

        # Transformer-like attention over [center + K neighbors]
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)

        # Final head
        self.fc   = nn.Linear(embed_dim, n_classes)

    def _encode_seq(self, x):
        """
        x: (B, L, 4) one-hot
        returns: (B, D) embedding
        """
        x = x.permute(0, 2, 1)            # (B, 4, L)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, C, L)
        x = self.gap(x).squeeze(-1)       # (B, C)
        x = self.proj(x)                  # (B, D)
        return x

    def _encode_neighbors(self, nb):
        """
        nb: (B, K, L, 4) or None
        returns: (B, K, D) or None
        """
        if nb is None or nb.shape[1] == 0:
            return None
        B, K, L, A = nb.shape
        nb_flat = nb.reshape(B * K, L, A)    # (B*K, L, 4)
        nb_emb  = self._encode_seq(nb_flat)  # (B*K, D)
        D = nb_emb.shape[-1]
        nb_emb  = nb_emb.view(B, K, D)       # (B, K, D)
        return nb_emb

    def forward(self, x, neighbors=None):
        """
        x: (B, L, 4)
        neighbors: (B, K, L, 4) or None
        """
        x_emb  = self._encode_seq(x).unsqueeze(1)      # (B, 1, D)
        nb_tok = self._encode_neighbors(neighbors)     # (B, K, D) or None

        if nb_tok is not None:
            tokens = torch.cat([x_emb, nb_tok], dim=1) # (B, 1+K, D)
        else:
            tokens = x_emb                              # (B, 1, D)

        attn_out, _ = self.attn(tokens, tokens, tokens) # (B, 1+K, D)
        pooled = attn_out.mean(dim=1)                   # (B, D)
        return self.fc(pooled)                          # (B, n_classes)
