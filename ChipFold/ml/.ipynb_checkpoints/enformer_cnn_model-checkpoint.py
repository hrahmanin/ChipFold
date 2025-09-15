# cnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(2024)

# --------------------------
# Your original base model
# --------------------------
class FlankCoreModel(nn.Module):
    def __init__(self, seq_len, n_head, kernel_size, n_feature=4, out_features=3):
        super().__init__()

        d_embed1 = 5
        d_embed2 = 5
        maxpool_size = 5
        d_embed3 = 11

        # capture core
        self.convblock1 = nn.Sequential(
            nn.Conv1d(in_channels=n_feature, out_channels=d_embed1, kernel_size=18, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=d_embed1),
            nn.MaxPool1d(kernel_size=maxpool_size, stride=maxpool_size)
        )

        # capture flank
        self.convblock2 = nn.Sequential(
            nn.Conv1d(in_channels=n_feature, out_channels=d_embed2, kernel_size=15, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=d_embed2),
            nn.MaxPool1d(kernel_size=maxpool_size, stride=maxpool_size)
        )

        # convolve on the new features
        self.convblock3 = nn.Sequential(
            nn.Conv1d(in_channels=d_embed1+d_embed2, out_channels=d_embed3, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=d_embed3),
        )

        self.flatten = nn.Flatten()
        flatten_out_features = self._calculate_num_out_features(n_feature, seq_len)
        self.linear = nn.Linear(in_features=flatten_out_features, out_features=out_features)

    def _calculate_num_out_features(self, n_feature, seq_len):
        with torch.no_grad():
            x = torch.zeros(1, n_feature, seq_len)
            out_1 = self.convblock1(x)
            out_2 = self.convblock2(x)
            out = torch.cat([out_1, out_2], dim=1)
            out = self.convblock3(out)
            out = self.flatten(out)
            return out.shape[1]

    def forward(self, x):
        out_1 = self.convblock1(x)
        out_2 = self.convblock2(x)
        out = torch.cat([out_1, out_2], dim=1)
        out = self.convblock3(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


# --------------------------
# Hybrid blocks
# --------------------------
class CNNEncoder(nn.Module):
    """
    Light CNN to embed the target sequence into d_model.
    """
    def __init__(self, seq_len: int, d_model: int = 64, n_feature: int = 4):
        super().__init__()
        d_embed1 = 32
        d_embed2 = 32
        d_embed3 = 64
        maxpool_size = 5

        self.block1 = nn.Sequential(
            nn.Conv1d(n_feature, d_embed1, kernel_size=18, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(d_embed1),
            nn.MaxPool1d(kernel_size=maxpool_size, stride=maxpool_size)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(n_feature, d_embed2, kernel_size=15, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(d_embed2),
            nn.MaxPool1d(kernel_size=maxpool_size, stride=maxpool_size)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(d_embed1 + d_embed2, d_embed3, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(d_embed3)
        )
        self.gap = nn.AdaptiveAvgPool1d(1)  # global average pool over length
        self.proj = nn.Linear(d_embed3, d_model)

    def forward(self, x):  # x: (B, 4, L)
        a = self.block1(x)
        b = self.block2(x)
        h = torch.cat([a, b], dim=1)  # (B, d1+d2, L')
        h = self.block3(h)
        h = self.gap(h).squeeze(-1)   # (B, d_embed3)
        out = self.proj(h)            # (B, d_model)
        return out


class NeighborAttention(nn.Module):
    """
    Cross-attention: target embedding attends to K neighbor feature vectors.
    neighbor_feats (K, F) -> MLP -> (K, d_model) for K/V.
    """
    def __init__(self, neighbor_feat_dim: int, d_model: int = 64, n_heads: int = 4, attn_dropout: float = 0.1):
        super().__init__()
        self.neighbor_mlp = nn.Sequential(
            nn.Linear(neighbor_feat_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.mha = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True, dropout=attn_dropout)
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.ReLU(inplace=True),
            nn.Linear(2*d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, target_emb, neighbor_feats, neighbor_mask):
        """
        target_emb:   (B, d_model)
        neighbor_feats:(B, K, F)
        neighbor_mask: (B, K)   1=valid, 0=pad
        """
        x = self.neighbor_mlp(neighbor_feats)              # (B, K, d_model)
        Q = self.q(target_emb).unsqueeze(1)                # (B, 1, d_model)
        Kproj = self.k(x)                                  # (B, K, d_model)
        Vproj = self.v(x)                                  # (B, K, d_model)
        key_padding_mask = (neighbor_mask == 0.0)          # True where to ignore
        attn_out, _ = self.mha(Q, Kproj, Vproj, key_padding_mask=key_padding_mask)  # (B,1,d_model)
        h = self.ln(target_emb + attn_out.squeeze(1))      # (B, d_model)
        h2 = self.ln2(h + self.ff(h))                      # (B, d_model)
        return h2


class HybridCTCFModel(nn.Module):
    """
    CNN encoder for the target site + attention over K neighbor feature vectors.
    """
    def __init__(self, seq_len: int, out_features: int = 3,
                 d_model: int = 64, n_heads: int = 4, neighbor_feat_dim: int = 32):
        super().__init__()
        self.encoder = CNNEncoder(seq_len=seq_len, d_model=d_model)
        self.nei_attn = NeighborAttention(neighbor_feat_dim=neighbor_feat_dim,
                                          d_model=d_model, n_heads=n_heads)
        self.head = nn.Linear(d_model, out_features)

    def forward(self, seq, neighbor_feats, neighbor_mask):
        """
        seq:           (B, 4, L)
        neighbor_feats:(B, K, F)
        neighbor_mask: (B, K)
        """
        tgt = self.encoder(seq)                                   # (B, d_model)
        fused = self.nei_attn(tgt, neighbor_feats, neighbor_mask) # (B, d_model)
        logits = self.head(fused)                                  # (B, out_features)
        return logits
