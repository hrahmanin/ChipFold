# enformer_cnn_hybrid.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalCNNEncoder(nn.Module):
    """
    CNN encoder that returns a compact embedding via global pooling.
    Input : (B, 4, L)
    Output: (B, d_model)
    """
    def __init__(self, n_feature=4, d_model=128):
        super().__init__()
        d_embed1 = 5
        d_embed2 = 5
        d_embed3 = 11
        maxpool_size = 5

        self.convblock1 = nn.Sequential(
            nn.Conv1d(n_feature, d_embed1, kernel_size=18, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(d_embed1),
            nn.MaxPool1d(kernel_size=maxpool_size, stride=maxpool_size)
        )
        self.convblock2 = nn.Sequential(
            nn.Conv1d(n_feature, d_embed2, kernel_size=15, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(d_embed2),
            nn.MaxPool1d(kernel_size=maxpool_size, stride=maxpool_size)
        )
        self.convblock3 = nn.Sequential(
            nn.Conv1d(d_embed1 + d_embed2, d_embed3, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(d_embed3)
        )
        self.proj = nn.Sequential(
            nn.Linear(d_embed3, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):               # x: (B,4,L)
        a = self.convblock1(x)          # (B,d1, L1)
        b = self.convblock2(x)          # (B,d2, L2) with L1==L2
        z = torch.cat([a, b], dim=1)    # (B,d1+d2, L1)
        z = self.convblock3(z)          # (B,d3, L3)
        z = z.mean(dim=-1)              # (B,d3) global avg pool
        z = self.proj(z)                # (B,D)
        return z


class CrossAttnBlock(nn.Module):
    """
    Single Transformer-style cross-attention:
    Query:   (B,1,D)  — target embedding
    Key/Val: (B,K,D)  — neighbor embeddings
    """
    def __init__(self, d_model=128, n_heads=4, ff_mult=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, kv, key_padding_mask=None):
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)  # (B,1,D)
        x = self.ln1(q + self.drop(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x + self.drop(ff_out))
        return x.squeeze(1)  # (B,D)


class NeighborFeatureEmbedder(nn.Module):
    """
    Projects raw neighbor features (B,K,F) -> (B,K,D)
    """
    def __init__(self, in_features, d_model=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model)
        )

    def forward(self, feats):
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return self.mlp(feats)


class EnformerCNNHybrid(nn.Module):
    """
    1) Local sequence -> LocalCNNEncoder -> target embedding (B,D)
    2) Neighbor features -> MLP -> neighbor embeddings (B,K,D)
    3) Cross-attention (q=target, kv=neighbors) -> context (B,D)
    4) Fuse [target || context] -> classifier
    """
    def __init__(self, seq_len, d_model=128, n_heads=4, out_features=3,
                 neighbor_feat_dim=0, dropout=0.1):
        super().__init__()
        self.local = LocalCNNEncoder(n_feature=4, d_model=d_model)
        self.nei = NeighborFeatureEmbedder(neighbor_feat_dim, d_model) if neighbor_feat_dim > 0 else None
        self.cross = CrossAttnBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_features)
        )

    def forward(self, x_seq, neigh_feats=None, neigh_mask=None):
        tgt = self.local(x_seq)                     # (B,D)
        if self.nei is None or neigh_feats is None:
            context = tgt
        else:
            kv = self.nei(neigh_feats)             # (B,K,D)
            q  = tgt.unsqueeze(1)                  # (B,1,D)
            context = self.cross(q, kv, key_padding_mask=neigh_mask)  # (B,D)
        fused = torch.cat([tgt, context], dim=1)   # (B,2D)
        logits = self.classifier(fused)            # (B,C)
        return logits
