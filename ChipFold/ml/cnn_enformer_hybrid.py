import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEnformerHybrid(nn.Module):
    def __init__(self, seq_len=618, embed_dim=64, kernel_size=15, n_channels=16, n_classes=3):
        super().__init__()
        # CNN encoder
        self.conv1 = nn.Conv1d(4, n_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(n_channels)
        self.pool1 = nn.MaxPool1d(2)

        # Projection to embedding
        self.proj = nn.Linear((seq_len // 2) * n_channels, embed_dim)

        # Transformer-like attention block
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)

        # Final classifier
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x, neighbor_embed=None):
        # CNN
        x = x.permute(0, 2, 1)  # (B, 4, L)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = x.flatten(1)

        # Embedding
        x_embed = self.proj(x).unsqueeze(1)  # (B, 1, D)

        # If neighbors exist, concatenate them
        if neighbor_embed is not None:
            tokens = torch.cat([x_embed, neighbor_embed], dim=1)
        else:
            tokens = x_embed

        # Attention
        attn_out, _ = self.attn(tokens, tokens, tokens)
        pooled = attn_out.mean(dim=1)

        return self.fc(pooled)
