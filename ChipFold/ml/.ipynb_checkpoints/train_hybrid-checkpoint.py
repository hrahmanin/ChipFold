# ChipFold/ml/train_hybrid.py
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from .cnn_enformer_hybrid import CNNEnformerHybrid

def _collate_with_neighbor_padding(batch):
    # batch: list of (seq[L,4], neighbors[K,L,4], y[...])
    seqs, neighs, ys = zip(*batch)
    seqs = torch.stack(seqs, dim=0)  # (B,L,4)
    ys   = torch.stack(ys,   dim=0)  # (B, C)

    L, C = seqs.shape[1], seqs.shape[2]
    Ks = [n.shape[0] for n in neighs]
    Kmax = max(Ks) if Ks else 0
    if Kmax == 0:
        neighbors = torch.zeros((len(batch), 0, L, C), dtype=seqs.dtype)
    else:
        padded = []
        for n in neighs:
            k = n.shape[0]
            if k == 0:
                pad = torch.zeros((Kmax, L, C), dtype=seqs.dtype)
            elif k < Kmax:
                pad = torch.cat([n, torch.zeros((Kmax - k, L, C), dtype=seqs.dtype)], dim=0)
            else:
                pad = n[:Kmax]
            padded.append(pad)
        neighbors = torch.stack(padded, dim=0)  # (B,Kmax,L,4)
    return seqs, neighbors, ys

def train_model(train_ds, val_ds, label_cols, epochs=25, patience=10, lr=1e-3, device="cuda"):
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=_collate_with_neighbor_padding)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, collate_fn=_collate_with_neighbor_padding)

    n_classes = len(label_cols)
    model = CNNEnformerHybrid(n_classes=n_classes).to(device)
    opt = Adam(model.parameters(), lr=lr)

    # choose loss by output size
    if n_classes == 1:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(epochs):
        model.train()
        for x, neighbors, y in train_loader:
            x = x.to(device)                       # (B,L,4)
            neighbors = neighbors.to(device)       # (B,K,L,4) padded; may be K=0
            y = y.to(device)                       # (B,1) or (B,C)

            # forward (let the model handle neighbors=None if K==0)
            nb = None if neighbors.shape[1] == 0 else neighbors
            out = model(x, nb)

            if n_classes == 1:
                loss = loss_fn(out.squeeze(1), y.squeeze(1))
            else:
                loss = loss_fn(torch.log_softmax(out, dim=1), y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, neighbors, y in val_loader:
                x = x.to(device)
                neighbors = neighbors.to(device)
                y = y.to(device)
                nb = None if neighbors.shape[1] == 0 else neighbors
                out = model(x, nb)
                if n_classes == 1:
                    val_loss += loss_fn(out.squeeze(1), y.squeeze(1)).item()
                else:
                    val_loss += loss_fn(torch.log_softmax(out, dim=1), y).item()
        val_loss /= max(1, len(val_loader))

        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), "hybrid_best.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    return model
