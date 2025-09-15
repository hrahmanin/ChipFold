import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from .cnn_enformer_hybrid import CNNEnformerHybrid

def train_model(train_ds, val_ds, label_cols, epochs=25, patience=10, lr=1e-3, device="cuda"):
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = CNNEnformerHybrid(n_classes=len(label_cols)).to(device)
    opt = Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(epochs):
        model.train()
        for x, neighbors, y in train_loader:
            x, neighbors, y = x.to(device), neighbors.to(device), y.to(device)
            out = model(x, neighbors if neighbors.shape[0] > 0 else None)
            loss = loss_fn(torch.log_softmax(out, dim=1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, neighbors, y in val_loader:
                x, neighbors, y = x.to(device), neighbors.to(device), y.to(device)
                out = model(x, neighbors if neighbors.shape[0] > 0 else None)
                val_loss += loss_fn(torch.log_softmax(out, dim=1), y).item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), "hybrid_best.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    return model
