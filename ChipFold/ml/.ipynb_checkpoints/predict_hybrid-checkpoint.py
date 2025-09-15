import torch
import pandas as pd
from torch.utils.data import DataLoader
from .cnn_enformer_hybrid import CNNEnformerHybrid

def predict_model(test_ds, label_cols, weights_path="hybrid_best.pt", device="cuda"):
    loader = DataLoader(test_ds, batch_size=32)
    model = CNNEnformerHybrid(n_classes=len(label_cols)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for x, neighbors, y in loader:
            x, neighbors, y = x.to(device), neighbors.to(device), y.to(device)
            out = model(x, neighbors if neighbors.shape[0] > 0 else None)
            preds.append(torch.softmax(out, dim=1).cpu())
            trues.append(y.cpu())

    return torch.cat(trues).numpy(), torch.cat(preds).numpy()
