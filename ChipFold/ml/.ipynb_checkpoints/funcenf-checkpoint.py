# funcs.py
import os
import random
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

from cnn_model import FlankCoreModel, HybridCTCFModel  # keep old, add new

torch.manual_seed(2024)
np.random.seed(2024)
random.seed(2024)

# -------------------- utilities --------------------

def dna_1hot(seq: str) -> np.ndarray:
    """One-hot encode a DNA sequence (A,C,G,T). Shape: (4, L)"""
    seq = seq.upper().strip()
    L = len(seq)
    out = np.zeros((4, L), dtype=np.float32)
    m = {'A':0, 'C':1, 'G':2, 'T':3}
    for i, nt in enumerate(seq):
        if nt in m:
            out[m[nt], i] = 1.0
        else:
            out[:, i] = 0.25  # N -> uniform
    return out


def fourier_features(dist_bp: np.ndarray, n_freq: int = 8, max_dist: float = 200_000.0) -> np.ndarray:
    """
    Distance encoding with sinusoidal features. Returns (K, 2*n_freq)
    """
    d = np.clip(dist_bp, 0.0, max_dist)
    min_scale = 10.0
    max_scale = max_dist
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num=n_freq)
    angles = (2.0 * np.pi * d[:, None] / scales[None, :]).astype(np.float32)
    feats = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
    return feats


def load_pfm(pfm_path: str) -> np.ndarray:
    """Load PFM (A,C,G,T rows)."""
    try:
        arr = np.loadtxt(pfm_path, dtype=float)
        if arr.shape[0] == 4:
            return arr
        return np.loadtxt(pfm_path, dtype=float, skiprows=1)
    except Exception:
        df = pd.read_csv(pfm_path, sep=None, engine="python", header=None, comment="#")
        arr = df.values.astype(float)
        if arr.shape[0] == 4:
            return arr
        return arr[1:, :]


def pfm_to_pwm(pfm: np.ndarray, bg: float = 0.25, eps: float = 1e-6) -> np.ndarray:
    col_sum = pfm.sum(axis=0, keepdims=True) + eps
    p = pfm / col_sum
    pwm = np.log((p + eps) / bg)
    return pwm


def scan_pwm_max(seq_1hot: np.ndarray, pwm: np.ndarray) -> Tuple[float, int, int]:
    """
    Score sequence with PWM on both strands; return (best_score, start, strand_sign)
    strand_sign = +1 for '+', -1 for '-'
    """
    k = pwm.shape[1]
    L = seq_1hot.shape[1]
    f_scores = np.array([(seq_1hot[:, i:i+k] * pwm).sum() for i in range(max(1, L-k+1))], dtype=np.float32)
    rc = seq_1hot[::-1, ::-1]
    r_scores = np.array([(rc[:, i:i+k] * pwm).sum() for i in range(max(1, L-k+1))], dtype=np.float32)
    f_max_idx = int(np.argmax(f_scores)) if len(f_scores) else 0
    r_max_idx = int(np.argmax(r_scores)) if len(r_scores) else 0
    f_max = float(f_scores[f_max_idx]) if len(f_scores) else -1e9
    r_max = float(r_scores[r_max_idx]) if len(r_scores) else -1e9
    if f_max >= r_max:
        return f_max, f_max_idx, +1
    else:
        return r_max, r_max_idx, -1


def build_neighbors(df: pd.DataFrame, k: int = 8, max_dist: int = 200_000):
    """
    For each row, pick up to k neighbors on same chrom within max_dist bp by center distance.
    Returns (neighbor_idx, neighbor_dist), shapes (N,k) and (N,k).
    """
    df = df.copy()
    df['mid'] = ((df['start'] + df['end']) * 0.5).astype(np.int64)
    neighbor_idx = -np.ones((len(df), k), dtype=np.int64)
    neighbor_dist = np.zeros((len(df), k), dtype=np.float32)

    for chrom, sub in df.groupby('chrom'):
        idxs = sub.index.values
        mids = sub['mid'].values
        order = np.argsort(mids)
        sorted_idx = idxs[order]
        sorted_mid = mids[order]

        for pos_in_order, row_idx in enumerate(sorted_idx):
            mid = sorted_mid[pos_in_order]
            lo = np.searchsorted(sorted_mid, mid - max_dist, side='left')
            hi = np.searchsorted(sorted_mid, mid + max_dist, side='right')
            cand_idx = sorted_idx[lo:hi]
            cand_mid = sorted_mid[lo:hi]
            mask = cand_idx != row_idx
            cand_idx = cand_idx[mask]
            cand_mid = cand_mid[mask]
            if cand_idx.size == 0:
                continue
            d = np.abs(cand_mid - mid)
            pick = np.argsort(d)[:k]
            chosen_idx = cand_idx[pick]
            chosen_dist = d[pick]
            fill = len(chosen_idx)
            neighbor_idx[row_idx, :fill] = chosen_idx
            neighbor_dist[row_idx, :fill] = chosen_dist.astype(np.float32)

    return neighbor_idx, neighbor_dist


# -------------------- Dataset --------------------
class HybridCTCFDataset(Dataset):
    def __init__(self,
                 df_path: str,
                 label_cols = ['Accessible','Bound','Nucleosome.occupied'],
                 neighbor_k: int = 8,
                 neighbor_max_dist: int = 200_000,
                 strength_source: str = 'label',  # 'label' or 'motif'
                 pfm_path: Optional[str] = None):
        """
        Expects CSV/TSV with columns: chrom,start,end,sequence, Accessible,Bound,Nucleosome.occupied
        Builds per-row:
          - target sequence one-hot (4, L)
          - K neighbor feature vectors:
                [Fourier(dist), neighbor_orient, rel_orient, strength_z, log1p_dist]
          - neighbor mask
        """
        self.df = pd.read_csv(df_path, sep=None, engine='python', comment='#')
        for c in ['chrom','start','end','sequence']:
            assert c in self.df.columns, f"Missing column '{c}'."
        self.df = self.df.dropna(subset=label_cols).reset_index(drop=True)
        self.label_cols = label_cols
        self.neighbor_k = neighbor_k
        self.neighbor_max_dist = neighbor_max_dist

        # neighbor index/dist
        self.neighbor_idx, self.neighbor_dist = build_neighbors(self.df, k=neighbor_k, max_dist=neighbor_max_dist)

        # orientation & strength
        self.pwm = None
        if pfm_path is not None:
            pfm = load_pfm(pfm_path)
            self.pwm = pfm_to_pwm(pfm)

        ori = np.ones((len(self.df),), dtype=np.float32)  # default +1
        strength = np.zeros((len(self.df),), dtype=np.float32)

        if self.pwm is not None:
            for i, seq in enumerate(self.df['sequence'].astype(str).values):
                score, _, sign = scan_pwm_max(dna_1hot(seq), self.pwm)
                ori[i] = float(sign)
                strength[i] = float(score)
        else:
            if strength_source == 'label' and 'Bound' in self.df.columns:
                strength = self.df['Bound'].values.astype(np.float32)
            else:
                strength[:] = 1.0

        self.orient = ori
        s_mean = float(np.mean(strength))
        s_std  = float(np.std(strength) + 1e-6)
        self.strength = (strength - s_mean) / s_std

        # labels
        self.labels = self.df[self.label_cols].values.astype(np.float32)

        # sequences (all same L)
        seqs = [dna_1hot(s) for s in self.df['sequence'].astype(str).values]
        self.seq_len = int(seqs[0].shape[1])
        for x in seqs:
            assert x.shape[1] == self.seq_len, "All sequences must have same length."
        self.seqs = np.stack(seqs, axis=0)  # (N, 4, L)

        # neighbor feature dim
        self.n_freq = 8
        self.feat_dim = 2*self.n_freq + 3 + 1  # fourier + neighbor_orient + rel_orient + strength_z + log1p_dist

    def __len__(self):
        return len(self.df)

    def _neighbor_feats_row(self, i: int):
        n_idx = self.neighbor_idx[i].copy()      # (K,)
        n_dist = self.neighbor_dist[i].copy()    # (K,)
        valid = n_idx >= 0
        K = self.neighbor_k

        feats = np.zeros((K, self.feat_dim), dtype=np.float32)
        mask = np.zeros((K,), dtype=np.float32)

        if valid.any():
            ff = fourier_features(n_dist[valid], n_freq=self.n_freq, max_dist=float(self.neighbor_max_dist))
            neighbor_orient = self.orient[n_idx[valid]]
            rel_orient = neighbor_orient * self.orient[i]
            neighbor_strength = self.strength[n_idx[valid]]
            logd = np.log1p(n_dist[valid])

            stacked = np.concatenate([
                ff,
                neighbor_orient[:, None],
                rel_orient[:, None],
                neighbor_strength[:, None],
                logd[:, None],
            ], axis=1)

            feats[:stacked.shape[0], :] = stacked
            mask[:stacked.shape[0]] = 1.0

        return feats, mask

    def __getitem__(self, i: int):
        x = torch.tensor(self.seqs[i], dtype=torch.float32)
        y = torch.tensor(self.labels[i], dtype=torch.float32)
        nfeats, nmask = self._neighbor_feats_row(i)
        nfeats = torch.tensor(nfeats, dtype=torch.float32)
        nmask  = torch.tensor(nmask,  dtype=torch.float32)
        return x, nfeats, nmask, y


# -------------------- Train / Predict / Plot --------------------

def train_hybrid_model(dataset: HybridCTCFDataset,
                       outdir: str,
                       batch_size: int = 64,
                       lr: float = 1e-3,
                       epochs: int = 20,
                       patience: int = 5,
                       min_delta: float = 0.0,
                       use_cuda_if_available: bool = True):
    """
    Trains HybridCTCFModel with KLDivLoss (labels are probabilities).
    Saves best weights to {outdir}/hybrid_weights.pt
    """
    os.makedirs(outdir, exist_ok=True)
    device = 'cuda' if (use_cuda_if_available and torch.cuda.is_available()) else 'cpu'

    # split train/val
    N = len(dataset)
    n_train = int(0.8 * N)
    n_val = N - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    model = HybridCTCFModel(seq_len=dataset.seq_len,
                            out_features=len(dataset.label_cols),
                            d_model=64, n_heads=4,
                            neighbor_feat_dim=dataset.feat_dim).to(device)

    loss_fn = nn.KLDivLoss(reduction='batchmean')
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float('inf')
    best_state = None
    no_improve = 0

    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for seq, nfeat, nmask, y in train_loader:
            seq, nfeat, nmask, y = seq.to(device), nfeat.to(device), nmask.to(device), y.to(device)
            opt.zero_grad()
            logits = model(seq, nfeat, nmask)
            loss = loss_fn(F.log_softmax(logits, dim=1), y)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item())
        tr_loss /= max(1, len(train_loader))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for seq, nfeat, nmask, y in val_loader:
                seq, nfeat, nmask, y = seq.to(device), nfeat.to(device), nmask.to(device), y.to(device)
                logits = model(seq, nfeat, nmask)
                loss = loss_fn(F.log_softmax(logits, dim=1), y)
                va_loss += float(loss.item())
        va_loss /= max(1, len(val_loader))

        print(f"Epoch {ep:02d}/{epochs} | Train {tr_loss:.4f} | Val {va_loss:.4f}")

        if va_loss < best_val - min_delta:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"⏹️ Early stopping at epoch {ep} (no improvement for {patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"🔙 Restored best model (Val {best_val:.4f}).")

    weights_path = os.path.join(outdir, "hybrid_weights.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"✅ Saved weights -> {weights_path}")
    return model, weights_path, device


def predict_hybrid(dataset: HybridCTCFDataset,
                   weights_path: str,
                   batch_size: int = 128,
                   use_cuda_if_available: bool = True):
    """
    Loads HybridCTCFModel and returns predicted probabilities (N, 3).
    """
    device = 'cuda' if (use_cuda_if_available and torch.cuda.is_available()) else 'cpu'
    model = HybridCTCFModel(seq_len=dataset.seq_len,
                            out_features=len(dataset.label_cols),
                            d_model=64, n_heads=4,
                            neighbor_feat_dim=dataset.feat_dim).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs_all = []
    with torch.no_grad():
        for seq, nfeat, nmask, _ in loader:
            seq, nfeat, nmask = seq.to(device), nfeat.to(device), nmask.to(device)
            logits = model(seq, nfeat, nmask)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probs_all.append(probs)
    return np.vstack(probs_all)  # (N, 3)


def plot_results(df: pd.DataFrame,
                 label_cols = ['Accessible','Bound','Nucleosome.occupied'],
                 pred_prefix: str = 'pred_',
                 outdir: str = '.'):
    os.makedirs(outdir, exist_ok=True)
    for name in label_cols:
        gt = df[name].values
        pdv = df[f'{pred_prefix}{name}'].values
        r = float(np.corrcoef(gt, pdv)[0,1])
        plt.figure(figsize=(4,4))
        plt.scatter(gt, pdv, s=6, alpha=0.4)
        lims = [0, 1]
        plt.plot(lims, lims, linestyle='--')
        plt.xlim(lims); plt.ylim(lims)
        plt.xlabel(f"True {name}")
        plt.ylabel(f"Pred {name}")
        plt.title(f"{name} (r={r:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"scatter_{name}.png"), dpi=150)
        plt.close()

    gt_idx = df[label_cols].values.argmax(axis=1)
    pred_idx = df[[f'{pred_prefix}{n}' for n in label_cols]].values.argmax(axis=1)
    acc = float(np.mean(gt_idx == pred_idx))
    print(f"Argmax accuracy: {acc:.3f}")
