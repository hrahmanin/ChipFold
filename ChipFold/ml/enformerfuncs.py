# enformerfuncs.py
import os
import math
import random
import numpy as np
import pandas as pd
import pysam
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from .enformer_cnn_hybrid import EnformerCNNHybrid

torch.manual_seed(2024)


# ========================== Safe PWM + sequence utils ==========================

def pfm_to_pwm(pfm, background_frequency=0.25, pseudocount=0.5, eps=1e-8):
    """
    Safe PWM: add pseudocounts, normalize columns, avoid log(0).
    Returns natural-log odds.
    """
    pfm = pfm.astype(np.float64) + pseudocount
    col_sums = pfm.sum(axis=0, keepdims=True)
    freqs = pfm / (col_sums + eps)
    pwm = np.log((freqs + eps) / max(background_frequency, eps))
    return pwm


def dna_1hot(seq, seq_len=None, n_uniform=False, n_sample=False):
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim:seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2

    seq = seq.upper()
    seq_code = np.zeros((seq_len, 4), dtype='float16' if n_uniform else 'bool')
    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if   nt == 'A': seq_code[i, 0] = 1
            elif nt == 'C': seq_code[i, 1] = 1
            elif nt == 'G': seq_code[i, 2] = 1
            elif nt == 'T': seq_code[i, 3] = 1
            else:
                if n_uniform:
                    seq_code[i, :] = 0.25
                elif n_sample:
                    ni = random.randint(0, 3)
                    seq_code[i, ni] = 1
    return seq_code


def scan_sequence(pwm, pwm_rc, seq_1hot_T):  # seq_1hot_T: (4,L)
    """
    Returns (orient, motif_start_idx, strength) with clamped finite strength.
    orient: +1 forward, -1 reverse
    """
    k = pwm.shape[1]
    L = seq_1hot_T.shape[1]
    scores = np.array(
        [np.nansum(pwm * seq_1hot_T[:, i:i+k]) for i in range(L - k + 1)],
        dtype=np.float64
    )
    scores_rc = np.array(
        [np.nansum(pwm_rc * seq_1hot_T[:, i:i+k]) for i in range(L - k + 1)],
        dtype=np.float64
    )
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    scores_rc = np.nan_to_num(scores_rc, nan=0.0, posinf=1e6, neginf=-1e6)

    i_f = int(np.argmax(scores))
    i_r = int(np.argmax(scores_rc))
    if scores[i_f] >= scores_rc[i_r]:
        strength = float(np.clip(scores[i_f], -1e4, 1e4))
        return +1, i_f, strength
    else:
        strength = float(np.clip(scores_rc[i_r], -1e4, 1e4))
        return -1, i_r, strength


def fetch_and_orient_from_fasta(
    bedfile,
    ref_genome_filepath,
    ctcfpfm,
    flanking_bp=15,
    core_bp=19
):
    """
    For each row (chrom,start,end), center on best motif,
    extract (4, 2*flanking_bp+core_bp), orient forward.
    Returns: seqs (N,4,L2), seq_len (L2), target_orients (N,)
    """
    df = pd.read_csv(bedfile, sep=None, engine='python', comment='#')
    peaks_table = df.iloc[:, :3].copy()
    ref_genome = pysam.FastaFile(ref_genome_filepath)

    ctcf_pfm = np.loadtxt(ctcfpfm, skiprows=1)
    ctcf_pwm = pfm_to_pwm(ctcf_pfm)
    ctcf_pwm_rc = pfm_to_pwm(np.flip(np.flip(ctcf_pfm, axis=0), axis=1))

    seqs, dirs = [], []
    for _, row in peaks_table.iterrows():
        chrom, start, end = row[['chrom', 'start', 'end']]
        start = int(start); end = int(end)

        seq = dna_1hot(ref_genome.fetch(chrom, start, end+1)).T  # (4,L)
        orient, motif_i, _ = scan_sequence(ctcf_pwm, ctcf_pwm_rc, seq)

        left  = start + motif_i - flanking_bp
        right = start + motif_i + flanking_bp + core_bp
        sub = dna_1hot(ref_genome.fetch(chrom, left, right)).T   # (4,L2)

        if orient == -1:
            sub = np.flip(sub, axis=0)
            sub = np.flip(sub, axis=1)

        dirs.append(orient)
        seqs.append(sub)

    seqs = np.array(seqs)
    seq_len = seqs.shape[-1]
    return seqs, seq_len, np.array(dirs, dtype=np.int8)


# ========================== Neighbor index + features (start-anchored) ==========================

def fourier_features(x, n_freq=8, min_period=1.0, max_period=200.0):
    """
    x: numpy array of distances in kb (absolute), shape (...,)
    Returns features (..., 2*n_freq) using log-spaced periods.
    """
    periods = np.logspace(np.log10(min_period), np.log10(max_period), num=n_freq)
    feats = []
    for p in periods:
        w = 2.0 * math.pi / p
        feats.append(np.sin(w * x))
        feats.append(np.cos(w * x))
    return np.stack(feats, axis=-1)  # (..., 2*n_freq)


def build_per_chrom_index(df):
    """
    Returns dict: chrom -> {'starts': array[int], 'rows': array[int]}
    Anchors are the 'start' positions only.
    """
    idx = {}
    for chrom, sub in df.groupby('chrom', sort=False):
        starts = sub['start'].to_numpy(dtype=np.int64)
        idx[chrom] = {'starts': starts, 'rows': sub.index.to_numpy()}
    return idx


def nearest_neighbors_for_row(chrom, start_anchor, chrom_index, K=8, window_kb=250.0, self_row=None):
    """
    Returns (neighbor_row_indices, distances_bp) for the K nearest within the window,
    using 'start' as the anchor for both target and neighbors.
    """
    if chrom not in chrom_index:
        return np.array([], dtype=int), np.array([], dtype=int)

    starts = chrom_index[chrom]['starts']
    rows = chrom_index[chrom]['rows']

    dists = starts - start_anchor
    mask = np.abs(dists) <= int(window_kb * 1000)
    cand_rows = rows[mask]
    cand_dists = dists[mask]

    if self_row is not None and cand_rows.size:
        keep = cand_rows != self_row
        cand_rows = cand_rows[keep]
        cand_dists = cand_dists[keep]

    if cand_rows.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    order = np.argsort(np.abs(cand_dists), kind="mergesort")[:K]
    return cand_rows[order], cand_dists[order]


def neighbor_features_from_df(df, row_i, chrom_index, ctcf_pwm, ctcf_pwm_rc,
                              K=8, window_kb=250.0, ref_fasta=None,
                              use_seq_col_if_available=True):
    """
    Build neighbor features for row_i (start-anchored).

    Feature per neighbor (F=20):
      [Fourier(|dist_kb|) (16), dist_sign (1), target_orient (1),
       neigh_orient (1), neigh_strength (1)]

    Returns:
      feats: (K, F) float32
      mask:  (K,) bool  (True = PAD / ignore in attention)
    """
    row = df.iloc[row_i]
    chrom = row['chrom']
    s = int(row['start']); e = int(row['end'])
    anchor = s  # start-anchored only

    neigh_rows, neigh_dists_bp = nearest_neighbors_for_row(
        chrom, anchor, chrom_index, K=K, window_kb=window_kb, self_row=row_i
    )

    # Target orientation
    if use_seq_col_if_available and 'sequence' in df.columns and isinstance(row['sequence'], str):
        seq_T = dna_1hot(row['sequence']).T
        t_orient, _, _ = scan_sequence(ctcf_pwm, ctcf_pwm_rc, seq_T)
    else:
        assert ref_fasta is not None, "ref_fasta required if no 'sequence' column"
        fasta = pysam.FastaFile(ref_fasta)
        seq_T = dna_1hot(fasta.fetch(chrom, s, e+1)).T
        t_orient, _, _ = scan_sequence(ctcf_pwm, ctcf_pwm_rc, seq_T)

    # If no neighbors, return all-PAD zeros
    if neigh_rows.size == 0:
        Fdim = 2*8 + 4
        feats = np.zeros((K, Fdim), dtype='float32')
        mask = np.ones((K,), dtype=bool)
        return feats, mask

    feats = []
    for j, dist_bp in zip(neigh_rows, neigh_dists_bp):
        nrow = df.iloc[j]
        n_chrom = nrow['chrom']; n_start = int(nrow['start']); n_end = int(nrow['end'])

        # neighbor orientation & strength
        if use_seq_col_if_available and 'sequence' in df.columns and isinstance(nrow['sequence'], str):
            n_seq_T = dna_1hot(nrow['sequence']).T
            n_orient, _, n_strength = scan_sequence(ctcf_pwm, ctcf_pwm_rc, n_seq_T)
        else:
            assert ref_fasta is not None, "ref_fasta required if no 'sequence' column"
            fasta = pysam.FastaFile(ref_fasta)
            n_seq_T = dna_1hot(fasta.fetch(n_chrom, n_start, n_end+1)).T
            n_orient, _, n_strength = scan_sequence(ctcf_pwm, ctcf_pwm_rc, n_seq_T)

        n_strength = float(np.clip(n_strength, -100.0, 100.0))
        dist_kb = float(dist_bp) / 1000.0
        if not np.isfinite(dist_kb):
            dist_kb = 0.0

        ff = fourier_features(np.array([abs(dist_kb)], dtype=float),
                              n_freq=8, min_period=1.0, max_period=200.0).reshape(-1)  # (16,)
        sign = np.sign(dist_kb)  # -1 (left) or +1 (right)
        feat = np.concatenate([
            ff, np.array([sign, float(t_orient), float(n_orient), float(n_strength)], dtype=float)
        ], axis=0)
        feats.append(feat.astype('float32'))

    # pad to K
    if len(feats) < K:
        Fdim = feats[0].shape[0]
        feats += [np.zeros((Fdim,), dtype='float32')] * (K - len(feats))

    feats = np.stack(feats, axis=0) if len(feats) else np.zeros((K, 2*8+4), dtype='float32')
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    mask = np.zeros((K,), dtype=bool)
    if len(neigh_rows) < K:
        mask[len(neigh_rows):] = True  # mark padded rows as PAD
    return feats, mask


# ========================== Dataset ==========================

class HybridCTCFOccupancyDataset(Dataset):
    def __init__(self, bedfile_path,
                 label_cols=['Accessible', 'Bound', 'Nucleosome.occupied'],
                 ref_genome_fasta='/project2/fudenber_735/genomes/mm10/mm10.fa',
                 ctcfpfm='files/MA0139.1.pfm',
                 flanking_bp=15, core_bp=19,
                 K=8, window_kb=250.0):
        """
        Yields: (x_seq: (4,L), neigh_feats: (K,F), neigh_mask: (K,), y: (C,))
        """
        self.df = pd.read_csv(bedfile_path, sep=None, engine='python', comment='#')
        self.df = self.df[self.df[label_cols].notnull().all(axis=1)].reset_index(drop=True)
        self.y = self.df[label_cols].values.astype('float32')
        self.label_cols = label_cols

        self.seqs, self.seq_len, self.target_orients = fetch_and_orient_from_fasta(
            bedfile_path, ref_genome_fasta, ctcfpfm, flanking_bp=flanking_bp, core_bp=core_bp
        )

        ctcf_pfm = np.loadtxt(ctcfpfm, skiprows=1)
        self.ctcf_pwm = pfm_to_pwm(ctcf_pfm)
        self.ctcf_pwm_rc = pfm_to_pwm(np.flip(np.flip(ctcf_pfm, axis=0), axis=1))

        self.chrom_index = build_per_chrom_index(self.df)
        self.K = K
        self.window_kb = window_kb
        self.ref_genome_fasta = ref_genome_fasta

        feats0, _ = neighbor_features_from_df(
            self.df, 0, self.chrom_index, self.ctcf_pwm, self.ctcf_pwm_rc,
            K=self.K, window_kb=self.window_kb, ref_fasta=self.ref_genome_fasta
        )
        self.neigh_feat_dim = feats0.shape[1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.tensor(self.seqs[idx], dtype=torch.float32)  # (4,L)
        feats, mask = neighbor_features_from_df(
            self.df, idx, self.chrom_index, self.ctcf_pwm, self.ctcf_pwm_rc,
            K=self.K, window_kb=self.window_kb, ref_fasta=self.ref_genome_fasta
        )
        nf = torch.tensor(feats, dtype=torch.float32)          # (K,F)
        nm = torch.tensor(mask, dtype=torch.bool)              # (K,)
        y = torch.tensor(self.y[idx], dtype=torch.float32)     # (C,)
        x = torch.nan_to_num(x,  nan=0.0, posinf=0.0, neginf=0.0)
        nf = torch.nan_to_num(nf, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y,  nan=0.0, posinf=0.0, neginf=0.0)
        return x, nf, nm, y


# ========================== Loss, metrics, train, predict ==========================

def _choose_loss(num_outputs, use_kldiv):
    if num_outputs == 1:
        return nn.BCEWithLogitsLoss()
    return nn.KLDivLoss(reduction='batchmean') if use_kldiv else nn.CrossEntropyLoss()


@torch.no_grad()
def _epoch_metrics(logits, y, loss_fn):
    if logits.shape[1] == 1:
        # Pearson r for single-output
        y_true = y.squeeze(1).cpu()
        y_pred = torch.sigmoid(logits.squeeze(1)).cpu()
        yt = y_true - y_true.mean()
        yp = y_pred - y_pred.mean()
        denom = torch.sqrt((yt**2).sum() * (yp**2).sum()) + 1e-8
        pearson = (yt * yp).sum() / denom
        return float(pearson.item())
    else:
        # accuracy for multi-class
        y_true = torch.argmax(y, dim=1).cpu()
        y_pred = torch.argmax(logits, dim=1).cpu()
        acc = (y_true == y_pred).float().mean()
        return float(acc.item())


def train_hybrid_model(
    bedfile_path,
    ref_genome_fasta='/project2/fudenber_735/genomes/mm10/mm10.fa',
    ctcfpfm='files/MA0139.1.pfm',
    save_weights_path='files/hybrid_flankcore_weights.pt',
    label_cols=['Accessible', 'Bound', 'Nucleosome.occupied'],
    K=8, window_kb=250.0,
    batch_size=32, epochs=20, lr=1e-3, use_kldiv=False,
    early_stopping=True, patience=5, min_delta=0.0, restore_best=True,
    device_override=None
):
    dev = device_override or ('cuda' if torch.cuda.is_available() else 'cpu')
    ds = HybridCTCFOccupancyDataset(
        bedfile_path, label_cols=label_cols,
        ref_genome_fasta=ref_genome_fasta, ctcfpfm=ctcfpfm,
        K=K, window_kb=window_kb
    )
    num_outputs = ds.y.shape[1]
    seq_len = ds.seq_len

    n_train = int(0.8 * len(ds))
    n_val = len(ds) - n_train
    tr, va = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)

    model = EnformerCNNHybrid(
        seq_len=seq_len, d_model=128, n_heads=4, out_features=num_outputs,
        neighbor_feat_dim=ds.neigh_feat_dim, dropout=0.1
    ).to(dev)

    loss_fn = _choose_loss(num_outputs, use_kldiv)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float('inf')
    best_state = None
    no_improve = 0

    for ep in range(1, epochs+1):
        # ---- train ----
        model.train()
        tr_loss_sum = 0.0
        tr_batches = 0
        for x, nf, nm, y in tr_loader:
            x, nf, nm, y = x.to(dev), nf.to(dev), nm.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(x, nf, nm)
            if num_outputs == 1:
                loss = loss_fn(logits.squeeze(1), y.squeeze(1))
            elif isinstance(loss_fn, nn.KLDivLoss):
                loss = loss_fn(F.log_softmax(logits, dim=1), y)
            else:
                loss = loss_fn(logits, torch.argmax(y, dim=1))

            if not torch.isfinite(loss):
                print("⚠️ Non-finite train loss encountered; skipping batch.")
                continue

            loss.backward()
            opt.step()
            tr_loss_sum += float(loss.item())
            tr_batches += 1

        tr_loss = tr_loss_sum / max(1, tr_batches)

        # ---- validate ----
        model.eval()
        va_loss_sum = 0.0
        va_batches = 0
        metric_vals = []
        with torch.no_grad():
            for x, nf, nm, y in va_loader:
                x, nf, nm, y = x.to(dev), nf.to(dev), nm.to(dev), y.to(dev)
                logits = model(x, nf, nm)

                if num_outputs == 1:
                    l = loss_fn(logits.squeeze(1), y.squeeze(1))
                elif isinstance(loss_fn, nn.KLDivLoss):
                    l = loss_fn(F.log_softmax(logits, dim=1), y)
                else:
                    l = loss_fn(logits, torch.argmax(y, dim=1))

                if not torch.isfinite(l):
                    print("⚠️ Non-finite val loss encountered; skipping batch.")
                    continue

                va_loss_sum += float(l.item())
                va_batches += 1
                metric_vals.append(_epoch_metrics(logits, y, loss_fn))

        va_loss = va_loss_sum / max(1, va_batches)
        metric = float(np.mean(metric_vals)) if metric_vals else float('nan')
        mname = "Pearson(r)" if num_outputs == 1 else "Acc"
        print(f"Epoch {ep:02d}/{epochs} | Train {tr_loss:.4f} | Val {va_loss:.4f} | {mname} {metric:.3f}")

        # early stopping
        if early_stopping:
            if va_loss < best_val - min_delta:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"⏹️ Early stopping at epoch {ep} (no improvement ≥ {min_delta} for {patience} epochs).")
                    break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"🔙 Restored best model (Val {best_val:.4f}).")

    os.makedirs(os.path.dirname(save_weights_path), exist_ok=True)
    torch.save(model.state_dict(), save_weights_path)
    print(f"✅ Saved hybrid model to {save_weights_path}")
    return model, ds.seq_len, ds.neigh_feat_dim, ds.label_cols


@torch.no_grad()
def predict_hybrid(
    bedfile_path,
    model_weights_path,
    ref_genome_fasta='/project2/fudenber_735/genomes/mm10/mm10.fa',
    ctcfpfm='files/MA0139.1.pfm',
    out_features=3,
    label_cols=['Accessible', 'Bound', 'Nucleosome.occupied'],
    K=8, window_kb=250.0,
    device_override=None,
    batch_size=64
):
    dev = device_override or ('cuda' if torch.cuda.is_available() else 'cpu')

    ds = HybridCTCFOccupancyDataset(
        bedfile_path, label_cols=label_cols,
        ref_genome_fasta=ref_genome_fasta, ctcfpfm=ctcfpfm,
        K=K, window_kb=window_kb
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model = EnformerCNNHybrid(
        seq_len=ds.seq_len, d_model=128, n_heads=4, out_features=out_features,
        neighbor_feat_dim=ds.neigh_feat_dim, dropout=0.0
    ).to(dev)

    state = torch.load(model_weights_path, map_location=dev)
    model.load_state_dict(state)
    model.eval()

    preds = []
    for x, nf, nm, _ in dl:
        x, nf, nm = x.to(dev), nf.to(dev), nm.to(dev)
        logits = model(x, nf, nm)
        if out_features == 1:
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            preds.append(probs[:, None])
        else:
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds.append(probs)
    preds = np.concatenate(preds, axis=0)

    df = ds.df.copy()
    if out_features == 1:
        base = label_cols[0] if label_cols else 'occupancy'
        df[f'predicted_{base}'] = preds[:, 0]
    else:
        names = [f'predicted_{c}' for c in label_cols]
        for i, n in enumerate(names):
            df[n] = preds[:, i]

    out_path = f'{bedfile_path}_with_predicted_hybrid.csv'
    df.to_csv(out_path, index=False)
    print(f"✅ Saved predictions to {out_path}")
    return df
