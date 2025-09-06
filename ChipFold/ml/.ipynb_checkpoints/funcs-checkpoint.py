# funcs.py
import os
import random
import numpy as np
import pandas as pd
import pysam
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from .cnn_model import FlankCoreModel as CtcfOccupPredictor

torch.manual_seed(2024)
device = 'cpu'  # prediction is fast, so GPU is not necessary


# -------------------- utilities --------------------

def pfm_to_pwm(pfm, background_frequency=0.25):
    s = pfm.sum(axis=0)
    pwm = np.log((pfm / s) / background_frequency)
    return pwm


def dna_1hot(seq, seq_len=None, n_uniform=False, n_sample=False):
    """One-hot encode a sequence (A,C,G,T)."""
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
            if nt == 'A':
                seq_code[i, 0] = 1
            elif nt == 'C':
                seq_code[i, 1] = 1
            elif nt == 'G':
                seq_code[i, 2] = 1
            elif nt == 'T':
                seq_code[i, 3] = 1
            else:
                if n_uniform:
                    seq_code[i, :] = 0.25
                elif n_sample:
                    ni = random.randint(0, 3)
                    seq_code[i, ni] = 1
    return seq_code


def scan_sequence(pwm, pwm_rc, seq):
    k = pwm.shape[1]  # motif length
    scores = np.array([
        np.nansum((pwm * seq[:, i:i + k]))
        for i in range(seq.shape[1] - k + 1)
    ])
    scores_rc = np.array([
        np.nansum((pwm_rc * seq[:, i:i + k]))
        for i in range(seq.shape[1] - k + 1)
    ])

    motif_position = scores.argmax()
    rc_motif_position = scores_rc.argmax()

    if scores[motif_position] > scores_rc[rc_motif_position]:
        return "+", motif_position
    else:
        return "-", rc_motif_position


def fetch_and_orient_from_fasta(
    bedfile,
    ref_genome_filepath='/project/fudenber_735/genomes/mm10/mm10.fa',
    ctcfpfm='data/MA0139.1.pfm',
    flanking_bp=15,
    core_bp=19
):
    # auto-detect CSV/TSV
    df = pd.read_csv(bedfile, sep=None, engine='python', comment='#')
    peaks_table = df.iloc[:, :3].copy()  # expects first 3 cols: chrom, start, end
    ref_genome = pysam.FastaFile(ref_genome_filepath)

    ctcf_pfm = np.loadtxt(ctcfpfm, skiprows=1)
    ctcf_pwm = pfm_to_pwm(ctcf_pfm)

    ctcf_pfm_rc = np.flip(ctcf_pfm, axis=[0])
    ctcf_pfm_rc = np.flip(ctcf_pfm_rc, axis=[1])
    ctcf_pwm_rc = pfm_to_pwm(ctcf_pfm_rc)

    seqs = []
    for _, row in peaks_table.iterrows():
        chrom, start, end = row[['chrom', 'start', 'end']]
        # initial window to locate motif
        seq = dna_1hot(ref_genome.fetch(chrom, int(start), int(end) + 1))  # inclusive end
        seq = seq.T
        direction, ctcf_start = scan_sequence(ctcf_pwm, ctcf_pwm_rc, seq)

        # extract centered window around motif (flanks + core)
        left = int(start + ctcf_start - flanking_bp)
        right = int(start + ctcf_start + flanking_bp + core_bp)
        seq2 = dna_1hot(ref_genome.fetch(chrom, left, right)).T

        if direction == "-":
            seq2 = np.flip(seq2, axis=[0])
            seq2 = np.flip(seq2, axis=[1])

        seqs.append(seq2)

    seqs = np.array(seqs)
    seq_len = 2 * flanking_bp + core_bp
    return seqs, seq_len


# -------------------- predict --------------------

def predict_ctcf_occupancy(
    ctcf_bed,
    ctcfpfm='data/MA0139.1.pfm',
    model_weights_path='data/model_weights.pt',
    out_features=3,              # set to 1 for single-output models
    pred_device=None,
    class_names=None,            # optional names for multi-class columns
    bound_index=1                # which class to also expose as a single "bound" prob
):
    if pred_device is None:
        pred_device = device

    # sequences
    seqs, seq_len = fetch_and_orient_from_fasta(ctcf_bed, ctcfpfm=ctcfpfm)
    seqs = torch.tensor(seqs, dtype=torch.float32, device=pred_device)

    # input table
    peaks_table = pd.read_csv(ctcf_bed, sep=None, engine='python', comment='#')

    # model
    best_model = CtcfOccupPredictor(
        seq_len=seq_len, n_head=11, kernel_size=3, out_features=out_features
    ).to(pred_device)

    state = torch.load(model_weights_path, map_location=pred_device)
    if isinstance(state, dict) and 'state_dict' in state:
        best_model.load_state_dict(state['state_dict'])
    else:
        best_model.load_state_dict(state)

    best_model.eval()
    with torch.no_grad():
        logits = best_model(seqs)

        if logits.shape[1] == 1:
            # single-output: sigmoid
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()  # (N,)
            peaks_table['predicted_occupancy'] = probs
        else:
            # multi-class: softmax
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()      # (N, C)

            if class_names and len(class_names) == probs.shape[1]:
                for i, name in enumerate(class_names):
                    peaks_table[f'pred_{name}'] = probs[:, i]
            else:
                for i in range(probs.shape[1]):
                    peaks_table[f'pred_class{i}'] = probs[:, i]

            bi = bound_index if bound_index < probs.shape[1] else 1
            peaks_table['predicted_bound_prob'] = probs[:, bi]

    out_path = f'{ctcf_bed}_with_predicted_occupancy.csv'
    peaks_table.to_csv(out_path, index=False)
    print(f"✅ Saved predictions to {out_path}")
    return peaks_table


# -------------------- dataset & train --------------------

class CTCFOccupancyDataset(Dataset):
    def __init__(self, bedfile,
                 label_cols=['Accessible', 'Bound', 'Nucleosome.occupied'],
                 ctcfpfm='files/MA0139.1.pfm'):
        self.df = pd.read_csv(bedfile, sep=None, engine='python', comment='#')
        self.df = self.df[self.df[label_cols].notnull().all(axis=1)]
        self.labels = self.df[label_cols].values.astype("float32")
        self.seqs, self.seq_len = fetch_and_orient_from_fasta(bedfile, ctcfpfm=ctcfpfm)  # (N, 4, L)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = torch.tensor(self.seqs[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def train_ctcf_model(
    bedfile_path='files/sites_with_freqs_and_seqs.csv',
    ctcfpfm='files/MA0139.1.pfm',
    save_weights_path='files/flankcore_trained_weights.pt',
    batch_size=32,
    epochs=20,
    lr=1e-3,
    device_override=None,
    use_kldiv=False,
    label_cols=['Accessible', 'Bound', 'Nucleosome.occupied']
):
    dev = device_override or ('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    dataset = CTCFOccupancyDataset(bedfile_path, ctcfpfm=ctcfpfm, label_cols=label_cols)
    num_outputs = dataset[0][1].shape[0]
    seq_len = dataset.seq_len

    # split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # model
    model = CtcfOccupPredictor(seq_len=seq_len, n_head=11, kernel_size=3, out_features=num_outputs).to(dev)

    # loss
    if num_outputs == 1:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.KLDivLoss(reduction='batchmean') if use_kldiv else nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            logits = model(x)

            if num_outputs == 1:
                loss = loss_fn(logits.squeeze(1), y.squeeze(1))
            elif isinstance(loss_fn, nn.KLDivLoss):
                loss = loss_fn(F.log_softmax(logits, dim=1), y)          # y are probs
            else:
                loss = loss_fn(logits, torch.argmax(y, dim=1))           # y are one-hot

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # val
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
                if num_outputs == 1:
                    vloss = loss_fn(logits.squeeze(1), y.squeeze(1))
                elif isinstance(loss_fn, nn.KLDivLoss):
                    vloss = loss_fn(F.log_softmax(logits, dim=1), y)
                else:
                    vloss = loss_fn(logits, torch.argmax(y, dim=1))
                total_val_loss += vloss.item()

        print(f"Epoch {epoch+1}/{epochs} | Train {total_train_loss/len(train_loader):.4f} | Val {total_val_loss/len(val_loader):.4f}")

    # save
    os.makedirs(os.path.dirname(save_weights_path), exist_ok=True)
    torch.save(model.state_dict(), save_weights_path)
    print(f"✅ Model weights saved at {save_weights_path}")
    return model
