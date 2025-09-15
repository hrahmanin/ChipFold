# ChipFold/ml/funcs_neighbors_dataset.py
import torch
from torch.utils.data import Dataset

class NeighborSeqDataset(Dataset):
    def __init__(self, df, label_cols, neighbor_map, seq_to_onehot):
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.neighbor_map = neighbor_map
        self.seq_to_onehot = seq_to_onehot

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # sequence -> (L,4) float32 tensor
        seq = self.seq_to_onehot(row["sequence"])
        seq = torch.as_tensor(seq, dtype=torch.float32)

        # label(s) as float32 tensor
        y_vals = row[self.label_cols].astype(float).values
        y = torch.as_tensor(y_vals, dtype=torch.float32)

        # neighbors -> variable K x L x 4 (each as float32 tensor)
        neighbors_list = []
        for nb in self.neighbor_map.get(idx, []):
            hits = self.df[
                (self.df.chrom == nb["chrom"]) &
                (self.df.start == nb["start"]) &
                (self.df.end   == nb["end"])
            ]
            if len(hits) > 0:
                nb_seq = self.seq_to_onehot(hits.iloc[0]["sequence"])
                neighbors_list.append(torch.as_tensor(nb_seq, dtype=torch.float32))

        if len(neighbors_list) > 0:
            neighbors = torch.stack(neighbors_list, dim=0)  # (K,L,4)
        else:
            neighbors = torch.zeros((0, seq.shape[0], seq.shape[1]), dtype=torch.float32)

        return seq, neighbors, y
