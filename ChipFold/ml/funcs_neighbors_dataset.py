import torch
from torch.utils.data import Dataset

class NeighborSeqDataset(Dataset):
    def __init__(self, df, label_cols, neighbor_map, seq_to_onehot):
        self.df = df
        self.label_cols = label_cols
        self.neighbor_map = neighbor_map
        self.seq_to_onehot = seq_to_onehot

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = self.seq_to_onehot(row["sequence"])
        y = torch.tensor(row[self.label_cols].values, dtype=torch.float32)

        neighbors = []
        for nb in self.neighbor_map.get(idx, []):
            nb_seq = self.df.loc[(self.df.chrom == nb["chrom"]) & 
                                 (self.df.start == nb["start"])].sequence.values
            if len(nb_seq) > 0:
                neighbors.append(self.seq_to_onehot(nb_seq[0]))

        if len(neighbors) > 0:
            neighbors = torch.stack(neighbors)
        else:
            neighbors = torch.zeros((0, seq.shape[0], seq.shape[1]))

        return torch.tensor(seq, dtype=torch.float32), neighbors, y
