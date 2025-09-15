import numpy as np
import pandas as pd

def add_neighbor_features(df, max_distance=30000, k_neighbors=5):
    """
    For each site, add up to k neighbors within max_distance.
    """
    df_sorted = df.sort_values(["chrom", "start"]).reset_index(drop=True)
    neighbor_map = {}

    for i, row in df_sorted.iterrows():
        chrom, start, end = row["chrom"], row["start"], row["end"]
        center = (start + end) // 2

        neighbors = []
        for j in range(max(0, i - k_neighbors), min(len(df_sorted), i + k_neighbors + 1)):
            if i == j: continue
            other = df_sorted.iloc[j]
            if other["chrom"] != chrom: continue

            dist = abs(center - (other["start"] + other["end"]) // 2)
            if dist <= max_distance:
                neighbors.append({
                    "chrom": other["chrom"],
                    "start": other["start"],
                    "end": other["end"],
                    "dist": dist
                })
        neighbor_map[i] = neighbors
    return neighbor_map
