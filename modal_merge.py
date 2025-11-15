# modal_merge.py
import modal
import numpy as np

app = modal.App("merge-datasets")

# Persistent volume so merged dataset survives across runs
v = modal.Volume.from_name("chess-data", create_if_missing=True)

@app.function(volumes={"/data": v}, timeout=60*30)
def merge_all():
    paths = [
        "/root/2025-01_elite_positions.npy",
        "/root/2021-10_niko_positions.npy",
        "/root/2021-11_niko_positions.npy",
    ]

    arrays = []
    for p in paths:
        print("Loading:", p)
        arr = np.load(p, allow_pickle=True)
        arrays.append(arr)
        print(f" → {len(arr):,} positions")

    merged = np.concatenate(arrays)
    print(f"\nMerged total: {len(merged):,} positions")

    # Save into volume
    outpath = "/data/merged_positions.npy"
    np.save(outpath, merged)

    print(f"Saved merged dataset to volume at: {outpath}")
    return outpath
