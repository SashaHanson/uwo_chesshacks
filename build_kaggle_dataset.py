# build_kaggle_dataset.py
from chess_engine.dataset_kaggle import extract_from_kaggle
import numpy as np

print("Extracting Kaggle dataset...")
X, Y = extract_from_kaggle("games.csv", max_positions=200_000)

print("Saving numpy arrays...")
np.save("kaggle_states.npy", X)
np.save("kaggle_policies.npy", Y)

print("Done!")
print("Saved kaggle_states.npy and kaggle_policies.npy")
