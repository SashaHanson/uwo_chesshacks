import numpy as np

m1 = np.load("training/data/move_index_map.npy", allow_pickle=True)
m2 = np.load("src/weights/move_index_map.npy", allow_pickle=True)

print(type(m1), type(m2))
print(m1.shape, m2.shape)
print(m1 == m2)  # or compare dict(m1.item()) etc.

