# supervised_train.py
import numpy as np
import torch
import torch.optim as optim
from model import ChessNet

def warmstart_numpy(epochs=2, batch=256):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = np.load("elite_states.npy")
    Y = np.load("elite_policies.npy")

    model = ChessNet().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)

    N = len(X)
    for e in range(epochs):
        perm = np.random.permutation(N)

        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            s = torch.tensor(X[idx], dtype=torch.float32, device=device)
            p = torch.tensor(Y[idx], dtype=torch.float32, device=device)

            opt.zero_grad()
            logits, _ = model(s)
            loss = -(p * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss.backward()
            opt.step()

            if i % 20000 == 0:
                print(f"Epoch {e} step {i} loss={loss.item():.4f}")

    torch.save(model.state_dict(), "warmstart.pth")
    print("[✔] Saved warmstart.pth")
