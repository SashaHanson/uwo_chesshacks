# modal_warmstart.py
from modal import Image, Stub, gpu
import modal
import torch
import numpy as np
from chess_engine.model import ChessNet

stub = Stub("chess-warmstart")

image = (
    Image.debian_slim()
    .pip_install("torch", "numpy")
)

@stub.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 2,
)
def warmstart():
    device = "cuda"

    print("Loading encoded dataset...")
    X = np.load("/root/project/chess_engine/elite_states.npy")
    Y = np.load("/root/project/chess_engine/elite_policies.npy")

    model = ChessNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch = 256

    N = len(X)
    print("Dataset size:", N)

    for epoch in range(3):
        print("Epoch", epoch)
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
                print(f"{i}/{N} loss={loss.item():.4f}")

    torch.save(model.state_dict(), "/root/project/warmstart.pth")
    print("Saved warmstart.pth")
