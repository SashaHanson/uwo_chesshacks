# train_alphazero.py
import torch
import torch.optim as optim
import torch.nn as nn
from model import ChessNet
from selfplay import play_selfplay_game
from buffer import ReplayBuffer
from arena import arena

def train_alphazero(
        warmstart="warmstart.pth",
        iterations=200,
        games_per_iter=10,
        batch=256,
        minibatches=50,
        sims=300,
        gating_games=20):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    current = ChessNet().to(device)
    current.load_state_dict(torch.load(warmstart, map_location=device))

    new = ChessNet().to(device)
    new.load_state_dict(current.state_dict())

    buffer = ReplayBuffer()

    opt = optim.Adam(new.parameters(), lr=1e-4)

    for it in range(iterations):
        print(f"\n=== Iteration {it} ===")

        # 1) SELFPLAY
        for g in range(games_per_iter):
            data = play_selfplay_game(current, sims, device)
            for s, p, v in data:
                buffer.push(s, p, v)

        if len(buffer) < batch * 5:
            continue

        # 2) TRAINING
        for _ in range(minibatches):
            S, P, V = buffer.sample(batch)

            S = torch.tensor(S, dtype=torch.float32, device=device)
            P = torch.tensor(P, dtype=torch.float32, device=device)
            V = torch.tensor(V, dtype=torch.float32, device=device)

            opt.zero_grad()
            pred_p, pred_v = new(S)

            v_loss = nn.MSELoss()(pred_v, V)
            p_loss = -(P * torch.log_softmax(pred_p, dim=1)).sum(dim=1).mean()
            loss = v_loss + p_loss

            loss.backward()
            opt.step()

        # 3) GATING
        score = arena(new, current, games=gating_games, sims=200)

        if score > 2:
            print("[✔] New model accepted")
            current.load_state_dict(new.state_dict())
            torch.save(current.state_dict(), f"checkpoint_iter{it}.pth")
        else:
            print("[x] New model rejected")
