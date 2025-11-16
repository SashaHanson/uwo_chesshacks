import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

app = modal.App("chess-train")

PROJECT_ROOT = "/root/chesshacks"
TRAINING_LOCAL = "training"
TRAINING_REMOTE = f"{PROJECT_ROOT}/training"

# Shared volume
vol = modal.Volume.from_name("chess-training-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("numpy", "torch", "tqdm")
    .add_local_dir(local_path=TRAINING_LOCAL, remote_path=TRAINING_REMOTE)
)

#GPU_CONFIG = "L40S"



@app.function(
    image=image,
    gpu=modal.gpu.H100(count=2)
    timeout=80000,
    volumes={"/root/vol": vol},
    max_containers=10,
)
def train_model():

    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    from torch.cuda.amp import autocast, GradScaler
    from tqdm import tqdm

    # -----------------------------
    # HYPERPARAMETERS
    # -----------------------------
    BATCH_SIZE = 2048
    EPOCHS = 30                      # was 10 – more training for stronger weights
    BASE_LR = 1e-3
    MIN_LR = 1e-5
    WEIGHT_DECAY = 1e-4
    VALUE_LOSS_WEIGHT = 0.5          # weight for value loss relative to policy loss
    LABEL_SMOOTHING = 0.1
    MAX_GRAD_NORM = 1.0

    # --------------------------------------
    # Load dataset
    # --------------------------------------
    DATA_DIR = "/root/vol/data"

    X_PATH = os.path.join(DATA_DIR, "X.npy")
    Y_POLICY_PATH = os.path.join(DATA_DIR, "Y.npy")
    Y_VALUE_PATH = os.path.join(DATA_DIR, "Y_value.npy")
    MOVE_MAP_PATH = os.path.join(DATA_DIR, "move_index_map.npy")

    print("Loading dataset from:", DATA_DIR)

    if not os.path.exists(X_PATH):
        raise FileNotFoundError(f"Missing X.npy at {X_PATH}")
    if not os.path.exists(Y_POLICY_PATH):
        raise FileNotFoundError(f"Missing Y.npy at {Y_POLICY_PATH}")
    if not os.path.exists(Y_VALUE_PATH):
        raise FileNotFoundError(f"Missing Y_value.npy at {Y_VALUE_PATH}")
    if not os.path.exists(MOVE_MAP_PATH):
        raise FileNotFoundError(f"Missing move_index_map.npy at {MOVE_MAP_PATH}")

    X = np.load(X_PATH)
    Y_policy = np.load(Y_POLICY_PATH)
    Y_value = np.load(Y_VALUE_PATH)

    move_index_map = dict(np.load(MOVE_MAP_PATH, allow_pickle=True))
    output_dim = len(move_index_map)

    print("X:", X.shape)
    print("Policy labels:", Y_policy.shape)
    print("Value labels:", Y_value.shape)
    print("Number of move classes:", output_dim)

    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    Y_policy = torch.tensor(Y_policy, dtype=torch.long)
    Y_value = torch.tensor(Y_value, dtype=torch.float32)

    dataset = TensorDataset(X, Y_policy, Y_value)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # ============================
    # MODEL DEFINITION (unchanged)
    # ============================

    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()

            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)

            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            residual = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(out + residual)

    class AlphaZeroCNN(nn.Module):
        def __init__(self, output_dim, channels=192, num_blocks=12):
            super().__init__()

            self.conv_in = nn.Conv2d(12, channels, 3, padding=1)
            self.bn_in = nn.BatchNorm2d(channels)

            self.res_blocks = nn.Sequential(
                *[ResidualBlock(channels) for _ in range(num_blocks)]
            )

            # Policy head
            self.policy_conv = nn.Conv2d(channels, 2, 1)
            self.policy_bn = nn.BatchNorm2d(2)
            self.policy_fc = nn.Linear(2 * 8 * 8, output_dim)

            # Value head
            self.value_conv = nn.Conv2d(channels, 1, 1)
            self.value_bn = nn.BatchNorm2d(1)
            self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
            self.value_fc2 = nn.Linear(256, 1)

        def forward(self, x):
            # x: (batch, 768) → (batch, 12, 8, 8)
            x = x.view(x.size(0), 12, 8, 8)

            out = F.relu(self.bn_in(self.conv_in(x)))
            out = self.res_blocks(out)

            # Policy
            p = F.relu(self.policy_bn(self.policy_conv(out)))
            p = p.view(p.size(0), -1)
            p = self.policy_fc(p)

            # Value
            v = F.relu(self.value_bn(self.value_conv(out)))
            v = v.view(v.size(0), -1)
            v = F.relu(self.value_fc1(v))
            v = torch.tanh(self.value_fc2(v))

            return p, v

    # -----------------------------
    # TRAINING SETUP
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = AlphaZeroCNN(output_dim).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,
        fused=True
    )

    # policy + value losses
    policy_loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    value_loss_fn = nn.MSELoss()

    scaler = GradScaler()

    # cosine LR schedule over epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)

    # -----------------------------
    # TRAIN LOOP
    # -----------------------------
    for epoch in range(EPOCHS):
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for xb, ypb, yvb in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

            xb = xb.to(device, non_blocking=True)
            ypb = ypb.to(device, non_blocking=True)
            yvb = yvb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                policy_logits, value_pred = model(xb)
                value_pred = value_pred.squeeze(-1)

                loss_p = policy_loss_fn(policy_logits, ypb)
                loss_v = value_loss_fn(value_pred, yvb)
                loss = loss_p + VALUE_LOSS_WEIGHT * loss_v

            scaler.scale(loss).backward()

            # gradient clipping with AMP: unscale then clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            scaler.step(optimizer)
            scaler.update()

            total_policy_loss += loss_p.item()
            total_value_loss += loss_v.item()

        scheduler.step()

        avg_policy_loss = total_policy_loss / len(loader)
        avg_value_loss = total_value_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"LR: {current_lr:.6f} | "
            f"Policy Loss: {avg_policy_loss:.4f} | "
            f"Value Loss: {avg_value_loss:.4f}"
        )

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    save_path = "/root/vol/alphazero_model.pt"
    torch.save(model.state_dict(), save_path)
    vol.commit()
    print("Saved AlphaZero model to:", save_path)


@app.local_entrypoint()
def main():
    train_model.remote()
