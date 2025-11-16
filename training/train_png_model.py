import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from modal import gpu

app = modal.App("chess-train-hf")

PROJECT_ROOT = "/root/chesshacks"
TRAINING_LOCAL = "training"
TRAINING_REMOTE = f"{PROJECT_ROOT}/training"

vol = modal.Volume.from_name("chess-training-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("numpy", "torch", "tqdm")
    .add_local_dir(local_path=TRAINING_LOCAL, remote_path=TRAINING_REMOTE)
)

GPU_CONFIG = "A100:2"

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=80000,
    volumes={"/root/vol": vol},
    max_containers=1,
)
def train_model():
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    from tqdm import tqdm
    from torch.amp import autocast, GradScaler


    # ============================
    # HYPERPARAMETERS
    # ============================
    BATCH_SIZE = 8192
    EPOCHS = 25
    BASE_LR = 1e-3
    MIN_LR = 1e-5
    WEIGHT_DECAY = 1e-4
    VALUE_LOSS_WEIGHT = 0.5
    LABEL_SMOOTHING = 0.1
    MAX_GRAD_NORM = 1.0

    # ============================
    # LOAD DATA (FROM VOLUME)
    # ============================
    DATA_DIR = "/root/vol/data"

    X_PATH = f"{DATA_DIR}/X.npy"
    Y_PATH = f"{DATA_DIR}/Y.npy"
    V_PATH = f"{DATA_DIR}/Y_value.npy"
    MAP_PATH = f"{DATA_DIR}/move_index_map.npy"

    print("Loading dataset from volume:", DATA_DIR)

    X = np.load(X_PATH, mmap_mode="r")      # streaming: avoids RAM blowup
    Y = np.load(Y_PATH, mmap_mode="r")
    V = np.load(V_PATH, mmap_mode="r")

    move_index_map = dict(np.load(MAP_PATH, allow_pickle=True))
    output_dim = len(move_index_map)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("Value shape:", V.shape)
    print("Classes:", output_dim)

    # Convert to tensors
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.long)
    V_t = torch.tensor(V, dtype=torch.float32)

    dataset = TensorDataset(X_t, Y_t, V_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # ============================
    # MODEL
    # ============================
    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            r = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(out + r)

    class AlphaZeroCNN(nn.Module):
        def __init__(self, output_dim, channels=192, num_blocks=12):
            super().__init__()
            self.conv_in = nn.Conv2d(12, channels, 3, padding=1)
            self.bn_in = nn.BatchNorm2d(channels)
            self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

            # Policy head
            self.policy_conv = nn.Conv2d(channels, 2, 1)
            self.policy_bn = nn.BatchNorm2d(2)
            self.policy_fc = nn.Linear(2 * 64, output_dim)

            # Value head
            self.value_conv = nn.Conv2d(channels, 1, 1)
            self.value_bn = nn.BatchNorm2d(1)
            self.value_fc1 = nn.Linear(64, 256)
            self.value_fc2 = nn.Linear(256, 1)

        def forward(self, x):
            x = x.view(x.size(0), 12, 8, 8)

            h = F.relu(self.bn_in(self.conv_in(x)))
            h = self.res_blocks(h)

            # Policy
            p = F.relu(self.policy_bn(self.policy_conv(h)))
            p = p.view(p.size(0), -1)
            logits = self.policy_fc(p)

            # Value
            v = F.relu(self.value_bn(self.value_conv(h)))
            v = v.view(v.size(0), -1)
            v = F.relu(self.value_fc1(v))
            v = torch.tanh(self.value_fc2(v)).squeeze(-1)

            return logits, v

    # ============================
    # TRAINING SETUP
    # ============================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on:", device.upper())

    model = AlphaZeroCNN(output_dim).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY, fused=True
    )

    policy_loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    value_loss_fn = nn.MSELoss()

    scaler = GradScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)

    CHECKPOINT_DIR = "/root/vol"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ============================
    # TRAIN LOOP
    # ============================
    for epoch in range(EPOCHS):
        model.train()
        p_loss_total = 0.0
        v_loss_total = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for xb, y_pb, y_vb in pbar:
            xb = xb.to(device)
            y_pb = y_pb.to(device)
            y_vb = y_vb.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=True):
                logits, v_pred = model(xb)
                loss_p = policy_loss_fn(logits, y_pb)
                loss_v = value_loss_fn(v_pred, y_vb)
                loss = loss_p + VALUE_LOSS_WEIGHT * loss_v

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()

            p_loss_total += loss_p.item()
            v_loss_total += loss_v.item()

        scheduler.step()

        avg_p_loss = p_loss_total / len(loader)
        avg_v_loss = v_loss_total / len(loader)

        print(
            f"Epoch {epoch+1} | "
            f"LR {scheduler.get_last_lr()[0]:.6f} | "
            f"P Loss {avg_p_loss:.4f} | "
            f"V Loss {avg_v_loss:.4f}"
        )

        # ============================
        # SAVE CHECKPOINT EACH EPOCH
        # ============================
        epoch_ckpt_path = os.path.join(CHECKPOINT_DIR, f"alphazero_epoch_{epoch+1}.pt")
        latest_ckpt_path = os.path.join(CHECKPOINT_DIR, "alphazero_model.pt")

        torch.save(model.state_dict(), epoch_ckpt_path)
        torch.save(model.state_dict(), latest_ckpt_path)
        vol.commit()
        print(f"[Checkpoint] Saved epoch {epoch+1} to {epoch_ckpt_path} and {latest_ckpt_path}")

    print("Training finished.")


@app.local_entrypoint()
def main():
    train_model.remote()
