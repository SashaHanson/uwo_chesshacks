from .utils import chess_manager, GameContext

import chess
from chess import Move
import time
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===========================
# Device & paths
# ===========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
MODEL_PATH = os.path.join(WEIGHTS_DIR, "alphazero_model.pt")
MOVE_MAP_PATH = os.path.join(WEIGHTS_DIR, "move_index_map.npy")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Expected weights at {MODEL_PATH}. "
        "Copy alphazero_model.pt from Modal into src/weights/."
    )

if not os.path.exists(MOVE_MAP_PATH):
    raise FileNotFoundError(
        f"Expected move_index_map.npy at {MOVE_MAP_PATH}. "
        "Copy it from Modal into src/weights/."
    )

# ===========================
# Model definition
# ===========================

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        r = x
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + r)


class AlphaZeroCNN(nn.Module):
    def __init__(self, output_dim: int, channels=192, num_blocks=12):
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


# ===========================
# Load move index map
# ===========================

_raw = np.load(MOVE_MAP_PATH, allow_pickle=True)
move_to_index = dict(_raw)
index_to_move = {v: k for k, v in move_to_index.items()}
OUTPUT_DIM = len(move_to_index)

# ===========================
# Lazy-load model
# ===========================

_model = None

def get_model():
    global _model
    if _model is None:
        print("Lazy-loading AlphaZero model...")
        model = AlphaZeroCNN(OUTPUT_DIM)
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state, strict=True)
        model.to(DEVICE)
        model.eval()
        _model = model
        print("Model loaded.")
    return _model


# ===========================
# Board → 12×8×8 vector
# ===========================

def board_to_vector(board: chess.Board) -> np.ndarray:
    planes = np.zeros((12, 8, 8), dtype=np.float32)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece:
            continue

        r = chess.square_rank(sq)
        c = chess.square_file(sq)
        t = piece.piece_type - 1
        offset = 0 if piece.color == chess.WHITE else 6
        planes[offset + t, r, c] = 1.0

    return planes.reshape(-1)


# ===========================
# Policy + Value from net
# ===========================

def policy_from_net(board: chess.Board, legal_moves):
    x = torch.tensor(
        board_to_vector(board),
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    with torch.no_grad():
        logits, _ = get_model()(x)
        logits = logits[0].cpu().numpy()

    # Mask everything, then fill in legal moves
    mask = np.full_like(logits, -1e9, dtype=np.float32)
    filtered = []

    for mv in legal_moves:
        u = mv.uci()
        if u in move_to_index:
            idx = move_to_index[u]
            mask[idx] = logits[idx]
            filtered.append(mv)

    if not filtered:
        # fallback if model never saw a move (rare)
        return {mv: 1.0 / len(legal_moves) for mv in legal_moves}

    mask -= mask.max()
    exp = np.exp(mask)
    probs = exp / exp.sum()

    return {mv: float(probs[move_to_index[mv.uci()]]) for mv in filtered}


def evaluate_position(board: chess.Board, root_player: bool) -> float:
    """
    root_player: True for White, False for Black (same as board.turn).
    We assume the net outputs value from POV of side to move.
    We convert to POV of root_player.
    """
    x = torch.tensor(
        board_to_vector(board),
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    with torch.no_grad():
        _, v = get_model()(x)
        v = float(v.item())

    # If it's not root_player to move, flip the sign
    if board.turn != root_player:
        v = -v

    return max(-1.0, min(1.0, v))


# ===========================
# Improved MCTS (AlphaZero-lite)
# ===========================

# Simple transposition table
TT = {}
TT_MAX = 50000


class MCTSNode:
    def __init__(self, board, parent, prior, root_player):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.N = 0          # visit count
        self.W = 0.0        # total value from root POV
        self.Q = 0.0        # mean value from root POV
        self.root_player = root_player  # True = White, False = Black

    def expanded(self):
        return len(self.children) > 0


def select_child(node, c_puct):
    total_N = sum(child.N for child in node.children.values()) + 1e-8

    best_score = -1e9
    best_mv = None
    best_child = None

    for mv, child in node.children.items():
        U = c_puct * child.prior * np.sqrt(total_N) / (1 + child.N)
        score = child.Q + U
        if score > best_score:
            best_score = score
            best_mv = mv
            best_child = child

    return best_mv, best_child


def expand(node, add_noise=False):
    board = node.board

    # Don't expand terminal nodes
    if board.outcome() is not None:
        return

    legal = list(board.legal_moves)
    if not legal:
        return

    priors = policy_from_net(board, legal)

    # Dirichlet noise at root only
    if add_noise:
        alpha = 0.3
        eps = 0.25
        noise = np.random.dirichlet([alpha] * len(legal))
        for i, mv in enumerate(legal):
            if mv in priors:
                priors[mv] = (1 - eps) * priors[mv] + eps * float(noise[i])

    for mv in legal:
        b = board.copy()
        b.push(mv)
        node.children[mv] = MCTSNode(
            board=b,
            parent=node,
            prior=priors.get(mv, 1e-3),
            root_player=node.root_player,
        )


def position_key(board: chess.Board):
    """
    Safe position key for transposition table.
    Uses FEN + side to move.
    """
    return (board.turn, board.fen())


def mcts(root_board, n_sim=30, c_puct=1.3):
    key = position_key(root_board)
    root_player = root_board.turn  # True = White, False = Black

    # Reuse root node from TT if we have it
    if key in TT:
        root = TT[key]
        root.parent = None
    else:
        root = MCTSNode(root_board.copy(), None, 1.0, root_player)
        expand(root, add_noise=True)
        TT[key] = root

    for _ in range(n_sim):
        node = root
        path = [node]

        # ====================
        # SELECTION
        # ====================
        while node.expanded() and node.board.outcome() is None:
            _, node = select_child(node, c_puct)
            path.append(node)

        # ====================
        # EVALUATION (root POV)
        # ====================
        value = evaluate_position(node.board, root_player=root_player)

        # ====================
        # EXPANSION
        # ====================
        if node.board.outcome() is None:
            expand(node)

        # ====================
        # BACKUP
        # Store values from root POV, no sign flipping needed
        # ====================
        for n in reversed(path):
            n.N += 1
            n.W += value
            n.Q = n.W / n.N

    # Limit TT size
    if len(TT) > TT_MAX:
        TT.clear()

    visits = {mv: child.N for mv, child in root.children.items()}
    return visits


# ===========================
# ENTRYPOINTS
# ===========================

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    board = ctx.board
    legal = list(board.legal_moves)

    if not legal:
        ctx.logProbabilities({})
        raise ValueError("No legal moves")

    start = time.time()

    # 30 simulations, tuned for CPU
    visits = mcts(board, n_sim=15, c_puct=1.3)

    total = sum(visits.values()) + 1e-8
    probs = {mv: visits.get(mv, 0) / total for mv in legal}

    ctx.logProbabilities(probs)

    best_move = max(visits.items(), key=lambda kv: kv[1])[0]

    print(f"MCTS chose {best_move.uci()} in {time.time() - start:.2f}s")

    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    # Optionally clear TT between games:
    # global TT
    # TT = {}
    pass
