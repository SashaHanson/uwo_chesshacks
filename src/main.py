from .utils import chess_manager, GameContext

from chess import Move
import chess
import random
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ===========================
# Device & paths
# ===========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
MODEL_PATH = os.path.join(WEIGHTS_DIR, "alphazero_model.pt")
MOVE_MAP_PATH = os.path.join(WEIGHTS_DIR, "move_index_map.npy")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Expected model weights at {MODEL_PATH}. "
        "Make sure you downloaded alphazero_model.pt into src/weights/."
    )

if not os.path.exists(MOVE_MAP_PATH):
    raise FileNotFoundError(
        f"Expected move_index_map.npy at {MOVE_MAP_PATH}. "
        "Make sure you downloaded it into src/weights/."
    )


# ===========================
# Model
# ===========================

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
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
    def __init__(self, output_dim: int, channels=192, num_blocks=12):
        super().__init__()

        self.conv_in = nn.Conv2d(12, channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        self.policy_conv = nn.Conv2d(channels, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, output_dim)

        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1, 12, 8, 8)

        h = F.relu(self.bn_in(self.conv_in(x)))
        h = self.res_blocks(h)

        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value.squeeze(-1)


# ===========================
# Load move index map
# ===========================

_raw_map = np.load(MOVE_MAP_PATH, allow_pickle=True)
if isinstance(_raw_map, np.ndarray):
    if _raw_map.ndim == 0:
        _raw_map = _raw_map.item()
    else:
        _raw_map = dict(_raw_map.tolist())

if not isinstance(_raw_map, dict):
    _raw_map = dict(_raw_map)

if all(isinstance(k, str) for k in _raw_map.keys()):
    move_to_index = _raw_map
    index_to_move = {v: k for k, v in _raw_map.items()}
else:
    index_to_move = _raw_map
    move_to_index = {v: k for k, v in _raw_map.items()}

OUTPUT_DIM = len(index_to_move)


# ===========================
# LAZY MODEL LOADING (IMPORTANT)
# ===========================

_model = None

def get_model():
    """Load the model the first time it's needed, not during server startup."""
    global _model
    if _model is None:
        print("Lazy-loading AlphaZero model...")
        model = AlphaZeroCNN(OUTPUT_DIM)
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        _model = model
        print("Model loaded.")
    return _model


# ===========================
# Encoding
# ===========================

def board_to_vector(board: chess.Board) -> np.ndarray:
    planes = np.zeros((12, 8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        row = chess.square_rank(square)
        col = chess.square_file(square)

        type_idx = piece.piece_type - 1
        offset = 0 if piece.color == chess.WHITE else 6
        planes[offset + type_idx, row, col] = 1.0

    return planes.reshape(-1)


# ===========================
# Policy + Value
# ===========================

def policy_from_net(board: chess.Board, legal_moves):
    x_np = board_to_vector(board)
    x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        logits, _ = get_model()(x)
        logits = logits[0].cpu().numpy()

    # ====== ILLEGAL-MOVE MASKING ======
    mask = np.full_like(logits, -1e9)
    for mv in legal_moves:
        idx = move_to_index[mv.uci()]
        mask[idx] = logits[idx]

    mask -= mask.max()
    exp = np.exp(mask)
    probs_all = exp / np.sum(exp)

    priors = {}
    for mv in legal_moves:
        idx = move_to_index[mv.uci()]
        priors[mv] = float(probs_all[idx])

    return priors


def evaluate_position(board: chess.Board, root_player_is_white: bool) -> float:
    x_np = board_to_vector(board)
    x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        _, value = get_model()(x)
        v = float(value.item())

    if board.turn != (chess.WHITE if root_player_is_white else chess.BLACK):
        v = -v

    return max(-1.0, min(1.0, v))


# ===========================
# MCTS
# ===========================

class MCTSNode:
    def __init__(self, board: chess.Board, parent, prior: float, root_player_is_white: bool):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.root_player_is_white = root_player_is_white

    def is_expanded(self):
        return len(self.children) > 0


def select_child(node: MCTSNode, c_puct: float):
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


def expand_node(node: MCTSNode, dirichlet_noise=False):
    board = node.board
    if board.outcome() is not None:
        return

    legal_moves = list(board.legal_moves)
    priors = policy_from_net(board, legal_moves)

    # ====== APPLY DIRICHLET NOISE AT ROOT ======
    if dirichlet_noise:
        alpha = 0.3
        eps = 0.25
        noise = np.random.dirichlet([alpha] * len(legal_moves))
        for i, mv in enumerate(legal_moves):
            priors[mv] = (1 - eps) * priors[mv] + eps * noise[i]

    for mv in legal_moves:
        child_board = board.copy()
        child_board.push(mv)
        node.children[mv] = MCTSNode(
            board=child_board,
            parent=node,
            prior=priors[mv],
            root_player_is_white=node.root_player_is_white
        )


def mcts_search(root_board: chess.Board, n_simulations=100, c_puct=1.5):
    root_player_is_white = root_board.turn
    root = MCTSNode(root_board.copy(), parent=None, prior=1.0, root_player_is_white=root_player_is_white)

    expand_node(root, dirichlet_noise=True)

    for _ in range(n_simulations):
        node = root
        path = [node]

        while node.is_expanded() and node.board.outcome() is None:
            _, node = select_child(node, c_puct)
            path.append(node)

        v = evaluate_position(node.board, root_player_is_white)

        if node.board.outcome() is None:
            expand_node(node)

        sign = 1
        for n in reversed(path):
            n.N += 1
            n.W += sign * v
            n.Q = n.W / n.N
            sign = -sign

    visit_counts = {mv: child.N for mv, child in root.children.items()}
    return root, visit_counts


# ===========================
# Entry points
# ===========================

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    board = ctx.board
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves")

    start = time.time()

    _, visit_counts = mcts_search(board, n_simulations=100, c_puct=1.5)

    total_visits = sum(visit_counts.values()) + 1e-8
    move_probs = {mv: visit_counts.get(mv, 0)
                  / total_visits for mv in legal_moves}

    ctx.logProbabilities(move_probs)

    best_move = max(visit_counts.items(), key=lambda kv: kv[1])[0]

    print(
        f"MCTS chose {best_move.uci()} after {total_visits:.0f} visits in {time.time() - start:.2f}s")

    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    pass
