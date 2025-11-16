from .utils import chess_manager, GameContext
from chess import Move
import random
import time
# Additional imports
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

# Precompute where weights live (src/weights/policy_model.pt) and which device to use.
# This way the model only loads once, and we reuse it across every move request.
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "policy_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    # Simple AlphaZero-style residual block: two conv+BN layers with a skip connection.
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return torch.relu(x + residual)


class PolicyNet(nn.Module):
    # Mirrors the training-time architecture: 12→128 stem, 4 residual blocks, then the policy head.
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(12, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 4096),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.stem(x)))


PIECE_ORDER = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]


def board_to_planes(board) -> np.ndarray:
    # Encode current board state into 12 stacked 8x8 planes (AlphaZero-style one-hot tensor).
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    for square in range(64):
        piece = board.piece_at(square)
        if not piece:
            continue
        idx = PIECE_ORDER.index(piece.symbol())
        row = 7 - (square // 8)
        col = square % 8
        planes[idx, row, col] = 1.0
    return planes


def move_to_index(move: Move) -> int:
    return move.from_square * 64 + move.to_square


def load_model() -> tuple[PolicyNet | None, bool]:
    # Lazily load the policy network weights once at startup.
    if not WEIGHTS_PATH.exists():
        print(f"[warning] weights not found at {WEIGHTS_PATH}")
        return None, False
    else:
        print(f"[model] Found weights at {WEIGHTS_PATH}")

    model = PolicyNet().to(DEVICE)
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("[warning] mismatch loading weights:", missing, unexpected)
    model.eval()
    print(f"[model] Loaded weights from {WEIGHTS_PATH}")
    return model, True


MODEL, MODEL_READY = load_model()

MCTS_SIMULATIONS = 128
C_PUCT = 1.5


class MCTSNode:
    def __init__(self, board, parent=None, move: Move | None = None, prior: float = 0.0):
        # Each node tracks the board state, tree connection, visit counts and running value sum.
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children: dict[Move, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0

    def q_value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def expanded(self) -> bool:
        return bool(self.children)


def select_child(node: MCTSNode) -> MCTSNode:
    # Pick the child with highest UCT score = Q + exploration bonus.
    best_score = float("-inf")
    best_child = None
    for child in node.children.values():
        ucb = child.q_value() + C_PUCT * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
        if ucb > best_score:
            best_score = ucb
            best_child = child
    return best_child


def evaluate_board(board) -> tuple[dict[Move, float], float]:
    # Use the neural net to produce policy priors (and crude value estimate) for expansion.
    legal_moves = list(board.generate_legal_moves())
    if not legal_moves:
        return {}, terminal_value(board)

    planes = board_to_planes(board)
    tensor = torch.from_numpy(planes).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(tensor).squeeze(0).cpu()

    move_ids = torch.tensor([move_to_index(m) for m in legal_moves], dtype=torch.long)
    scores = logits[move_ids]
    probs = torch.softmax(scores, dim=0)
    policy = {move: float(prob) for move, prob in zip(legal_moves, probs.tolist())}
    value = float(probs.max().item() * 2 - 1)
    return policy, value


def expand_node(node: MCTSNode, policy: dict[Move, float]) -> None:
    # Add child nodes for all legal moves with their policy priors.
    if node.board.is_game_over(claim_draw=True) or node.children:
        return
    for move, prior in policy.items():
        next_board = node.board.copy(stack=False)
        next_board.push(move)
        node.children[move] = MCTSNode(next_board, parent=node, move=move, prior=prior)


def terminal_value(board) -> float:
    # Value from the current player's perspective: -1 for checkmate, 0 otherwise.
    if board.is_checkmate():
        return -1.0
    return 0.0


def backup(path: list[MCTSNode], value: float) -> None:
    # Propagate evaluation results back up the tree, alternating sign for each ply.
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += value
        value = -value


def run_mcts(board) -> tuple[Move | None, dict[Move, float]]:
    # Full MCTS loop: selection, expansion, evaluation, backup for a fixed number of simulations.
    root = MCTSNode(board.copy(stack=False))
    if board.is_game_over(claim_draw=True):
        return None, {}

    for _ in range(MCTS_SIMULATIONS):
        node = root
        path = [node]

        while node.expanded() and not node.board.is_game_over(claim_draw=True):
            node = select_child(node)
            path.append(node)

        if node.board.is_game_over(claim_draw=True):
            value = terminal_value(node.board)
        else:
            policy, value = evaluate_board(node.board)
            if policy:
                expand_node(node, policy)
            else:
                value = terminal_value(node.board)

        backup(path, value)

    visit_total = sum(child.visit_count for child in root.children.values())
    if visit_total == 0:
        return None, {}

    move_probs = {
        child.move: child.visit_count / visit_total
        for child in root.children.values()
        if child.move is not None
    }
    best_child = max(root.children.values(), key=lambda c: c.visit_count)
    return best_child.move, move_probs

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Cooking move...")
    print(ctx.board.move_stack)
    time.sleep(0.1)

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    if MODEL_READY and MODEL is not None:
        # Primary inference path: rely on the neural net + MCTS search
        selected_move, search_probs = run_mcts(ctx.board)
        if selected_move and search_probs:
            ctx.logProbabilities(search_probs)
            return selected_move
        # If MCTS somehow fails to return a move, fall back to pure policy evaluation.
        planes = board_to_planes(ctx.board)
        tensor = torch.from_numpy(planes).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = MODEL(tensor).squeeze(0).cpu()

        move_ids = torch.tensor([move_to_index(m) for m in legal_moves])
        legal_logits = logits[move_ids]
        probs_tensor = torch.softmax(legal_logits, dim=0)
        move_probs = {
            move: float(prob) for move, prob in zip(legal_moves, probs_tensor.tolist())
        }
        ctx.logProbabilities(move_probs)
        best_idx = int(torch.argmax(probs_tensor))
        return legal_moves[best_idx]

    move_weights = [random.random() for _ in legal_moves]
    total_weight = sum(move_weights)
    # Normalize so probabilities sum to 1
    move_probs = {
        move: weight / total_weight
        for move, weight in zip(legal_moves, move_weights)
    }
    ctx.logProbabilities(move_probs)

    return random.choices(legal_moves, weights=move_weights, k=1)[0]


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
