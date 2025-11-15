from .utils import chess_manager, GameContext
from chess import Move
import random
import time
import torch

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

# --- Begin My code ---
import torch
from chess_engine.encoder import encode_with_history
from collections import deque
from chess_engine.model import ChessNet
from chess_engine.move_index import move_to_index, index_to_move

history_queue = deque(maxlen=40)
model = ChessNet()
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()
# --- End My code ---

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    # --- Begin My code ---
        
    history_queue.append(ctx.board.copy())
    x = encode_with_history(ctx.board, history_queue, 40)

    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        policy_logits, _ = model(x)

    policy = torch.softmax(policy_logits[0], dim=0)

    legal_moves = list(ctx.board.generate_legal_moves())
    scored = []

    for m in legal_moves:
        idx = move_to_index(m, ctx.board)
        scored.append((policy[idx].item(), m))

    scored.sort(reverse=True)
    return scored[0][1]

    # --- End My code ---

    '''
    print("Cooking move...")
    print(ctx.board.move_stack)
    time.sleep(0.1)

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    move_weights = [random.random() for _ in legal_moves]
    total_weight = sum(move_weights)
    # Normalize so probabilities sum to 1
    move_probs = {
        move: weight / total_weight
        for move, weight in zip(legal_moves, move_weights)
    }
    ctx.logProbabilities(move_probs)

    return random.choices(legal_moves, weights=move_weights, k=1)[0]
    '''

@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    history_queue.clear()
    pass
