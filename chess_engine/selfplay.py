# selfplay.py
import numpy as np
import chess
from collections import deque

from mcts import MCTS, MCTSNode
from encoder import encode_with_history
from move_index import move_to_index, index_to_move

def play_selfplay_game(model, simulations=400, device="cpu"):
    board = chess.Board()
    history = deque(maxlen=40)
    samples = []

    mcts_engine = MCTS(model, simulations=simulations, device=device)

    while not board.is_game_over():
        history.append(board.copy())

        encode_fn = lambda b: encode_with_history(b, history, 40)

        # Run MCTS
        root = MCTSNode(board.copy())
        mcts_engine.search(root, encode_fn)

        # Build π distribution
        pi = np.zeros(4672, dtype=np.float32)
        for idx, child in root.children.items():
            pi[idx] = child.N
        pi /= pi.sum()

        samples.append((encode_fn(board), pi))

        # Choose move stochastically
        move_idx = np.random.choice(np.arange(4672), p=pi)
        move = index_to_move(move_idx, board)
        if move not in board.legal_moves:
            move = next(iter(board.legal_moves))

        board.push(move)

    result = board.result()
    z = 1 if result == "1-0" else -1 if result == "0-1" else 0

    final = []
    player = 1
    for s, p in samples:
        final.append((s, p, z * player))
        player = -player

    return final
