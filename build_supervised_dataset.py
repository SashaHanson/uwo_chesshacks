# build_supervised_dataset.py
import numpy as np
import chess
from chess_engine.encoder import encode_single_state
from chess_engine.move_index import MOVE_SPACE

def build_supervised(fens_path, moves_path, out_states="elite_states.npy", out_policies="elite_policies.npy"):
    fens = np.load(fens_path)
    move_indices = np.load(moves_path)

    X = []
    Y = []

    for fen, idx in zip(fens, move_indices):
        board = chess.Board(fen)
        planes = encode_single_state(board)
        X.append(planes)

        y = np.zeros(len(MOVE_SPACE), dtype=np.float32)
        y[idx] = 1.0
        Y.append(y)

    X = np.stack(X).astype(np.float32)
    Y = np.stack(Y).astype(np.float32)

    np.save(out_states, X)
    np.save(out_policies, Y)

    print("Saved:")
    print(f"  {out_states}  (states)")
    print(f"  {out_policies} (policies)")
