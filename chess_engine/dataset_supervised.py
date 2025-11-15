# dataset_supervised.py
import chess
import chess.pgn
import numpy as np
from encoder import encode_single_state
from move_index import move_to_index

def extract_supervised_dataset(pgn_path, max_positions=200_000):
    X = []
    Y = []

    with open(pgn_path, "r") as f:
        game = chess.pgn.read_game(f)

        while game and len(X) < max_positions:
            board = game.board()

            for move in game.mainline_moves():
                idx = move_to_index(move, board)
                if idx is not None:
                    X.append(encode_single_state(board))   # shape (18, 8, 8)
                    y = np.zeros(4672, dtype=np.float32)
                    y[idx] = 1.0
                    Y.append(y)
                board.push(move)

            game = chess.pgn.read_game(f)

    X = np.stack(X)
    Y = np.stack(Y)
    return X, Y
