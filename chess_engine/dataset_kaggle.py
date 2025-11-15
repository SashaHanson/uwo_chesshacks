# dataset_kaggle.py
import pandas as pd
import chess.pgn
import io
import numpy as np
from chess_engine.encoder import encode_single_state
from chess_engine.move_index import move_to_index

def extract_from_kaggle(csv_path, max_positions=200_000):
    df = pd.read_csv(csv_path)

    # Kaggle dataset column names
    df["white_rating"] = pd.to_numeric(df["white_rating"], errors="coerce").fillna(0)
    df["black_rating"] = pd.to_numeric(df["black_rating"], errors="coerce").fillna(0)
    df["MaxElo"] = df[["white_rating", "black_rating"]].max(axis=1)

    # Curriculum: lowest → highest Elo first
    df = df.sort_values("MaxElo")

    X, Y = [], []

    for _, row in df.iterrows():
        moves = row["moves"]
        if not isinstance(moves, str) or len(moves) < 3:
            continue

        # Build PGN block
        pgn_text = f"""
[Event "?"]
[Site "?"]
[Date "????.??.??"]
[Round "?"]
[White "?"]
[Black "?"]
[WhiteElo "{int(row['white_rating'])}"]
[BlackElo "{int(row['black_rating'])}"]
[Result "{row.get('winner', '*')}"]

{moves}
"""

        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            continue

        board = game.board()

        for move in game.mainline_moves():
            idx = move_to_index(move, board)
            if idx is None:
                board.push(move)
                continue

            X.append(encode_single_state(board))
            y = np.zeros(4672, dtype=np.float32)
            y[idx] = 1.0
            Y.append(y)

            board.push(move)

            if len(X) >= max_positions:
                return np.stack(X), np.stack(Y)

    return np.stack(X), np.stack(Y)
