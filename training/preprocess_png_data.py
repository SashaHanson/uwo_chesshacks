import modal

app = modal.App("chess-preprocess-hf")

# Shared volume with training data
vol = modal.Volume.from_name("chess-training-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "numpy",
        "tqdm",
        "python-chess",
        "datasets",
    )
)

@app.function(image=image, timeout=86400, volumes={"/root/vol": vol})
def run_preprocessing():
    import os
    import numpy as np
    import chess
    from tqdm import tqdm
    from datasets import load_dataset

    # ---------------------------------------
    # Output directory inside Modal volume
    # ---------------------------------------
    DATA_DIR = "/root/vol/data"
    os.makedirs(DATA_DIR, exist_ok=True)

    OUT_X = os.path.join(DATA_DIR, "X.npy")
    OUT_Y = os.path.join(DATA_DIR, "Y.npy")
    OUT_V = os.path.join(DATA_DIR, "Y_value.npy")
    OUT_MAP = os.path.join(DATA_DIR, "move_index_map.npy")

    # ---------------------------------------
    # Max positions to process
    # ---------------------------------------
    MAX_POSITIONS = 15_000_000  # ≈1 million

    # ---------------------------------------
    # Board → 768 float vector
    # ---------------------------------------
    def board_to_vector(board: chess.Board) -> np.ndarray:
        planes = np.zeros((12, 8, 8), dtype=np.float32)

        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                row = chess.square_rank(sq)
                col = chess.square_file(sq)
                t = piece.piece_type - 1
                offset = 0 if piece.color == chess.WHITE else 6
                planes[offset + t, row, col] = 1.0

        return planes.reshape(-1)

    # ---------------------------------------
    # Game result → scalar
    # ---------------------------------------
    def result_to_value(result_string, stm_white: bool) -> float:
        if result_string == "1-0":
            return 1.0 if stm_white else -1.0
        elif result_string == "0-1":
            return -1.0 if stm_white else 1.0
        return 0.0  # draw or unknown

    # ---------------------------------------
    # Move → label
    # ---------------------------------------
    move_index_map = {}

    def move_to_label(move: chess.Move):
        u = move.uci()
        if u not in move_index_map:
            move_index_map[u] = len(move_index_map)
        return move_index_map[u]

    # ---------------------------------------
    # Buffers for data
    # ---------------------------------------
    X_list = []
    Y_list = []
    V_list = []

    # ---------------------------------------
    # LOAD DATASET
    # ---------------------------------------
    print("Loading HF dataset: angeluriot/chess_games")
    dataset = load_dataset("angeluriot/chess_games", split="train")

    print(f"Dataset has {len(dataset)} games")

    # detect moves + result columns
    has_moves_uci = "moves_uci" in dataset.column_names
    has_moves_san = "moves_san" in dataset.column_names

    result_col = "result" if "result" in dataset.column_names else None
    if result_col is None:
        print("⚠ WARNING: dataset has NO 'result' column — all games treated as draws.")

    total_positions = 0

    # ---------------------------------------
    # PROCESS GAMES
    # ---------------------------------------
    for game in tqdm(dataset, desc="Games"):
        if total_positions >= MAX_POSITIONS:
            break

        board = chess.Board()

        # Choose fastest representation
        if has_moves_uci:
            moves = game["moves_uci"]
            use_uci = True
        else:
            moves = game["moves_san"]
            use_uci = False

        # Read result safely
        result = game[result_col] if result_col else "1/2-1/2"

        for mv_str in moves:
            if total_positions >= MAX_POSITIONS:
                break

            try:
                move = (
                    chess.Move.from_uci(mv_str)
                    if use_uci
                    else board.parse_san(mv_str)
                )
            except:
                # skip corrupted games
                break

            # Encode board
            X_list.append(board_to_vector(board))

            # Move label
            Y_list.append(move_to_label(move))

            # Game value
            V_list.append(result_to_value(result, board.turn == chess.WHITE))

            total_positions += 1
            board.push(move)

    # ---------------------------------------
    # SAVE OUTPUT
    # ---------------------------------------
    print(f"Collected {total_positions} positions. Saving...")

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.int64)
    V = np.array(V_list, dtype=np.float32)

    np.save(OUT_X, X)
    np.save(OUT_Y, Y)
    np.save(OUT_V, V)
    np.save(OUT_MAP, np.array(list(move_index_map.items()), dtype=object))

    # commit to modal volume
    vol.commit()

    print("DONE.")
    print("Positions:", len(X))
    print("Unique moves:", len(move_index_map))
    print("Saved to:", DATA_DIR)


@app.local_entrypoint()
def main():
    run_preprocessing.remote()
