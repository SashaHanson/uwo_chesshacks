import modal

app = modal.App("chess-preprocess")

PROJECT_ROOT = "/root/chesshacks"
TRAINING_LOCAL = "training"
TRAINING_REMOTE = f"{PROJECT_ROOT}/training"

# Shared volume (same one used by training)
vol = modal.Volume.from_name("chess-training-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "numpy",
        "pandas",
        "tqdm",
        "python-chess",
    )
    .add_local_dir(
        local_path=TRAINING_LOCAL,
        remote_path=TRAINING_REMOTE,
    )
)


@app.function(image=image, timeout=1000, volumes={"/root/vol": vol})
def run_preprocessing():
    import os
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    import chess

    # --------------------------------------
    # Paths
    # --------------------------------------
    DATA_PATH = os.path.join(TRAINING_REMOTE, "data", "games.csv")

    # WRITE all outputs into the shared volume
    DATA_DIR = "/root/vol/data"
    os.makedirs(DATA_DIR, exist_ok=True)

    OUT_X = os.path.join(DATA_DIR, "X.npy")
    OUT_Y = os.path.join(DATA_DIR, "Y.npy")
    OUT_V = os.path.join(DATA_DIR, "Y_value.npy")
    OUT_MAP = os.path.join(DATA_DIR, "move_index_map.npy")

    MAX_GAMES = 20000

    print("Loading CSV:", DATA_PATH)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CSV file not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # ---- column checks ----
    if "moves" not in df.columns:
        raise KeyError("CSV must contain a 'moves' column with SAN moves.")

    # Try to find a result-like column, but don't require it
    result_col = None
    for cand in ["result", "Result", "outcome", "Outcome", "winner"]:
        if cand in df.columns:
            result_col = cand
            break

    if result_col is None:
        print(
            "WARNING: No result column found in games.csv. "
            "All value labels will be 0.0 (treated as draws)."
        )

    if MAX_GAMES is not None:
        df = df.iloc[:MAX_GAMES]

    # ======================================
    # Board encoding (must match training)
    # ======================================
    def board_to_vector(board: chess.Board) -> np.ndarray:
        """Encode board as 768-dim vector: 64 squares × 12 piece planes."""
        planes = np.zeros((64, 12), dtype=np.float32)
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None:
                continue
            t = piece.piece_type - 1  # 0..5
            offset = 0 if piece.color == chess.WHITE else 6
            planes[sq, offset + t] = 1.0
        return planes.flatten()  # (768,)

    # --------------------------------------
    # Convert game result string → scalar
    # --------------------------------------
    def game_result_to_scalar(result_str) -> float:
        """Map game result to [-1, 1]. If missing/unknown → 0.0."""
        if not isinstance(result_str, str):
            return 0.0

        s = result_str.strip()
        if s == "1-0":
            return 1.0
        if s == "0-1":
            return -1.0
        if s in ["1/2-1/2", "1/2 - 1/2", "0.5-0.5", "draw", "Draw"]:
            return 0.0

        # unknown / anything else → treat as draw
        return 0.0

    # final value from POV of side to move
    def position_value(result_str, stm_white: bool) -> float:
        g = game_result_to_scalar(result_str)
        return g if stm_white else -g

    # ======================================
    # move → label index
    # ======================================
    def move_to_label(move: chess.Move, move_index_map: dict) -> int:
        uci = move.uci()
        if uci not in move_index_map:
            move_index_map[uci] = len(move_index_map)
        return move_index_map[uci]

    X_list: list[np.ndarray] = []
    Y_list: list[int] = []
    V_list: list[float] = []
    move_index_map: dict[str, int] = {}

    print(f"Processing {len(df)} games...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Games"):
        moves_str = row["moves"]
        if not isinstance(moves_str, str) or not moves_str.strip():
            continue

        # result string for this game, if we have a column, else None
        game_result = row[result_col] if result_col is not None else None

        board = chess.Board()
        for san in moves_str.split():
            try:
                move = board.parse_san(san)
            except Exception:
                # bad SAN in this game; skip remaining moves
                break

            # features BEFORE move
            X_list.append(board_to_vector(board))

            # policy label (move index)
            Y_list.append(move_to_label(move, move_index_map))

            # value label from side-to-move POV
            V_list.append(position_value(game_result, board.turn == chess.WHITE))

            # play move
            board.push(move)

    # Convert to arrays
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.int64)
    V = np.array(V_list, dtype=np.float32)

    print("Saving arrays to shared volume...")
    np.save(OUT_X, X)
    np.save(OUT_Y, Y)
    np.save(OUT_V, V)

    # save move_index_map in (uci, idx) pair format (matches training)
    move_map_array = np.array(list(move_index_map.items()), dtype=object)
    np.save(OUT_MAP, move_map_array)

    # Persist volume changes
    vol.commit()

    print("\nDONE!")
    print("Positions:", len(X))
    print("Unique moves:", len(move_index_map))
    print("X shape:", X.shape)
    print("Y (policy) shape:", Y.shape)
    print("Y_value shape:", V.shape)
    print("Saved map to:", OUT_MAP)


@app.local_entrypoint()
def main():
    run_preprocessing.remote()
