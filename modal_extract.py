import modal

app = modal.App("lichess-extractor")

# Build image: pip_install FIRST, add_local_dir LAST
image = (
    modal.Image.debian_slim()
    .pip_install("python-chess", "zstandard", "requests", "numpy")
    .add_local_dir(local_path=".", remote_path="/root/project")  # FIXED ORDER
)

volume = modal.Volume.from_name("chess-data", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=60 * 60 * 4
)
def run_extractor(
    month: str = "2024-11",
    min_elo: int = 2300,
    max_games: int = 50_000,
    max_positions: int = 200_000,
    move_sample_interval: int = 1,
    min_move_number: int = 6,
):
    import sys
    sys.path.append("/root/project")

    import requests
    import zstandard
    import numpy as np
    import chess.pgn
    import io
    import re
    import time

    from chess_engine.move_index import move_to_index

    HEADER_REGEX = re.compile(
        r'\[WhiteElo "(\d+)"\].*?\[BlackElo "(\d+)"\]', re.DOTALL
    )

    url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
    print(f"Downloading {url} ...")

    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    compressed = io.BytesIO()
    for chunk in resp.iter_content(1024 * 1024):
        compressed.write(chunk)
    compressed.seek(0)

    print("Starting extraction...")

    dctx = zstandard.ZstdDecompressor()
    stream_reader = dctx.stream_reader(compressed)
    text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")

    fens = []
    moves = []

    game_counter = 0
    elite_games = 0
    last_print = time.time()

    header_buf = []
    inside_header = False

    for raw_line in text_stream:
        line = raw_line.strip()

        if line.startswith("[Event "):
            inside_header = True
            header_buf = [line]
            continue

        if inside_header:
            header_buf.append(line)
            if line == "":
                inside_header = False
                game_counter += 1

                match = HEADER_REGEX.search("\n".join(header_buf))
                if not match:
                    continue

                w_elo = int(match.group(1))
                b_elo = int(match.group(2))

                if max(w_elo, b_elo) < min_elo:
                    continue

                elite_games += 1

                if time.time() - last_print >= 3:
                    print(f"Games: {game_counter:,} | Elite: {elite_games:,} | Samples: {len(fens):,}")
                    last_print = time.time()

                game = chess.pgn.read_game(io.StringIO("\n".join(header_buf) + "\n"))
                if game is None:
                    continue

                board = game.board()
                move_no = 0

                for move in game.mainline_moves():
                    move_no += 1
                    fen_before = board.fen()

                    if move_no >= min_move_number and move_no % move_sample_interval == 0:
                        idx = move_to_index(move, board)
                        if idx is not None:
                            fens.append(fen_before)
                            moves.append(idx)

                            if len(fens) >= max_positions:
                                break

                    board.push(move)

                if len(fens) >= max_positions:
                    break

    np.save(f"/data/{month}_fens.npy", np.array(fens))
    np.save(f"/data/{month}_moves.npy", np.array(moves, dtype=np.int32))

    print("\n=== Extraction finished ===")
    print(f"Elite games: {elite_games:,}")
    print(f"Positions saved: {len(fens):,}")
    print(f"Saved to /data/{month}_fens.npy and /data/{month}_moves.npy")
