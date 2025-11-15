# modal_niko_extract.py
import modal
import io
import requests
import numpy as np

app = modal.App("niko-extractor")

IMAGE = (
    modal.Image.debian_slim()
    .pip_install(
        "python-chess",
        "requests",
        "zstandard",
        "bz2file",
        "numpy"
    )
)

@app.function(image=IMAGE, timeout=60*60*4)
def extract_from_niko(month: str):
    import chess.pgn
    import zstandard
    import bz2file
    import zipfile
    from chess_engine.move_index import move_to_index

    base = "https://database.nikonoel.fr"

    candidate_urls = [
        f"{base}/lichess_elite_{month}.pgn.zst",
        f"{base}/lichess_elite_{month}.pgn.bz2",
        f"{base}/lichess_elite_{month}.pgn.zip",
        f"{base}/lichess_elite_{month}.zip",
        f"{base}/lichess_elite_{month}",
    ]

    url = None
    content_type = ""
    for u in candidate_urls:
        head = requests.head(u)
        if head.status_code == 200:
            url = u
            content_type = head.headers.get("Content-Type", "").lower()
            break

    if url is None:
        raise FileNotFoundError(f"No archive found for month {month}")

    print(f"Found: {url}")
    print(f"Content-Type: {content_type}")

    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    data = io.BytesIO()
    for chunk in resp.iter_content(1024 * 1024):
        data.write(chunk)
    data.seek(0)

    # detect format
    if "zst" in url or "zstd" in content_type:
        fmt = "zst"
    elif "bz2" in url or "bz2" in content_type:
        fmt = "bz2"
    else:
        try:
            zipfile.ZipFile(data).testzip()
            fmt = "zip"
        except:
            fmt = "zip"

    print(f"Detected format: {fmt}")

    if fmt == "zst":
        dctx = zstandard.ZstdDecompressor()
        stream = dctx.stream_reader(data)
        text_stream = io.TextIOWrapper(stream, encoding="utf-8", errors="ignore")
    elif fmt == "bz2":
        stream = bz2file.BZ2File(data)
        text_stream = io.TextIOWrapper(stream, encoding="utf-8", errors="ignore")
    else:
        data.seek(0)
        with zipfile.ZipFile(data) as z:
            pgn_name = next((n for n in z.namelist() if n.endswith(".pgn")), None)
            if pgn_name is None:
                raise RuntimeError("ZIP has no .pgn")
            stream = z.open(pgn_name)
            text_stream = io.TextIOWrapper(stream, encoding="utf-8", errors="ignore")

    print("Parsing PGN...")

    fens = []
    moves = []
    buffer = []

    for line in text_stream:
        if line.startswith("[Event "):
            buffer = [line]
        else:
            buffer.append(line)
            if line.strip() == "":
                game = chess.pgn.read_game(io.StringIO("".join(buffer)))
                if game:
                    board = game.board()
                    for move in game.mainline_moves():
                        fen_before = board.fen()
                        idx = move_to_index(move, board)
                        if idx is not None:
                            fens.append(fen_before)
                            moves.append(idx)
                        board.push(move)

    fens = np.array(fens)
    moves = np.array(moves, dtype=np.int32)

    out_fens = f"/root/{month}_niko_fens.npy"
    out_moves = f"/root/{month}_niko_move_indices.npy"

    np.save(out_fens, fens)
    np.save(out_moves, moves)

    print(f"✓ Saved {len(fens):,} samples → {out_fens}")
    print(f"✓ Saved moves → {out_moves}")

    return out_fens, out_moves
