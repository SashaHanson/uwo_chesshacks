import modal

app = modal.App("lichess-decompress-v2")

# Use the exact modal image structure that worked in your extractor
image = (
    modal.Image.debian_slim()
    .pip_install("requests", "zstandard")       # only deps needed for decompress
    .add_local_dir(local_path=".", remote_path="/root/project")
)

# Same volume you already used
volume = modal.Volume.from_name("chess-data", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=60 * 60 * 4,      # generous timeout
)
def decompress_month(month: str):
    import requests
    import zstandard as zstd
    import io

    url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
    print(f"Downloading: {url}")

    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    # Load compressed data into memory buffer
    compressed = io.BytesIO()
    for chunk in resp.iter_content(1024 * 1024):
        compressed.write(chunk)
    compressed.seek(0)

    print("Download complete. Starting decompression...")

    dctx = zstd.ZstdDecompressor()

    output_path = f"/data/{month}.pgn"
    with open(output_path, "w", encoding="utf-8") as f_out:
        with dctx.stream_reader(compressed) as reader:
            while True:
                chunk = reader.read(1024 * 1024)
                if not chunk:
                    break
                f_out.write(chunk.decode("utf-8", errors="ignore"))

    print(f"Decompression complete. PGN saved to {output_path}")
