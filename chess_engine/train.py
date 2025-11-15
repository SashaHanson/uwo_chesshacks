from encoder import encode_with_history
from chess.pgn import read_game
import chess

def load_pgn_and_encode(pgn_path, history_len=40):
    pgn = open(pgn_path)
    game = chess.pgn.read_game(pgn)
    board = game.board()

    history = deque(maxlen=history_len)
    history.append(board.copy())

    for move in game.mainline_moves():
        board.push(move)
        history.append(board.copy())

    encoded = encode_with_history(board, history, history_len)
    return encoded