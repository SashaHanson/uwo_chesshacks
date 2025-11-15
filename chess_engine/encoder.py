# encoder.py
import chess
import numpy as np

# 18 planes per state: 12 pieces + 1 side + 4 castling + 1 halfmove
def encode_single_state(board: chess.Board):
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    # --- 1. Piece bitboards (12 planes)
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    for sq, piece in board.piece_map().items():
        color_offset = 0 if piece.color == chess.WHITE else 6
        idx = piece_map[piece.piece_type] + color_offset
        planes[idx, sq // 8, sq % 8] = 1.0

    # --- 2. Side to move
    planes[12] = 1.0 if board.turn == chess.WHITE else 0.0

    # --- 3. Castling rights
    planes[13] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # --- 4. Halfmove clock (50-move rule)
    planes[17] = board.halfmove_clock / 50.0

    return planes


# For full 40-history stacking
def encode_with_history(board, history_queue, history_len=40):
    states = []

    # Current board first
    states.append(encode_single_state(board))

    # Up to 40 historical states
    for past in list(history_queue)[-history_len:]:
        states.append(encode_single_state(past))

    # Pad if not enough history
    while len(states) < history_len + 1:
        states.append(np.zeros((18, 8, 8), dtype=np.float32))

    return np.concatenate(states, axis=0)
