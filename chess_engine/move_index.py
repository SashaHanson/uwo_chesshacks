# move_index.py
import chess

# 4672 universal move indices
MOVE_SPACE = []

# Generate all from->to moves
for from_sq in range(64):
    for to_sq in range(64):
        if from_sq != to_sq:
            MOVE_SPACE.append((from_sq, to_sq, None))

# Promotions
for from_sq in range(64):
    rank = from_sq // 8
    if rank == 6:  # white pawn promotions
        for to_sq in range(from_sq + 8, from_sq + 8 + 1):
            for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                MOVE_SPACE.append((from_sq, to_sq, promo))
    if rank == 1:  # black pawn promotions
        for to_sq in range(from_sq - 8, from_sq - 8 + 1):
            for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                MOVE_SPACE.append((from_sq, to_sq, promo))

MOVE_INDEX = {mv: i for i, mv in enumerate(MOVE_SPACE)}


def move_to_index(move: chess.Move, board: chess.Board):
    promo = move.promotion if move.promotion else None
    key = (move.from_square, move.to_square, promo)
    return MOVE_INDEX.get(key)


def index_to_move(idx: int, board: chess.Board):
    from_sq, to_sq, promo = MOVE_SPACE[idx]
    return chess.Move(from_sq, to_sq, promo)
