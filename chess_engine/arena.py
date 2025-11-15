# arena.py
import chess
from mcts import MCTS, MCTSNode
from encoder import encode_single_state
from move_index import index_to_move

def play_single_game(modelA, modelB, sims=200, device="cpu"):
    board = chess.Board()
    players = [modelA, modelB]
    turn = 0

    while not board.is_game_over():
        model = players[turn]

        encode_fn = lambda b: encode_single_state(b)

        root = MCTSNode(board.copy())
        mcts = MCTS(model, simulations=sims, device=device)
        mcts.search(root, encode_fn)

        best_idx = max(root.children.items(), key=lambda x: x[1].N)[0]
        board.push(index_to_move(best_idx, board))

        turn ^= 1

    return board.result()


def arena(modelA, modelB, games=20, sims=200):
    score = 0
    for g in range(games):
        r = play_single_game(modelA, modelB, sims)
        if r == "1-0": score += 1
        elif r == "0-1": score -= 1
    return score
