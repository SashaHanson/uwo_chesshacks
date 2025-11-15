# mcts.py
import math
import numpy as np
import chess
from move_index import move_to_index, index_to_move

class MCTSNode:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = None  # prior distribution over moves


class MCTS:
    def __init__(self, model, simulations=400, c_puct=1.5, device="cpu"):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.device = device

    # --------------------------------------------------------

    def search(self, root, encode_fn):
        for _ in range(self.simulations):
            node = root
            path = [node]

            # 1) SELECTION
            while node.children:
                node = self._select(node)
                path.append(node)

            # 2) EXPANSION + EVALUATION
            value = self._expand_and_evaluate(node, encode_fn)

            # 3) BACKPROP
            for n in path:
                n.N += 1
                n.W += value
                n.Q = n.W / n.N
                value = -value

    # --------------------------------------------------------

    def _select(self, node):
        best = None
        best_score = -1e9

        for idx, child in node.children.items():
            U = self.c_puct * child.P * math.sqrt(node.N + 1) / (1 + child.N)
            score = child.Q + U
            if score > best_score:
                best_score = score
                best = child
        return best

    # --------------------------------------------------------

    def _expand_and_evaluate(self, node, encode_fn):
        board = node.board

        # Terminal?
        if board.is_game_over():
            outcome = board.result()
            return 1 if outcome == "1-0" else -1 if outcome == "0-1" else 0

        # Encode state
        state = encode_fn(board)
        x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            policy_logits, value = self.model(x)
        value = value.item()

        # Convert logits → masked priors
        priors = torch.softmax(policy_logits[0], dim=0).cpu().numpy()

        # Expand children
        for move in board.legal_moves:
            idx = move_to_index(move, board)
            child_board = board.copy()
            child_board.push(move)
            child = MCTSNode(child_board, parent=node)
            node.children[idx] = child
            child.P = priors[idx]

        return value
