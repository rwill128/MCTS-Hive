import random
from functools import lru_cache

from .connect_four import ConnectFour


class MinimaxConnectFourPlayer:
    """Simple depth-limited minimax player for Connect Four."""

    def __init__(self, game: ConnectFour, perspective_player: str, depth: int = 4):
        self.game = game
        self.perspective = perspective_player
        self.depth = depth

    def _serialize(self, state):
        board = tuple(tuple(cell for cell in row) for row in state["board"])
        return board, state["current_player"]

    @lru_cache(maxsize=None)
    def _minimax(self, board_serialized, to_move, depth):
        board, _ = board_serialized
        state = {
            "board": [list(row) for row in board],
            "current_player": to_move,
        }
        outcome = self.game.getGameOutcome(state)
        if outcome == self.perspective:
            return 1.0
        if outcome == "Draw":
            return 0.0
        if outcome is not None:
            return -1.0
        if depth == 0:
            return self._heuristic_value(state)

        next_player = self.game.getOpponent(to_move)
        actions = self.game.getLegalActions(state)
        scores = []
        for action in actions:
            next_state = self.game.applyAction(state, action)
            ser = self._serialize(next_state)
            score = self._minimax(ser, next_state["current_player"], depth - 1)
            scores.append(score)
        if to_move == self.perspective:
            return max(scores)
        else:
            return min(scores)

    def _heuristic_value(self, state):
        board = state["board"]
        player = self.perspective
        opp = self.game.getOpponent(player)
        score = 0.0
        for r in range(self.game.ROWS):
            for c in range(self.game.COLS):
                for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    cells = []
                    for i in range(4):
                        rr = r + dr * i
                        cc = c + dc * i
                        if 0 <= rr < self.game.ROWS and 0 <= cc < self.game.COLS:
                            cells.append(board[rr][cc])
                        else:
                            break
                    if len(cells) != 4:
                        continue
                    if opp not in cells:
                        count = cells.count(player)
                        score += pow(10.0, count - 1)
                    elif player not in cells:
                        count = cells.count(opp)
                        score -= pow(10.0, count - 1)
        # Clamp score to [-1, 1] for safety
        if score > 0:
            return min(1.0, score / 10000.0)
        else:
            return max(-1.0, score / 10000.0)

    def search(self, state):
        actions = self.game.getLegalActions(state)
        best_score = -float("inf")
        best_actions = []
        for action in actions:
            next_state = self.game.applyAction(state, action)
            ser = self._serialize(next_state)
            score = self._minimax(ser, next_state["current_player"], self.depth - 1)
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
        return random.choice(best_actions)
