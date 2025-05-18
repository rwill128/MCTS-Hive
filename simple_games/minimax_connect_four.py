import random
from functools import lru_cache

from .connect_four import ConnectFour

try:
    import pygame
except ImportError:  # pragma: no cover - allow headless use
    pygame = None


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
            # Keep the OS compositor happy by consuming window events regularly.
            if pygame is not None and hasattr(pygame, "get_init") and pygame.get_init():
                pygame.event.pump()

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

    def search(self, state, value_callback=None):
        actions = self.game.getLegalActions(state)
        winning_actions = []
        scores = {}
        for action in actions:
            next_state = self.game.applyAction(state, action)
            if self.game.getGameOutcome(next_state) == self.perspective:
                winning_actions.append(action)
                score = 1.0
            else:
                ser = self._serialize(next_state)
                score = self._minimax(ser, next_state["current_player"], self.depth - 1)
            scores[action] = score

        if winning_actions:
            best_actions = winning_actions
        else:
            best_score = max(scores.values())
            best_actions = [a for a, s in scores.items() if s == best_score]

        if value_callback is not None:
            value_callback(scores)

        return random.choice(best_actions)
