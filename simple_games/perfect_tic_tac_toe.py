import random
from functools import lru_cache

from .tic_tac_toe import TicTacToe


class PerfectTicTacToePlayer:
    """Minimax-based player that never loses."""

    def __init__(self, game: TicTacToe, perspective_player: str):
        self.game = game
        self.perspective = perspective_player

    def _serialize(self, state):
        # Preserve ``None`` cells exactly so that cached boards remain
        # compatible with the game's native representation.  Older cache
        # entries may still contain ``'-'`` in place of ``None`` but new
        # serializations no longer substitute that sentinel value.
        board = tuple(tuple(cell for cell in row) for row in state["board"])
        return board, state["current_player"]

    @lru_cache(maxsize=None)
    def _minimax(self, board_serialized, to_move):
        board, _ = board_serialized
        state = {
            # Convert legacy ``'-'`` placeholders back to ``None`` so that
            # caches created with older versions remain usable.
            "board": [
                [None if cell == '-' else cell for cell in row]
                for row in board
            ],
            "current_player": to_move,
        }
        outcome = self.game.getGameOutcome(state)
        if outcome == self.perspective:
            return 1
        if outcome == 'Draw':
            return 0
        if outcome is not None:
            return -1

        next_player = self.game.getOpponent(to_move)
        actions = self.game.getLegalActions(state)
        scores = []
        for action in actions:
            next_state = self.game.applyAction(state, action)
            ser = self._serialize(next_state)
            score = self._minimax(ser, next_state["current_player"])
            scores.append(score)
        if to_move == self.perspective:
            return max(scores)
        else:
            return min(scores)

    def search(self, state):
        actions = self.game.getLegalActions(state)
        best_score = -float("inf")
        best_actions = []
        for action in actions:
            next_state = self.game.applyAction(state, action)
            ser = self._serialize(next_state)
            score = self._minimax(ser, next_state["current_player"])
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
        return random.choice(best_actions)
