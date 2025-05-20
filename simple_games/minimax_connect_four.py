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
        # Pre-compute playability: the first empty row in each column
        next_empty_row = [self.game.ROWS] * self.game.COLS
        for c in range(self.game.COLS):
            for r in range(self.game.ROWS):
                if board[r][c] is None:
                    next_empty_row[c] = r
                    break

        def line_playable(line):
            for rr, cc in line:
                if board[rr][cc] is None and next_empty_row[cc] == rr:
                    return True
            return False

        DOUBLE_THREAT_BONUS = 5000.0

        for r in range(self.game.ROWS):
            for c in range(self.game.COLS):
                for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    line = []
                    for i in range(4):
                        rr = r + dr * i
                        cc = c + dc * i
                        if 0 <= rr < self.game.ROWS and 0 <= cc < self.game.COLS:
                            line.append((rr, cc))
                        else:
                            break
                    if len(line) != 4:
                        continue
                    cells = [board[x][y] for x, y in line]
                    if opp not in cells and any(cells):
                        count = cells.count(player)
                        if count == 0:
                            continue
                        score += pow(10.0, count - 1)
                    elif player not in cells and any(cells):
                        count = cells.count(opp)
                        if count == 0:
                            continue
                        score -= pow(10.0, count - 1)

        # Double-threat detection
        for c in range(self.game.COLS):
            r = next_empty_row[c]
            if r >= self.game.ROWS:
                continue
            threat_lines = 0
            for dr, dc in ((1,0), (0,1), (1,1), (1,-1)):
                line = []
                for i in range(-3,1):
                    rr = r + dr*i
                    cc = c + dc*i
                    segment = []
                    for j in range(4):
                        rrr = rr + dr*j
                        ccc = cc + dc*j
                        if 0 <= rrr < self.game.ROWS and 0 <= ccc < self.game.COLS:
                            segment.append((rrr, ccc))
                    if len(segment) != 4 or (r,c) not in segment:
                        continue
                    cells = [board[x][y] for x,y in segment]
                    if opp in cells:
                        continue
                    if cells.count(player)==3:
                        threat_lines +=1
            if threat_lines >=2:
                score += DOUBLE_THREAT_BONUS

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
            if value_callback is not None:
                value_callback(dict(scores))

        if winning_actions:
            best_actions = winning_actions
        else:
            best_score = max(scores.values())
            best_actions = [a for a, s in scores.items() if s == best_score]

        if value_callback is not None:
            value_callback(scores)

        return random.choice(best_actions)
