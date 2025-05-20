import random
from typing import List, Tuple


class TicTacToe:
    """Minimal TicTacToe implementation with optional precomputation."""

    def __init__(self, use_precomputed_lines: bool = True):
        self.use_precomputed_lines = use_precomputed_lines
        if self.use_precomputed_lines:
            lines: List[List[Tuple[int, int]]] = []
            for r in range(3):
                lines.append([(r, c) for c in range(3)])
            for c in range(3):
                lines.append([(r, c) for r in range(3)])
            lines.append([(i, i) for i in range(3)])
            lines.append([(i, 2 - i) for i in range(3)])
            self._precomputed_lines = lines
    def getInitialState(self):
        board = [[None]*3 for _ in range(3)]
        return {"board": board, "current_player": "X"}

    def copyState(self, state):
        return {"board": [row[:] for row in state["board"]],
                "current_player": state["current_player"]}

    def getCurrentPlayer(self, state):
        return state["current_player"]

    def getOpponent(self, player):
        return "O" if player == "X" else "X"

    def getLegalActions(self, state):
        actions = []
        for r in range(3):
            for c in range(3):
                if state["board"][r][c] is None:
                    actions.append((r, c))
        return actions

    def applyAction(self, state, action):
        r, c = action
        new_state = self.copyState(state)
        assert new_state["board"][r][c] is None
        new_state["board"][r][c] = state["current_player"]
        new_state["current_player"] = self.getOpponent(state["current_player"])
        return new_state

    def getGameOutcome(self, state):
        board = state["board"]
        if self.use_precomputed_lines:
            for line in self._precomputed_lines:
                r0, c0 = line[0]
                first = board[r0][c0]
                if first is None:
                    continue
                won = True
                for r, c in line[1:]:
                    if board[r][c] != first:
                        won = False
                        break
                if won:
                    return first
        else:
            lines = []
            for r in range(3):
                lines.append(board[r])
            for c in range(3):
                lines.append([board[r][c] for r in range(3)])
            lines.append([board[i][i] for i in range(3)])
            lines.append([board[i][2 - i] for i in range(3)])
            for line in lines:
                if line[0] is not None and all(cell == line[0] for cell in line):
                    return line[0]
        if all(cell is not None for row in board for cell in row):
            return "Draw"
        return None

    def isTerminal(self, state):
        return self.getGameOutcome(state) is not None

    def get_action_size(self) -> int:
        return 9

    def simulateRandomPlayout(self, state, perspectivePlayer, max_depth=9, eval_func=None, weights=None):
        temp_state = self.copyState(state)
        depth = 0
        while not self.isTerminal(temp_state) and depth < max_depth:
            legal = self.getLegalActions(temp_state)
            if not legal:
                break
            action = random.choice(legal)
            temp_state = self.applyAction(temp_state, action)
            depth += 1
        outcome = self.getGameOutcome(temp_state)
        if outcome == perspectivePlayer:
            return 1.0
        elif outcome == "Draw" or outcome is None:
            return 0.0
        else:
            return -1.0

    def evaluateState(self, perspectivePlayer, state, weights=None):
        outcome = self.getGameOutcome(state)
        if outcome == perspectivePlayer:
            return 1.0
        elif outcome == "Draw" or outcome is None:
            return 0.0
        else:
            return -1.0
