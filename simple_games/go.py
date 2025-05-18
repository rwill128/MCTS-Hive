import random
from typing import List, Tuple, Optional


class Go:
    """A very small Go implementation with area scoring.

    This is intentionally simplified: the Ko rule is ignored and
    suicide is treated as illegal. The board defaults to 5x5 but a
    different size can be provided when constructing the game.
    """

    def __init__(self, size: int = 5):
        self.size = size

    def getInitialState(self) -> dict:
        board = [[None] * self.size for _ in range(self.size)]
        return {"board": board, "current_player": "B", "passes": 0}

    def copyState(self, state: dict) -> dict:
        return {
            "board": [row[:] for row in state["board"]],
            "current_player": state["current_player"],
            "passes": state.get("passes", 0),
        }

    def getCurrentPlayer(self, state: dict) -> str:
        return state["current_player"]

    def getOpponent(self, player: str) -> str:
        return "W" if player == "B" else "B"

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _adjacent(self, r: int, c: int) -> List[Tuple[int, int]]:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.size and 0 <= cc < self.size:
                yield rr, cc

    def _collect_group(self, board: List[List[Optional[str]]], r: int, c: int) -> Tuple[List[Tuple[int, int]], set]:
        color = board[r][c]
        group = []
        liberties = set()
        stack = [(r, c)]
        seen = set(stack)
        while stack:
            rr, cc = stack.pop()
            group.append((rr, cc))
            for nr, nc in self._adjacent(rr, cc):
                if board[nr][nc] is None:
                    liberties.add((nr, nc))
                elif board[nr][nc] == color and (nr, nc) not in seen:
                    stack.append((nr, nc))
                    seen.add((nr, nc))
        return group, liberties

    def _remove_captured(self, board: List[List[Optional[str]]], player: str) -> None:
        visited = set()
        for r in range(self.size):
            for c in range(self.size):
                if board[r][c] == player and (r, c) not in visited:
                    group, libs = self._collect_group(board, r, c)
                    visited.update(group)
                    if not libs:
                        for gr, gc in group:
                            board[gr][gc] = None

    # ------------------------------------------------------------
    # Interface methods
    # ------------------------------------------------------------
    def getLegalActions(self, state: dict) -> List[Tuple[int, int]]:
        actions = []
        for r in range(self.size):
            for c in range(self.size):
                if state["board"][r][c] is None:
                    # check for suicide
                    temp = self.copyState(state)
                    temp["board"][r][c] = state["current_player"]
                    self._remove_captured(temp["board"], self.getOpponent(state["current_player"]))
                    group, libs = self._collect_group(temp["board"], r, c)
                    if libs:  # not suicide
                        actions.append((r, c))
        actions.append("pass")
        return actions

    def applyAction(self, state: dict, action) -> dict:
        new_state = self.copyState(state)
        board = new_state["board"]
        if action == "pass":
            new_state["passes"] = state.get("passes", 0) + 1
        else:
            r, c = action
            assert board[r][c] is None
            board[r][c] = state["current_player"]
            self._remove_captured(board, self.getOpponent(state["current_player"]))
            new_state["passes"] = 0
        new_state["current_player"] = self.getOpponent(state["current_player"])
        return new_state

    def getGameOutcome(self, state: dict) -> Optional[str]:
        board = state["board"]
        if state.get("passes", 0) >= 2 or all(cell is not None for row in board for cell in row):
            black = sum(cell == "B" for row in board for cell in row)
            white = sum(cell == "W" for row in board for cell in row)
            if black > white:
                return "B"
            elif white > black:
                return "W"
            else:
                return "Draw"
        return None

    def isTerminal(self, state: dict) -> bool:
        return self.getGameOutcome(state) is not None

    def simulateRandomPlayout(self, state: dict, perspectivePlayer: str, max_depth: int = 200, eval_func=None, weights=None) -> float:
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

    def evaluateState(self, perspectivePlayer: str, state: dict, weights=None) -> float:
        outcome = self.getGameOutcome(state)
        if outcome == perspectivePlayer:
            return 1.0
        elif outcome == "Draw" or outcome is None:
            return 0.0
        else:
            return -1.0
