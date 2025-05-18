import random

class Chess:
    BOARD_SIZE = 8

    def getInitialState(self):
        board = [[None] * self.BOARD_SIZE for _ in range(self.BOARD_SIZE)]
        board[0] = ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"]
        board[1] = ["bP"] * self.BOARD_SIZE
        board[6] = ["wP"] * self.BOARD_SIZE
        board[7] = ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]
        return {"board": board, "current_player": "w"}

    def copyState(self, state):
        return {"board": [row[:] for row in state["board"]],
                "current_player": state["current_player"]}

    def getCurrentPlayer(self, state):
        return state["current_player"]

    def getOpponent(self, player):
        return "b" if player == "w" else "w"

    def in_bounds(self, r, c):
        return 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE

    def getLegalActions(self, state):
        board = state["board"]
        player = state["current_player"]
        actions = []
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                piece = board[r][c]
                if not piece or piece[0] != player:
                    continue
                p = piece[1]
                if p == "P":
                    dr = -1 if player == "w" else 1
                    if self.in_bounds(r + dr, c) and board[r + dr][c] is None:
                        actions.append(((r, c), (r + dr, c)))
                        start_row = 6 if player == "w" else 1
                        if r == start_row and board[r + 2 * dr][c] is None:
                            actions.append(((r, c), (r + 2 * dr, c)))
                    for dc in (-1, 1):
                        rr, cc = r + dr, c + dc
                        if self.in_bounds(rr, cc) and board[rr][cc] and board[rr][cc][0] != player:
                            actions.append(((r, c), (rr, cc)))
                elif p == "N":
                    for dr, dc in ((2, 1), (1, 2), (-1, 2), (-2, 1),
                                   (-2, -1), (-1, -2), (1, -2), (2, -1)):
                        rr, cc = r + dr, c + dc
                        if self.in_bounds(rr, cc) and (board[rr][cc] is None or board[rr][cc][0] != player):
                            actions.append(((r, c), (rr, cc)))
                elif p in ("B", "R", "Q"):
                    directions = []
                    if p in ("B", "Q"):
                        directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                    if p in ("R", "Q"):
                        directions += [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    for dr, dc in directions:
                        rr, cc = r + dr, c + dc
                        while self.in_bounds(rr, cc):
                            if board[rr][cc] is None:
                                actions.append(((r, c), (rr, cc)))
                            else:
                                if board[rr][cc][0] != player:
                                    actions.append(((r, c), (rr, cc)))
                                break
                            rr += dr
                            cc += dc
                elif p == "K":
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            rr, cc = r + dr, c + dc
                            if self.in_bounds(rr, cc) and (board[rr][cc] is None or board[rr][cc][0] != player):
                                actions.append(((r, c), (rr, cc)))
        return actions

    def applyAction(self, state, action):
        (r, c), (rr, cc) = action
        new_state = self.copyState(state)
        board = new_state["board"]
        piece = board[r][c]
        board[r][c] = None
        board[rr][cc] = piece
        if piece[1] == "P":
            if (piece[0] == "w" and rr == 0) or (piece[0] == "b" and rr == self.BOARD_SIZE - 1):
                board[rr][cc] = piece[0] + "Q"
        new_state["current_player"] = self.getOpponent(state["current_player"])
        return new_state

    def getGameOutcome(self, state):
        board = state["board"]
        pieces = [p for row in board for p in row if p]
        white_king = any(p == "wK" for p in pieces)
        black_king = any(p == "bK" for p in pieces)
        if not white_king:
            return "b"
        if not black_king:
            return "w"
        if not self.getLegalActions(state):
            return "Draw"
        return None

    def isTerminal(self, state):
        return self.getGameOutcome(state) is not None

    def simulateRandomPlayout(self, state, perspectivePlayer, max_depth=200, eval_func=None, weights=None):
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
