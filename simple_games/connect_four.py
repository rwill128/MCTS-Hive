import random

class ConnectFour:
    ROWS = 6
    COLS = 7

    def getInitialState(self):
        board = [[None]*self.COLS for _ in range(self.ROWS)]
        return {"board": board, "current_player": "X"}

    def copyState(self, state):
        return {"board": [row[:] for row in state["board"]],
                "current_player": state["current_player"]}

    def getCurrentPlayer(self, state):
        return state["current_player"]

    def getOpponent(self, player):
        return "O" if player == "X" else "X"

    def getLegalActions(self, state):
        board = state["board"]
        actions = []
        for c in range(self.COLS):
            if board[self.ROWS-1][c] is None:
                actions.append(c)
        return actions

    def applyAction(self, state, action):
        col = action
        new_state = self.copyState(state)
        board = new_state["board"]
        for r in range(self.ROWS):
            if board[r][col] is None:
                board[r][col] = state["current_player"]
                break
        new_state["current_player"] = self.getOpponent(state["current_player"])
        return new_state

    def _check_line(self, board, r, c, dr, dc):
        player = board[r][c]
        if player is None:
            return None
        for i in range(1,4):
            rr = r + dr*i
            cc = c + dc*i
            if not (0 <= rr < self.ROWS and 0 <= cc < self.COLS):
                return None
            if board[rr][cc] != player:
                return None
        return player

    def getGameOutcome(self, state):
        board = state["board"]
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if board[r][c] is None:
                    continue
                for dr, dc in ((1,0), (0,1), (1,1), (1,-1)):
                    winner = self._check_line(board, r, c, dr, dc)
                    if winner:
                        return winner
        if all(board[self.ROWS-1][c] is not None for c in range(self.COLS)):
            return "Draw"
        return None

    def isTerminal(self, state):
        return self.getGameOutcome(state) is not None

    def simulateRandomPlayout(self, state, perspectivePlayer, max_depth=42, eval_func=None, weights=None):
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
