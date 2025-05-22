import random

class ConnectFour:
    ROWS = 6
    COLS = 7

    def __init__(self, use_precomputed_lines: bool = True):
        """Create a Connect Four game instance.

        Parameters
        ----------
        use_precomputed_lines : bool, optional
            If True (default), evaluation uses a precomputed list of all
            possible four-in-a-row lines on the board.  This avoids rebuilding
            the list on every call and slightly improves performance.  Setting
            the flag to ``False`` restores the original behaviour.
        """

        self.use_precomputed_lines = use_precomputed_lines

        # Precompute every possible line of four cells once.  The old
        # implementation rebuilt these nested loops inside
        # ``evaluateState``.  Building the list once upfront lets that method
        # iterate directly over the coordinates.
        lines = []
        for r in range(self.ROWS):
            for c in range(self.COLS):
                for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    cells = []
                    for i in range(4):
                        rr = r + dr * i
                        cc = c + dc * i
                        if 0 <= rr < self.ROWS and 0 <= cc < self.COLS:
                            cells.append((rr, cc))
                        else:
                            break
                    if len(cells) == 4:
                        lines.append(cells)
        self._precomputed_lines = lines

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

    def get_action_size(self):
        return self.COLS

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
        """Evaluate ``state`` from the perspective of ``perspectivePlayer``.

        The previous implementation returned ``1``, ``0`` or ``-1`` based only
        on terminal outcomes.  This version mirrors the heuristic used by the
        :class:`MinimaxConnectFourPlayer` so non‑terminal positions also receive
        meaningful scores.
        """

        outcome = self.getGameOutcome(state)
        if outcome == perspectivePlayer:
            return 1.0
        if outcome == "Draw":
            return 0.0
        if outcome is not None:
            return -1.0

        # Non-terminal position – compute a heuristic based on potential lines
        board = state["board"]
        player = perspectivePlayer
        opp = self.getOpponent(player)

        # Pre-compute playability: the first empty row in each column
        next_empty_row = [self.ROWS] * self.COLS  # default: column is full
        for c in range(self.COLS):
            for r in range(self.ROWS):
                if board[r][c] is None:
                    next_empty_row[c] = r
                    break

        def line_playable(line):
            """Return True if at least one empty square in *line* is currently playable."""
            for rr, cc in line:
                if board[rr][cc] is None and next_empty_row[cc] == rr:
                    return True
            return False

        DOUBLE_THREAT_BONUS = 5000.0  # extra score for each double-threat cell

        score = 0.0
        if self.use_precomputed_lines:
            for line in self._precomputed_lines:
                cells = [board[r][c] for r, c in line]
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

            # --- Double-threat detection -----------------------------------
            for c in range(self.COLS):
                r = next_empty_row[c]
                if r >= self.ROWS:
                    continue  # column full
                threat_lines = 0
                for line in self._precomputed_lines:
                    if (r, c) not in line:
                        continue
                    cells = [board[x][y] for x, y in line]
                    if opp in cells:
                        continue
                    if cells.count(player) == 3:
                        threat_lines += 1
                if threat_lines >= 2:
                    score += DOUBLE_THREAT_BONUS
        else:
            for r in range(self.ROWS):
                for c in range(self.COLS):
                    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                        line = []
                        for i in range(4):
                            rr = r + dr * i
                            cc = c + dc * i
                            if 0 <= rr < self.ROWS and 0 <= cc < self.COLS:
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

            # Double-threat detection in on-the-fly mode
            for c in range(self.COLS):
                r = next_empty_row[c]
                if r >= self.ROWS:
                    continue
                threat_lines = 0
                for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    line = []
                    for i in range(-3, 1):  # include (r,c) as one of the 4 positions
                        rr = r + dr * i
                        cc = c + dc * i
                        segment = []
                        for j in range(4):
                            rrr = rr + dr * j
                            ccc = cc + dc * j
                            if 0 <= rrr < self.ROWS and 0 <= ccc < self.COLS:
                                segment.append((rrr, ccc))
                        if len(segment) != 4 or (r, c) not in segment:
                            continue
                        cells = [board[x][y] for x, y in segment]
                        if opp in cells:
                            continue
                        if cells.count(player) == 3:
                            threat_lines += 1
                if threat_lines >= 2:
                    score += DOUBLE_THREAT_BONUS

        # Clamp heuristic score to [-1, 1] for consistency with terminal values
        if score > 0:
            return min(1.0, score / 10000.0)
        else:
            return max(-1.0, score / 10000.0)
