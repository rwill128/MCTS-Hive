import random
from asyncio import sleep

from HivePocket.DrawGame import drawStatePygame


class HiveGame:
    """
    A *simplified* Hive implementation for demonstration with a generic MCTS framework.

    Major simplifications/assumptions:
      - Board is tracked using axial coordinates (q, r).
      - Each coordinate can have a stack of pieces (top of stack is last in list).
      - Movement rules are partial and simplified (no strict 'sliding' constraints, no climbing).
      - We do check if each Queen is fully surrounded for a terminal state.
      - We treat "no legal moves" as an immediate loss for the current player to avoid MCTS contradictions.
    """

    def __init__(self):
        # Standard set for each player (no expansions):
        # 1x Queen, 2x Spider, 2x Beetle, 3x Grasshopper, 3x Soldier Ant
        self.INITIAL_PIECES = {
            "Queen": 1,
            "Spider": 2,
            "Beetle": 2,
            "Grasshopper": 3,
            "Ant": 3
        }

        # Axial coordinate neighbors (q, r)
        self.DIRECTIONS = [
            (1, 0),   # East
            (-1, 0),  # West
            (0, 1),   # Southeast
            (0, -1),  # Northwest
            (1, -1),  # Northeast
            (-1, 1)   # Southwest
        ]

    # ---------------------------------------------------------
    # 1. Initialization & Game State
    # ---------------------------------------------------------
    def getInitialState(self):
        """
        Returns the initial state of the simplified Hive game.
        """
        state = {
            "board": {},
            "current_player": "Player1",
            "pieces_in_hand": {
                "Player1": self.INITIAL_PIECES.copy(),
                "Player2": self.INITIAL_PIECES.copy()
            },
            "move_number": 0
        }
        return state

    def copyState(self, state):
        """
        Returns a deep copy of the state dictionary.
        """
        new_board = {}
        for coord, stack in state["board"].items():
            # Shallow copy of the stack list is enough for this simplified approach
            new_board[coord] = stack.copy()

        new_pieces_in_hand = {
            "Player1": state["pieces_in_hand"]["Player1"].copy(),
            "Player2": state["pieces_in_hand"]["Player2"].copy()
        }

        new_state = {
            "board": new_board,
            "current_player": state["current_player"],
            "pieces_in_hand": new_pieces_in_hand,
            "move_number": state["move_number"]
        }
        return new_state

    # ---------------------------------------------------------
    # 2. Core Hive logic
    # ---------------------------------------------------------
    def getOpponent(self, player):
        return "Player2" if player == "Player1" else "Player1"

    def getAdjacentCells(self, q, r):
        for dq, dr in self.DIRECTIONS:
            yield (q + dq, r + dr)

    def isQueenSurrounded(self, state, player):
        """
        Checks if player's Queen is on the board and fully surrounded by pieces.
        """
        board = state["board"]
        # Find the queen
        for (q, r), stack in board.items():
            for piece in stack:
                if piece == (player, "Queen"):
                    # Check neighbors for occupancy
                    neighbors_occupied = 0
                    for (nq, nr) in self.getAdjacentCells(q, r):
                        if (nq, nr) in board and len(board[(nq, nr)]) > 0:
                            neighbors_occupied += 1
                        else:
                            return False
                    return True
        return False

    # ---------------------------------------------------------
    # 3. Action generation
    # ---------------------------------------------------------
    def getLegalActions(self, state):
        """
        Returns all legal actions for the current player: placements + moves
        """
        return self.placePieceActions(state) + self.movePieceActions(state)

    def placePieceActions(self, state):
        """
        PLACE actions of the form ("PLACE", insectType, (q, r))
        """
        player = state["current_player"]
        board = state["board"]
        pieces_in_hand = state["pieces_in_hand"][player]
        actions = []

        # If the player has no pieces left to place, skip.
        if all(count == 0 for count in pieces_in_hand.values()):
            return actions

        # If the board is empty, place anywhere — fix at (0,0) for simplicity.
        if len(board) == 0:
            for insectType, count in pieces_in_hand.items():
                if count > 0:
                    actions.append(("PLACE", insectType, (0, 0)))
            return actions

        if len(board) == 1:
            for insectType, count in pieces_in_hand.items():
                if count > 0:
                    actions.append(("PLACE", insectType, (0, 1)))
            return actions

        # Otherwise, must place adjacent to at least one friendly piece
        # (Simplified check: any stack that contains at least one piece of current_player.)
        friendly_cells = set()
        for (q, r), stack in board.items():
            if any(p[0] == player for p in stack):
                friendly_cells.add((q, r))

        potential_spots = set()
        for (q, r) in friendly_cells:
            for (nq, nr) in self.getAdjacentCells(q, r):
                # If that cell is empty, we can place there
                if (nq, nr) not in board or len(board[(nq, nr)]) == 0:
                    potential_spots.add((nq, nr))

        # For each piece in hand, for each potential spot
        for insectType, count in pieces_in_hand.items():
            if count > 0:
                for (tq, tr) in potential_spots:
                    actions.append(("PLACE", insectType, (tq, tr)))

        return actions

    def movePieceActions(self, state):
        """
        MOVE actions of the form ("MOVE", (from_q, from_r), (to_q, to_r)).
        Simplified movement for demonstration.
        """
        board = state["board"]
        player = state["current_player"]
        actions = []

        # Gather all top pieces belonging to the current player
        player_cells = []
        for (q, r), stack in board.items():
            if stack and stack[-1][0] == player:
                insectType = stack[-1][1]
                player_cells.append((q, r, insectType))

        for (q, r, insectType) in player_cells:
            if insectType in ["Queen", "Beetle"]:
                # Move to adjacent empty cell
                for (nq, nr) in self.getAdjacentCells(q, r):
                    if (nq, nr) not in board or len(board[(nq, nr)]) == 0:
                        actions.append(("MOVE", (q, r), (nq, nr)))

            elif insectType == "Spider":
                # Exactly 3 steps along empty neighbors
                reachable_3 = self._bfsExactSteps(board, (q, r), steps=3)
                for (dest_q, dest_r) in reachable_3:
                    if (dest_q, dest_r) != (q, r):
                        actions.append(("MOVE", (q, r), (dest_q, dest_r)))

            elif insectType == "Ant":
                # Up to 8 steps along empty neighbors (demo limit)
                reachable_8 = self._bfsUpToSteps(board, (q, r), max_steps=8)
                for (dest_q, dest_r) in reachable_8:
                    if (dest_q, dest_r) != (q, r):
                        actions.append(("MOVE", (q, r), (dest_q, dest_r)))

            elif insectType == "Grasshopper":
                # Jump over exactly one occupied cell if next cell is empty
                for (dq, dr) in self.DIRECTIONS:
                    over_cell = (q + dq, r + dr)
                    landing_cell = (q + 2*dq, r + 2*dr)
                    if (over_cell in board and len(board[over_cell]) > 0
                            and ((landing_cell not in board) or len(board[landing_cell]) == 0)):
                        actions.append(("MOVE", (q, r), landing_cell))

        return actions

    def getOtherPlayer(self, currentPlayer):
        """
        Returns the current player to move ('Player1' or 'Player2').
        """
        if currentPlayer == "Player1":
            return "Player2"
        elif currentPlayer == "Player2":
            return "Player1"

    def _bfsExactSteps(self, board, start, steps=3):
        """
        Return all reachable cells in exactly `steps` steps,
        traveling on empty adjacent cells only.
        """
        visited = set()
        q0, r0 = start
        frontier = [(q0, r0, 0)]
        results = set()

        while frontier:
            q, r, dist = frontier.pop()
            if dist == steps:
                results.add((q, r))
                continue
            if dist > steps:
                continue

            for (nq, nr) in self.getAdjacentCells(q, r):
                if (nq, nr) not in board or len(board[(nq, nr)]) == 0:
                    if (nq, nr, dist+1) not in visited:
                        visited.add((nq, nr, dist+1))
                        frontier.append((nq, nr, dist+1))
        return results

    def _bfsUpToSteps(self, board, start, max_steps=8):
        """
        Return all reachable cells in up to `max_steps` steps
        traveling on empty adjacent cells only.
        """
        visited = set()
        q0, r0 = start
        frontier = [(q0, r0, 0)]
        results = {(q0, r0)}

        while frontier:
            q, r, dist = frontier.pop()
            if dist >= max_steps:
                continue

            for (nq, nr) in self.getAdjacentCells(q, r):
                if (nq, nr) not in board or len(board[(nq, nr)]) == 0:
                    if (nq, nr) not in results:
                        results.add((nq, nr))
                    if (nq, nr, dist+1) not in visited:
                        visited.add((nq, nr, dist+1))
                        frontier.append((nq, nr, dist+1))

        return results

    # ---------------------------------------------------------
    # 4. MCTS-required interface
    # ---------------------------------------------------------
    def isTerminal(self, state):
        """
        We say the game ends if either queen is surrounded,
        OR if the current player has no moves at all (treated as a loss).
        """
        if self.isQueenSurrounded(state, "Player1"):
            return True
        if self.isQueenSurrounded(state, "Player2"):
            return True

        # If no moves are possible, we treat it as terminal (loss for current player)
        if not self.getLegalActions(state):
            return True

        return False

    def applyAction(self, state, action):
        """
        Applies the given action ("PLACE", insectType, (q, r)) or
        ("MOVE", (from_q, from_r), (to_q, to_r)).
        Returns the next state.
        """
        new_state = self.copyState(state)
        board = new_state["board"]
        player = new_state["current_player"]

        if action[0] == "PLACE":
            # ("PLACE", insectType, (q, r))
            _, insectType, (q, r) = action
            new_state["pieces_in_hand"][player][insectType] -= 1
            if (q, r) not in board:
                board[(q, r)] = []
            board[(q, r)].append((player, insectType))

        elif action[0] == "MOVE":
            # ("MOVE", (from_q, from_r), (to_q, to_r))
            _, (fq, fr), (tq, tr) = action
            piece = board[(fq, fr)].pop()
            if len(board[(fq, fr)]) == 0:
                del board[(fq, fr)]
            if (tq, tr) not in board:
                board[(tq, tr)] = []
            board[(tq, tr)].append(piece)

        # Switch to next player
        new_state["current_player"] = self.getOpponent(player)
        new_state["move_number"] += 1
        return new_state

    def getGameOutcome(self, state):
        """
        Returns:
         - "Player1" if Player1 wins,
         - "Player2" if Player2 wins,
         - "Draw" if both queens are simultaneously surrounded (rare),
         - or None if not terminal yet.

        We also handle the "no moves" scenario by awarding victory to the other player.
        """
        p1_surrounded = self.isQueenSurrounded(state, "Player1")
        p2_surrounded = self.isQueenSurrounded(state, "Player2")

        if p1_surrounded and p2_surrounded:
            # both queens simultaneously surrounded => draw
            return "Draw"
        elif p1_surrounded:
            return "Player2"
        elif p2_surrounded:
            return "Player1"

        # If not strictly queen-surrounded terminal, maybe we have no moves for the current player
        if not self.getLegalActions(state):
            # current player loses, so the other player is the winner
            current = state["current_player"]
            opponent = self.getOpponent(current)
            return opponent

        # Not terminal
        return None

    def getCurrentPlayer(self, state):
        return state["current_player"]

    def simulateRandomPlayout(self, state):
        """
        Playout from 'state' by choosing random legal actions until terminal.
        """
        temp_state = self.copyState(state)
        while not self.isTerminal(temp_state):
            self.printState(temp_state)
            sleep(1)
            legal = self.getLegalActions(temp_state)
            if not legal:
                # No moves => break or treat as terminal
                break
            action = random.choice(legal)
            temp_state = self.applyAction(temp_state, action)
            drawStatePygame(temp_state)
        return temp_state

    # ---------------------------------------------------------
    # 5. Print / Debug
    # ---------------------------------------------------------
    def printState(self, state):
        """
        Quick debug print of the current board.
        """
        board = state["board"]
        current_player = state["current_player"]
        move_number = state["move_number"]

        print(f"Move#: {move_number}, Current Player: {current_player}")
        if not board:
            print("Board is empty.")
        else:
            for (q, r), stack in sorted(board.items()):
                print(f"Cell ({q},{r}): {stack}")
        print("Pieces in hand:")
        for pl, counts in state["pieces_in_hand"].items():
            print(f"  {pl}: {counts}")
        print("-" * 50)
