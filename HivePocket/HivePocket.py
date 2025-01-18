import random

class HiveGame:
    """
    A *simplified* Hive implementation for demonstration with a generic MCTS framework.

    Major simplifications/assumptions:
      - Board is tracked using axial coordinates (q, r).
      - Each coordinate can have a stack of pieces (top of stack is last in list).
      - Movement rules are partial: we check adjacency for slides, but do not fully implement
        the "sliding around corners" constraint or climbing on top for Beetles.
      - We do check if each Queen is fully surrounded (terminal check).
      - Queen placement deadlines and the "one hive" rule are only partially enforced.
      - For a real Hive game, you may need a more robust geometry/movement code and
        additional checks (e.g., no splitting the hive, forced queen placement by 4th move).
    """

    # ---------------------------------------------------------
    # 1. Initialization & Game State
    # ---------------------------------------------------------
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

        # For adjacency in an axial coordinate system (q, r):
        self.DIRECTIONS = [
            (1, 0),  # East
            (-1, 0), # West
            (0, 1),  # Southeast
            (0, -1), # Northwest
            (1, -1), # Northeast
            (-1, 1)  # Southwest
        ]

    def getInitialState(self):
        """
        Returns the initial state of the simplified Hive game.
        The state dictionary will have:
          - 'board': dictionary mapping (q, r) -> list of pieces [ (player, insectType), ... ]
          - 'current_player': 'Player1' or 'Player2'
          - 'pieces_in_hand': {
                'Player1': {'Queen':1, 'Spider':2, 'Beetle':2, 'Grasshopper':3, 'Ant':3},
                'Player2': {'Queen':1, 'Spider':2, 'Beetle':2, 'Grasshopper':3, 'Ant':3}
            }
          - 'move_number': integer counting total moves so far
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

    # ---------------------------------------------------------
    # 2. Utility / Helper methods
    # ---------------------------------------------------------
    def getOpponent(self, player):
        return "Player2" if player == "Player1" else "Player1"

    def getAdjacentCells(self, q, r):
        """
        Returns the 6 neighboring cells in axial coords.
        """
        for dq, dr in self.DIRECTIONS:
            yield (q + dq, r + dr)

    def copyState(self, state):
        """
        Returns a deep-ish copy of the state.
        For dictionary-of-dictionaries or dictionary-of-lists,
        we need to do it carefully.
        """
        new_board = {}
        for coord, stack in state["board"].items():
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

    def isQueenSurrounded(self, state, player):
        """
        Checks if player's Queen is on the board and is completely surrounded.
        Surrounded means all 6 adjacent cells are occupied by at least one piece.
        If queen not placed yet or not fully surrounded, return False.
        """
        board = state["board"]
        # Find the queen's cell (if any)
        for (q, r), stack in board.items():
            for piece in stack:
                if piece == (player, "Queen"):
                    # Check if all neighbors are occupied
                    neighbors_occupied = 0
                    for nq, nr in self.getAdjacentCells(q, r):
                        if (nq, nr) in board and len(board[(nq, nr)]) > 0:
                            neighbors_occupied += 1
                        else:
                            # Found an empty neighbor, so not surrounded
                            return False
                    # If we get here, all neighbors are occupied
                    return True
        # Queen not on the board => not "surrounded" in the typical sense.
        return False

    def placePieceActions(self, state):
        """
        Returns a list of legal "place" actions for the current player,
        i.e. placing one of the in-hand pieces adjacent to existing pieces
        (or anywhere if no pieces are on the board yet).

        Action format example:
            ("PLACE", insectType, (q, r))
        """
        actions = []
        player = state["current_player"]
        pieces_in_hand = state["pieces_in_hand"][player]

        # If no pieces left to place, no place actions
        if all(count == 0 for count in pieces_in_hand.values()):
            return actions

        board = state["board"]
        placed_any_piece = (len(board) > 0)

        # If board is empty, the only "legal" adjacency doesn't matter;
        # you can place on (0,0) to start, for instance.
        # Let's just say the first piece is placed at (0, 0) for simplicity.
        if not placed_any_piece:
            for insectType, count in pieces_in_hand.items():
                if count > 0:
                    actions.append(("PLACE", insectType, (0, 0)))
            return actions

        # If there are existing pieces, the new piece must be placed adjacent to at least
        # one friendly piece (in the real rules, you must place next to your own pieces only,
        # but let's allow adjacency to any piece from your side for simplicity).
        # We'll gather all cells on the board that are empty but adjacent to at least one
        # piece of the current player.
        potential_cells = set()
        for (q, r), stack in board.items():
            # Check if top piece (or any piece) belongs to current player
            # (Official Hive requires adjacency to your own hive,
            #  we simplify by checking at least one piece in the stack belongs to the current player.)
            if any(p[0] == player for p in stack):
                # Add all empty neighboring cells as potential placements
                for nq, nr in self.getAdjacentCells(q, r):
                    if (nq, nr) not in board or len(board[(nq, nr)]) == 0:
                        potential_cells.add((nq, nr))

        # Now create place actions for each piece in hand, for each potential cell
        for insectType, count in pieces_in_hand.items():
            if count > 0:
                for cell in potential_cells:
                    actions.append(("PLACE", insectType, cell))

        return actions

    def movePieceActions(self, state):
        """
        Returns a list of "move" actions for the current player,
        e.g. ("MOVE", (from_q, from_r), (to_q, to_r)).

        In real Hive, movement depends on the insect type, sliding constraints,
        and the "one hive" rule. We'll do a heavily simplified approach:
          - Queen: can move to any adjacent cell if it is empty.
          - Beetle: same as Queen in this simplified version.
          - Spider: can move up to 3 steps, but each step must be to an adjacent empty cell.
          - Grasshopper: can jump over exactly one occupied cell to land on the next empty cell in a straight line.
          - Ant: can move any number of steps along adjacent empty cells.

        No full check for "splitting the hive," no climbing on top for Beetle, etc.
        This is just for demonstration of generating moves for MCTS.
        """
        board = state["board"]
        player = state["current_player"]
        actions = []

        # Gather all (q, r) cells that contain at least one piece belonging to current player
        player_cells = []
        for (q, r), stack in board.items():
            if any(p[0] == player for p in stack):
                # We track each piece individually so we know which piece is on top.
                # In real Hive, only the top piece can move from a stack.
                if stack[-1][0] == player:
                    # The top piece belongs to the current player
                    top_piece = stack[-1]
                    player_cells.append((q, r, top_piece[1]))  # (q, r, insectType)

        for (q, r, insectType) in player_cells:
            if insectType == "Queen" or insectType == "Beetle":
                # Queen/Beetle can move to an adjacent empty cell
                for nq, nr in self.getAdjacentCells(q, r):
                    if (nq, nr) not in board or len(board[(nq, nr)]) == 0:
                        actions.append(("MOVE", (q, r), (nq, nr)))

            elif insectType == "Spider":
                # Spider moves exactly 3 steps.
                # We'll do a brute-force BFS of length exactly 3 along empty adjacent cells.
                spider_paths = self._bfsExactSteps(board, (q, r), steps=3)
                for (dest_q, dest_r) in spider_paths:
                    if (dest_q, dest_r) != (q, r):
                        actions.append(("MOVE", (q, r), (dest_q, dest_r)))

            elif insectType == "Ant":
                # Ant can move any number of steps along empty adjacent cells
                # We'll do BFS up to some limit (say 8 steps to avoid huge expansions).
                # Real Hive has no strict limit, but let's cap it for demonstration.
                ant_paths = self._bfsUpToSteps(board, (q, r), max_steps=8)
                for (dest_q, dest_r) in ant_paths:
                    if (dest_q, dest_r) != (q, r):
                        actions.append(("MOVE", (q, r), (dest_q, dest_r)))

            elif insectType == "Grasshopper":
                # Grasshopper in real Hive jumps any distance in a straight line
                # over at least one occupied cell, landing on the next empty cell.
                # We'll do a simplified version: jump exactly over 1 occupied cell if possible.
                for dq, dr in self.DIRECTIONS:
                    over_cell = (q + dq, r + dr)
                    landing_cell = (q + 2*dq, r + 2*dr)
                    if (over_cell in board and len(board[over_cell]) > 0 and
                            ((landing_cell not in board) or len(board[landing_cell]) == 0)):
                        actions.append(("MOVE", (q, r), landing_cell))

        return actions

    def _bfsExactSteps(self, board, start, steps=3):
        """
        Return all reachable cells in exactly `steps` steps,
        traveling only on empty adjacent hexes.
        """
        visited = set()
        q0, r0 = start
        frontier = [(q0, r0, 0)]  # (q, r, distance)
        results = set()

        while frontier:
            q, r, dist = frontier.pop()
            if dist == steps:
                results.add((q, r))
                continue
            if dist > steps:
                continue

            for (nq, nr) in self.getAdjacentCells(q, r):
                # Cell must be empty or not exist in board yet
                if (nq, nr) not in board or len(board[(nq, nr)]) == 0:
                    # We track visited with (q, r, dist) or we might skip some paths
                    # but for simplicity let's store (q, r, dist) to avoid re-visiting
                    # the same state.
                    if (nq, nr, dist+1) not in visited:
                        visited.add((nq, nr, dist+1))
                        frontier.append((nq, nr, dist+1))

        return results

    def _bfsUpToSteps(self, board, start, max_steps=8):
        """
        Return all reachable cells in up to `max_steps` steps,
        traveling only on empty adjacent hexes.
        """
        visited = set()
        q0, r0 = start
        frontier = [(q0, r0, 0)]
        results = set([(q0, r0)])

        while frontier:
            q, r, dist = frontier.pop()
            if dist >= max_steps:
                continue

            for (nq, nr) in self.getAdjacentCells(q, r):
                # Must be empty or not exist in board
                if (nq, nr) not in board or len(board[(nq, nr)]) == 0:
                    if (nq, nr) not in results:
                        results.add((nq, nr))
                    if (nq, nr, dist+1) not in visited:
                        visited.add((nq, nr, dist+1))
                        frontier.append((nq, nr, dist+1))
        return results

    # ---------------------------------------------------------
    # 3. MCTS-required Interface
    # ---------------------------------------------------------
    def getLegalActions(self, state):
        """
        Returns a list of legal actions for the current player.
        We'll combine "placement" actions (if the player still has pieces in hand)
        and "move" actions (if any of their pieces on the board can legally move).
        """
        legal_actions = []
        legal_actions.extend(self.placePieceActions(state))
        legal_actions.extend(self.movePieceActions(state))
        return legal_actions

    def applyAction(self, state, action):
        """
        Applies the given action to the current state and returns the resulting next state.
        Action can be of the form:
          ("PLACE", insectType, (q, r))
          ("MOVE", (from_q, from_r), (to_q, to_r))
        """
        new_state = self.copyState(state)
        board = new_state["board"]
        player = new_state["current_player"]

        action_type = action[0]

        if action_type == "PLACE":
            _, insectType, (q, r) = action
            # Remove one piece from the player's hand
            new_state["pieces_in_hand"][player][insectType] -= 1
            if (q, r) not in board:
                board[(q, r)] = []
            board[(q, r)].append((player, insectType))

        elif action_type == "MOVE":
            _, (from_q, from_r), (to_q, to_r) = action
            stack_from = board[(from_q, from_r)]
            # Pop the top piece
            piece = stack_from.pop()
            if len(stack_from) == 0:
                # If the stack is empty now, remove the cell from board dict
                del board[(from_q, from_r)]
            # Place the piece on the destination
            if (to_q, to_r) not in board:
                board[(to_q, to_r)] = []
            board[(to_q, to_r)].append(piece)

        # Switch to next player
        new_state["current_player"] = self.getOpponent(player)
        new_state["move_number"] += 1

        return new_state

    def isTerminal(self, state):
        """
        The game ends when either Queen is surrounded.
        (We won't handle draws or repeated states here;
         you could expand if you need more precise conditions.)
        """
        # If either queen is surrounded, return True
        if self.isQueenSurrounded(state, "Player1"):
            return True
        if self.isQueenSurrounded(state, "Player2"):
            return True
        return False

    def getReward(self, final_state, root_player):
        """
        Reward:
          +1 if root_player wins (opponent's queen is surrounded first),
          -1 if root_player loses (their queen is surrounded),
           0 otherwise (in case of a draw or no conclusive result).
        """
        opponent = self.getOpponent(root_player)
        root_surrounded = self.isQueenSurrounded(final_state, root_player)
        opp_surrounded  = self.isQueenSurrounded(final_state, opponent)

        if not root_surrounded and opp_surrounded:
            return 1.0
        elif root_surrounded and not opp_surrounded:
            return -1.0
        else:
            return 0.0

    def getGameOutcome(self, state):
        """
        Returns a string describing the outcome of the game if terminal:
          - "Player1" if Player1 has won
          - "Player2" if Player2 has won
          - "Draw" if the board is full and no one has won
          - None if the game is not yet terminal
        """
        current_player = state["current_player"]
        opponent = self.getOpponent(current_player)
        root_surrounded = self.isQueenSurrounded(state, current_player)
        opp_surrounded  = self.isQueenSurrounded(state, opponent)

        if not root_surrounded and opp_surrounded:
            return current_player
        elif root_surrounded and not opp_surrounded:
            return opponent
        else:
            # If board is full and no winner => draw
            if root_surrounded and opp_surrounded:
                return "Draw"
            else:
                # Game not finished
                return None


    def getCurrentPlayer(self, state):
        """
        Returns the current player.
        """
        return state["current_player"]

    def simulateRandomPlayout(self, state):
        """
        Simulates a random playout from the given state until the game ends.
        Returns the final state.
        """
        temp_state = self.copyState(state)

        while not self.isTerminal(temp_state):
            legal_actions = self.getLegalActions(temp_state)
            if not legal_actions:
                # No moves available: break (this might be an edge condition
                # if you want to handle forced pass, etc.)
                break
            action = random.choice(legal_actions)
            temp_state = self.applyAction(temp_state, action)

        return temp_state

    def printState(self, state):
        """
        Prints the simplified board state in a human-friendly format.
        We'll just list non-empty cells and top pieces for quick reference.
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
