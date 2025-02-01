import random
from collections import deque


def hex_distance(q1, r1, q2, r2):
    x1, z1 = q1, r1
    y1 = -x1 - z1
    x2, z2 = q2, r2
    y2 = -x2 - z2
    return (abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)) // 2

def find_queen_position(board, player):
    for (q, r), stack in board.items():
        for piece_owner, piece_type in stack:
            if piece_owner == player and piece_type == "Queen":
                return (q, r)
    return None

class HiveGame:
    """
    A Hive implementation for demonstration with a generic MCTS framework.

    Major simplifications/assumptions:
      - Board is tracked using axial coordinates (q, r).
      - Each coordinate can have a stack of pieces (top of stack is last in list).
      - Movement rules are partial and simplified (no strict 'sliding' constraints, no climbing).
      - We do check if each Queen is fully surrounded for a terminal state.
      - We treat "no legal moves" as an immediate loss for the current player to avoid MCTS contradictions.
    """
    def __init__(self):
        self.INITIAL_PIECES = {
            "Queen": 1,
            "Spider": 2,
            "Beetle": 2,
            "Grasshopper": 3,
            "Ant": 3
        }
        self.DIRECTIONS = [(1, 0), (-1, 0), (0, 1),
                           (0, -1), (1, -1), (-1, 1)]
        # Cache for connectivity checks.
        self._connectivity_cache = {}
        self._legal_moves_cache = {}


    def board_key(self, board):
        """
        Create an immutable, canonical representation of the board.
        We assume board is a dict mapping coordinates to a list of pieces.
        Each piece is a tuple (player, insect). Sorting ensures consistent ordering.
        """
        # For each cell, sort the stack (if needed) and then sort by coordinate.
        return tuple(sorted((coord, tuple(stack)) for coord, stack in board.items()))

    def board_hash(self, board):
        """
        Creates an immutable, canonical representation of the board and returns its hash.
        Each board is a dict mapping coordinates (tuples) to a list of pieces.
        We convert each list into a tuple and sort by coordinate.
        """
        # For each cell, sort the stack (if needed) and then sort the items by coordinate.
        items = tuple(sorted((coord, tuple(stack)) for coord, stack in board.items()))
        return hash(items)

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
        """
        Returns a generator of the six neighbors of cell (q, r) using the pre-stored directions.
        (This is already very simple; if needed, you can inline its logic in other methods.)
        """
        # Using a generator expression avoids the overhead of constructing a list.
        return ((q + dq, r + dr) for dq, dr in self.DIRECTIONS)

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
                    for (nq, nr) in self.getAdjacentCells(q, r):
                        if (nq, nr) not in board or len(board[(nq, nr)]) == 0:
                            return False
                    return True
        return False

    # ---------------------------------------------------------
    # 3. Action generation
    # ---------------------------------------------------------
    def getLegalActions(self, state):
        """
        Returns all legal actions for the current player, using a cache to avoid
        recomputation on board states we've seen before.
        """
        board = state["board"]
        key = self.board_hash(board)
        if key in self._legal_moves_cache:
            cached_moves = self._legal_moves_cache[key]
        else:
            cached_moves = self.placePieceActions(state) + self.movePieceActions(state)
            self._legal_moves_cache[key] = cached_moves

        current_player = state["current_player"]
        if find_queen_position(board, current_player) is None:
            move_count = state["move_number"] // 2 + 1
            if move_count >= 3:
                queen_actions = [action for action in cached_moves if action[0] == "PLACE" and action[1] == "Queen"]
                return queen_actions
        return cached_moves

    def clearCaches(self):
        """
        Clears caches. Call this after applying an action to ensure that subsequent
        move generation uses the updated board.
        """
        self._connectivity_cache.clear()
        self._legal_moves_cache.clear()

    def placePieceActions(self, state):
        player = state["current_player"]
        opponent = self.getOpponent(player)
        board = state["board"]
        pieces_in_hand = state["pieces_in_hand"][player]
        actions = []
        if all(count == 0 for count in pieces_in_hand.values()):
            return actions
        if len(board) == 0:
            for insectType, count in pieces_in_hand.items():
                if count > 0:
                    actions.append(("PLACE", insectType, (0, 0)))
            return actions
        if len(board) == 1:
            for insectType, count in pieces_in_hand.items():
                if count > 0 and (0, 1) not in board:
                    actions.append(("PLACE", insectType, (0, 1)))
                elif count > 0 and (0, 0) not in board:
                    actions.append(("PLACE", insectType, (0, 0)))
            return actions
        friendly_cells = set()
        for (q, r), stack in board.items():
            if any(p[0] == player for p in stack):
                friendly_cells.add((q, r))
        potential_spots = set()
        for (q, r) in friendly_cells:
            for (nq, nr) in self.getAdjacentCells(q, r):
                if (nq, nr) not in board or len(board[(nq, nr)]) == 0:
                    potential_spots.add((nq, nr))
        valid_spots = []
        for (tq, tr) in potential_spots:
            adjacent_to_enemy = False
            for (nq, nr) in self.getAdjacentCells(tq, tr):
                if (nq, nr) in board and any(p[0] == opponent for p in board[(nq, nr)]):
                    adjacent_to_enemy = True
                    break
            if not adjacent_to_enemy:
                valid_spots.append((tq, tr))
        for insectType, count in pieces_in_hand.items():
            if count > 0:
                for (tq, tr) in valid_spots:
                    actions.append(("PLACE", insectType, (tq, tr)))
        return actions

    def getAntDestinationsEdge(self, board, q, r, max_steps=20):
        """
        Ant moves up to `max_steps` steps around the edge of the hive,
        adding every reachable intermediate cell as a valid destination.
        """
        directions = [(1, 0), (1, -1), (0, -1),
                      (-1, 0), (-1, 1), (0, 1)]
        results = set()

        def tryPath(startQ, startR, startDirIndex, directionIncrement):
            path = [(startQ, startR)]
            curQ, curR = startQ, startR
            dirIndex = startDirIndex

            for step in range(max_steps):
                dirIndex = (dirIndex + directionIncrement) % 6
                dQ, dR = directions[dirIndex]
                nextQ = curQ + dQ
                nextR = curR + dR

                if (nextQ, nextR) in board and len(board[(nextQ, nextR)]) > 0:
                    return

                if (nextQ, nextR) in path:
                    return

                neighbors_occupied = any(
                    ((adjQ, adjR) in board and len(board[(adjQ, adjR)]) > 0)
                    for (adjQ, adjR) in self.getAdjacentCells(nextQ, nextR)
                )
                if not neighbors_occupied:
                    return

                results.add((nextQ, nextR))
                path.append((nextQ, nextR))
                curQ, curR = nextQ, nextR

        for i in range(6):
            tryPath(q, r, i, +1)
            tryPath(q, r, i, -1)

        return results

    def getGrasshopperJumps(self, board, q, r):
        """
        Grasshopper jumps in each direction by leaping over consecutive occupied cells.
        """
        possible_destinations = []
        for (dq, dr) in self.DIRECTIONS:
            step_count = 1
            while True:
                check_q = q + dq * step_count
                check_r = r + dr * step_count
                if (check_q, check_r) not in board or len(board[(check_q, check_r)]) == 0:
                    if step_count > 1:
                        possible_destinations.append((check_q, check_r))
                    break
                step_count += 1
        return possible_destinations

    def movePieceActions(self, state):
        """
        MOVE actions: ("MOVE", (from_q, from_r), (to_q, to_r)).
        Implements movement for Queen, Beetle, Grasshopper, Spider, and Ant.
        Instead of modifying the board in place (and then trying to restore it),
        we simulate moves on a temporary copy of the board. This avoids leaving the board
        in an inconsistent state that might cause KeyError when applying moves.
        """
        board = state["board"]
        player = state["current_player"]
        actions = []

        # Gather candidate cells (from a copy of board keys) to avoid issues if board is modified.
        player_cells = []
        for (q, r), stack in list(board.items()):
            if stack and stack[-1][0] == player:
                insectType = stack[-1][1]
                player_cells.append((q, r, insectType))

        def board_copy(board):
            """Make a shallow copy of board, copying each stack."""
            return {coord: stack[:] for coord, stack in board.items()}

        for (q, r, insectType) in player_cells:
            if insectType == "Queen":
                temp_board = board_copy(board)
                # Remove the queen from the current cell.
                piece = temp_board[(q, r)].pop()
                if not temp_board[(q, r)]:
                    del temp_board[(q, r)]
                for (nq, nr) in self.getAdjacentCells(q, r):
                    # Queen can only move to an empty cell.
                    if (nq, nr) in temp_board and len(temp_board[(nq, nr)]) > 0:
                        continue
                    if not self.canSlide(q, r, nq, nr, temp_board):
                        continue
                    # Simulate moving the queen.
                    temp_board.setdefault((nq, nr), []).append(piece)
                    if self.isBoardConnected(temp_board, self.getAdjacentCells):
                        actions.append(("MOVE", (q, r), (nq, nr)))
                    temp_board[(nq, nr)].pop()
                    if not temp_board[(nq, nr)]:
                        del temp_board[(nq, nr)]
                        # No need to restore temp_board; it is local.
             # In movePieceActions, inside the loop over candidate cells:
            elif insectType == "Beetle":
                # Make a copy of the board to simulate moves.
                temp_board = board_copy(board)
                # Determine the stack height at (q, r) in the current board.
                stack_height = len(temp_board.get((q, r), []))
                # Iterate over each neighbor.
                for (nq, nr) in self.getAdjacentCells(q, r):
                    # Work on a fresh copy for each candidate move.
                    sim_board = board_copy(temp_board)
                    # Remove the beetle from (q, r).
                    piece = sim_board[(q, r)].pop()
                    if not sim_board[(q, r)]:
                        del sim_board[(q, r)]
                    # For beetles on top (stack_height > 1), allow movement into any adjacent cell.
                    if stack_height > 1:
                        # For a beetle on top, we do not enforce the sliding constraint.
                        sim_board.setdefault((nq, nr), []).append(piece)
                        if self.isBoardConnected(sim_board, self.getAdjacentCells):
                            actions.append(("MOVE", (q, r), (nq, nr)))
                        # No need to remove piece from sim_board here, since sim_board is local.
                    else:
                        # For a solitary beetle, enforce sliding rules as before.
                        # Skip if the destination cell is occupied.
                        if (nq, nr) in sim_board and len(sim_board[(nq, nr)]) > 0:
                            # Even for a solitary beetle, sometimes climbing may be allowed—but if you want
                            # to follow sliding constraints strictly, you can skip here.
                            continue
                        if not self.canSlide(q, r, nq, nr, sim_board):
                            continue
                        sim_board.setdefault((nq, nr), []).append(piece)
                        if self.isBoardConnected(sim_board, self.getAdjacentCells):
                            actions.append(("MOVE", (q, r), (nq, nr)))
                            # (No need to restore sim_board; it is discarded after each candidate.)
            elif insectType == "Grasshopper":
                temp_board = board_copy(board)
                jumps = self.getGrasshopperJumps(temp_board, q, r)
                for (tq, tr) in jumps:
                    piece = temp_board[(q, r)].pop()
                    if not temp_board[(q, r)]:
                        del temp_board[(q, r)]
                    still_connected_after_removal = self.isBoardConnected(temp_board, self.getAdjacentCells)
                    if not still_connected_after_removal:
                        temp_board.setdefault((q, r), []).append(piece)
                        continue
                    temp_board.setdefault((tq, tr), []).append(piece)
                    valid_new_spot = True
                    if len(temp_board[(tq, tr)]) == 1:
                        neighbors_occupied = any(
                            ((xq, xr) in temp_board and temp_board[(xq, xr)])
                            for (xq, xr) in self.getAdjacentCells(tq, tr)
                        )
                        if not neighbors_occupied:
                            valid_new_spot = False
                    still_connected_after_placement = self.isBoardConnected(temp_board, self.getAdjacentCells)
                    temp_board[(tq, tr)].pop()
                    if not temp_board.get((tq, tr)):
                        temp_board.pop((tq, tr), None)
                    temp_board.setdefault((q, r), []).append(piece)
                    if still_connected_after_placement and valid_new_spot:
                        actions.append(("MOVE", (q, r), (tq, tr)))
            elif insectType == "Spider":
                temp_board = board_copy(board)
                piece = temp_board[(q, r)].pop()
                if not temp_board[(q, r)]:
                    del temp_board[(q, r)]
                possible_ends = self.getSpiderDestinations(temp_board, (q, r))
                temp_board.setdefault((q, r), []).append(piece)
                for dest in possible_ends:
                    piece = temp_board[(q, r)].pop()
                    if not temp_board[(q, r)]:
                        del temp_board[(q, r)]
                    temp_board.setdefault(dest, []).append(piece)
                    if self.isBoardConnected(temp_board, self.getAdjacentCells):
                        actions.append(("MOVE", (q, r), dest))
                    temp_board[dest].pop()
                    if not temp_board.get(dest):
                        temp_board.pop(dest, None)
                    temp_board.setdefault((q, r), []).append(piece)
            elif insectType == "Ant":
                temp_board = board_copy(board)
                piece = temp_board[(q, r)].pop()
                if not temp_board[(q, r)]:
                    del temp_board[(q, r)]
                possible_ends = self.getAntDestinations(temp_board, (q, r))
                temp_board.setdefault((q, r), []).append(piece)
                for dest in possible_ends:
                    piece = temp_board[(q, r)].pop()
                    if not temp_board[(q, r)]:
                        del temp_board[(q, r)]
                    temp_board.setdefault(dest, []).append(piece)
                    if self.isBoardConnected(temp_board, self.getAdjacentCells):
                        actions.append(("MOVE", (q, r), dest))
                    temp_board[dest].pop()
                    if not temp_board.get(dest):
                        temp_board.pop(dest, None)
                    temp_board.setdefault((q, r), []).append(piece)
            # End for each piece.
        return actions

    def getSpiderDestinations(self, board, start):
        """
        Returns the set of cells that the spider can reach by moving exactly 3 steps,
        following a path of empty cells where each step satisfies the sliding constraint.
        The spider may not revisit cells on its path.
        """
        results = set()

        def dfs(path, steps):
            cur = path[-1]
            if steps == 3:
                results.add(cur)
                return
            for neighbor in self.getAdjacentCells(*cur):
                if neighbor in path:
                    continue
                if neighbor in board and len(board[neighbor]) > 0:
                    continue
                # Check that the neighbor "hugs" the hive: at least one adjacent cell is occupied.
                if not any(((adj in board) and (len(board[adj]) > 0))
                           for adj in self.getAdjacentCells(*neighbor)):
                    continue
                # Check sliding from current cell to neighbor.
                if not self.canSlide(cur[0], cur[1], neighbor[0], neighbor[1], board):
                    continue
                dfs(path + [neighbor], steps + 1)

        dfs([start], 0)  # Start with a list containing the starting tuple.
        return results

    # --- New: Revised Ant Move Generation ---
    def getAntDestinations(self, board, start):
        """
        Returns all empty cells reachable by the ant from 'start'
        via a sliding move. The ant is removed from the board
        (the caller must do so) so that its original cell is empty.
        A neighbor cell is accepted if:
          - It is empty.
          - It is adjacent to at least one occupied cell (i.e. "hugs" the hive).
          - The ant can slide from the current cell into it (using canSlide).
        """
        results = set()
        visited = set()
        frontier = [start]
        while frontier:
            cur = frontier.pop(0)
            for neighbor in self.getAdjacentCells(*cur):
                if neighbor in board and len(board[neighbor]) > 0:
                    continue
                if not any(((adj in board) and (len(board[adj]) > 0)) for adj in self.getAdjacentCells(*neighbor)):
                    continue
                if not self.canSlide(cur[0], cur[1], neighbor[0], neighbor[1], board):
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    results.add(neighbor)
                    frontier.append(neighbor)
        return results

    def canSlide(self, from_q, from_r, to_q, to_r, board):
        """
        Determines whether a piece can slide from (from_q, from_r) to (to_q, to_r)
        according to Hive sliding rules.
        The piece must be able to “squeeze” between the two cells adjacent to the edge
        between the from and to cells. We do this by checking the two adjacent directions.
        """
        dq = to_q - from_q
        dr = to_r - from_r
        move_dir = (dq, dr)
        # For a pointy-topped hex grid using axial coordinates, we define:
        adjacent_mapping = {
            (1, 0): [(0, 1), (1, -1)],
            (0, 1): [(-1, 1), (1, 0)],
            (-1, 1): [(0, 1), (-1, 0)],
            (-1, 0): [(-1, 1), (0, -1)],
            (0, -1): [(1, -1), (-1, 0)],
            (1, -1): [(1, 0), (0, -1)]
        }
        if move_dir not in adjacent_mapping:
            return False
        adj_dirs = adjacent_mapping[move_dir]
        blocked = 0
        for ad in adj_dirs:
            cell = (from_q + ad[0], from_r + ad[1])
            if cell in board and len(board[cell]) > 0:
                blocked += 1
        return blocked < 2

    def _makeTempState(self, original_state, temp_board):
        """
        Helper to create a temporary state referencing the current board progress.
        """
        temp_state = self.copyState(original_state)
        temp_state["board"] = {}
        for coord, stack in temp_board.items():
            temp_state["board"][coord] = stack[:]
        return temp_state

    def getOtherPlayer(self, currentPlayer):
        if currentPlayer == "Player1":
            return "Player2"
        elif currentPlayer == "Player2":
            return "Player1"

    def isBoardConnected(self, board, getAdjacentCells=None):
        """
        Returns True if all occupied cells in 'board' form one connected component.
        Uses union-find and caches the result keyed by the board's hash.
        """
        key = self.board_hash(board)
        if key in self._connectivity_cache:
            return self._connectivity_cache[key]

        # Build a list of occupied cells.
        occupied_cells = [cell for cell, stack in board.items() if stack]
        if not occupied_cells:
            self._connectivity_cache[key] = True
            return True

        # Union-Find initialization.
        parent = {cell: cell for cell in occupied_cells}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx = find(x)
            ry = find(y)
            if rx != ry:
                parent[ry] = rx

        # Use the precomputed directions.
        directions = self.DIRECTIONS
        for cell in occupied_cells:
            q, r = cell
            for dq, dr in directions:
                neighbor = (q + dq, r + dr)
                if neighbor in board and board[neighbor]:
                    union(cell, neighbor)

        roots = {find(cell) for cell in occupied_cells}
        result = (len(roots) == 1)
        self._connectivity_cache[key] = result
        return result

    def getSpiderDestinationsEdge(self, board, q, r):
        directions = [(1, 0), (1, -1), (0, -1),
                      (-1, 0), (-1, 1), (0, 1)]
        results = set()

        def tryPath(startQ, startR, startDirIndex, directionIncrement):
            path = [(startQ, startR)]
            curQ, curR = startQ, startR
            dirIndex = startDirIndex
            for _ in range(3):
                dirIndex = (dirIndex + directionIncrement) % 6
                dQ, dR = directions[dirIndex]
                nextQ = curQ + dQ
                nextR = curR + dR
                if (nextQ, nextR) in board and len(board[(nextQ, nextR)]) > 0:
                    return
                if (nextQ, nextR) in path:
                    return
                neighbors_occupied = any(
                    ((adjQ, adjR) in board and len(board[(adjQ, adjR)]) > 0)
                    for (adjQ, adjR) in self.getAdjacentCells(nextQ, nextR)
                )
                if not neighbors_occupied:
                    return
                path.append((nextQ, nextR))
                curQ, curR = nextQ, nextR
            results.add((curQ, curR))

        for i in range(6):
            tryPath(q, r, i, +1)
            tryPath(q, r, i, -1)
        return results

    def weightedActionChoice(self, state, actions):
        board = state["board"]
        current_player = state["current_player"]
        enemy_player = self.getOpponent(current_player)
        enemy_queen_pos = find_queen_position(board, enemy_player)
        if enemy_queen_pos is None:
            return random.choice(actions)
        (eq_q, eq_r) = enemy_queen_pos
        weighted_moves = []
        for action in actions:
            if action[0] == "PLACE":
                _, insectType, (q, r) = action
                final_q, final_r = q, r
            elif action[0] == "MOVE":
                _, (fq, fr), (tq, tr) = action
                final_q, final_r = tq, tr
            else:
                final_q, final_r = 0, 0
            dist = hex_distance(final_q, final_r, eq_q, eq_r)
            steepness = 2.0
            weight = 1.0 / (1.0 + steepness * dist**2)
            weighted_moves.append((action, weight))
        total_weight = sum(w for (_, w) in weighted_moves)
        if total_weight == 0.0:
            return random.choice(actions)
        rnd = random.random() * total_weight
        cumulative = 0.0
        for (act, w) in weighted_moves:
            cumulative += w
            if rnd < cumulative:
                return act
        return actions[-1]

    def _bfsExactSteps(self, board, start, steps=3):
        visited = set()
        q0, r0 = start
        frontier = [(q0, r0, 0)]
        results = set()
        while frontier:
            q, r, dist = frontier.pop()
            if dist == steps:
                results.add((q, r))
                continue
            elif dist > steps:
                continue
            for (nq, nr) in self.getAdjacentCells(q, r):
                if (nq, nr) not in board or len(board[(nq, nr)]) == 0:
                    next_state = (nq, nr, dist+1)
                    if next_state not in visited and dist+1 <= steps:
                        visited.add(next_state)
                        frontier.append(next_state)
        return results

    def _bfsUpToSteps(self, board, start, max_steps=8):
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
        if self.isQueenSurrounded(state, "Player1"):
            return True
        if self.isQueenSurrounded(state, "Player2"):
            return True
        if not self.getLegalActions(state):
            return True
        return False

    def applyAction(self, state, action):
        new_state = self.copyState(state)
        board = new_state["board"]
        player = new_state["current_player"]
        if action[0] == "PLACE":
            _, insectType, (q, r) = action
            new_state["pieces_in_hand"][player][insectType] -= 1
            if (q, r) not in board:
                board[(q, r)] = []
            board[(q, r)].append((player, insectType))
        elif action[0] == "MOVE":
            _, (fq, fr), (tq, tr) = action
            piece = board[(fq, fr)].pop()
            if len(board[(fq, fr)]) == 0:
                del board[(fq, fr)]
            if (tq, tr) not in board:
                board[(tq, tr)] = []
            board[(tq, tr)].append(piece)
        new_state["current_player"] = self.getOpponent(player)
        new_state["move_number"] += 1

        # Clear caches so that subsequent calls use the new board state.
        self.clearCaches()
        return new_state

    def evaluateState(self, state):
        """
        Evaluates the board state heuristically from Player1's perspective.

        Positive scores indicate an advantage for Player1; negative for Player2.

        Factors included:
          - Queen mobility: More liberties for our queen is good; fewer liberties for the enemy queen is good.
          - Average distance of non-queen friendly pieces to the enemy queen: Closer is better.
        """
        outcome = self.getGameOutcome(state)
        if outcome is not None:
            # Use high scores for terminal positions.
            if outcome == "Player1":
                return 10000
            elif outcome == "Player2":
                return -10000
            else:
                return 0

        board = state["board"]
        # Find queen positions.
        our_queen = find_queen_position(board, "Player1")
        enemy_queen = find_queen_position(board, "Player2")

        our_liberties = 0
        enemy_liberties = 0

        # Count adjacent empty cells (liberties) for each queen.
        if our_queen is not None:
            for neighbor in self.getAdjacentCells(*our_queen):
                if neighbor not in board or len(board[neighbor]) == 0:
                    our_liberties += 1
        if enemy_queen is not None:
            for neighbor in self.getAdjacentCells(*enemy_queen):
                if neighbor not in board or len(board[neighbor]) == 0:
                    enemy_liberties += 1

        # A simple heuristic: score based on queen liberties.
        # The more free spaces our queen has and the fewer the enemy queen has, the better.
        score = (our_liberties - enemy_liberties) * 20

        # Next, compute an average distance measure for non-queen friendly pieces to the enemy queen.
        if enemy_queen is not None:
            total_distance = 0
            count = 0
            for (q, r), stack in board.items():
                if stack and stack[-1][0] == "Player1" and stack[-1][1] != "Queen":
                    total_distance += hex_distance(q, r, enemy_queen[0], enemy_queen[1])
                    count += 1
            if count > 0:
                avg_distance = total_distance / count
                # If our pieces are closer to the enemy queen, that is an advantage.
                # You can adjust the constant below to scale the effect.
                score += (20 - avg_distance) * 5

        return score


    def getGameOutcome(self, state):
        p1_surrounded = self.isQueenSurrounded(state, "Player1")
        p2_surrounded = self.isQueenSurrounded(state, "Player2")
        if p1_surrounded and p2_surrounded:
            return "Draw"
        elif p1_surrounded:
            return "Player2"
        elif p2_surrounded:
            return "Player1"
        if not self.getLegalActions(state):
            current = state["current_player"]
            opponent = self.getOpponent(current)
            return opponent
        return None

    def getCurrentPlayer(self, state):
        return state["current_player"]

    def simulateRandomPlayout(self, state, max_depth=10):
        temp_state = self.copyState(state)
        depth = 0
        while not self.isTerminal(temp_state) and depth < max_depth:
            legal = self.getLegalActions(temp_state)
            if not legal:
                break
            action = self.weightedActionChoice(temp_state, legal)
            temp_state = self.applyAction(temp_state, action)
            depth += 1
        # Instead of returning the state, return the heuristic evaluation.
        return self.evaluateState(temp_state)

    # ---------------------------------------------------------
    # 5. Print / Debug
    # ---------------------------------------------------------
    def printState(self, state):
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
