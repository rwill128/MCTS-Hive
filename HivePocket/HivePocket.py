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
    """
    INITIAL_PIECES = {
        "Queen": 1,
        "Spider": 2,
        "Beetle": 2,
        "Grasshopper": 3,
        "Ant": 3
    }

    DIRECTIONS = [(1, 0), (-1, 0), (0, 1),
                  (0, -1), (1, -1), (-1, 1)]

    def __init__(self):

        # Cache for connectivity checks.
        self._connectivity_cache = {}
        self._legal_moves_cache = {}
        self._can_slide_cache = {}


    def board_key(self, board):
        """
        Create an immutable, canonical representation of the board.
        """
        return tuple(sorted((coord, tuple(stack)) for coord, stack in board.items()))

    def board_hash(self, board):
        """
        Creates an immutable, canonical representation of the board and returns its hash.
        """
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

    def getGameOutcome(self, state):
        p1_surrounded = self.isQueenSurrounded(state, "Player1")
        p2_surrounded = self.isQueenSurrounded(state, "Player2")
        if p1_surrounded and p2_surrounded:
            return "Draw"
        elif p1_surrounded:
            return "Player2"
        elif p2_surrounded:
            return "Player1"
        # No "no moves => end" check here anymore!
        return None

    def copyState(self, state):
        """
        Returns a deep copy of the state dictionary.
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

    # ---------------------------------------------------------
    # 2. Core Hive logic
    # ---------------------------------------------------------
    def getOpponent(self, player):
        return "Player2" if player == "Player1" else "Player1"

    def getAdjacentCells(self, q, r):
        """
        Returns a generator of the six neighbors of cell (q, r).
        """
        return ((q + dq, r + dr) for dq, dr in self.DIRECTIONS)

    def isQueenSurrounded(self, state, player):
        """
        Checks if player's Queen is on the board and fully surrounded.
        """
        board = state["board"]
        for (q, r), stack in board.items():
            for piece in stack:
                if piece == (player, "Queen"):
                    for (nq, nr) in self.getAdjacentCells(q, r):
                        if (nq, nr) not in board or not board[(nq, nr)]:
                            return False
                    return True
        return False

    def state_key(self, state):
        board = state["board"]
        current_player = state["current_player"]
        # Sort pieces_in_hand so the order is canonical.
        pieces_in_hand = tuple(sorted((player, tuple(sorted(pieces.items())))
                                      for player, pieces in state["pieces_in_hand"].items()))
        return (self.board_hash(board), current_player, pieces_in_hand)

    # ---------------------------------------------------------
    # 3. Action generation
    # ---------------------------------------------------------
    def getLegalActions(self, state):
        board = state["board"]  # existing
        key = self.state_key(state)
        if key in self._legal_moves_cache:
            return self._legal_moves_cache[key]

        # 1. Generate normal piece placements and movements
        place_actions = self.placePieceActions(state)
        move_actions = self.movePieceActions(state)
        all_actions = place_actions + move_actions

        # 2. If no actions are possible, allow a PASS
        if not all_actions:
            all_actions = [("PASS",)]

        # 3. Handle the Queen-placement rule
        current_player = state["current_player"]
        if find_queen_position(board, current_player) is None:
            pieces_placed = sum(
                1 for _, stack in board.items()
                for p_owner, _ in stack if p_owner == current_player
            )
            if pieces_placed >= 3:
                # Filter out everything that's not "PLACE Queen" or "PASS"
                queen_or_pass = [
                    action for action in all_actions
                    if (action[0] == "PLACE" and action[1] == "Queen")
                       or (action[0] == "PASS")
                ]
                self._legal_moves_cache[key] = queen_or_pass
                return queen_or_pass

        self._legal_moves_cache[key] = all_actions
        return all_actions

    def clearCaches(self):
        """Clears all caches."""
        self._connectivity_cache.clear()
        self._legal_moves_cache.clear()
        self._can_slide_cache.clear()

    def placePieceActions(self, state):
        """Generates legal placement actions."""
        player = state["current_player"]
        opponent = self.getOpponent(player)
        board = state["board"]
        pieces_in_hand = state["pieces_in_hand"][player]
        actions = []

        if all(count == 0 for count in pieces_in_hand.values()):
            return actions

        if len(board) == 0:  # First placement
            for insectType, count in pieces_in_hand.items():
                if count > 0:
                    actions.append(("PLACE", insectType, (0, 0)))
            return actions
        #Second placement.
        if len(board) == 1:
            for insectType, count in pieces_in_hand.items():
                if count > 0:
                    for coord, _ in board.items(): #Safe since len(board) == 1
                        q, r = coord
                    for neighbor in self.getAdjacentCells(q, r):
                        actions.append(("PLACE", insectType, neighbor))
            return actions


        friendly_cells = set()
        for (q, r), stack in board.items():
            if any(p[0] == player for p in stack):
                friendly_cells.add((q, r))

        potential_spots = set()
        for (q, r) in friendly_cells:
            for (nq, nr) in self.getAdjacentCells(q, r):
                if (nq, nr) not in board or not board[(nq, nr)]:
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

    def getGrasshopperJumps(self, board, q, r):
        """Calculates Grasshopper jump destinations."""
        possible_destinations = []
        for (dq, dr) in self.DIRECTIONS:
            step_count = 1
            while True:
                check_q = q + dq * step_count
                check_r = r + dr * step_count
                if (check_q, check_r) not in board or not board[(check_q, check_r)]:
                    if step_count > 1:
                        possible_destinations.append((check_q, check_r))
                    break
                step_count += 1
        return possible_destinations

    def movePieceActions(self, state, debug=False):
        board = state["board"]
        player = state["current_player"]
        actions = []

        # We only consider the top piece in each stack if it belongs to the current player
        player_cells = {
            (q, r, stack[-1][1])
            for (q, r), stack in board.items()
            if stack and stack[-1][0] == player
        }

        def board_copy(bd):
            return {coord: st[:] for coord, st in bd.items()}

        if debug:
            print(f"[DEBUG] movePieceActions called for {player}")

        for (q, r, insectType) in player_cells:
            temp_board = board_copy(board)

            if debug:
                print(f"  [DEBUG] Considering {insectType} at {(q, r)}")
                print(f"         Stack before removal: {board[(q, r)]}")

            piece = temp_board[(q, r)].pop()  # Remove the top piece
            if not temp_board[(q, r)]:
                del temp_board[(q, r)]

            # 1. Check if removing this piece splits the hive
            if not self.isBoardConnected(temp_board):
                if debug:
                    print("    [DEBUG] Skipping. Removing this piece breaks hive connectivity (pinned).")
                continue
            else:
                if debug:
                    print("    [DEBUG] Lifting piece does NOT break the hive.")

            # 2. Get possible destinations based on insect type
            if insectType == "Queen" or insectType == "Beetle":
                destinations = self.getAdjacentCells(q, r)
            elif insectType == "Grasshopper":
                destinations = self.getGrasshopperJumps(temp_board, q, r)
            elif insectType == "Spider":
                destinations = self.getSpiderDestinations(temp_board, (q, r))
            elif insectType == "Ant":
                destinations = self.getAntDestinations(temp_board, (q, r))
            else:
                destinations = []

            if debug:
                print(f"    [DEBUG] Potential destinations = {list(destinations)}")

            # 3. Validate each destination
            for (tq, tr) in destinations:
                # (a) Occupancy check (non-Beetles cannot land on occupied cells)
                if insectType != "Beetle" and (tq, tr) in temp_board and temp_board[(tq, tr)]:
                    if debug:
                        print(f"      [DEBUG] Destination {(tq, tr)} is occupied, skipping (only Beetle can stack).")
                    continue

                # (b) Sliding check (for Queen, e.g.)
                if insectType not in ["Beetle", "Grasshopper"]:
                    can_slide = self.canSlide(q, r, tq, tr, temp_board, debug=debug)
                    if not can_slide:
                        if debug:
                            print(f"      [DEBUG] Slide from {(q, r)} to {(tq, tr)} is blocked, skipping.")
                        continue

                # (c) Check final connectivity if we place the piece there
                temp_board.setdefault((tq, tr), []).append(piece)
                if not self.isBoardConnected(temp_board):
                    temp_board[(tq, tr)].pop()  # Undo
                    if not temp_board[(tq, tr)]:
                        del temp_board[(tq, tr)]
                    if debug:
                        print(f"      [DEBUG] Move to {(tq, tr)} breaks hive connectivity afterwards, skipping.")
                    continue

                # If we reach here, it’s a valid move
                temp_board[(tq, tr)].pop()  # Undo before next iteration
                if not temp_board[(tq, tr)]:
                    del temp_board[(tq, tr)]
                actions.append(("MOVE", (q, r), (tq, tr)))

                if debug:
                    print(f"      [DEBUG] Valid move => ({(q, r)} -> {(tq, tr)})")

        if debug:
            print(f"[DEBUG] Total legal moves for {player}: {actions}")
        return actions

    def is_locally_connected(self, board, q, r):
        """Checks if placing a piece at (q, r) would connect it to the hive."""
        for nq, nr in self.getAdjacentCells(q, r):
            if (nq, nr) in board and board[(nq, nr)]:
                return True
        return False

    def would_disconnect(self, board, q, r):
        """Checks if removing the piece at (q, r) would disconnect the hive."""
        if not board.get((q,r)): return False
        temp_board = {coord: stack[:] for coord, stack in board.items()}
        temp_board[(q, r)].pop()
        if not temp_board[(q, r)]: del temp_board[(q, r)]
        return not self.isBoardConnected(temp_board)

    def getSpiderPaths(self, board, start):
        """
        Returns a list of full paths (each a list of cells) representing a
        legal three‐step spider move from the starting cell.
        """
        results = []
        def dfs(path, steps):
            cur = path[-1]
            if steps == 3:
                results.append(path)
                return
            for neighbor in self.getAdjacentCells(*cur):
                if neighbor in path:
                    continue
                if neighbor in temp_board and temp_board[neighbor]:
                    continue
                # Place the spider temporarily.
                temp_board.setdefault(neighbor, []).append(("SpiderDebug", "Spider"))
                # Check connectivity and sliding from cur to neighbor.
                if self.isBoardConnected(temp_board) and self.canSlide(cur[0], cur[1], neighbor[0], neighbor[1], temp_board):
                    dfs(path + [neighbor], steps + 1)
                # Backtrack.
                temp_board[neighbor].pop()
                if not temp_board[neighbor]:
                    del temp_board[neighbor]
        # Work on a copy so we don’t affect the original board.
        temp_board = {c: st[:] for c, st in board.items()}
        if start in temp_board and temp_board[start]:
            temp_board[start].pop()
            if not temp_board[start]:
                del temp_board[start]
        # Get valid first moves.
        valid_first = []
        for first in self.getAdjacentCells(*start):
            if first in temp_board and temp_board[first]:
                continue
            temp_board.setdefault(first, []).append(("SpiderDebug", "Spider"))
            if self.isBoardConnected(temp_board) and self.canSlide(start[0], start[1], first[0], first[1], temp_board):
                valid_first.append(first)
            temp_board[first].pop()
            if not temp_board[first]:
                del temp_board[first]
        for first in valid_first:
            dfs([start, first], 1)
        return results


    def getSpiderDestinations_debug(self, board, start):
        """
        Debug version of getSpiderDestinations with detailed logging.
        It performs a DFS for exactly 3 moves (as per Hive rules) and logs
        decisions about occupancy, connectivity, and sliding.
        """
        results = set()
        debug_log = []

        def dfs(path, steps):
            cur = path[-1]
            debug_log.append(f"DFS at {cur} with path {path} (steps taken: {steps})")
            if steps == 3:
                debug_log.append(f"Reached 3 steps; adding destination {cur}")
                results.add(cur)
                return

            for neighbor in self.getAdjacentCells(*cur):
                if neighbor in path:
                    debug_log.append(f"Skipping neighbor {neighbor} (already in path).")
                    continue
                if neighbor in temp_board and temp_board[neighbor]:
                    debug_log.append(f"Skipping neighbor {neighbor} (occupied: {temp_board[neighbor]}).")
                    continue

                # Temporarily place the spider here.
                temp_board.setdefault(neighbor, []).append(("SpiderDebug", "Spider"))
                # Check connectivity first.
                if not self.isBoardConnected(temp_board):
                    debug_log.append(f"Placing spider at {neighbor} disconnects the hive; backtracking.")
                    temp_board[neighbor].pop()
                    if not temp_board[neighbor]:
                        del temp_board[neighbor]
                    continue
                # Check if the spider can slide from the current cell to this neighbor.
                if not self.canSlide(cur[0], cur[1], neighbor[0], neighbor[1], temp_board):
                    debug_log.append(f"Cannot slide from {cur} to {neighbor}; backtracking.")
                    temp_board[neighbor].pop()
                    if not temp_board[neighbor]:
                        del temp_board[neighbor]
                    continue

                debug_log.append(f"Valid slide from {cur} to {neighbor}; recursing DFS.")
                dfs(path + [neighbor], steps + 1)

                # Backtrack: remove the temporarily placed spider.
                debug_log.append(f"Backtracking from {neighbor} to {cur}.")
                temp_board[neighbor].pop()
                if not temp_board[neighbor]:
                    del temp_board[neighbor]

        # Make a copy of the board so we can simulate moves without affecting the original.
        temp_board = {c: st[:] for c, st in board.items()}

        # Remove the spider from its starting cell.
        if start in temp_board and temp_board[start]:
            temp_board[start].pop()
            if not temp_board[start]:
                del temp_board[start]

        debug_log.append(f"Starting DFS for spider at {start}")

        # For the first step, gather valid neighbors.
        valid_starts = []
        for first_neighbor in self.getAdjacentCells(*start):
            if first_neighbor in temp_board and temp_board[first_neighbor]:
                debug_log.append(f"First neighbor {first_neighbor} is occupied; skipping.")
                continue
            # Simulate moving the spider to this neighbor.
            temp_board.setdefault(first_neighbor, []).append(("SpiderDebug", "Spider"))
            if self.isBoardConnected(temp_board) and self.canSlide(start[0], start[1], first_neighbor[0], first_neighbor[1], temp_board):
                debug_log.append(f"First neighbor {first_neighbor} is a valid starting move.")
                valid_starts.append(first_neighbor)
            else:
                debug_log.append(f"First neighbor {first_neighbor} failed connectivity or sliding; not valid.")
            # Undo the temporary move.
            temp_board[first_neighbor].pop()
            if not temp_board[first_neighbor]:
                del temp_board[first_neighbor]

        debug_log.append(f"Valid first moves: {valid_starts}")

        # Run DFS from each valid first neighbor.
        for fn in valid_starts:
            debug_log.append(f"Starting DFS branch from first neighbor {fn}")
            dfs([start, fn], 1)

        # Print the entire debug log.
        for log in debug_log:
            print(log)
        print(f"Spider destinations found: {results}")

        return results


    def getSpiderDestinations(self, board, start):
        results = set()

        def dfs(path, steps):
            cur = path[-1]
            if steps == 3:
                results.add(cur)
                return

            for neighbor in self.getAdjacentCells(*cur):
                if neighbor in path:
                    continue
                if neighbor in temp_board and temp_board[neighbor]:
                    continue

                # Place spider
                temp_board.setdefault(neighbor, []).append(("SpiderOwner","Spider"))

                # Check connectivity and canSlide
                if self.isBoardConnected(temp_board) and self.canSlide(cur[0], cur[1], neighbor[0], neighbor[1], temp_board):
                    dfs(path + [neighbor], steps + 1)

                # **Backtrack**: remove the spider you just placed
                temp_board[neighbor].pop()
                if not temp_board[neighbor]:
                    del temp_board[neighbor]

        # "Lift" the spider from its start so that cell is empty:
        temp_board = {c: st[:] for c, st in board.items()}
        if start in temp_board and temp_board[start]:
            temp_board[start].pop()
            if not temp_board[start]:
                del temp_board[start]

        # For the first step away from 'start', check which adjacent cells
        # are empty and pass connectivity + sliding:
        valid_starts = []
        for first_neighbor in self.getAdjacentCells(*start):
            if first_neighbor in temp_board and temp_board[first_neighbor]:
                continue
            # Test connectivity with spider at 'first_neighbor':
            temp_board.setdefault(first_neighbor, []).append(("SpiderOwner","Spider"))
            if self.isBoardConnected(temp_board) and self.canSlide(start[0], start[1],
                                                                   first_neighbor[0], first_neighbor[1],
                                                                   temp_board):
                valid_starts.append(first_neighbor)
            # Undo so we can try the next neighbor
            temp_board[first_neighbor].pop()
            if not temp_board[first_neighbor]:
                del temp_board[first_neighbor]

        # Now run 3-step DFS from each valid first step
        for fn in valid_starts:
            dfs([start, fn], 1)

        return results

    # --- New: Revised Ant Move Generation ---
    def getAntDestinations(self, board, start):
        results = set()
        visited = {start}
        frontier = [start]
        while frontier:
            cur = frontier.pop(0)
            for neighbor in self.getAdjacentCells(*cur):
                # Skip if the neighbor is occupied.
                if neighbor in board and board[neighbor]:
                    continue
                # The destination must be adjacent to at least one piece.
                if not any(adj in board and board[adj] for adj in self.getAdjacentCells(*neighbor)):
                    continue
                # Only enforce the sliding condition for the first step out of the start.
                if cur == start and not self.canSlide(start[0], start[1], neighbor[0], neighbor[1], board):
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    results.add(neighbor)
                    frontier.append(neighbor)
        return results

    def canSlide(self, from_q, from_r, to_q, to_r, board, debug=False):
        key = (self.board_hash(board), from_q, from_r, to_q, to_r)
        if key in self._can_slide_cache:
            result = self._can_slide_cache[key]
            if debug:
                print(f"[DEBUG] canSlide({(from_q, from_r)} -> {(to_q, to_r)}) found in cache: {result}")
            return result

        if debug:
            print(f"[DEBUG] canSlide checking from {(from_q, from_r)} to {(to_q, to_r)}")

        dq = to_q - from_q
        dr = to_r - from_r
        move_dir = (dq, dr)

        adjacent_mapping = {
            (1, 0):  [(0, 1), (1, -1)],
            (0, 1):  [(-1, 1), (1, 0)],
            (-1, 1): [(0, 1), (-1, 0)],
            (-1, 0): [(-1, 1), (0, -1)],
            (0, -1): [(1, -1), (-1, 0)],
            (1, -1): [(1, 0), (0, -1)]
        }
        if move_dir not in adjacent_mapping:
            if debug:
                print(f"  [DEBUG] Invalid direction: {move_dir} not recognized.")
            self._can_slide_cache[key] = False
            return False

        adj_dirs = adjacent_mapping[move_dir]
        blocked_count = 0
        for adj_q, adj_r in adj_dirs:
            neighbor1 = (from_q + adj_q, from_r + adj_r)
            neighbor2 = (to_q + adj_q, to_r + adj_r)

            # If the piece is effectively "squeezed" by occupied neighbors on both sides,
            # sliding is blocked
            if (neighbor1 in board and board[neighbor1]) and (neighbor2 in board and board[neighbor2]):
                blocked_count += 1

        result = (blocked_count == 0)
        self._can_slide_cache[key] = result

        if debug:
            if result:
                print("  [DEBUG] Slide is possible.")
            else:
                print(f"  [DEBUG] Slide blocked because blocked_count={blocked_count}")

        return result

    def isBoardConnected(self, board, getAdjacentCells=None):
        """Checks if the board is connected (using Union-Find)."""
        key = self.board_hash(board)
        if key in self._connectivity_cache:
            return self._connectivity_cache[key]

        occupied_cells = [cell for cell, stack in board.items() if stack]
        if not occupied_cells:
            self._connectivity_cache[key] = True
            return True

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
                # Handle both 3-element and 4-element move tuples.
                if len(action) == 3:
                    _, (fq, fr), (tq, tr) = action
                elif len(action) == 4:
                    _, (fq, fr), (tq, tr), path = action
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

        # PASS action
        if action[0] == "PASS":
            new_state["current_player"] = self.getOpponent(player)
            new_state["move_number"] += 1
            self.clearCaches()
            return new_state

        # PLACE action
        if action[0] == "PLACE":
            _, insectType, (q, r) = action
            new_state["pieces_in_hand"][player][insectType] -= 1
            if (q, r) not in board:
                board[(q, r)] = []
            board[(q, r)].append((player, insectType))

        # MOVE action
        elif action[0] == "MOVE":
            # Check if action contains a full path (i.e. 4 elements)
            if len(action) == 3:
                _, (fq, fr), (tq, tr) = action
            elif len(action) == 4:
                # Unpack the full spider move, ignoring the intermediate path.
                _, (fq, fr), (tq, tr), path = action
            else:
                raise ValueError(f"Unexpected MOVE action format: {action}")

            if (fq, fr) not in board or not any(p[0] == player for p in board[(fq, fr)]):
                raise ValueError(f"Invalid move: {action} from state: {state}\n")

            piece = board[(fq, fr)].pop()
            if len(board[(fq, fr)]) == 0:
                del board[(fq, fr)]
            if (tq, tr) not in board:
                board[(tq, tr)] = []
            board[(tq, tr)].append(piece)

        # Switch to the other player
        new_state["current_player"] = self.getOpponent(player)
        new_state["move_number"] += 1

        self.clearCaches()
        return new_state

    def evaluateState(self, state):
        """
        A more nuanced Hive heuristic.
        """
        outcome = self.getGameOutcome(state)
        if outcome is not None:
            if outcome == self.getOpponent(state["current_player"]):
                # The player who just moved (the opponent of current) is the winner
                return +10000
            elif outcome == "Draw":
                return 0
            else:
                # Then the current_player must be the winner
                # because your outcome-check says the other side lost
                return -10000


        board = state["board"]
        p1_queen_pos = find_queen_position(board, "Player1")
        p2_queen_pos = find_queen_position(board, "Player2")

        # 1. Queen Surrounding & Liberties
        p1_liberties = p2_liberties = 0
        p1_surround_count = 0
        p2_surround_count = 0

        if p1_queen_pos:
            p1_liberties = sum(
                1 for nq, nr in self.getAdjacentCells(*p1_queen_pos)
                if (nq, nr) not in board or not board[(nq, nr)]
            )
            p1_surround_count = sum(
                1 for nq, nr in self.getAdjacentCells(*p1_queen_pos)
                if (nq, nr) in board and board[(nq, nr)]
            )

        if p2_queen_pos:
            p2_liberties = sum(
                1 for nq, nr in self.getAdjacentCells(*p2_queen_pos)
                if (nq, nr) not in board or not board[(nq, nr)]
            )
            p2_surround_count = sum(
                1 for nq, nr in self.getAdjacentCells(*p2_queen_pos)
                if (nq, nr) in board and board[(nq, nr)]
            )

        # Freedoms vs. being surrounded
        # Weighted so that being near fully surrounded is severe.
        # e.g.  Each queen has 6 adjacent cells in a typical hex setting.
        # We can scale the difference in "how close each queen is to being fully surrounded."
        queen_factor = 50
        queen_score = queen_factor * ((p1_surround_count - p2_surround_count))

        # Add in the difference in liberties as well.
        # If your queen has more free neighbors than theirs, that is good for you.
        # We'll scale by 10 for example.
        liberties_factor = 10
        queen_score += liberties_factor * (p1_liberties - p2_liberties)

        # 2. Mobility & Pinning
        p1_movable_pieces = self.countMovablePieces(board, "Player1")
        p2_movable_pieces = self.countMovablePieces(board, "Player2")
        mobility_factor = 3
        mobility_score = mobility_factor * (p1_movable_pieces - p2_movable_pieces)

        # 3. Early-Game Placement Bonus or Penalty
        # This is still a decent approach, but let's keep it simpler:
        p1_placed = sum(1 for _, stack in board.items() for (owner, _) in stack if owner == "Player1")
        p2_placed = sum(1 for _, stack in board.items() for (owner, _) in stack if owner == "Player2")
        move_number = state["move_number"]

        # We'll do a small bonus for placing more pieces in the early game
        # so as to avoid pass-happy or do-nothing strategies.
        early_factor = 2
        early_game_bonus = 0
        if move_number < 10:
            early_game_bonus = early_factor * (p1_placed - p2_placed)

        # 4. Combine
        score = queen_score + mobility_score + early_game_bonus

        # Return from perspective of the current player
        if state["current_player"] == "Player2":
            return -score
        else:
            return score


    def countMovablePieces(self, board, player):
        """
        Count how many pieces the player can actually move without breaking
        connectivity. (This is different than just summing empty neighbors.)
        """
        movable_count = 0
        for (q, r), stack in board.items():
            if stack and stack[-1][0] == player:
                # Temporarily remove piece
                temp_board = {coord: st[:] for coord, st in board.items()}
                piece = temp_board[(q, r)].pop()
                if not temp_board[(q, r)]:
                    del temp_board[(q, r)]
                # Check connectivity
                if not self.isBoardConnected(temp_board):
                    # This piece is pinned - cannot move at all
                    continue
                # If still connected, see if there's at least one valid move
                # that is possible for that piece.
                moves = self.getSinglePieceMoves(temp_board, (q, r), piece)
                if moves:
                    movable_count += 1

        return movable_count


    def getSinglePieceMoves(self, board, from_coord, piece):
        """
        Return a list of valid (from_coord, to_coord) for the single piece
        ignoring the entire player's set, just focusing on this piece's move logic.
        """
        # Re-insert piece so movement logic uses an accurate board.
        if from_coord not in board:
            board[from_coord] = []
        board[from_coord].append(piece)
        # Then reuse your normal move logic but restrict it to just that single piece:
        # We'll do something like a special version of "movePieceActions" that
        # only returns moves for that piece.
        # For brevity, re-use the code from `movePieceActions` but filter by from_coord.
        # ...
        # Remove piece again at the end or just copy board in the beginning.
        #
        # (You can adapt your `movePieceActions` routine.)

        # For a short example:
        insectType = piece[1]
        (q, r) = from_coord
        board_copy = {coord: st[:] for coord, st in board.items()}
        board_copy[(q, r)].pop()  # remove the piece
        if not board_copy[(q, r)]:
            del board_copy[(q, r)]

        if insectType == "Queen" or insectType == "Beetle":
            destinations = self.getAdjacentCells(q, r)
        elif insectType == "Grasshopper":
            destinations = self.getGrasshopperJumps(board_copy, q, r)
        elif insectType == "Spider":
            destinations = self.getSpiderDestinations(board_copy, (q, r))
        elif insectType == "Ant":
            destinations = self.getAntDestinations(board_copy, (q, r))
        else:
            destinations = []

        valid_moves = []
        for to_coord in destinations:
            # Check occupancy, connectivity, etc.
            to_q, to_r = to_coord
            # If it's a beetle, you can land on top of an occupied cell.
            # Otherwise you cannot land on a cell that is occupied.
            if insectType != "Beetle" and to_coord in board_copy and board_copy[to_coord]:
                continue

            # Possibly check sliding if needed.
            # Then "simulate" placing the piece there.
            board_copy.setdefault(to_coord, []).append(piece)
            if self.isBoardConnected(board_copy):
                valid_moves.append((from_coord, to_coord))
            # Undo
            board_copy[to_coord].pop()
            if not board_copy[to_coord]:
                del board_copy[to_coord]

        return valid_moves

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