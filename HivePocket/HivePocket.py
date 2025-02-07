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

    def movePieceActions(self, state):
        board = state["board"]
        player = state["current_player"]
        actions = []

        player_cells = {(q, r, stack[-1][1]) for (q, r), stack in board.items() if stack and stack[-1][0] == player}

        def board_copy(board):
            return {coord: stack[:] for coord, stack in board.items()}

        # print(f"movePieceActions: Player = {player}")  # Log current player

        for (q, r, insectType) in player_cells:
            temp_board = board_copy(board)
            # print(f"  Considering piece: ({q}, {r}), {insectType}")  # Log piece
            # print(f"    temp_board before removal: {temp_board}")

            piece = temp_board[(q, r)].pop()
            if not temp_board[(q, r)]:
                del temp_board[(q, r)]

            if not self.isBoardConnected(temp_board):
                # print(f"    Removing ({q}, {r}) disconnects the hive. Skipping.")
                continue

            if insectType == "Queen":
                destinations = self.getAdjacentCells(q, r)
            elif insectType == "Beetle":
                destinations = self.getAdjacentCells(q, r)
            elif insectType == "Grasshopper":
                destinations = self.getGrasshopperJumps(temp_board, q, r)
            elif insectType == "Spider":
                destinations = self.getSpiderDestinations(temp_board, (q, r))
            elif insectType == "Ant":
                destinations = self.getAntDestinations(temp_board, (q, r))
            else:
                destinations = []

            # print(f"    Destinations: {destinations}")

            for to_q, to_r in destinations:
                # --- CORRECTED OCCUPANCY CHECK ---
                if insectType != "Beetle" and (to_q, to_r) in temp_board and temp_board[(to_q, to_r)]:
                    continue

                if insectType not in ["Beetle", "Grasshopper", "Ant", "Spider"] and not self.canSlide(q, r, to_q, to_r, temp_board):
                    # print(f"    canSlide({q}, {r}, {to_q}, {to_r}) returned False. Skipping.")
                    continue

                # Inside your loop over possible destinations:
                temp_board.setdefault((to_q, to_r), []).append(piece)
                if not self.isBoardConnected(temp_board):
                    # The move disconnects the hive, so undo and skip this move.
                    temp_board[(to_q, to_r)].pop()
                    if not temp_board.get((to_q, to_r)):
                        temp_board.pop((to_q, to_r), None)
                    continue  # Skip adding this move.
                # If we get here, the move is valid.
                actions.append(("MOVE", (q, r), (to_q, to_r)))
                temp_board[(to_q, to_r)].pop()
                if not temp_board.get((to_q, to_r)):
                    temp_board.pop((to_q, to_r), None)


        # print(f"movePieceActions returning: {actions}")  # Log returned actions
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
                if neighbor in board and board[neighbor]:  # Corrected occupancy check
                    continue
                if not any(adj in board and board[adj] for adj in self.getAdjacentCells(*neighbor)):
                    continue
                if not self.canSlide(cur[0], cur[1], neighbor[0], neighbor[1], board):
                    continue
                dfs(path + [neighbor], steps + 1)

        # Check sliding for the *initial* move from 'start'.
        valid_starts = []
        for first_neighbor in self.getAdjacentCells(*start):
            if first_neighbor in board and board[first_neighbor]: #Corrected initial check
                continue
            if not any(adj in board and board[adj] for adj in self.getAdjacentCells(*first_neighbor)):
                continue
            if self.canSlide(start[0], start[1], first_neighbor[0], first_neighbor[1], board):
                valid_starts.append(first_neighbor)

        for first_step in valid_starts:
            dfs([start, first_step], 1)  # Start DFS from each valid first step

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

    def canSlide(self, from_q, from_r, to_q, to_r, board):
        # Include move parameters in the cache key.
        key = (self.board_hash(board), from_q, from_r, to_q, to_r)
        if key in self._can_slide_cache:
            return self._can_slide_cache[key]

        dq = to_q - from_q
        dr = to_r - from_r
        move_dir = (dq, dr)
        adjacent_mapping = {
            (1, 0): [(0, 1), (1, -1)],
            (0, 1): [(-1, 1), (1, 0)],
            (-1, 1): [(0, 1), (-1, 0)],
            (-1, 0): [(-1, 1), (0, -1)],
            (0, -1): [(1, -1), (-1, 0)],
            (1, -1): [(1, 0), (0, -1)]
        }
        if move_dir not in adjacent_mapping:
            self._can_slide_cache[key] = False
            return False

        adj_dirs = adjacent_mapping[move_dir]
        blocked_count = 0
        for adj_q, adj_r in adj_dirs:
            neighbor1 = (from_q + adj_q, from_r + adj_r)
            neighbor2 = (to_q + adj_q, to_r + adj_r)
            if (neighbor1 in board and board[neighbor1]) and (neighbor2 in board and board[neighbor2]):
                blocked_count += 1

        result = blocked_count == 0
        self._can_slide_cache[key] = result
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
            # Just switch the current player, update move_number, clear caches
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
            _, (fq, fr), (tq, tr) = action
            if (fq, fr) not in board or not any(p[0] == player for p in board[(fq, fr)]):
                raise ValueError(
                    f"Invalid move: {action} from state: {state}\n"
                )

            piece = board[(fq, fr)].pop()
            if len(board[(fq, fr)]) == 0:
                del board[(fq, fr)]
            if (tq, tr) not in board:
                board[(tq, tr)] = []
            board[(tq, tr)].append(piece)

        # Switch to the other player
        new_state["current_player"] = self.getOpponent(player)
        new_state["move_number"] += 1

        # Clear caches so subsequent calls see the updated board
        self.clearCaches()
        return new_state

    def evaluateState(self, state):
        outcome = self.getGameOutcome(state)
        if outcome is not None:
            current_player = state["current_player"]
            if outcome == current_player:
                return +10000
            elif outcome == "Draw":
                return 0
            else:
                return -10000

        board = state["board"]
        p1_queen = find_queen_position(board, "Player1")
        p2_queen = find_queen_position(board, "Player2")

        p1_liberties = sum(1 for neighbor in self.getAdjacentCells(*p1_queen) if neighbor not in board or not board[neighbor]) if p1_queen else 0
        p2_liberties = sum(1 for neighbor in self.getAdjacentCells(*p2_queen) if neighbor not in board or not board[neighbor]) if p2_queen else 0

        score = (p1_liberties - p2_liberties) * 20

        # --- Piece Mobility (Simplified Example) ---
        p1_mobility = 0
        p2_mobility = 0
        for (q, r), stack in board.items():
            if stack:
                top_piece = stack[-1]
                if top_piece[0] == "Player1" and top_piece[1] != "Queen":
                    # Very basic mobility count: number of empty neighbors.
                    p1_mobility += sum(1 for nq, nr in self.getAdjacentCells(q, r) if (nq, nr) not in board or not board[(nq, nr)])
                elif top_piece[0] == "Player2" and top_piece[1] != "Queen":
                    p2_mobility += sum(1 for nq, nr in self.getAdjacentCells(q, r) if (nq, nr) not in board or not board[(nq, nr)])

        score += (p1_mobility - p2_mobility) * 5

        # --- Threats to Queen (Simplified) ---
        p1_threats = 0
        p2_threats = 0
        if p1_queen:
            for nq, nr in self.getAdjacentCells(*p1_queen):
                if (nq, nr) in board and board[(nq, nr)] and board[(nq, nr)][-1][0] == "Player2":
                    p1_threats += 1
        if p2_queen:
            for nq, nr in self.getAdjacentCells(*p2_queen):
                if (nq, nr) in board and board[(nq, nr)] and board[(nq, nr)][-1][0] == "Player1":
                    p2_threats += 1
        score += (p2_threats - p1_threats) * 10


        # --- Encourage Piece Placement (Early Game) ---
        p1_placed = sum(1 for _, stack in board.items() for piece, _ in stack if piece == "Player1")
        p2_placed = sum(1 for _, stack in board.items() for piece, _ in stack if piece == "Player2")

        # Scale the placement bonus by the move number (early game emphasis)
        move_number = state["move_number"]
        placement_bonus = 0
        if move_number < 10:  # First 5 turns for each player
            placement_bonus = (p1_placed - p2_placed) * 30 * (10-move_number)/10 # More important in earlier turns.
        score += placement_bonus

        return score

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