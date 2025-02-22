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

            # Inside movePieceActions, after computing 'destinations'
            for (tq, tr) in destinations:
                # For Beetle and Grasshopper moves, we don’t need to check sliding.
                # For Queen moves, we want to enforce sliding.
                # For Spider and Ant moves, the DFS should already have validated sliding for each step.
                if insectType == "Queen":
                    if not self.canSlide(q, r, tq, tr, temp_board, debug=debug):
                        if debug:
                            print(f"      [DEBUG] Slide from {(q, r)} to {(tq, tr)} is blocked, skipping.")
                        continue

                # (Optionally, if you want extra safety for Spider/Ant moves that lack a full DFS path,
                #  you might check that the move is adjacent—but that’s not how their movement works.)

                # Check occupancy and connectivity as before.
                if insectType != "Beetle" and (tq, tr) in temp_board and temp_board[(tq, tr)]:
                    if debug:
                        print(f"      [DEBUG] Destination {(tq, tr)} is occupied, skipping (only Beetle can stack).")
                    continue

                temp_board.setdefault((tq, tr), []).append(piece)
                if not self.isBoardConnected(temp_board):
                    temp_board[(tq, tr)].pop()  # Undo
                    if not temp_board[(tq, tr)]:
                        del temp_board[(tq, tr)]
                    if debug:
                        print(f"      [DEBUG] Move to {(tq, tr)} breaks hive connectivity afterwards, skipping.")
                    continue

                # Undo simulation and record the action.
                temp_board[(tq, tr)].pop()
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

    def getSpiderDestinations(self, board, start):
        results = set()

        # Create a temporary board without the Spider
        temp_board = {coord: st[:] for coord, st in board.items()}
        if start in temp_board and temp_board[start]:
            temp_board[start].pop()
            if not temp_board[start]:
                del temp_board[start]

        # Check if the hive remains connected without the Spider
        if not self.isBoardConnected(temp_board):
            return results  # Spider is pinned

        def dfs(current, path, steps):
            if steps == 3:
                # Simulate placing the Spider at 'current' and check connectivity
                temp_board[current] = [("Player1", "Spider")]
                if self.isBoardConnected(temp_board):
                    results.add(current)
                del temp_board[current]
                return

            # Get adjacent cells
            for neighbor in self.getAdjacentCells(current[0], current[1]):
                if neighbor in path:
                    continue  # Avoid backtracking
                # Ensure the neighbor is empty
                if neighbor not in temp_board or not temp_board[neighbor]:
                    # Ensure the neighbor is adjacent to the hive
                    if any(adj in temp_board and temp_board[adj] for adj in self.getAdjacentCells(neighbor[0], neighbor[1])):
                        # Check if sliding is possible
                        if self.canSlide(current[0], current[1], neighbor[0], neighbor[1], temp_board):
                            dfs(neighbor, path + [neighbor], steps + 1)

        # Start DFS from the starting position
        dfs(start, [start], 0)
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
        """
        Determines if a piece can slide from (from_q, from_r) to (to_q, to_r).
        The move is possible if at least one of the two common neighbors is empty.
        """
        # Find the two common neighbors
        adj_from = set(self.getAdjacentCells(from_q, from_r))
        adj_to = set(self.getAdjacentCells(to_q, to_r))
        common_neighbors = adj_from & adj_to
        # Remove from and to if they are in the sets
        common_neighbors.discard((from_q, from_r))
        common_neighbors.discard((to_q, to_r))
        if len(common_neighbors) != 2:
            if debug:
                print(f"[DEBUG] Unexpected number of common neighbors: {common_neighbors}")
            return False  # Should not happen for adjacent cells

        # Check if both common neighbors are occupied
        both_occupied = all((n in board and board[n]) for n in common_neighbors)
        result = not both_occupied
        if debug:
            if result:
                print(f"[DEBUG] Slide from {(from_q, from_r)} to {(to_q, to_r)} is possible.")
            else:
                print(f"[DEBUG] Slide from {(from_q, from_r)} to {(to_q, to_r)} is blocked.")
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

    def randomActionChoice(self, state, actions):
        """
        Simply selects a random action from the provided actions list.
        """
        return random.choice(actions)


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

        new_state["move_number"] += 1


        # if not self.isTerminal(new_state):
        new_state["current_player"] = self.getOpponent(player)

        self.clearCaches()
        return new_state

    def evaluateState(self, perspectivePlayer, state, weights=None):
        """
        Evaluate the board state from the perspective of the current player.

        The heuristic is broken into 4 parts:
          1. Queen Surrounding & Liberties
          2. Mobility & Pinning
          3. Early-Game Placement Bonus or Penalty
          4. Combine the scores

        The final score is returned from the perspective of the current player.

        You can pass in a dictionary 'weights' to adjust the relative importance of each factor.
        Expected keys (with default values) are:
          - queen_factor: default 50
          - liberties_factor: default 10
          - mobility_factor: default 3
          - early_factor: default 2
        """
        # Set default weights if none are provided
        default_weights = {
            "queen_factor": 50,
            "liberties_factor": 10,
            "mobility_factor": 3,
            "early_factor": 2,
        }

        if weights is None:
            weights = default_weights
        else:
            # Merge provided weights with defaults
            for key, value in default_weights.items():
                weights.setdefault(key, value)

        # Assert that all necessary weights exist and are numeric
        for key in default_weights.keys():
            assert key in weights, f"Missing weight: {key}"
            assert isinstance(weights[key], (int, float)), f"Weight '{key}' must be an int or float."

        # Check for a terminal outcome
        outcome = self.getGameOutcome(state)
        if outcome is not None:
            if outcome == perspectivePlayer:
                # The player who just moved (the opponent of current) is the winner
                return 10000
            elif outcome == "Draw":
                return 0
            else:
                # Then the current_player must be the winner
                return -10000

        board = state["board"]
        # Assert board is a dictionary.
        assert isinstance(board, dict), "Board must be a dictionary."

        p1_queen_pos = find_queen_position(board, perspectivePlayer)
        p2_queen_pos = find_queen_position(board, self.getOpponent(perspectivePlayer))

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

        # Calculate queen score
        queen_score = weights["queen_factor"] * (p1_surround_count - p2_surround_count)
        queen_score += weights["liberties_factor"] * (p1_liberties - p2_liberties)

        # Assertion: If neither queen is found, then queen_score should be 0.
        if not p1_queen_pos and not p2_queen_pos:
            assert queen_score == 0, "Expected queen_score to be 0 when no queens are present."

        # 2. Mobility & Pinning
        current_player_movable = self.countMovablePieces(board, perspectivePlayer)
        opponent_movable_pieces = self.countMovablePieces(board, self.getOpponent(perspectivePlayer))

        # Assert that the counts are non-negative integers.
        assert isinstance(current_player_movable, int) and current_player_movable >= 0, \
            "Movable pieces count for current player should be a non-negative integer."
        assert isinstance(opponent_movable_pieces, int) and opponent_movable_pieces >= 0, \
            "Movable pieces count for opponent should be a non-negative integer."

        mobility_diff = current_player_movable - opponent_movable_pieces
        mobility_score = weights["mobility_factor"] * mobility_diff

        # Assertions regarding the mobility score's sign:
        if mobility_diff < 0:
            assert mobility_score < 0, (
                f"Expected negative mobility_score when current player's movable pieces are fewer than opponent's, "
                f"got {mobility_score}"
            )
        elif mobility_diff > 0:
            assert mobility_score > 0, (
                f"Expected positive mobility_score when current player's movable pieces exceed opponent's, "
                f"got {mobility_score}"
            )
        else:
            assert mobility_score == 0, "Expected mobility_score to be 0 when both players have equal movable pieces."

        # 3. Early-Game Placement Bonus or Penalty
        current_player_placed = sum(1 for _, stack in board.items() for (owner, _) in stack if owner == perspectivePlayer)
        early_game_bonus = weights["early_factor"] * current_player_placed

        # Assert that the early game bonus is non-negative.
        assert early_game_bonus >= 0, "Early game bonus should be non-negative."

        # 4. Combine the factors into the final score
        score = queen_score + mobility_score + early_game_bonus

        # Assertion: Final score should equal the sum of its individual components.
        computed_score = queen_score + mobility_score + early_game_bonus
        assert score == computed_score, "Final score does not match the sum of its individual components."

        # Return score from the perspective of the current player.
        # If current_player is "Player2", you might want to flip the sign (commented out below):
        # if state["current_player"] == "Player2":
        #     return -score
        # else:
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

    def simulateRandomPlayout(self, state, perspectivePlayer, max_depth=10, eval_func=None, weights=None):
        """
        Plays a random (or weighted-random) sequence of moves up to max_depth,
        then returns a numeric evaluation of the final position.
        """
        temp_state = self.copyState(state)
        depth = 0
        while not self.isTerminal(temp_state) and depth < max_depth:
            legal = self.getLegalActions(temp_state)
            if not legal:
                break
            # Weighted random or any style of random move
            action = self.randomActionChoice(temp_state, legal)
            temp_state = self.applyAction(temp_state, action)
            depth += 1

        evaluation_score = self.evaluateState(perspectivePlayer, temp_state, weights=weights)
        if self.isTerminal(temp_state) and self.isQueenSurrounded(temp_state, self.getOpponent(perspectivePlayer)):
            assert evaluation_score > 0

        if self.isTerminal(temp_state) and self.isQueenSurrounded(temp_state, perspectivePlayer):
            assert evaluation_score < 0

        return evaluation_score

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