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
        self.in_move_generation = False


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
        self.in_move_generation = True
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
                if debug and insectType == "Queen":
                    print(f"    [DEBUG] Skipping Queen at {(q, r)}. Removing this piece breaks hive connectivity (pinned).")
                if debug and insectType != "Queen":
                    print("    [DEBUG] Skipping. Removing this piece breaks hive connectivity (pinned).")
                continue
            else:
                if debug and insectType == "Queen":
                    print(f"    [DEBUG] Lifting Queen at {(q, r)} does NOT break the hive.")

            # 2. Get possible destinations based on insectType
            if insectType == "Queen" or insectType == "Beetle":
                destinations = self.getAdjacentCells(q, r)
            elif insectType == "Grasshopper":
                destinations = self.getGrasshopperJumps(temp_board, q, r)
            elif insectType == "Spider":
                destinations = self.getSpiderDestinations(temp_board, (q, r), piece[0])
            elif insectType == "Ant":
                destinations = self.getAntDestinations(temp_board, (q, r))
            else:
                destinations = []

            if debug and insectType == "Queen":
                print(f"    [DEBUG] Potential destinations for Queen at {(q, r)}: {list(destinations)}")

            # Evaluate each destination
            for (tq, tr) in destinations:
                # For Queen moves, enforce sliding
                if insectType == "Queen":
                    if not self.canSlide(q, r, tq, tr, temp_board, debug=debug):
                        if debug:
                            print(f"      [DEBUG] Slide from {(q, r)} to {(tq, tr)} is blocked for Queen, skipping.")
                        continue
                    else:
                        if debug:
                            print(f"      [DEBUG] Slide from {(q, r)} to {(tq, tr)} is possible for Queen.")

                # Check occupancy (Queen can’t stack)
                if insectType != "Beetle" and (tq, tr) in temp_board and temp_board[(tq, tr)]:
                    if debug and insectType == "Queen":
                        print(f"      [DEBUG] Destination {(tq, tr)} is occupied, skipping for Queen (only Beetle can stack).")
                    if debug and insectType != "Queen":
                        print(f"      [DEBUG] Destination {(tq, tr)} is occupied, skipping (only Beetle can stack).")
                    continue

                # Simulate move and check connectivity
                temp_board.setdefault((tq, tr), []).append(piece)
                if not self.isBoardConnected(temp_board):
                    temp_board[(tq, tr)].pop()  # Undo
                    if not temp_board[(tq, tr)]:
                        del temp_board[(tq, tr)]
                    if debug and insectType == "Queen":
                        print(f"      [DEBUG] Move to {(tq, tr)} breaks hive connectivity for Queen, skipping.")
                    if debug and insectType != "Queen":
                        print(f"      [DEBUG] Move to {(tq, tr)} breaks hive connectivity afterwards, skipping.")
                    continue
                else:
                    if debug and insectType == "Queen":
                        print(f"      [DEBUG] Move to {(tq, tr)} maintains hive connectivity for Queen.")

                # Undo simulation and record the action
                temp_board[(tq, tr)].pop()
                if not temp_board[(tq, tr)]:
                    del temp_board[(tq, tr)]
                actions.append(("MOVE", (q, r), (tq, tr)))
                if debug and insectType == "Queen":
                    print(f"      [DEBUG] Valid move for Queen => ({(q, r)} -> {(tq, tr)})")

        if debug:
            print(f"[DEBUG] Total legal moves for {player}: {actions}")

        self.in_move_generation = False
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

    def getSpiderDestinations(self, board, start, owner):
        """
        Return every hex the spider (three‑step crawler) can end on, obeying
        sliding and hive‑connectivity rules.  Works whether `board` already has
        the spider removed or not.
        """
        results         = set()
        original_owner  = owner
        temp            = {c: s[:] for c, s in board.items()}

        # Ensure the spider is *not* on the board copy
        if start in temp and temp[start]:
            temp[start].pop()
            if not temp[start]:
                del temp[start]

        # If taking the spider off breaks the hive it is pinned
        if not self.isBoardConnected(temp):
            return results

        def dfs(cur, path, depth):
            if depth == 3:
                temp[cur] = [(original_owner, "Spider")]
                if self.isBoardConnected(temp):
                    results.add(cur)
                del temp[cur]
                return

            for nb in self.getAdjacentCells(*cur):
                if nb in path:                          # no U‑turns / back‑tracking
                    continue
                if nb in temp and temp[nb]:             # must land on empty hex
                    continue
                if not any((adj in temp and temp[adj])  # destination must touch hive
                           for adj in self.getAdjacentCells(*nb)):
                    continue

                # Put the spider back on its current hex so canSlide can evaluate gap
                temp[cur] = [(original_owner, "Spider")]
                if self.canSlide(cur[0], cur[1], nb[0], nb[1], temp):
                    del temp[cur]                       # lift it again for recursion
                    dfs(nb, path + [nb], depth + 1)

        dfs(start, [start], 0)
        return results

    # --- New: Revised Ant Move Generation ---
    def getAntDestinations(self, board, start):
        """
        Flood‑fill all empty hexes reachable by the ant while
        *respecting the slide rule on **every** step*.
        """
        visited   = {start}
        frontier  = [start]
        results   = set()

        while frontier:
            cq, cr = frontier.pop(0)
            for nq, nr in self.getAdjacentCells(cq, cr):
                dest = (nq, nr)

                # 1️⃣ destination must be empty
                if dest in board and board[dest]:
                    continue

                # 2️⃣ destination must touch the hive
                if not any((aq, ar) in board and board[(aq, ar)]
                           for aq, ar in self.getAdjacentCells(*dest)):
                    continue

                # 3️⃣ slide rule at *this* step
                if not self.canSlide(cq, cr, nq, nr, board):
                    continue

                if dest not in visited:
                    visited.add(dest)
                    results.add(dest)
                    frontier.append(dest)
        return results

    def canSlide(self, from_q, from_r, to_q, to_r, board, debug=False):
        """
        True ⇢ the gap between (from_q,from_r) and (to_q,to_r) is wide enough.
        Rule: of the two flanking hexes that touch *both* cells, at least one must be empty.
              If the edge only has one such hex (happens on the rim of the hive),
              sliding is always allowed.
        """
        adj_from = set(self.getAdjacentCells(from_q, from_r))
        adj_to   = set(self.getAdjacentCells(to_q, to_r))
        common   = (adj_from & adj_to) - {(from_q, from_r), (to_q, to_r)}

        # Rim case: less than two flanking hexes ⇒ nothing can pinch you.
        if len(common) < 2:
            return True

        blocked = all((c in board and board[c]) for c in common)
        if debug:
            status = "blocked" if blocked else "open"
            print(f"[DEBUG] slide {status} {from_q,from_r} → {to_q,to_r} via {tuple(common)}")
        return not blocked

    def isBoardConnected(self, board, getAdjacentCells=None):
        """Checks if the board is connected (using Union-Find)."""
        key = self.board_hash(board)
        if key in self._connectivity_cache and not self.in_move_generation:
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

    # ------------------------------------------------------------------
    # New heuristic
    # ------------------------------------------------------------------
    def evaluateState(self, perspectivePlayer, state, weights=None):
        """
        Static evaluation of `state` from `perspectivePlayer`.

        Feature list (all signed from my point of view):
        -------------------------------------------------
        1. queen_ring_diff        : (# enemy stones around their queen)
                                    - (# my stones around my queen)

        2. liberty_diff           : (enemy queen liberties) - (my queen liberties)

        3. distance_pressure      : Σ (piece_value / (1+dist_to_enemy_Q))
                                    - Σ_enemy (piece_value / (1+dist_to_my_Q))

        4. mobility_diff          : movable_top_pieces(my) - movable_top_pieces(enemy)

        5. development_bonus      : min(my_tiles, 4) * taper  (taper vanishes after move 8)

        All coefficients live in *weights* so you can tune them in the JSON.
        """

        # ---------- default coefficients ----------
        base = dict(
            queen_ring_factor      = 60.0,
            liberty_factor         = 15.0,
            distance_pressure      = 25.0,
            mobility_factor        = 5.0,
            development_factor     = 6.0,
            dev_taper_moves        = 8        # fade bonus from move 4 to move 12
        )
        if weights is None:
            weights = base
        else:
            for k, v in base.items():
                weights.setdefault(k, v)

        # ---------- terminal outcome ----------
        outcome = self.getGameOutcome(state)
        if outcome is not None:
            return  10000 if outcome == perspectivePlayer else \
                -10000 if outcome not in (None, "Draw") else 0

        board = state["board"]
        me, opp = perspectivePlayer, self.getOpponent(perspectivePlayer)
        my_q  = find_queen_position(board, me)
        op_q  = find_queen_position(board, opp)

        # ---------- helper: surround count & liberties ----------
        def surround_liberties(pos):
            if pos is None:
                return 0, 0
            ring = lib = 0
            for nq, nr in self.getAdjacentCells(*pos):
                if (nq, nr) in board and board[(nq, nr)]:
                    ring += 1
                else:
                    lib  += 1
            return ring, lib

        my_ring, my_lib = surround_liberties(my_q)
        op_ring, op_lib = surround_liberties(op_q)

        # ---------- distance‑weighted pressure ----------
        PIECE_VALUE = dict(Queen=0, Beetle=2, Spider=1.5, Ant=1, Grasshopper=1)
        my_press = op_press = 0.0
        for (q, r), stack in board.items():
            if not stack:
                continue
            owner, typ = stack[-1]
            val = PIECE_VALUE.get(typ, 1)
            if owner == me and op_q is not None:
                d = hex_distance(q, r, *op_q)
                my_press += val / (1 + d)
            elif owner == opp and my_q is not None:
                d = hex_distance(q, r, *my_q)
                op_press += val / (1 + d)

        # ---------- mobility ----------
        my_mob  = self.countMovablePieces(board, me)
        op_mob  = self.countMovablePieces(board, opp)

        # ---------- development (early) ----------
        my_tiles = sum(1 for _, s in board.items() for o, _ in s if o == me)
        move_no  = state["move_number"]
        taper    = max(0.0, 1.0 - max(0, move_no - 4) / weights["dev_taper_moves"])
        dev_bonus = weights["development_factor"] * min(my_tiles, 4) * taper

        # ---------- aggregate ----------
        score = 0.0
        score += weights["queen_ring_factor"] * (op_ring - my_ring)
        score += weights["liberty_factor"]    * (op_lib  - my_lib)
        score += weights["distance_pressure"] * (my_press - op_press)
        score += weights["mobility_factor"]   * (my_mob   - op_mob)
        score += dev_bonus
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
            destinations = self.getSpiderDestinations(board_copy, (q, r), piece[0])
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

    def simulateRandomPlayout(self, state, perspectivePlayer, max_depth=1000, eval_func=None, weights=None):
        temp_state = self.copyState(state)
        depth = 0
        while not self.isTerminal(temp_state) and depth < max_depth:
            legal = self.getLegalActions(temp_state)
            if not legal:
                break
            action = self.randomActionChoice(temp_state, legal)
            temp_state = self.applyAction(temp_state, action)
            depth += 1
        outcome = self.getGameOutcome(temp_state)
        if outcome == perspectivePlayer:
            return 1.0
        elif outcome == "Draw":
            return 0.0
        else:
            return -1.0
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