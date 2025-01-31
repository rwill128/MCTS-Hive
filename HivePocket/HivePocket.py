import random


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
        # Standard set for each player (no expansions):
        # 1x Queen, 2x Spider, 2x Beetle, 3x Grasshopper, 3x Ant
        self.INITIAL_PIECES = {
            "Queen": 1,
            "Spider": 2,
            "Beetle": 2,
            "Grasshopper": 3,
            "Ant": 3
        }
        self.DIRECTIONS = [
            (1, 0), (-1, 0), (0, 1),
            (0, -1), (1, -1), (-1, 1)
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
                    for (nq, nr) in self.getAdjacentCells(q, r):
                        if (nq, nr) not in board or len(board[(nq, nr)]) == 0:
                            return False
                    return True
        return False

    # ---------------------------------------------------------
    # 3. Action generation
    # ---------------------------------------------------------
    def getLegalActions(self, state):
        actions = self.placePieceActions(state) + self.movePieceActions(state)
        current_player = state["current_player"]
        if find_queen_position(state["board"], current_player) is None:
            move_count = state["move_number"] // 2 + 1
            if move_count >= 3:
                queen_actions = [action for action in actions if action[0] == "PLACE" and action[1] == "Queen"]
                return queen_actions
        return actions

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
        """
        board = state["board"]
        player = state["current_player"]
        actions = []
        player_cells = []
        for (q, r), stack in board.items():
            if stack and stack[-1][0] == player:
                insectType = stack[-1][1]
                player_cells.append((q, r, insectType))
        for (q, r, insectType) in player_cells:
            if insectType == "Queen":
                piece = board[(q, r)].pop()
                if len(board[(q, r)]) == 0:
                    del board[(q, r)]
                for (nq, nr) in self.getAdjacentCells(q, r):
                    if (nq, nr) in board and len(board[(nq, nr)]) > 0:
                        continue
                    if not self.canSlide(q, r, nq, nr, board):
                        continue
                    board.setdefault((nq, nr), []).append(piece)
                    if self.isBoardConnected(board, self.getAdjacentCells):
                        actions.append(("MOVE", (q, r), (nq, nr)))
                    board[(nq, nr)].pop()
                    if len(board[(nq, nr)]) == 0:
                        del board[(nq, nr)]
                board.setdefault((q, r), []).append(piece)
            elif insectType == "Beetle":
                for (nq, nr) in self.getAdjacentCells(q, r):
                    piece = board[(q, r)].pop()
                    if len(board[(q, r)]) == 0:
                        del board[(q, r)]
                    still_connected_after_removal = self.isBoardConnected(board, self.getAdjacentCells)
                    if not still_connected_after_removal:
                        board.setdefault((q, r), []).append(piece)
                        continue
                    board.setdefault((nq, nr), []).append(piece)
                    valid_new_spot = True
                    if len(board[(nq, nr)]) == 1:
                        neighbors_occupied = any(
                            ((xq, xr) in board and board[(xq, xr)])
                            for (xq, xr) in self.getAdjacentCells(nq, nr)
                        )
                        if not neighbors_occupied:
                            valid_new_spot = False
                    still_connected_after_placement = self.isBoardConnected(board, self.getAdjacentCells)
                    board[(nq, nr)].pop()
                    if len(board[(nq, nr)]) == 0:
                        del board[(nq, nr)]
                    board.setdefault((q, r), []).append(piece)
                    if still_connected_after_placement and valid_new_spot:
                        actions.append(("MOVE", (q, r), (nq, nr)))
            elif insectType == "Grasshopper":
                jumps = self.getGrasshopperJumps(board, q, r)
                for (tq, tr) in jumps:
                    piece = board[(q, r)].pop()
                    if len(board[(q, r)]) == 0:
                        del board[(q, r)]
                    still_connected_after_removal = self.isBoardConnected(board, self.getAdjacentCells)
                    if not still_connected_after_removal:
                        board.setdefault((q, r), []).append(piece)
                        continue
                    board.setdefault((tq, tr), []).append(piece)
                    valid_new_spot = True
                    if len(board[(tq, tr)]) == 1:
                        neighbors_occupied = any(
                            ((xq, xr) in board and board[(xq, xr)])
                            for (xq, xr) in self.getAdjacentCells(tq, tr)
                        )
                        if not neighbors_occupied:
                            valid_new_spot = False
                    still_connected_after_placement = self.isBoardConnected(board, self.getAdjacentCells)
                    board[(tq, tr)].pop()
                    if len(board[(tq, tr)]) == 0:
                        del board[(tq, tr)]
                    board.setdefault((q, r), []).append(piece)
                    if still_connected_after_placement and valid_new_spot:
                        actions.append(("MOVE", (q, r), (tq, tr)))
            elif insectType == "Spider":
                piece = board[(q, r)].pop()
                if len(board[(q, r)]) == 0:
                    del board[(q, r)]
                possible_ends = self.getSpiderDestinations(board, (q, r))
                board.setdefault((q, r), []).append(piece)
                for dest in possible_ends:
                    piece = board[(q, r)].pop()
                    if len(board[(q, r)]) == 0:
                        del board[(q, r)]
                    board.setdefault(dest, []).append(piece)
                    if self.isBoardConnected(board, self.getAdjacentCells):
                        actions.append(("MOVE", (q, r), dest))
                    board[dest].pop()
                    if len(board[dest]) == 0:
                        del board[dest]
                    board.setdefault((q, r), []).append(piece)
            elif insectType == "Ant":
                piece = board[(q, r)].pop()
                if len(board[(q, r)]) == 0:
                    del board[(q, r)]
                possible_ends = self.getAntDestinations(board, (q, r))
                board.setdefault((q, r), []).append(piece)
                for dest in possible_ends:
                    piece = board[(q, r)].pop()
                    if len(board[(q, r)]) == 0:
                        del board[(q, r)]
                    board.setdefault(dest, []).append(piece)
                    if self.isBoardConnected(board, self.getAdjacentCells):
                        actions.append(("MOVE", (q, r), dest))
                    board[dest].pop()
                    if len(board[dest]) == 0:
                        del board[dest]
                    board.setdefault((q, r), []).append(piece)
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

    def isBoardConnected(self, board, getAdjacentCells):
        occupied_cells = [coord for coord, stack in board.items() if stack]
        if not occupied_cells:
            return True
        visited = set()
        to_visit = [occupied_cells[0]]
        while to_visit:
            current = to_visit.pop()
            if current not in visited:
                visited.add(current)
                for (nq, nr) in getAdjacentCells(*current):
                    if (nq, nr) in board and board[(nq, nr)]:
                        to_visit.append((nq, nr))
        return len(visited) == len(occupied_cells)

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
        return new_state

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

    def simulateRandomPlayout(self, state):
        temp_state = self.copyState(state)
        while not self.isTerminal(temp_state):
            legal = self.getLegalActions(temp_state)
            if not legal:
                break
            action = self.weightedActionChoice(temp_state, legal)
            temp_state = self.applyAction(temp_state, action)
        return temp_state

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
