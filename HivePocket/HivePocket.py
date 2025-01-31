import random
from collections import deque

from HivePocket.DrawGame import drawStatePygame


# Optional import for debugging visualization
# from HivePocket.DrawGame import drawStatePygame

def hex_distance(q1, r1, q2, r2):
    """
    Compute the hex distance between two axial coordinates (q1, r1) and (q2, r2).
    """
    x1, z1 = q1, r1
    y1 = -x1 - z1
    x2, z2 = q2, r2
    y2 = -x2 - z2
    return (abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)) // 2

def find_queen_position(board, player):
    """
    Look for the given player's Queen on the board. Return (q, r) or None if not found.
    """
    for (q, r), stack in board.items():
        for piece_owner, piece_type in stack:
            if piece_owner == player and piece_type == "Queen":
                return (q, r)
    return None

def edge_hugging_distance(board, start, goal, getNeighbors):
    """
    Returns the BFS distance from start to goal, but only through empty cells
    that are adjacent to an occupied cell (“hug the hive”).
    If goal is unreachable, returns float('inf').
    """
    if start == goal:
        return 0

    visited = set()
    queue = deque()
    queue.append((start, 0))
    visited.add(start)

    while queue:
        cell, dist = queue.popleft()
        if cell == goal:
            return dist

        for neighbor in getNeighbors(cell):
            # neighbor must be empty, but also adjacent to at least one occupied cell:
            if neighbor not in visited:
                if is_valid_hive_edge_cell(board, neighbor, getNeighbors):
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return float('inf')

def is_valid_hive_edge_cell(board, cell, getNeighbors):
    """
    A cell is valid for "hive-hugging" movement if:
      - It is empty (no pieces on it).
      - It has at least one neighbor that is occupied.
    """
    if cell not in board or len(board[cell]) == 0:
        # Cell is empty => check hugging
        for neigh in getNeighbors(cell):
            if neigh in board and len(board[neigh]) > 0:
                return True
    return False

class HiveGame:
    """
    A *simplified* Hive implementation demonstrating adjacency storage for faster lookups.
    """

    def __init__(self):
        # Standard set for each player
        self.INITIAL_PIECES = {
            "Queen": 1,
            "Spider": 2,
            "Beetle": 2,
            "Grasshopper": 3,
            "Ant": 3
        }

        # For reference, we still keep the 6 hex directions if we need them
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
        Returns the initial state of the simplified Hive game, including adjacency map.
        """
        state = {
            "board": {},  # (q,r) -> list of (player, insectType)
            "adjacency": {},  # (q,r) -> set of neighbor-coords
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
        Returns a deep(ish) copy of the state dictionary.
        """
        new_board = {}
        for coord, stack in state["board"].items():
            new_board[coord] = stack.copy()

        new_pieces_in_hand = {
            "Player1": state["pieces_in_hand"]["Player1"].copy(),
            "Player2": state["pieces_in_hand"]["Player2"].copy()
        }

        # For adjacency, we copy each set
        new_adjacency = {}
        for coord, neighbors in state["adjacency"].items():
            new_adjacency[coord] = neighbors.copy()

        new_state = {
            "board": new_board,
            "adjacency": new_adjacency,
            "current_player": state["current_player"],
            "pieces_in_hand": new_pieces_in_hand,
            "move_number": state["move_number"]
        }
        return new_state

    # ---------------------------------------------------------
    # 2. Adjacency Management
    # ---------------------------------------------------------
    def getNeighbors(self, state, cell):
        """
        Return the set of neighbors from the adjacency map.
        If cell not in adjacency, return empty set by default.
        """
        return state["adjacency"].get(cell, set())

    def addCellToAdjacency(self, state, cell):
        """
        Ensure 'cell' is in state["adjacency"], link it to existing neighbors.
        We only do this for newly occupied cells or newly relevant empty cells.
        """
        if cell not in state["adjacency"]:
            state["adjacency"][cell] = set()

        # For each of the 6 directions, see if the neighbor is also in adjacency:
        q, r = cell
        for dq, dr in self.DIRECTIONS:
            neighbor = (q + dq, r + dr)
            if neighbor in state["adjacency"]:
                # Link them
                state["adjacency"][cell].add(neighbor)
                state["adjacency"][neighbor].add(cell)

    def removeCellFromAdjacency(self, state, cell):
        """
        Remove 'cell' from adjacency entirely (if it's no longer occupied or relevant).
        This means unlinking it from neighbors as well.

        In Hive, you *might* want to keep empty cells that are next to the hive
        in adjacency for movement checks. That depends on your design.
        For simplicity, here we remove cells that become empty.
        """
        if cell in state["adjacency"]:
            # Unlink from neighbors
            neighbors = state["adjacency"][cell]
            for n in neighbors:
                if n in state["adjacency"]:
                    state["adjacency"][n].discard(cell)
            # Now remove cell
            del state["adjacency"][cell]

    # ---------------------------------------------------------
    # 3. Core Hive logic
    # ---------------------------------------------------------
    def getOpponent(self, player):
        return "Player2" if player == "Player1" else "Player1"

    def isQueenSurrounded(self, state, player):
        board = state["board"]
        # Find the queen
        for (q, r), stack in board.items():
            for piece in stack:
                if piece == (player, "Queen"):
                    # Check 6 neighbors for occupancy
                    neighbors_occupied = 0
                    for neigh in self.getNeighbors(state, (q, r)):
                        if neigh in board and len(board[neigh]) > 0:
                            neighbors_occupied += 1
                        else:
                            return False
                    return True
        return False

    # ---------------------------------------------------------
    # 4. Action generation
    # ---------------------------------------------------------
    def getLegalActions(self, state):
        # For demonstration, keep it as in your code
        return self.placePieceActions(state) + self.movePieceActions(state)

    def placePieceActions(self, state):
        """
        PLACE actions of the form ("PLACE", insectType, (q, r))
        """
        player = state["current_player"]
        opponent = self.getOpponent(player)
        board = state["board"]
        pieces_in_hand = state["pieces_in_hand"][player]
        actions = []

        if all(count == 0 for count in pieces_in_hand.values()):
            return actions

        # (A) If the board is empty => place at (0,0)
        if len(board) == 0:
            for insectType, count in pieces_in_hand.items():
                if count > 0:
                    actions.append(("PLACE", insectType, (0, 0)))
            return actions

        # (B) If the board has exactly 1 cell => simplified approach
        if len(board) == 1:
            for insectType, count in pieces_in_hand.items():
                if count > 0 and (0, 1) not in board:
                    actions.append(("PLACE", insectType, (0, 1)))
                elif count > 0 and (0, 0) not in board:
                    actions.append(("PLACE", insectType, (0, 0)))
            return actions

        # (C) Normal rule: must place adjacent to at least one friendly piece,
        # and cannot be adjacent to any enemy piece.
        friendly_cells = set()
        for (q, r), stack in board.items():
            if any(p[0] == player for p in stack):
                friendly_cells.add((q, r))

        potential_spots = set()
        for fc in friendly_cells:
            for neigh in self.getNeighbors(state, fc):
                # If neighbor is empty, it's a candidate
                if neigh not in board or len(board[neigh]) == 0:
                    potential_spots.add(neigh)

        valid_spots = []
        for spot in potential_spots:
            # Check if it's adjacent to any enemy piece
            adjacent_to_enemy = False
            for neigh in self.getNeighbors(state, spot):
                if neigh in board and any(p[0] == opponent for p in board[neigh]):
                    adjacent_to_enemy = True
                    break
            if not adjacent_to_enemy:
                valid_spots.append(spot)

        for insectType, count in pieces_in_hand.items():
            if count > 0:
                for spot in valid_spots:
                    actions.append(("PLACE", insectType, spot))

        return actions

    # Example partial implementations of movement for a few pieces:

    def movePieceActions(self, state):
        board = state["board"]
        player = state["current_player"]
        actions = []

        # Gather all top pieces belonging to current player
        player_cells = []
        for (q, r), stack in board.items():
            if stack and stack[-1][0] == player:
                insectType = stack[-1][1]
                player_cells.append((q, r, insectType))

        for (q, r, insectType) in player_cells:
            if insectType == "Beetle":
                actions.extend(self._beetleMoves(state, q, r))
            elif insectType == "Grasshopper":
                actions.extend(self._grasshopperMoves(state, q, r))
            elif insectType == "Spider":
                actions.extend(self._spiderMoves(state, q, r))
            elif insectType == "Ant":
                actions.extend(self._antMoves(state, q, r))
        return actions

    # Below, you can keep the same logic as your original code—just replace
    # references to "getAdjacentCells(q, r)" with self.getNeighbors(state, (q, r)).

    def _beetleMoves(self, state, q, r):
        board = state["board"]
        moves = []
        piece = board[(q, r)][-1]
        for neigh in self.getNeighbors(state, (q, r)):
            # Temporarily remove piece, check connectivity, place piece, etc.
            # (Same pattern as your original remove-check-place-check.)
            from_cell = (q, r)
            to_cell = neigh

            # 1) Remove
            board[from_cell].pop()
            if len(board[from_cell]) == 0:
                del board[from_cell]
                self.removeCellFromAdjacency(state, from_cell)

            # 2) Check connectivity
            still_connected = self.isBoardConnected(state)
            if not still_connected:
                # revert
                board.setdefault(from_cell, []).append(piece)
                self.addCellToAdjacency(state, from_cell)
                continue

            # 3) Place piece on neighbor
            board.setdefault(to_cell, []).append(piece)
            self.addCellToAdjacency(state, to_cell)

            # 4) Check adjacency to avoid “floating” if you want
            valid_spot = True
            if len(board[to_cell]) == 1:  # was empty
                # Must be adjacent to at least one occupied cell
                if not any(len(board[n]) > 0 for n in self.getNeighbors(state, to_cell) if n in board):
                    valid_spot = False

            still_connected_2 = self.isBoardConnected(state)

            # revert
            board[to_cell].pop()
            if len(board[to_cell]) == 0:
                del board[to_cell]
                self.removeCellFromAdjacency(state, to_cell)

            board.setdefault(from_cell, []).append(piece)
            self.addCellToAdjacency(state, from_cell)

            if still_connected_2 and valid_spot:
                moves.append(("MOVE", (q, r), to_cell))

        return moves

    def _grasshopperMoves(self, state, q, r):
        # Example, same logic as your getGrasshopperJumps code,
        # but enumerating directions with self.DIRECTIONS
        board = state["board"]
        piece = board[(q, r)][-1]
        moves = []

        possible_destinations = self.getGrasshopperJumps(board, q, r)
        for (tq, tr) in possible_destinations:
            # Same remove-check-place-check
            from_cell = (q, r)
            to_cell = (tq, tr)

            # 1) Remove
            board[from_cell].pop()
            if len(board[from_cell]) == 0:
                del board[from_cell]
                self.removeCellFromAdjacency(state, from_cell)

            # 2) Check connectivity
            still_connected = self.isBoardConnected(state)
            if not still_connected:
                board.setdefault(from_cell, []).append(piece)
                self.addCellToAdjacency(state, from_cell)
                continue

            # 3) Place
            board.setdefault(to_cell, []).append(piece)
            self.addCellToAdjacency(state, to_cell)

            # 4) Check adjacency
            valid_spot = True
            if len(board[to_cell]) == 1:
                if not any(len(board[n]) > 0 for n in self.getNeighbors(state, to_cell) if n in board):
                    valid_spot = False

            still_connected_2 = self.isBoardConnected(state)

            # revert
            board[to_cell].pop()
            if len(board[to_cell]) == 0:
                del board[to_cell]
                self.removeCellFromAdjacency(state, to_cell)

            board.setdefault(from_cell, []).append(piece)
            self.addCellToAdjacency(state, from_cell)

            if still_connected_2 and valid_spot:
                moves.append(("MOVE", from_cell, to_cell))

        return moves

    def getGrasshopperJumps(self, board, q, r):
        """
        Grasshopper jumps in the 6 directions until it finds an empty cell.
        """
        possible_destinations = []
        for (dq, dr) in self.DIRECTIONS:
            step = 1
            while True:
                check_q = q + dq * step
                check_r = r + dr * step
                if (check_q, check_r) not in board or len(board[(check_q, check_r)]) == 0:
                    # If we haven't jumped over anything, no valid jump
                    if step > 1:
                        possible_destinations.append((check_q, check_r))
                    break
                step += 1
        return possible_destinations

    def _spiderMoves(self, state, q, r):
        # Reuse your "getSpiderDestinationsEdge" logic, but replace adjacency calls
        board = state["board"]
        piece = board[(q, r)][-1]
        moves = []

        possible_ends = self.getSpiderDestinationsEdge(state, q, r)
        for (tq, tr) in possible_ends:
            from_cell = (q, r)
            to_cell = (tq, tr)
            # remove-check-place-check
            board[from_cell].pop()
            if len(board[from_cell]) == 0:
                del board[from_cell]
                self.removeCellFromAdjacency(state, from_cell)

            still_connected = self.isBoardConnected(state)
            if not still_connected:
                board.setdefault(from_cell, []).append(piece)
                self.addCellToAdjacency(state, from_cell)
                continue

            board.setdefault(to_cell, []).append(piece)
            self.addCellToAdjacency(state, to_cell)

            # adjacency check
            valid_spot = True
            if len(board[to_cell]) == 1:
                if not any(len(board[n]) > 0 for n in self.getNeighbors(state, to_cell) if n in board):
                    valid_spot = False

            still_connected_2 = self.isBoardConnected(state)

            # revert
            board[to_cell].pop()
            if len(board[to_cell]) == 0:
                del board[to_cell]
                self.removeCellFromAdjacency(state, to_cell)

            board.setdefault(from_cell, []).append(piece)
            self.addCellToAdjacency(state, from_cell)

            if still_connected_2 and valid_spot:
                moves.append(("MOVE", from_cell, to_cell))
        return moves

    def _antMoves(self, state, q, r):
        # Reuse "getAntDestinationsEdge" logic or BFS logic
        board = state["board"]
        piece = board[(q, r)][-1]
        moves = []

        possible_ends = self.getAntDestinationsEdge(state, q, r, max_steps=20)
        for (tq, tr) in possible_ends:
            from_cell = (q, r)
            to_cell = (tq, tr)

            # remove-check-place-check
            board[from_cell].pop()
            if len(board[from_cell]) == 0:
                del board[from_cell]
                self.removeCellFromAdjacency(state, from_cell)

            still_connected = self.isBoardConnected(state)
            if not still_connected:
                board.setdefault(from_cell, []).append(piece)
                self.addCellToAdjacency(state, from_cell)
                continue

            board.setdefault(to_cell, []).append(piece)
            self.addCellToAdjacency(state, to_cell)

            valid_spot = True
            if len(board[to_cell]) == 1:
                if not any(len(board[n]) > 0 for n in self.getNeighbors(state, to_cell) if n in board):
                    valid_spot = False

            still_connected_2 = self.isBoardConnected(state)

            # revert
            board[to_cell].pop()
            if len(board[to_cell]) == 0:
                del board[to_cell]
                self.removeCellFromAdjacency(state, to_cell)

            board.setdefault(from_cell, []).append(piece)
            self.addCellToAdjacency(state, from_cell)

            if still_connected_2 and valid_spot:
                moves.append(("MOVE", from_cell, to_cell))

        return moves

    # Implement these two “getSpiderDestinationsEdge” and “getAntDestinationsEdge”
    # similarly to your original code, but replace references to directions with
    # direct adjacency or “hugging the hive” checks as needed.

    def getSpiderDestinationsEdge(self, state, q, r):
        # Simplified example: still do 3-step “turning” logic
        # but each step we pick next neighbor in adjacency after
        # rotating direction index. For brevity, re-use your approach:
        results = set()
        # ...
        # (Your specialized spider “turn” logic can remain the same.)
        return results

    def getAntDestinationsEdge(self, state, q, r, max_steps=20):
        # For demonstration, do a BFS that can make up to max_steps around
        # adjacent cells that remain “edge hugging”. Or re-use your prior logic.
        results = set()
        # ...
        return results

    # ---------------------------------------------------------
    # 5. Connectivity Check
    # ---------------------------------------------------------
    def isBoardConnected(self, state):
        """
        Returns True if all occupied cells form one connected component (ignoring empty).
        We'll do a BFS/DFS over adjacency but only on occupied cells.
        """
        board = state["board"]
        adjacency = state["adjacency"]

        occupied = [c for c in board if len(board[c]) > 0]
        if not occupied:
            return True

        visited = set()
        stack = [occupied[0]]
        while stack:
            cell = stack.pop()
            if cell not in visited:
                visited.add(cell)
                # Add occupied neighbors
                for neigh in adjacency.get(cell, []):
                    if neigh in board and len(board[neigh]) > 0:
                        stack.append(neigh)

        return len(visited) == len(occupied)

    # ---------------------------------------------------------
    # 6. MCTS-required interface
    # ---------------------------------------------------------
    def isTerminal(self, state):
        if self.isQueenSurrounded(state, "Player1"):
            return True
        if self.isQueenSurrounded(state, "Player2"):
            return True
        if not self.getLegalActions(state):
            return True
        return False

    def getOtherPlayer(self, currentPlayer):
        """
        Returns the current player to move ('Player1' or 'Player2').
        """
        if currentPlayer == "Player1":
            return "Player2"
        elif currentPlayer == "Player2":
            return "Player1"


    def applyAction(self, state, action):
        new_state = self.copyState(state)
        board = new_state["board"]
        adjacency = new_state["adjacency"]
        player = new_state["current_player"]

        if action[0] == "PLACE":
            # ("PLACE", insectType, (q, r))
            _, insectType, (q, r) = action
            new_state["pieces_in_hand"][player][insectType] -= 1

            if (q, r) not in board:
                board[(q, r)] = []
                # Make sure we add the cell to adjacency
                self.addCellToAdjacency(new_state, (q, r))

            board[(q, r)].append((player, insectType))

        elif action[0] == "MOVE":
            # ("MOVE", (fq, fr), (tq, tr))
            _, (fq, fr), (tq, tr) = action
            piece = board[(fq, fr)].pop()
            if len(board[(fq, fr)]) == 0:
                del board[(fq, fr)]
                self.removeCellFromAdjacency(new_state, (fq, fr))

            if (tq, tr) not in board:
                board[(tq, tr)] = []
                self.addCellToAdjacency(new_state, (tq, tr))

            board[(tq, tr)].append(piece)

        # Switch player
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
            action = random.choice(legal)
            temp_state = self.applyAction(temp_state, action)

            drawStatePygame(temp_state)

        return temp_state

    # ---------------------------------------------------------
    # 7. Debug printing
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

        print("Adjacency:")
        for c, neighs in state["adjacency"].items():
            print(f"  {c} -> {neighs}")

        print("Pieces in hand:")
        for pl, counts in state["pieces_in_hand"].items():
            print(f"  {pl}: {counts}")
        print("-" * 50)
