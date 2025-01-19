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
                    # print("Found a surrounded queen")
                    # drawStatePygame(state)
                    return True
        return False

    # ---------------------------------------------------------
    # 3. Action generation
    # ---------------------------------------------------------
    def getLegalActions(self, state):
        """
        Returns all legal actions for the current player: placements + moves
        """
        # drawStatePygame(state)
        return self.placePieceActions(state) + self.movePieceActions(state)
        # return self.placePieceActions(state)

    def placePieceActions(self, state):
        """
        PLACE actions of the form ("PLACE", insectType, (q, r)).

        - If the board is empty, place anywhere (simplified => fix at (0,0)).
        - If the board has exactly 1 cell, place at (0,1) (demo simplification).
        - Otherwise, you must place adjacent to at least one friendly piece
          AND you cannot place adjacent to any enemy piece.
        """
        player = state["current_player"]
        opponent = self.getOpponent(player)
        board = state["board"]
        pieces_in_hand = state["pieces_in_hand"][player]
        actions = []

        # If the player has no pieces left to place, skip.
        if all(count == 0 for count in pieces_in_hand.values()):
            return actions

        # If the board is empty => place at (0,0).
        if len(board) == 0:
            for insectType, count in pieces_in_hand.items():
                if count > 0:
                    actions.append(("PLACE", insectType, (0, 0)))
            return actions

        # If the board has exactly 1 cell => place at (0,1) for demo's sake
        # (Though in real Hive rules, you'd also check adjacency to the opponent's piece.)
        if len(board) == 1:
            for insectType, count in pieces_in_hand.items():
                if count > 0:
                    actions.append(("PLACE", insectType, (0, 1)))
            return actions

        # --------------------------
        # Otherwise, the normal rule:
        # - Must place adjacent to at least one friendly piece.
        # - Cannot be adjacent to any enemy piece.
        # --------------------------

        # 1. Identify all 'friendly_cells' that contain at least one piece of the current_player.
        friendly_cells = set()
        for (q, r), stack in board.items():
            if any(p[0] == player for p in stack):
                friendly_cells.add((q, r))

        # 2. Build a set of potential_spots by taking all empty cells adjacent to those friendly_cells.
        potential_spots = set()
        for (q, r) in friendly_cells:
            for (nq, nr) in self.getAdjacentCells(q, r):
                # If that cell is empty, it is a candidate
                if (nq, nr) not in board or len(board[(nq, nr)]) == 0:
                    potential_spots.add((nq, nr))

        # 3. Filter out any spot that is adjacent to an enemy piece.
        valid_spots = []
        for (tq, tr) in potential_spots:
            adjacent_to_enemy = False
            for (nq, nr) in self.getAdjacentCells(tq, tr):
                if (nq, nr) in board and any(p[0] == opponent for p in board[(nq, nr)]):
                    adjacent_to_enemy = True
                    break
            if not adjacent_to_enemy:
                valid_spots.append((tq, tr))

        # 4. For each piece in hand, add place-actions at these valid_spots.
        for insectType, count in pieces_in_hand.items():
            if count > 0:
                for (tq, tr) in valid_spots:
                    actions.append(("PLACE", insectType, (tq, tr)))

        return actions

    def getGrasshopperJumps(self, board, q, r):
        """
        Grasshopper can jump in each of the 6 directions.
        Starting from the cell immediately next to (q,r),
        keep going while each cell is occupied,
        stop at the first cell that is unoccupied => that's the landing spot.

        If you never find an unoccupied cell before running out
        of 'occupied cells', then there's no valid jump in that direction.

        Returns a list of (tq, tr) possible destinations.
        """
        possible_destinations = []
        for (dq, dr) in self.DIRECTIONS:
            # Step at least once in that direction
            step_count = 1
            while True:
                check_q = q + dq*step_count
                check_r = r + dr*step_count

                if (check_q, check_r) not in board or len(board[(check_q, check_r)]) == 0:
                    # If it's unoccupied, then:
                    # - We must have jumped over at least one occupied cell
                    #   (meaning step_count > 1, otherwise there's no "jump")
                    if step_count > 1:
                        possible_destinations.append((check_q, check_r))
                    break  # stop checking further in this direction

                # if we found an occupied cell, keep going
                step_count += 1

        return possible_destinations


    def movePieceActions(self, state):
        """
        MOVE actions of the form ("MOVE", (from_q, from_r), (to_q, to_r)).

        This example only implements Beetle movement with connectivity checks.
        Other pieces (Ant, Grasshopper, etc.) are commented out or omitted for brevity.
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
            if insectType == "Beetle":
                # For each adjacent cell to (q,r)
                for (nq, nr) in self.getAdjacentCells(q, r):
                    # A Beetle can move onto an occupied cell (climb on top)
                    # or onto an empty cell, provided we don't break the hive.

                    # -------------------------------
                    # STEP 1: Remove the Beetle and check connectivity
                    # -------------------------------
                    piece = board[(q, r)].pop()  # remove top piece
                    # If that cell is now empty, remove its entry from board for cleanliness
                    if len(board[(q, r)]) == 0:
                        del board[(q, r)]

                    still_connected_after_removal = self.isBoardConnected(board, self.getAdjacentCells)

                    if not still_connected_after_removal:
                        # Put the piece back and skip this neighbor
                        board.setdefault((q, r), []).append(piece)
                        continue

                    # -------------------------------
                    # STEP 2: Place the Beetle on (nq, nr) temporarily
                    # -------------------------------
                    board.setdefault((nq, nr), []).append(piece)

                    # Optionally check that the new location is adjacent to at least one piece
                    # (to avoid "jumping off" into empty space).
                    # But if your Beetle can always climb, maybe you skip this.
                    # We'll do a simplified check:
                    #   - if (nq, nr) was empty, ensure it has at least one neighbor occupied
                    #     so the Beetle isn't floating by itself.
                    valid_new_spot = True
                    if len(board[(nq, nr)]) == 1:  # means it was empty, now just 1 piece (the Beetle)
                        neighbors_occupied = False
                        for (xq, xr) in self.getAdjacentCells(nq, nr):
                            if (xq, xr) in board and board[(xq, xr)]:
                                neighbors_occupied = True
                                break
                        if not neighbors_occupied:
                            valid_new_spot = False

                    # Check connectivity after placing
                    still_connected_after_placement = self.isBoardConnected(board, self.getAdjacentCells)

                    # Restore: remove Beetle from new location
                    board[(nq, nr)].pop()
                    if len(board[(nq, nr)]) == 0:
                        del board[(nq, nr)]

                    # Put the Beetle back in original location
                    board.setdefault((q, r), []).append(piece)

                    # If the new spot was valid and the board remains connected, we have a legal move
                    if still_connected_after_placement and valid_new_spot:
                        actions.append(("MOVE", (q, r), (nq, nr)))

            # --- GRASSHOPPER (new logic) ---
            elif insectType == "Grasshopper":
                # Compute all possible destinations
                jumps = self.getGrasshopperJumps(board, q, r)

                for (tq, tr) in jumps:
                    # Same pattern: remove the piece, check connectivity,
                    # place the piece, check connectivity, restore state.
                    piece = board[(q, r)].pop()
                    if len(board[(q, r)]) == 0:
                        del board[(q, r)]

                    still_connected_after_removal = self.isBoardConnected(board, self.getAdjacentCells)
                    if not still_connected_after_removal:
                        board.setdefault((q, r), []).append(piece)
                        continue

                    # Place Grasshopper at (tq, tr)
                    board.setdefault((tq, tr), []).append(piece)

                    # Optionally, check adjacency to avoid "floating" if that matters to you:
                    valid_new_spot = True
                    if len(board[(tq, tr)]) == 1:  # newly placed piece => was empty
                        neighbors_occupied = False
                        for (xq, xr) in self.getAdjacentCells(tq, tr):
                            if (xq, xr) in board and board[(xq, xr)]:
                                neighbors_occupied = True
                                break
                        if not neighbors_occupied:
                            valid_new_spot = False

                    still_connected_after_placement = self.isBoardConnected(board, self.getAdjacentCells)

                    # Restore
                    board[(tq, tr)].pop()
                    if len(board[(tq, tr)]) == 0:
                        del board[(tq, tr)]
                    board.setdefault((q, r), []).append(piece)

                    # If good, we add the move
                    if still_connected_after_placement and valid_new_spot:
                        actions.append(("MOVE", (q, r), (tq, tr)))

            # --- SPIDER ---
            elif insectType == "Spider":
                # 1) Temporarily remove the Spider so (q, r) is considered empty
                piece = board[(q, r)].pop()
                if len(board[(q, r)]) == 0:
                    del board[(q, r)]

                # 2) Compute all possible 3-step edge destinations
                possible_ends = self.getSpiderDestinationsEdge(board, q, r)

                # 3) Put the Spider back so we can do remove–check–place–check for each end cell
                board.setdefault((q, r), []).append(piece)

                # 4) For each candidate end cell (tq, tr), check connectivity, floating, etc.
                for (tq, tr) in possible_ends:
                    # Remove the Spider again
                    piece = board[(q, r)].pop()
                    if len(board[(q, r)]) == 0:
                        del board[(q, r)]

                    still_connected_after_removal = self.isBoardConnected(board, self.getAdjacentCells)
                    if not still_connected_after_removal:
                        # revert
                        board.setdefault((q, r), []).append(piece)
                        continue

                    # Place Spider at (tq, tr)
                    board.setdefault((tq, tr), []).append(piece)

                    # (Optional) Check “floating” if you want to ensure
                    # the new location is still adjacent to the hive
                    valid_new_spot = True
                    # for (xq, xr) in self.getAdjacentCells(tq, tr):
                    #     if (xq, xr) in board and board[(xq, xr)]:
                    #         break
                    # else:
                    #     valid_new_spot = False

                    still_connected_after_placement = self.isBoardConnected(board, self.getAdjacentCells)

                    # revert
                    board[(tq, tr)].pop()
                    if len(board[(tq, tr)]) == 0:
                        del board[(tq, tr)]
                    board.setdefault((q, r), []).append(piece)

                    if still_connected_after_placement and valid_new_spot:
                        print("Found a valid spider move")
                        print("Drawing original state:")
                        drawStatePygame(state)
                        temp_state = self._makeTempState(state, board)
                        valid_spider_move = ("MOVE", (q, r), (tq, tr))
                        new_state = self.applyAction(temp_state, valid_spider_move)
                        print("Drawing spider move")
                        drawStatePygame(new_state)
                        actions.append(valid_spider_move)

        return actions

    def _makeTempState(self, original_state, temp_board):
        """
        Helper to create a state object that references the *current* board in progress
        so we can draw it without mutating the original state object forever.
        """
        # We'll copy everything else but override 'board' with temp_board references.
        temp_state = self.copyState(original_state)
        # Wipe out the board in the copy, then shallow-copy the temp_board contents.
        temp_state["board"] = {}
        for coord, stack in temp_board.items():
            temp_state["board"][coord] = stack[:]
        return temp_state


    def getOtherPlayer(self, currentPlayer):
        """
        Returns the current player to move ('Player1' or 'Player2').
        """
        if currentPlayer == "Player1":
            return "Player2"
        elif currentPlayer == "Player2":
            return "Player1"

    def isBoardConnected(self, board, getAdjacentCells):
        """
        Returns True if all occupied cells in 'board' form a single connected
        component (ignoring empty cells).
        """

        # Collect all occupied cells
        occupied_cells = [coord for coord, stack in board.items() if stack]
        if not occupied_cells:
            return True  # Empty board or no pieces => trivially "connected" in this simplified logic

        # We'll do a BFS/DFS from the first occupied cell
        visited = set()
        to_visit = [occupied_cells[0]]
        while to_visit:
            current = to_visit.pop()
            if current not in visited:
                visited.add(current)
                # For each neighbor that is occupied, visit it
                for (nq, nr) in getAdjacentCells(*current):
                    if (nq, nr) in board and board[(nq, nr)]:
                        to_visit.append((nq, nr))

        # If we've visited all occupied cells, the board is connected
        return len(visited) == len(occupied_cells)

    def getSpiderDestinationsEdge(self, board, q, r):
        """
        Spider moves exactly 3 steps around the edge of the hive,
        in one rotational direction (clockwise or counterclockwise)
        without revisiting tiles.

        - We define 6 directions in a clockwise order.
        - For each direction i, we try:
            - turning consistently "clockwise" (directionIncrement=+1)
            - turning consistently "counterclockwise" (directionIncrement=-1)
        - We do exactly 3 steps, skipping any path that hits an occupied cell
          or tries to revisit a cell on the same path.
        """

        # 6 directions in "clockwise" order
        directions = [(1, 0), (1, -1), (0, -1),
                      (-1, 0), (-1, 1), (0, 1)]

        results = set()

        def tryPath(startQ, startR, startDirIndex, directionIncrement):
            """
            Attempt a 3-step crawl from (startQ, startR).
            At each step we rotate the direction index by +1 or -1,
            move there if it's empty, and stop if blocked.
            """
            path = [(startQ, startR)]
            curQ, curR = startQ, startR
            dirIndex = startDirIndex

            for _ in range(3):
                # Rotate direction index
                dirIndex = (dirIndex + directionIncrement) % 6
                dQ, dR = directions[dirIndex]
                nextQ = curQ + dQ
                nextR = curR + dR

                # 1) Must be empty
                if (nextQ, nextR) in board and len(board[(nextQ, nextR)]) > 0:
                    return  # blocked by occupied cell

                # 2) Must not revisit the same cell in this path
                if (nextQ, nextR) in path:
                    return  # would revisit same tile => stop

                # 3) (Optional) If you want to ensure "hugging the hive":
                #    check that (nextQ, nextR) has at least one occupied neighbor:
                #
                # neighbors_occupied = False
                # for (adjQ, adjR) in self.getAdjacentCells(nextQ, nextR):
                #     if (adjQ, adjR) in board and len(board[(adjQ, adjR)]) > 0:
                #         neighbors_occupied = True
                #         break
                # if not neighbors_occupied:
                #     return  # not hugging the hive => invalid

                path.append((nextQ, nextR))
                curQ, curR = nextQ, nextR

            # If we get here, we completed 3 steps
            # The final cell (curQ, curR) is a valid end-position
            results.add((curQ, curR))

        # Try each of the 6 directions as a "starting orientation,"
        # in both clockwise (+1) and counterclockwise (-1) modes:
        for i in range(6):
            tryPath(q, r, i, +1)  # clockwise
            tryPath(q, r, i, -1)  # counterclockwise

        return results


    def _bfsExactSteps(self, board, start, steps=3):
        """
        Return all reachable cells in exactly `steps` steps,
        traveling on empty adjacent cells only.

        Potential issues:
          1) The starting cell is 'occupied' by the piece itself,
             so BFS won't move off it unless you temporarily remove the piece.
          2) LIFO vs. FIFO (the code is using .pop() from a list, which is DFS order)
             but for 'all reachable cells in exactly X steps', it usually doesn't break the result,
             as long as you track visited states properly.
          3) You might want to allow revisiting the same cell at different 'dist' levels.
             The code does handle that by storing (nq, nr, dist+1) in visited, which is good.
        """
        visited = set()  # will store (q, r, dist)
        q0, r0 = start
        frontier = [(q0, r0, 0)]
        results = set()

        while frontier:
            q, r, dist = frontier.pop()  # LIFO => DFS order
            if dist == steps:
                # Once we hit exactly 'steps' moves, record (q, r) and do NOT expand further
                results.add((q, r))
                continue
            elif dist > steps:
                # Shouldn't happen if we check dist before pushing,
                # but we have a guard anyway
                continue

            for (nq, nr) in self.getAdjacentCells(q, r):
                # Must be empty to proceed
                if (nq, nr) not in board or len(board[(nq, nr)]) == 0:
                    next_state = (nq, nr, dist+1)
                    if next_state not in visited and dist+1 <= steps:
                        visited.add(next_state)
                        frontier.append(next_state)

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
            legal = self.getLegalActions(temp_state)
            if not legal:
                # No moves => break or treat as terminal
                break
            action = random.choice(legal)
            temp_state = self.applyAction(temp_state, action)
            # drawStatePygame(temp_state)
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
