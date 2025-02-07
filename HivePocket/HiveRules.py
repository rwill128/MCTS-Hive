class HiveRules:
    DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

    @staticmethod
    def get_adjacent_cells(q, r):
        return ((q + dq, r + dr) for dq, dr in HiveRules.DIRECTIONS)

    @staticmethod
    def is_board_connected(board):
        occupied_cells = [cell for cell, stack in board.items() if stack]
        if not occupied_cells:
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

        directions = HiveRules.DIRECTIONS
        for cell in occupied_cells:
            q, r = cell
            for dq, dr in directions:
                neighbor = (q + dq, r + dr)
                if neighbor in board and board[neighbor]:
                    union(cell, neighbor)

        roots = {find(cell) for cell in occupied_cells}
        result = (len(roots) == 1)
        return result

    @staticmethod
    def can_slide(board, from_q, from_r, to_q, to_r):
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
            return False

        adj_dirs = adjacent_mapping[move_dir]
        blocked_count = 0
        for adj_q, adj_r in adj_dirs:
            neighbor1 = (from_q + adj_q, from_r + adj_r)
            neighbor2 = (to_q + adj_q, to_r + adj_r)
            if (neighbor1 in board and board[neighbor1]) and (neighbor2 in board and board[neighbor2]):
                blocked_count += 1

        return blocked_count == 0

    @staticmethod
    def get_ant_destinations(board, start):
        results = set()
        visited = {start}
        frontier = [start]
        while frontier:
            cur = frontier.pop(0)
            for neighbor in HiveRules.get_adjacent_cells(*cur):
                if neighbor in board and board[neighbor]:
                    continue
                if not any(adj in board and board[adj] for adj in HiveRules.get_adjacent_cells(*neighbor)):
                    continue
                if not HiveRules.can_slide(board, cur[0], cur[1], neighbor[0], neighbor[1], board):
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    results.add(neighbor)
                    frontier.append(neighbor)
        return results

    @staticmethod
    def get_spider_destinations(board, start):
        results = set()

        def dfs(path, steps):
            cur = path[-1]
            if steps == 3:
                results.add(cur)
                return
            for neighbor in HiveRules.get_adjacent_cells(*cur):
                if neighbor in path:
                    continue
                if neighbor in board and board[neighbor]:
                    continue
                if not any(adj in board and board[adj] for adj in HiveRules.get_adjacent_cells(*neighbor)):
                    continue
                if not HiveRules.can_slide(board, cur[0], cur[1], neighbor[0], neighbor[1], board):
                    continue
                dfs(path + [neighbor], steps + 1)

        valid_starts = []
        for first_neighbor in HiveRules.get_adjacent_cells(*start):
            if first_neighbor in board and board[first_neighbor]:
                continue
            if not any(adj in board and board[adj] for adj in HiveRules.get_adjacent_cells(*first_neighbor)):
                continue
            if HiveRules.can_slide(board, start[0], start[1], first_neighbor[0], first_neighbor[1], board):
                valid_starts.append(first_neighbor)

        for first_step in valid_starts:
            dfs([start, first_step], 1)

        return results

    @staticmethod
    def get_grasshopper_jumps(board, q, r):
        possible_destinations = []
        for (dq, dr) in HiveRules.DIRECTIONS:
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
    @staticmethod
    def find_queen_position(board, player):
        for (q, r), stack in board.items():
            for piece_owner, piece_type in stack:
                if piece_owner == player and piece_type == "Queen":
                    return (q, r)
        return None

    @staticmethod
    def is_queen_surrounded(board, player):
        queen_pos = HiveRules.find_queen_position(board, player)
        if queen_pos is None:
            return False  # Queen not on the board

        for neighbor in HiveRules.get_adjacent_cells(*queen_pos):
            if neighbor not in board or not board[neighbor]:
                return False  # Found an empty neighbor

        return True  # All neighbors are occupied

    @staticmethod
    def board_hash(board):
        items = tuple(sorted((coord, tuple(stack)) for coord, stack in board.items()))
        return hash(items)

    @staticmethod
    def get_legal_actions(state):
        board = state.board
        player = state.current_player

        all_moves = HiveRules.place_piece_actions(state) + HiveRules.move_piece_actions(state)

        if HiveRules.find_queen_position(board, player) is None:
            pieces_placed = sum(1 for _, stack in board.items() for p_owner, _ in stack if p_owner == player)
            if pieces_placed >= 3:
                queen_actions = [action for action in all_moves if action[0] == "PLACE" and action[1] == "Queen"]
                return queen_actions

        return all_moves
    @staticmethod
    def place_piece_actions(state):
        player = state.current_player
        opponent = state.get_opponent()
        board = state.board
        pieces_in_hand = state.pieces_in_hand[player]
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
                if count > 0:
                    for coord, _ in board.items(): #Safe since len(board) == 1
                        q, r = coord
                    for neighbor in HiveRules.get_adjacent_cells(q, r):
                        actions.append(("PLACE", insectType, neighbor))
            return actions


        friendly_cells = set()
        for (q, r), stack in board.items():
            if any(p[0] == player for p in stack):
                friendly_cells.add((q, r))

        potential_spots = set()
        for (q, r) in friendly_cells:
            for (nq, nr) in HiveRules.get_adjacent_cells(q, r):
                if (nq, nr) not in board or len(board[(nq, nr)]) == 0:
                    potential_spots.add((nq, nr))

        valid_spots = []
        for (tq, tr) in potential_spots:
            adjacent_to_enemy = False
            for (nq, nr) in HiveRules.get_adjacent_cells(tq, tr):
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

    @staticmethod
    def move_piece_actions(state):
        board = state.board
        player = state.current_player
        actions = []

        # CORRECTED: Only include cells where the *top* piece is the player's.
        player_cells = {(q, r, stack[-1][1]) for (q, r), stack in board.items() if stack and stack[-1][0] == player}

        def board_copy(board):
            return {coord: stack[:] for coord, stack in board.items()}

        for (q, r, insectType) in player_cells:
            temp_board = board_copy(board)
            piece = temp_board[(q, r)].pop()
            if not temp_board[(q, r)]:
                del temp_board[(q, r)]

            if not HiveRules.is_board_connected(temp_board):
                continue

            if insectType == "Queen":
                destinations = HiveRules.get_adjacent_cells(q, r)
            elif insectType == "Beetle":
                destinations = HiveRules.get_adjacent_cells(q, r)
            elif insectType == "Grasshopper":
                destinations = HiveRules.get_grasshopper_jumps(temp_board, q, r)
            elif insectType == "Spider":
                destinations = HiveRules.get_spider_destinations(temp_board, (q, r))
            elif insectType == "Ant":
                destinations = HiveRules.get_ant_destinations(temp_board, (q, r))
            else:
                destinations = []

            for to_q, to_r in destinations:
                if (to_q, to_r) in temp_board and any(p[0] == player for p in temp_board[(to_q, to_r)]) and insectType != "Beetle":
                    continue

                if insectType not in ["Beetle", "Grasshopper", "Ant", "Spider"] and not HiveRules.can_slide(temp_board, q, r, to_q, to_r):
                    continue

                temp_board.setdefault((to_q, to_r), []).append(piece)
                actions.append(("MOVE", (q, r), (to_q, to_r)))
                temp_board[(to_q, to_r)].pop()
                if not temp_board.get((to_q, to_r)):
                    temp_board.pop((to_q, to_r), None)

        return actions

    @staticmethod
    def is_legal_action(state, action):
        # Check if a *single* action is legal (for testing, etc.)
        for a in HiveRules.get_legal_actions(state):
            if a == action:
                return True
        return False