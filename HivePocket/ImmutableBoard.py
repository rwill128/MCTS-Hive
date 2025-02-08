import functools
from collections import defaultdict

from HivePocket import CONNECTIVITY_CACHE, CAN_SLIDE_CACHE


class ImmutableBoard:
    """
    Immutable representation of a Hive board:
      board_map: a dict (q, r) -> list of (player, pieceType),
                 or we might store just the entire stack as a tuple.
    """

    __slots__ = ("board_map", "_sorted_items", "_hash")

    def __init__(self, board_map):
        # board_map is presumably { (q,r): [(player, pieceType), ...], ... }
        # BUT we will store the stacks as tuples so they're also immutable.
        # We'll store them in a normal dict for quick access, but we'll
        # compute a sorted tuple for hashing.

        # We do a shallow copy, but each stack we store as a tuple so it’s immutable.
        object.__setattr__(self, "board_map", {
            coord: tuple(stack) for coord, stack in board_map.items()
        })

        # Precompute a sorted representation for hashing
        sorted_items = tuple(sorted(
            (coord, self.board_map[coord]) for coord in self.board_map
        ))
        object.__setattr__(self, "_sorted_items", sorted_items)

        # Precompute a hash
        object.__setattr__(self, "_hash", hash(sorted_items))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, ImmutableBoard):
            return False
        return self._sorted_items == other._sorted_items

    def __repr__(self):
        return f"ImmutableBoard({self._sorted_items})"

    @classmethod
    def empty(cls):
        return cls({})

    def getStack(self, q, r):
        """Return the stack (tuple of (player, insectType)) at (q,r) or an empty tuple."""
        return self.board_map.get((q, r), ())

    def withPieceAdded(self, q, r, piece):
        """
        Return a new ImmutableBoard with 'piece' appended to the stack at (q,r).
        'piece' is (player, insectType).
        """
        new_map = dict(self.board_map)  # shallow copy
        old_stack = new_map.get((q, r), ())
        new_stack = old_stack + (piece,)
        new_map[(q, r)] = new_stack
        return ImmutableBoard(new_map)

    def withPieceRemoved(self, q, r):
        """
        Return a new ImmutableBoard with the *top piece* removed from (q,r).
        If that empties the stack, remove the cell entirely.
        """
        old_stack = self.getStack(q, r)
        if not old_stack:
            raise ValueError(f"No piece at {(q, r)} to remove.")

        new_map = dict(self.board_map)
        new_stack = old_stack[:-1]  # remove top piece
        if new_stack:
            new_map[(q, r)] = new_stack
        else:
            del new_map[(q, r)]

        return ImmutableBoard(new_map)

    def occupiedCells(self):
        """Return a list/tuple of coords that have a non-empty stack."""
        return [coord for coord, stack in self.board_map.items() if stack]

def isBoardConnected(board: ImmutableBoard):
    """
    Return True if the entire hive is in one connected component.
    We key the result by board's hash.
    """
    board_key = hash(board)
    if board_key in CONNECTIVITY_CACHE:
        return CONNECTIVITY_CACHE[board_key]

    occupied = board.occupiedCells()
    if not occupied:
        CONNECTIVITY_CACHE[board_key] = True
        return True

    # We'll do a union-find or DFS approach
    visited = set()

    def neighbors(q, r):
        DIRECTIONS = [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]
        for dq, dr in DIRECTIONS:
            yield (q + dq, r + dr)

    def dfs(start):
        stack = [start]
        visited.add(start)
        while stack:
            cell = stack.pop()
            q, r = cell
            for ncell in neighbors(q, r):
                if board.getStack(*ncell):
                    if ncell not in visited:
                        visited.add(ncell)
                        stack.append(ncell)

    first = occupied[0]
    dfs(first)
    result = (len(visited) == len(occupied))

    CONNECTIVITY_CACHE[board_key] = result
    return result

def canSlide(from_q, from_r, to_q, to_r, board: ImmutableBoard):
    """
    Check if a single-step slide from (from_q, from_r) to (to_q, to_r) is blocked or not.
    We'll cache by (hash(board), from_q, from_r, to_q, to_r).
    """
    key = (hash(board), from_q, from_r, to_q, to_r)
    if key in CAN_SLIDE_CACHE:
        return CAN_SLIDE_CACHE[key]

    # Same logic you had before:
    dq = to_q - from_q
    dr = to_r - from_r
    move_dir = (dq, dr)
    adjacent_mapping = {
        (1, 0):  [(0,1), (1,-1)],
        (0, 1):  [(-1,1), (1,0)],
        (-1, 1): [(0,1), (-1,0)],
        (-1, 0): [(-1,1),(0,-1)],
        (0, -1): [(1,-1),(-1,0)],
        (1, -1): [(1,0),(0,-1)],
    }
    if move_dir not in adjacent_mapping:
        CAN_SLIDE_CACHE[key] = False
        return False

    adj_dirs = adjacent_mapping[move_dir]
    blocked_count = 0
    for (adj_dq, adj_dr) in adj_dirs:
        neighbor1 = (from_q+adj_dq, from_r+adj_dr)
        neighbor2 = (to_q+adj_dq,   to_r+adj_dr)
        if board.getStack(*neighbor1) and board.getStack(*neighbor2):
            blocked_count += 1

    result = (blocked_count == 0)
    CAN_SLIDE_CACHE[key] = result
    return result
