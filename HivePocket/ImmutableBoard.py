import functools
from collections import defaultdict

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
