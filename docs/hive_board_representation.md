# Hive Board Representation and Symmetry

The `HivePocket` module models the hexagonal board using **axial coordinates**
`(q, r)`. Each coordinate identifies a hex cell and maps to a *stack* of
pieces on that cell. A board state is therefore represented as a dictionary:

```python
board = {
    (q, r): [(owner, piece_type), ...],
    ...
}
```

Pieces are stacked in a list with the bottom piece first and the top piece last.
Only the top piece of each stack can move. The coordinate `(0, 0)` is arbitrary
but all movement functions rely on the relative axial directions:

```
      \  (0,1)  /
(-1,0) - (0,0) - (1,0)
      /  (0,-1) \
```

The six neighbour deltas are `(1,0)`, `(-1,0)`, `(0,1)`, `(0,-1)`, `(1,-1)` and
`(-1,1)`.

## Canonicalisation with Symmetry

Hive positions can be rotated or reflected without changing the underlying
state. To avoid storing duplicate transpositions the function
`canonical_board_key` computes a canonical representation that is invariant
under all 12 rotations and reflections.

1. **Convert axial coordinates to cube coordinates** `(x, y, z)` using
   `x = q`, `z = r` and `y = -x - z`.
2. **Generate all 12 symmetry transforms**:
   - six rotations in 60° steps using `_rotate_cube`;
   - after rotating, optionally reflect across the `x = z` axis with
     `_reflect_cube`.
3. **Apply each transform** to every occupied coordinate, sort the resulting
   items and pick the lexicographically smallest tuple. This unique tuple is the
   canonical board key.

The process in pseudo-code:

```
for k in range(12):
    items = []
    for (q, r), stack in board.items():
        q2, r2 = transform(q, r, k)  # rotate then reflect
        items.append(((q2, r2), tuple(stack)))
    representations.append(tuple(sorted(items)))
return min(representations)
```

Using this key allows the MCTS cache to recognise board states that are the same
up to rotation or reflection, greatly reducing duplication.
