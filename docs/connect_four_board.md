# Connect Four Board Representation

This project represents the Connect Four board as a 6×7 list of lists. Each cell contains one of three values:

- `None` if the cell is empty
- `'X'` for player X
- `'O'` for player O

Row index `0` corresponds to the **bottom** of the board while row index `5` is the **top**. Column index `0` is the **leftmost** column. When a piece is dropped into a column it occupies the lowest available row in that column and higher pieces have larger row indices.

The following diagram shows the relationship between board indices and the visual layout:

```
row5: [ ][ ][ ][ ][ ][ ][ ]
row4: [ ][ ][ ][ ][ ][ ][ ]
row3: [ ][ ][ ][ ][ ][ ][ ]
row2: [ ][ ][ ][ ][ ][ ][ ]
row1: [ ][ ][ ][ ][ ][ ][ ]
row0: [ ][ ][ ][ ][ ][ ][ ]
       0  1  2  3  4  5  6  <- column indices
```

Here is an example board where player `X` is about to win vertically in column `0`:

```python
board = [
    ["X", None, None, None, None, None, None],  # row 0 (bottom)
    ["X", None, None, None, None, None, None],  # row 1
    ["X", None, None, None, None, None, None],  # row 2
    [None, None, None, None, None, None, None], # row 3
    [None, None, None, None, None, None, None], # row 4
    [None, None, None, None, None, None, None], # row 5 (top)
]
current_player = "X"
```

Such a representation makes it straightforward to craft board states directly in tests or scripts. Place pieces using the numeric indices above and remember that row 0 is the bottom of the stack.
