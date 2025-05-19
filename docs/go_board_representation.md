# Go Board Representation

The `Go` class in `simple_games/go.py` implements a minimal version of the game for small board sizes (5×5 by default).  Every board state is represented as a dictionary:

```python
state = {
    "board": [
        [cell00, cell01, ..., cell0N],
        ...,
        [cellN0, cellN1, ..., cellNN],
    ],
    "current_player": "B" or "W",
    "passes": int,  # consecutive pass moves
}
```

Each `cellXY` entry is either `"B"`, `"W"` or `None`.  Row index `0` corresponds to the **top** of the board and column index `0` is the **leftmost** column.  The optional `passes` counter tracks how many pass moves have occurred in a row—two consecutive passes end the game.

## Coordinate System

```
(0,0) (0,1) ... (0,N)
(1,0) (1,1) ... (1,N)
 ...   ...       ...
(N,0) (N,1) ... (N,N)
```

During play, actions are `(row, column)` tuples.  A special action `"pass"` represents passing the turn.  The `applyAction` method updates the board and flips the `current_player` field.  Captured stones are removed by `_remove_captured` which checks liberties for each group.

## Capturing Logic

To determine captures, `_collect_group` performs a flood fill from a stone, gathering all connected stones of the same colour and recording their liberties.  After a move, `_remove_captured` iterates over the opponent's stones and removes any groups without liberties.  Suicide is considered illegal; the move is only allowed if the newly placed stone's group still has at least one liberty after removing opponent captures.

## End of Game and Scoring

A game ends when both players pass or the board fills up.  The winner is determined by **area scoring**—whoever has more stones on the board.  If the counts are equal the game is a draw.

This compact representation and rule set keep the implementation readable while retaining the tactical flavour of Go.  Because everything is pure Python lists and tuples, creating test positions or feeding states to search algorithms is straightforward.
