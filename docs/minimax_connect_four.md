# Minimax Search for Connect Four

`simple_games/minimax_connect_four.py` implements a depth-limited minimax player. The algorithm explores move sequences to a fixed depth and chooses the move that maximises the worst-case score from the perspective player's view.

Connect Four boards are represented as 6×7 lists of columns as described in [`connect_four_board.md`](connect_four_board.md). The minimax player serialises this board to an immutable tuple so that repeated positions can be cached with `functools.lru_cache`.

## Algorithm Overview

1. **Serialize** the board and current player.
2. **Recursive search** with memoisation via `@lru_cache` in `_minimax`.
3. **Heuristic evaluation** when a terminal state or depth limit is reached.
4. **Return** the best or worst score depending on which player is to move.

The core recursion can be summarised as:

```python
@lru_cache(maxsize=None)
def _minimax(board_serialized, to_move, depth):
    if terminal or depth == 0:
        return heuristic_value(state)
    scores = []
    for action in legal_moves(state):
        next_state = applyAction(state, action)
        ser = serialize(next_state)
        score = _minimax(ser, next_state["current_player"], depth - 1)
        scores.append(score)
    return max(scores) if to_move == perspective else min(scores)
```

A depth-limited tree expansion therefore looks like:

```
root (X to move)
├─ column 0 -> score s0
├─ column 1 -> score s1
└─ ...
```

where each `score` is computed by exploring alternating turns until the specified depth or a game outcome.

## Heuristic

`_heuristic_value` assigns a numeric score based on potential four-in-a-row lines. For each group of four cells, if the opponent does not occupy any of them the score increases exponentially with the number of the perspective player's pieces. Lines containing only the opponent's pieces subtract from the score. The final value is clamped to `[-1, 1]` so that terminal results and heuristic results share the same range.

```python
if opp not in cells:
    count = cells.count(player)
    score += 10 ** (count - 1)
elif player not in cells:
    count = cells.count(opp)
    score -= 10 ** (count - 1)
```

This heuristic allows the minimax player to evaluate non-terminal boards meaningfully and guides the search towards promising moves.

## Choosing a Move

The public `search` method enumerates all legal actions from the current state, calls `_minimax` on each resulting state, and selects an action with the highest score. Immediate winning moves are detected first and chosen outright.

The implementation demonstrates how a small depth-limited minimax search can be combined with memoisation and a simple heuristic to produce a reasonable Connect Four opponent.

