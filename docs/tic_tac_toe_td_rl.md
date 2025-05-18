# Temporal-Difference Learning for Tic-Tac-Toe

`examples/ttt_selfplay_rl.py` implements a tiny reinforcement learning loop that trains
an agent to play Tic-Tac-Toe using **state value estimates**.  No neural network is
involved—the agent simply stores a numerical value for every unique board position.
These values are updated via self-play episodes.

The approach mirrors the classical TD(0) algorithm:

```
V(s) <- V(s) + lr * (R - V(s))
```

Where:

- `V(s)` is the stored value for board state `s` from X's perspective.
- `lr` is the learning rate.
- `R` is the return received at the end of the episode.

After each training game the final reward propagates backward through all visited
states.  Because players alternate turns, the sign of the reward flips after every
move.  This lets a single value table serve both players.

## Board Representation

The Tic-Tac-Toe board is represented as a 3×3 list of lists with entries
`"X"`, `"O"` or `None`.  In the value table each state is stored as:

```python
key = (
    (
        cell_00, cell_01, cell_02,
        cell_10, cell_11, cell_12,
        cell_20, cell_21, cell_22,
    ),
    current_player,
)
```

This immutable tuple format allows states to be used as dictionary keys and ensures
symmetry between episodes.

## Training Loop

1. Play one complete game using an ε-greedy policy based on the current values.
2. Record each intermediate state.
3. At the end of the game, compute `R` as +1 for X win, −1 for O win, 0 for draw.
4. Iterate backwards through the visited states applying the TD update rule.

The following diagram illustrates the backward value updates:

```
state_0 --move--> state_1 --move--> ... --move--> state_N (terminal)
                                    ^            |
                                    |------------|
                                          R
```

At each step the reward `R` is negated to account for the opponent's perspective
on the previous move.

## Visualising Learning

Running the script launches a Tkinter UI.  Each square displays the current value
of taking that move.  Green tones indicate positions favourable for X and red
tones for O.  Buttons allow stepping through updates, autoplaying, or resetting the
value table.

This example provides an approachable introduction to reinforcement learning
without relying on heavy dependencies.  By studying the code in
[`examples/ttt_selfplay_rl.py`](../examples/ttt_selfplay_rl.py) and observing the
UI you can experiment with learning rate, exploration and other parameters.
