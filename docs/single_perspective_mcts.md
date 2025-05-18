# Single Perspective Monte‑Carlo Tree Search

`SinglePerspectiveMCTS` is a lighter variant of the standard MCTS algorithm implemented in
[`mcts/single_perspective.py`](../mcts/single_perspective.py).  It only expands
nodes for **one** player's turns.  All opponent turns are simulated on the fly.
This greatly reduces the branching factor at the cost of modelling the opponent
with random play.

The core idea is illustrated below:

```
start state --(roll forward)--> perspective player's turn
   |
   |-- action A --> opponent random moves --> next state
   |-- action B --> opponent random moves --> next state
   |-- ...
```

Each tree node stores the state **after** any opponent moves.  When an action
is expanded from a node, the algorithm:

1. Applies the chosen move for the perspective player.
2. Rolls the game forward using `_roll_forward` until it is the perspective
   player's turn again or the game reaches a terminal state.
3. Creates a child node from the resulting state.

Simulation and back‑propagation proceed exactly like vanilla MCTS except no
sign flips are needed—every node value is already from the same player's
perspective.

## Search Loop Overview

`SinglePerspectiveMCTS.search` performs the following steps for a fixed number
of iterations:

1. **Selection** – descend the tree via `best_child` until a leaf is reached.
2. **Expansion** – pick one untried action and apply it, then roll forward
   opponent moves.
3. **Simulation** – run a random playout from the new state using the game's
   `simulateRandomPlayout` method.
4. **Back‑propagation** – add the simulation value to each ancestor node.

At the end of the iterations the child of the root with the highest visit count
is chosen.

## Rolling Forward

Opponent moves are handled by `_roll_forward`.  Starting from a given state it
repeatedly selects a random legal action while the game is not terminal and it
is **not** the perspective player's turn.  This means each node in the tree
corresponds to a state where the perspective player is to move.

```
while not terminal and current_player != perspective:
    action = random legal move
    state = apply action
```

This approach is well suited for quick training experiments or games where the
opponent's policy does not need to be modelled precisely.

## When to Use

`SinglePerspectiveMCTS` trades accuracy for speed.  Because it does not branch
on opponent moves the tree stays much smaller, allowing many more iterations in
the same amount of time.  It works best in settings where a rough opponent model
is sufficient or as a baseline before introducing a stronger adversary.
