# Monte-Carlo Tree Search Overview

This document explains the `MCTS` class in `mcts/Mcts.py`.  It implements the classical four-step MCTS procedure:

1. **Selection** – choose child nodes according to UCB1 until a leaf node is reached.
2. **Expansion** – expand one unexplored action from that leaf.
3. **Simulation** – play out the game randomly (or with a policy) from the new state.
4. **Back-propagation** – update the statistics of each node on the path.

The code maintains a tree of `MCTSNode` objects, each storing:

```python
@dataclass(slots=True)
class MCTSNode:
    state: dict               # game state after the move leading here
    parent: Optional[MCTSNode]
    forced_depth_left: int
    children: dict            # action -> child node
    visit_count: int
    total_value: float
```

Only **edge statistics** are stored in the parent (visit count and total value of the edge leading to a child).  Children are indexed by the action that reaches them so that statistics remain associated with the originating move.

## Selection

`_select` starts from the root and repeatedly chooses `best_child` while the node is fully expanded and non-terminal.  `best_child` implements UCB1:

```python
exploit = child.total_value / child.visit_count
explore = sqrt(2 * ln(parent_visits) / child.visit_count)
score   = exploit + c_param * explore
```

The child with the highest score is chosen, breaking ties randomly.  This balances exploration and exploitation.

```
root
 ├─ a1 (3/5)
 │    └─ ...
 └─ a2 (2/5)
```

In the tree above `best_child` would compare the average win rate of each action against the exploration bonus.  The algorithm continues downwards until a leaf is found.

## Expansion

If the selected node is not terminal and not yet fully expanded, `expand` picks one of the remaining legal actions, applies it, and creates a new child node.  The child inherits a `forced_depth_left` counter used by the optional forced‑move pruning logic.

```
leaf
 ├─ a_new  <- expanded this iteration
```

## Simulation

`simulateRandomPlayout` from the game implementation performs a random playout (or uses a provided evaluation function) up to a maximum depth.  It returns a value in `[-1, 1]` from the perspective player's view: `1` for a win, `0` for a draw, and `-1` for a loss.

## Back‑propagation

`_backpropagate` adds the simulation result to `total_value` and increments `visit_count` for every node along the path back to the root, flipping the sign each step because turns alternate between players.

```python
while node is not None:
    node.update(value)
    node = node.parent
    value = -value  # switch perspective
```

This accumulates statistics for UCB1 and gradually improves the action estimates.

## Choosing an Action

After all iterations, `_best_action` simply returns the child of the root with the highest `visit_count`—the move explored most often.  When `minimax_depth` is configured, a depth‑limited minimax search is run from the root and its result overrides the MCTS suggestion, enabling a hybrid strategy.

## Relationship to the Code

The following code locations correspond to the steps described above:

- `MCTS.search` – orchestrates the iterations.
- `MCTSNode.best_child` – UCB1 selection.
- `MCTSNode.expand` – adds a new child node.
- `MCTS._select` – walks the tree until expansion is needed.
- `MCTS._backpropagate` – updates values after simulation.
- `MCTS._best_action` – picks the final move.

Studying these functions alongside this document provides a complete picture of how Monte‑Carlo Tree Search is implemented in this repository.
