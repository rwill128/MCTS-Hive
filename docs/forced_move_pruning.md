# Forced Move Pruning in MCTS

The `MCTS` implementation includes an optional *forced move* pruning step
that tries to detect immediate wins (or threats) before expanding a node.
When a node is expanded with a non‑zero `forced_depth_left`, the algorithm
performs a depth limited search to see whether the current player can force
a win or avoid defeat within that many moves.

If a winning line is found, only the winning action is kept.  Otherwise the
children are pruned to the subset of moves that do not allow the opponent a
forced reply.  This reduces the branching factor in tactical situations
where the correct continuation is forced.

```
expand(node):
    if node.forced_depth_left > 0:
        win, safe = forced_move_check(node.state, node.forced_depth_left)
        if win is not None:
            allowed = [win]
        else:
            allowed = safe
        node.prune_to_actions(allowed)
        node.forced_depth_left = 0
    # normal expansion continues here
```

The helper `_forced_move_check` uses `_forced_check_depth_limited` to
verify forced wins recursively:

```
forced_move_check(state, depth):
    for action in legal_moves(state):
        next_state = apply(action, state)
        if is_terminal(next_state):
            if winner(next_state) == current_player(state):
                return action, []
            if winner is draw:
                add action to safe moves
            continue
        if forced_check_depth_limited(next_state, current_player, depth-1):
            return action, []           # winning line found
        if not forced_check_depth_limited(next_state, opponent, depth-1):
            add action to safe moves    # opponent has no forced win
    return None, safe moves
```

This procedure is much cheaper than a full minimax search yet can quickly
recognise sequences of mandatory moves.  It is particularly useful in
Hive where piece placements can create threats that must be answered
immediately.
