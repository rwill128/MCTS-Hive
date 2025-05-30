# MCTS-Hive

This repository contains a Monte-Carlo Tree Search implementation along with a collection of example scripts and simple board games used for experimentation. The `MCTS` class also exposes an optional depth-limited minimax search. Set the `minimax_depth` parameter when constructing `MCTS` to enable a hybrid approach. When enabled, the minimax result always overrides the MCTS suggestion.

## Directory overview

- `mcts/` – Core MCTS algorithms.
- `HivePocket/` – Simplified Hive board game implementation.
- `simple_games/` – Minimal games such as Tic-Tac-Toe, Connect Four and Chess.
- `examples/` – Scripts and experiments demonstrating usage of the library. The
  `examples/c4_zero.py` training script now applies a simple learning rate
 schedule during self-play training.  The new `examples/c4_zero_advanced.py`
 example implements an AlphaZero-style loop with policy soft targets and
 Dirichlet noise.  It now saves the replay buffer and latest checkpoint so
 running the script with no arguments will automatically resume training.
- `tests/` – Unit tests.
- `docs/` – Additional documentation including [`connect_four_board.md`](docs/connect_four_board.md)
  , [`hive_board_representation.md`](docs/hive_board_representation.md)
  , [`go_board_representation.md`](docs/go_board_representation.md)
  , [`single_perspective_mcts.md`](docs/single_perspective_mcts.md)
  , [`tic_tac_toe_td_rl.md`](docs/tic_tac_toe_td_rl.md)
  , [`forced_move_pruning.md`](docs/forced_move_pruning.md)
 , [`mcts_search_algorithm.md`](docs/mcts_search_algorithm.md)
 , [`minimax_connect_four.md`](docs/minimax_connect_four.md).

Player configuration JSON files for Connect Four and Tic-Tac-Toe remain in `c4_players/` and `ttt_players/` respectively.

Running the unit tests requires adding the repository root to `PYTHONPATH`:

```bash
PYTHONPATH=. python -m unittest discover tests
```
