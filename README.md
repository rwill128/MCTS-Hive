# MCTS-Hive

This repository contains a Monte-Carlo Tree Search implementation along with a collection of example scripts and simple board games used for experimentation. The `MCTS` class also exposes an optional depth-limited minimax search. Set the `minimax_depth` parameter when constructing `MCTS` to enable a hybrid approach. When enabled, the minimax result always overrides the MCTS suggestion.

## Directory overview

- `mcts/` – Core MCTS algorithms.
- `HivePocket/` – Simplified Hive board game implementation.
- `simple_games/` – Minimal games such as Tic-Tac-Toe, Connect Four and Chess.
- `examples/` – Scripts and experiments demonstrating usage of the library.
- `tests/` – Unit tests.
- `docs/` – Additional documentation including [`connect_four_board.md`](docs/connect_four_board.md)
  and [`single_perspective_mcts.md`](docs/single_perspective_mcts.md).

Player configuration JSON files for Connect Four and Tic-Tac-Toe remain in `c4_players/` and `ttt_players/` respectively.

Running the unit tests requires adding the repository root to `PYTHONPATH`:

```bash
PYTHONPATH=. python -m unittest discover tests
```
