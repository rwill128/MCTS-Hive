# MCTS-Hive

This repository contains a Monte-Carlo Tree Search implementation along with a collection of example scripts and simple board games used for experimentation.

## Directory overview

- `mcts/` – Core MCTS algorithms.
- `HivePocket/` – Simplified Hive board game implementation.
- `simple_games/` – Minimal games such as Tic-Tac-Toe and Connect Four.
- `examples/` – Scripts and experiments demonstrating usage of the library.
- `tests/` – Unit tests.

Player configuration JSON files for Connect Four and Tic-Tac-Toe remain in `c4_players/` and `ttt_players/` respectively.

Running the unit tests requires adding the repository root to `PYTHONPATH`:

```bash
PYTHONPATH=. python -m unittest discover tests
```
