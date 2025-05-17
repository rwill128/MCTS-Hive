#!/usr/bin/env python3
"""Round-robin MCTS tournament for Tic-Tac-Toe.

This script loads player configurations from ``ttt_players/*.json`` and
runs a continuous Elo tournament. Results are written to
``ttt_results.json`` after every game so progress is preserved between
runs. The tournament loops endlessly and can be stopped by typing
``quit`` (or pressing ``Ctrl+C``) after any game.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

from mcts.Mcts import MCTS
from simple_games.tic_tac_toe import TicTacToe

PLAYERS_DIR = Path("ttt_players")
RESULTS_FILE = Path("ttt_results.json")
K_FACTOR = 16


def expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400))


def update(r: float, s: float, e: float, k: float = K_FACTOR) -> float:
    return r + k * (s - e)


def load_players() -> Dict[str, dict]:
    players = {}
    for path in sorted(PLAYERS_DIR.glob("*.json")):
        with path.open() as f:
            players[path.stem] = json.load(f)
    if not players:
        raise FileNotFoundError("No player configs found in 'ttt_players/'.")
    return players


def load_results(names) -> dict:
    if RESULTS_FILE.exists():
        with RESULTS_FILE.open() as f:
            data = json.load(f)
    else:
        data = {
            "ratings": {n: 1500.0 for n in names},
            "pair_results": {},
            "pair_index": 0,
            "orientation": 0,
        }
    return data


def save_results(data: dict) -> None:
    with RESULTS_FILE.open("w") as f:
        json.dump(data, f, indent=2)


def next_game(pair_idx: int, orientation: int, total_pairs: int) -> Tuple[int, int]:
    orientation += 1
    if orientation == 2:
        orientation = 0
        pair_idx = (pair_idx + 1) % total_pairs
    return pair_idx, orientation


def play_one_game(game: TicTacToe, params_x: dict, params_o: dict, seed: int) -> int:
    random.seed(seed)
    mcts_x = MCTS(game=game, perspective_player="X", **params_x)
    mcts_o = MCTS(game=game, perspective_player="O", **params_o)
    state = game.getInitialState()
    while not game.isTerminal(state):
        to_move = game.getCurrentPlayer(state)
        if to_move == "X":
            action = mcts_x.search(state)
        else:
            action = mcts_o.search(state)
        state = game.applyAction(state, action)
    outcome = game.getGameOutcome(state)
    return 1 if outcome == "X" else -1 if outcome == "O" else 0


def run() -> None:
    game = TicTacToe()
    players = load_players()
    names = list(players)
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
    data = load_results(names)
    pair_idx = data.get("pair_index", 0)
    orientation = data.get("orientation", 0)

    g = 0
    try:
        while True:
            i, j = pairs[pair_idx]
            if orientation == 0:
                x_name, o_name = names[i], names[j]
            else:
                x_name, o_name = names[j], names[i]

            g += 1
            print(f"Game {g}: {x_name} (X) vs {o_name} (O)")
            result = play_one_game(
                game,
                players[x_name],
                players[o_name],
                seed=random.randint(0, 2**32 - 1),
            )

            key = (
                f"{x_name}_vs_{o_name}"
                if orientation == 0
                else f"{o_name}_vs_{x_name}"
            )
            record = data["pair_results"].setdefault(key, {"w": 0, "d": 0, "l": 0})
            if result == 1:
                record["w"] += 1
                score_a = 1.0
                print("  X wins")
            elif result == 0:
                record["d"] += 1
                score_a = 0.5
                print("  Draw")
            else:
                record["l"] += 1
                score_a = 0.0
                print("  O wins")

            ra = data["ratings"][x_name]
            rb = data["ratings"][o_name]
            ea = expected(ra, rb)
            eb = expected(rb, ra)
            data["ratings"][x_name] = update(ra, score_a, ea)
            data["ratings"][o_name] = update(rb, 1 - score_a, eb)

            pair_idx, orientation = next_game(pair_idx, orientation, len(pairs))
            data["pair_index"] = pair_idx
            data["orientation"] = orientation
            save_results(data)

            print("Current Elo standings:")
            for n, r in sorted(data["ratings"].items(), key=lambda x: -x[1]):
                print(f"  {n:<15} {r:6.1f}")
            print()

            try:
                # cmd = input("Press Enter to continue or type 'quit' to exit: ")
                cmd = "continue"
            except EOFError:
                cmd = ""
            if cmd.strip().lower() in {"q", "quit", "exit"}:
                break
    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved.")


def init_players() -> None:
    PLAYERS_DIR.mkdir(exist_ok=True)
    samples = {
        "iter20":  {"num_iterations": 20,  "max_depth": 9, "c_param": 1.4, "forced_check_depth": 0},
        "iter50":  {"num_iterations": 50,  "max_depth": 9, "c_param": 1.4, "forced_check_depth": 0},
        "iter100": {"num_iterations": 100, "max_depth": 9, "c_param": 1.4, "forced_check_depth": 0},
        "iter150": {"num_iterations": 150, "max_depth": 9, "c_param": 1.4, "forced_check_depth": 0},
        "iter200": {"num_iterations": 200, "max_depth": 9, "c_param": 1.4, "forced_check_depth": 0},
        "iter300": {"num_iterations": 300, "max_depth": 9, "c_param": 1.4, "forced_check_depth": 0},
        "iter400": {"num_iterations": 400, "max_depth": 9, "c_param": 1.4, "forced_check_depth": 0},
        "iter600": {"num_iterations": 600, "max_depth": 9, "c_param": 1.4, "forced_check_depth": 0},
        "c2":      {"num_iterations": 200, "max_depth": 9, "c_param": 2.0, "forced_check_depth": 0},
        "c3":      {"num_iterations": 200, "max_depth": 9, "c_param": 3.0, "forced_check_depth": 0},
    }
    for name, cfg in samples.items():
        path = PLAYERS_DIR / f"{name}.json"
        if not path.exists():
            with path.open("w") as f:
                json.dump(cfg, f, indent=2)
            print("wrote", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init-players",
        action="store_true",
        help="create sample configs in ttt_players/",
    )
    args = parser.parse_args()

    if args.init_players:
        init_players()
    else:
        run()
