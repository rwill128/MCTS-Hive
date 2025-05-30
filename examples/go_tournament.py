#!/usr/bin/env python3
"""Round-robin MCTS tournament for the Go game."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple, List

try:
    import pygame
except ImportError:  # pragma: no cover - allow headless use
    pygame = None

from mcts.Mcts import MCTS
from simple_games.go import Go
try:
    from simple_games.go_visualizer import init_display, draw_board
except Exception:  # pragma: no cover - pygame optional
    init_display = draw_board = None

PLAYERS_DIR = Path("../go_players")
RESULTS_FILE = Path("go_results.json")
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
        raise FileNotFoundError("No player configs found in 'go_players/'.")
    return players


def load_results(names) -> dict:
    if RESULTS_FILE.exists():
        with RESULTS_FILE.open() as f:
            data = json.load(f)
    else:
        data = {"ratings": {}, "pair_results": {}, "pair_index": 0, "orientation": 0}

    for n in names:
        data.setdefault("ratings", {}).setdefault(n, 1500.0)

    obsolete = set(data["ratings"].keys()) - set(names)
    for n in obsolete:
        del data["ratings"][n]

    return data


def save_results(data: dict) -> None:
    with RESULTS_FILE.open("w") as f:
        json.dump(data, f, indent=2)


def game_counts(data: dict, names: List[str]) -> Dict[str, int]:
    counts = {n: 0 for n in names}
    for key, record in data.get("pair_results", {}).items():
        a, _, b = key.partition("_vs_")
        games = record.get("w", 0) + record.get("d", 0) + record.get("l", 0)
        counts[a] = counts.get(a, 0) + games
        counts[b] = counts.get(b, 0) + games
    return counts


def choose_pair(names: List[str], data: dict) -> Tuple[Tuple[int, int], int]:
    counts = game_counts(data, names)
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]

    min_total = None
    candidates = []
    for i, j in pairs:
        total = counts.get(names[i], 0) + counts.get(names[j], 0)
        if min_total is None or total < min_total:
            min_total = total
            candidates = [(i, j)]
        elif total == min_total:
            candidates.append((i, j))

    i, j = random.choice(candidates)

    key_ab = f"{names[i]}_vs_{names[j]}"
    key_ba = f"{names[j]}_vs_{names[i]}"
    count_ab = sum(data.get("pair_results", {}).get(key_ab, {}).values())
    count_ba = sum(data.get("pair_results", {}).get(key_ba, {}).values())
    if count_ab < count_ba:
        orientation = 0
    elif count_ba < count_ab:
        orientation = 1
    else:
        orientation = random.randint(0, 1)

    return (i, j), orientation


def play_one_game(game: Go, params_b: dict, params_w: dict, seed: int, screen=None) -> int:
    random.seed(seed)
    b_player = MCTS(game=game, perspective_player="B", **params_b)
    w_player = MCTS(game=game, perspective_player="W", **params_w)
    state = game.getInitialState()
    if screen is not None:
        draw_board(screen, state["board"])
        pygame.event.pump()
    while not game.isTerminal(state):
        to_move = game.getCurrentPlayer(state)
        if to_move == "B":
            action = b_player.search(state)
        else:
            action = w_player.search(state)
        state = game.applyAction(state, action)
        if screen is not None:
            draw_board(screen, state["board"])
            pygame.event.pump()
            pygame.time.delay(300)
    outcome = game.getGameOutcome(state)
    return 1 if outcome == "B" else -1 if outcome == "W" else 0


def run(display: bool = True) -> None:
    game = Go(size=9)
    players = load_players()
    names = list(players)
    data = load_results(names)
    if display and pygame is not None and init_display is not None:
        screen = init_display(game.size)
    else:
        screen = None

    g = 0
    try:
        while True:
            (i, j), orientation = choose_pair(names, data)
            if orientation == 0:
                b_name, w_name = names[i], names[j]
            else:
                b_name, w_name = names[j], names[i]

            g += 1
            print(f"Game {g}: {b_name} (B) vs {w_name} (W)")
            result = play_one_game(game, players[b_name], players[w_name], seed=random.randint(0, 2**32 - 1), screen=screen)

            key = f"{b_name}_vs_{w_name}"
            record = data["pair_results"].setdefault(key, {"w": 0, "d": 0, "l": 0})
            if result == 1:
                record["w"] += 1
                score_a = 1.0
                print("  B wins")
            elif result == 0:
                record["d"] += 1
                score_a = 0.5
                print("  Draw")
            else:
                record["l"] += 1
                score_a = 0.0
                print("  W wins")

            ra = data["ratings"][b_name]
            rb = data["ratings"][w_name]
            ea = expected(ra, rb)
            eb = expected(rb, ra)
            data["ratings"][b_name] = update(ra, score_a, ea)
            data["ratings"][w_name] = update(rb, 1 - score_a, eb)

            save_results(data)

            print("Current Elo standings:")
            for n, r in sorted(data["ratings"].items(), key=lambda x: -x[1]):
                print(f"  {n:<15} {r:6.1f}")
            print()

            try:
                cmd = "continue"
            except EOFError:
                cmd = ""
            if cmd.strip().lower() in {"q", "quit", "exit"}:
                break
    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved.")
    finally:
        if screen is not None and pygame is not None:
            pygame.quit()


def init_players() -> None:
    PLAYERS_DIR.mkdir(exist_ok=True)
    samples = {
        "iter20": {"num_iterations": 20, "max_depth": 100, "c_param": 1.4, "forced_check_depth": 0},
        "iter50": {"num_iterations": 50, "max_depth": 100, "c_param": 1.4, "forced_check_depth": 0},
        "iter100": {"num_iterations": 100, "max_depth": 100, "c_param": 1.4, "forced_check_depth": 0},
        "iter200": {"num_iterations": 200, "max_depth": 100, "c_param": 1.4, "forced_check_depth": 0},
        "c1":      {"num_iterations": 100, "max_depth": 100, "c_param": 1.0, "forced_check_depth": 0},
    }
    for name, cfg in samples.items():
        path = PLAYERS_DIR / f"{name}.json"
        if not path.exists():
            with path.open("w") as f:
                json.dump(cfg, f, indent=2)
            print("wrote", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-players", action="store_true", help="create sample configs in go_players/")
    parser.add_argument("--no-display", action="store_true", help="run tournament without the pygame visualizer")
    args = parser.parse_args()

    if args.init_players:
        init_players()
    else:
        run(display=not args.no_display)
