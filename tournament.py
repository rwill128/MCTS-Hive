#!/usr/bin/env python3
"""
Hive MCTS tournament + live‑view mode
====================================

Two ways to run
---------------
▶ *Batch / headless* (fast):
   $ python tournament.py --games 20        # default mode, SDL dummy driver

▶ *Watch a single match* (visual):
   $ python tournament.py --gui --games 1   # real window, board animates

The GUI slows search (pygame draws every ply) but is handy for debugging
behaviour.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ------------------------------------------------------------------
# CLI flags need to be parsed *before* importing pygame so we know
# whether to run in dummy headless mode or open a real window.
# ------------------------------------------------------------------
GUI_MODE = "--gui" in sys.argv

if not GUI_MODE:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from HivePocket.HivePocket import HiveGame  # noqa: E402
from mcts.Mcts import MCTS                 # noqa: E402

PLAYERS_DIR = Path("players")

# -----------------------------------------------------
# Hex‑board drawing helpers (only used in GUI mode)
# -----------------------------------------------------
HEX_SIZE   = 35
OFFSET_X   = 400
OFFSET_Y   = 300


def hex_to_pixel(q: int, r: int) -> Tuple[int, int]:
    x = HEX_SIZE * math.sqrt(3) * (q + r / 2)
    y = HEX_SIZE * 1.5 * r
    return int(x + OFFSET_X), int(y + OFFSET_Y)


def polygon_corners(center):
    cx, cy = center
    pts = []
    for i in range(6):
        ang = math.radians(60 * i - 30)
        pts.append((cx + HEX_SIZE * math.cos(ang), cy + HEX_SIZE * math.sin(ang)))
    return pts


def draw_board(state: dict, surface):
    surface.fill((255, 255, 255))
    if not state["board"]:
        q_min = r_min = -3
        q_max = r_max = 3
    else:
        qs = [q for (q, _) in state["board"]]
        rs = [r for (_, r) in state["board"]]
        q_min, q_max = min(qs) - 2, max(qs) + 2
        r_min, r_max = min(rs) - 2, max(rs) + 2
    for q in range(q_min, q_max + 1):
        for r in range(r_min, r_max + 1):
            center = hex_to_pixel(q, r)
            pygame.draw.polygon(surface, (200, 200, 200), polygon_corners(center), 1)
    for (q, r), stack in state["board"].items():
        if stack:
            owner, insect = stack[-1]
            color = (0, 0, 255) if owner == "Player1" else (255, 0, 0)
            center = hex_to_pixel(q, r)
            pygame.draw.polygon(surface, color, polygon_corners(center))
            text = pygame.font.SysFont(None, 20).render(insect[0], True, (255, 255, 255))
            rect = text.get_rect(center=center)
            surface.blit(text, rect)


# -----------------------------------------------------
# Player parameter loading
# -----------------------------------------------------

def load_agents() -> Dict[str, dict]:
    if not PLAYERS_DIR.exists():
        raise FileNotFoundError("'players/' directory not found. Run --init-players first.")
    param_files = sorted(PLAYERS_DIR.glob("*.json"))
    if not param_files:
        raise FileNotFoundError("No *.json files in 'players/'.")
    agents = {}
    for path in param_files:
        with path.open() as f:
            params = json.load(f)
        agents[path.stem] = params
    return agents


# -----------------------------------------------------
# Single game execution (with optional GUI)
# -----------------------------------------------------

def play_one_game(params_white: dict, params_black: dict, seed: int, visualize: bool = False) -> int:
    """Return +1/0/‑1 from White's perspective."""
    random.seed(seed)
    game = HiveGame()
    state = game.getInitialState()

    mcts_white = MCTS(game, **params_white)
    mcts_black = MCTS(game, **params_black)

    if visualize:
        screen = pygame.display.set_mode((800, 600))
        surface = pygame.display.get_surface()
        draw_board(state, surface)
        pygame.display.flip()

    while not game.isTerminal(state):
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        current = state["current_player"]
        action = (mcts_white if current == "Player1" else mcts_black).search(state)
        state = game.applyAction(state, action)

        if visualize:
            draw_board(state, surface)
            pygame.display.flip()
            pygame.time.wait(300)  # slow it down so humans can follow

    outcome = game.getGameOutcome(state)
    if visualize:
        print("Game finished: winner =", outcome)
        pygame.time.wait(1500)
    return +1 if outcome == "Player1" else -1 if outcome == "Player2" else 0


# -----------------------------------------------------
# Elo helper
# -----------------------------------------------------

def expected(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

def update(r, s, e, k=16):
    return r + k * (s - e)

def elo(names: List[str], results: Dict[Tuple[int, int], Tuple[int, int, int]]):
    rating = {n: 1500.0 for n in names}
    for (i, j), (w, d, l) in results.items():
        tot = w + d + l
        if tot == 0:
            continue
        s_i = (w + 0.5 * d) / tot
        s_j = 1 - s_i
        e_i = expected(rating[names[i]], rating[names[j]])
        rating[names[i]] = update(rating[names[i]], s_i, e_i)
        rating[names[j]] = update(rating[names[j]], s_j, 1 - e_i)
    return rating


# -----------------------------------------------------
# Tournament driver
# -----------------------------------------------------

def run_tournament(num_games: int, visualize: bool):
    agents = load_agents()
    names = list(agents)
    n = len(names)
    results = {}
    start = time.time()

    for i in range(n):
        for j in range(i + 1, n):
            w = d = l = 0
            for k in range(num_games):
                seed = k
                score = play_one_game(agents[names[i]], agents[names[j]], seed, visualize)
                if score == 1:
                    w += 1
                elif score == 0:
                    d += 1
                else:
                    l += 1
                # Swap colours
                score = play_one_game(agents[names[j]], agents[names[i]], seed + 10000, visualize)
                if score == -1:
                    w += 1
                elif score == 0:
                    d += 1
                else:
                    l += 1
            results[(i, j)] = (w, d, l)
            print(f"{names[i]} vs {names[j]} -> W/D/L = {w}/{d}/{l}")

    rating = elo(names, results)
    print("\n=== Elo standings ===")
    for n, r in sorted(rating.items(), key=lambda x: -x[1]):
        print(f"{n:<20} {r:6.1f}")
    print(f"Tournament time: {time.time() - start:.1f}s")


# -----------------------------------------------------
# Sample params generator
# -----------------------------------------------------
SAMPLE = {
    "baseline": {
        "num_iterations": 400, "max_depth": 12, "c_param": 1.4,
        "forced_check_depth": 0,
        "weights": {"winning_score": 1e4, "queen_factor": 50, "liberties_factor": 10,
                    "mobility_factor": 3, "early_factor": 2},
        "perspective_player": "Player1"
    },
    "explore":   {
        "num_iterations": 400, "max_depth": 12, "c_param": 2.0,
        "forced_check_depth": 0,
        "weights": {"winning_score": 1e4, "queen_factor": 50, "liberties_factor": 10,
                    "mobility_factor": 3, "early_factor": 2},
        "perspective_player": "Player1"
    }
}

def init_players():
    PLAYERS_DIR.mkdir(exist_ok=True)
    for nm, p in SAMPLE.items():
        for pl in ("Player1", "Player2"):
            cfg = json.loads(json.dumps(p))
            cfg["perspective_player"] = pl
            out = PLAYERS_DIR / f"{nm}_{pl}.json"
            if not out.exists():
                with out.open("w") as f:
                    json.dump(cfg, f, indent=2)
                print("wrote", out)


# -----------------------------------------------------
# Main -------------------------------------------------
# -----------------------------------------------------
if __name__ == "__main__":
    if "--init-players" in sys.argv:
        init_players()
        sys.exit(0)

    # ensure pygame started (needed even in dummy mode)
    if not pygame.get_init():
        pygame.init()
    if not pygame.display.get_init():
        pygame.display.set_mode((1, 1))

    games = 10
    if "--games" in sys.argv:
        games = int(sys.argv[sys.argv.index("--games") + 1])

    run_tournament(games, GUI_MODE)
