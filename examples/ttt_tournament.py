#!/usr/bin/env python3
"""Round-robin MCTS tournament for Tic-Tac-Toe.

This script loads player configurations from ``ttt_players/*.json`` and
runs a continuous Elo tournament. Results are written to
``ttt_results.json`` after every game so progress is preserved between
runs. Players with fewer recorded games are matched first so new
configs quickly obtain ratings. The tournament loops endlessly and can
be stopped by typing ``quit`` (or pressing ``Ctrl+C``) after any game.
"""
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
from mcts.alpha_zero_mcts import AlphaZeroMCTS
from simple_games.tic_tac_toe import TicTacToe
from simple_games.perfect_tic_tac_toe import PerfectTicTacToePlayer

# Import necessary components from ttt_zero_advanced for the new player
# We need the network definition, the adapter, and the state encoding
from examples.ttt_zero_advanced import AdvancedTTTZeroNet, TicTacToeAdapter, encode_ttt_state, torch

try:
    from simple_games.ttt_visualizer import init_display, draw_board
except Exception:  # pragma: no cover - pygame optional
    init_display = draw_board = None

PLAYERS_DIR = Path("../ttt_players")
RESULTS_FILE = Path("ttt_results.json")
K_FACTOR = 16


# --- Define AlphaZeroTTTPlayer class here ---
class AlphaZeroTTTPlayer:
    def __init__(self, game_instance: TicTacToe, perspective_player: str, 
                 model_path: str, device: str = "cpu",
                 mcts_simulations: int = 50, c_puct: float = 1.0, 
                 dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25,
                 nn_channels: int = 32, nn_blocks: int = 2): # NN arch params
        if torch is None: # Check if torch was successfully imported earlier
            raise ImportError("PyTorch is required for AlphaZeroTTTPlayer but was not imported.")

        self.game_adapter = TicTacToeAdapter(game_instance) # Use the same adapter
        self.perspective_player = perspective_player
        self.device = torch.device(device)

        # Load the trained neural network
        self.net = AdvancedTTTZeroNet(ch=nn_channels, blocks=nn_blocks).to(self.device)
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.net.load_state_dict(state_dict)
        except Exception:
            try:
                state_dict_full = torch.load(model_path, map_location=self.device)
                if "net" in state_dict_full and isinstance(state_dict_full["net"], dict):
                    self.net.load_state_dict(state_dict_full["net"])
                elif isinstance(state_dict_full, dict):
                    self.net.load_state_dict(state_dict_full)
                else:
                    raise ValueError("Could not determine how to load model weights from checkpoint.")
            except Exception as e:
                raise IOError(f"Failed to load model from {model_path}: {e}")
        self.net.eval()

        # MCTS model function
        def mcts_model_fn(encoded_state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                policy_logits, value_estimates = self.net(encoded_state_batch)
            return policy_logits, value_estimates.unsqueeze(-1)

        self.mcts_instance = AlphaZeroMCTS(
            game_interface=self.game_adapter,
            model_fn=mcts_model_fn,
            device=self.device,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon
        )
        self.mcts_simulations = mcts_simulations

    def search(self, state: dict) -> Tuple[int,int]: 
        int_action, _ = self.mcts_instance.get_action_policy(
            root_state=state, 
            num_simulations=self.mcts_simulations, 
            temperature=0.0 
        )
        return self.game_adapter._int_to_action(int_action)
# --- End AlphaZeroTTTPlayer class definition ---


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


def play_one_game(
        game: TicTacToe,
        params_x: dict,
        params_o: dict,
        seed: int,
        screen=None,
) -> int:
    random.seed(seed)

    def make_player(cfg, role):
        player_type = cfg.get("type", "mcts") # Default to original MCTS if type not specified
        if player_type == "perfect":
            return PerfectTicTacToePlayer(game, perspective_player=role)
        elif player_type == "alphazero_mcts": # Our new player type
            # Extract necessary params from cfg for AlphaZeroTTTPlayer
            # Ensure model_path is absolute or relative to a known location if needed
            # For now, assume model_path in JSON is findable.
            model_path = cfg.get("model_path")
            if not model_path:
                raise ValueError("AlphaZero player config must specify 'model_path'.")
            
            # Pass through relevant MCTS and NN architecture parameters from JSON
            return AlphaZeroTTTPlayer(
                game_instance=game, 
                perspective_player=role, 
                model_path=model_path,
                device=cfg.get("device", "cpu"), # Allow device config in JSON
                mcts_simulations=cfg.get("mcts_simulations", 50),
                c_puct=cfg.get("c_puct", 1.0),
                dirichlet_alpha=cfg.get("dirichlet_alpha", 0.3), # Noise usually off for eval
                dirichlet_epsilon=cfg.get("dirichlet_epsilon", 0.0), # Turn off noise for eval by default
                nn_channels=cfg.get("nn_channels", 32), # Match defaults from ttt_zero_advanced
                nn_blocks=cfg.get("nn_blocks", 2)
            )
        else: # Fallback to original MCTS player
            # The original MCTS takes **cfg directly for its own params
            return MCTS(game=game, perspective_player=role, **cfg)

    mcts_x = make_player(params_x, "X")
    mcts_o = make_player(params_o, "O")
    state = game.getInitialState()
    if screen is not None:
        draw_board(screen, state["board"])
        pygame.event.pump()
    while not game.isTerminal(state):
        to_move = game.getCurrentPlayer(state)
        if to_move == "X":
            action = mcts_x.search(state)
        else:
            action = mcts_o.search(state)
        state = game.applyAction(state, action)
        if screen is not None:
            draw_board(screen, state["board"])
            pygame.event.pump()
            pygame.time.delay(300)
    outcome = game.getGameOutcome(state)
    return 1 if outcome == "X" else -1 if outcome == "O" else 0


def run(display: bool = True) -> None:
    game = TicTacToe()
    players = load_players()
    names = list(players)
    data = load_results(names)
    if display and pygame is not None and init_display is not None:
        screen = init_display()
    else:
        screen = None

    g = 0
    try:
        while True:
            (i, j), orientation = choose_pair(names, data)
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
                seed=random.randint(0, 2 ** 32 - 1),
                screen=screen,
            )

            # Keep orientation-specific results separated by always putting the
            # X player first.  The previous conditional collapsed both
            # orientations into the same key, miscounting games.
            key = f"{x_name}_vs_{o_name}"
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

            ra = data["ratings"].setdefault(x_name, 1500.0)
            rb = data["ratings"].setdefault(o_name, 1500.0)
            ea = expected(ra, rb)
            eb = expected(rb, ra)
            data["ratings"][x_name] = update(ra, score_a, ea)
            data["ratings"][o_name] = update(rb, 1 - score_a, eb)

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
    finally:
        if screen is not None and pygame is not None:
            pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init-players",
        action="store_true",
        help="create sample configs in ttt_players/",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="run tournament without the pygame visualizer",
    )
    args = parser.parse_args()

    run(display=True)
