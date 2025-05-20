#!/usr/bin/env python3
"""Round-robin MCTS tournament for Connect Four.

This script loads player configurations from ``c4_players/*.json`` and
runs a continuous Elo tournament. Results are written to
``c4_results.json`` after every game so progress is preserved between
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

from mcts.alpha_zero_mcts import AlphaZeroMCTS

try:
    import pygame
except ImportError:  # pragma: no cover - allow headless use
    pygame = None
try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None

from mcts.Mcts import MCTS
from simple_games.connect_four import ConnectFour
from simple_games.minimax_connect_four import MinimaxConnectFourPlayer
# Basic Zero network utilities
from examples.c4_zero import (
    ZeroC4Player,
    C4ZeroNet,
    load_weights,
    encode_state,
    HAS_TORCH,
)

# Try to import the *advanced* Zero network so tournament games can use
# checkpoints produced by ``c4_zero_advanced.py``.  If the import fails (PyTorch
# not available or file missing) the variable will be ``None`` and the code
# silently continues with the basic architecture.
try:
    from examples.c4_zero_advanced import AdvancedC4ZeroNet as C4AdvZeroNet  # type: ignore
except Exception:  # pragma: no cover – advanced network optional
    C4AdvZeroNet = None

# Import new style network from c4_zero_advanced for new models
from examples.c4_zero_advanced import AdvancedC4ZeroNet as AdvancedC4ZeroNet_NewStyle
from examples.c4_zero_advanced import ConnectFourAdapter as C4Adapter_NewStyle # Renaming for clarity
from examples.c4_zero_advanced import encode_c4_state # Using this encoding
from examples.ttt_zero_advanced import torch, nn, F, np # Common torch/np imports

# --- BEGIN Definition of Old Sequential Network Architecture ---
if torch is not None:
    class ResidualBlock_Old(nn.Module):
        def __init__(self, ch: int):
            super().__init__()
            self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.b1 = nn.BatchNorm2d(ch)
            self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.b2 = nn.BatchNorm2d(ch)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = F.relu(self.b1(self.c1(x)))
            y = self.b2(self.c2(y))
            return F.relu(x + y)

    class AdvancedC4ZeroNet_OldSequential(nn.Module):
        def __init__(self, ch: int = 128, blocks: int = 10):
            super().__init__()
            # Constants for C4 board dimensions, assuming they are available globally or passed
            # For safety, let's define them here if not already present from other imports
            BOARD_H_C4 = 6 # ConnectFour.ROWS
            BOARD_W_C4 = 7 # ConnectFour.COLS

            self.stem = nn.Sequential(
                nn.Conv2d(3, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(),
            )
            self.res = nn.Sequential(*[ResidualBlock_Old(ch) for _ in range(blocks)])
            self.policy = nn.Sequential(
                nn.Conv2d(ch, 2, 1, bias=True), # Original might have had bias=True here
                nn.BatchNorm2d(2), 
                nn.ReLU(),
                nn.Flatten(), 
                nn.Linear(2 * BOARD_H_C4 * BOARD_W_C4, BOARD_W_C4)
            )
            self.value = nn.Sequential(
                nn.Conv2d(ch, 1, 1, bias=True), # Original might have had bias=True here
                nn.BatchNorm2d(1), 
                nn.ReLU(),
                nn.Flatten(), 
                nn.Linear(BOARD_H_C4 * BOARD_W_C4, 64), 
                nn.ReLU(),
                nn.Linear(64, 1), 
                nn.Tanh()
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            x = self.res(self.stem(x))
            return self.policy(x), self.value(x).squeeze(1)
else:
    class AdvancedC4ZeroNet_OldSequential: pass # Placeholder if torch not found
# --- END Definition of Old Sequential Network Architecture ---


# --- AlphaZero Player for Connect Four (using new style by default) ---
class AlphaZeroC4Player:
    def __init__(self, game_instance: ConnectFour, perspective_player: str, 
                 model_path: str, device: str = "cpu",
                 mcts_simulations: int = 50, c_puct: float = 1.41, 
                 dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.0, # Epsilon 0 for eval
                 nn_channels: int = 128, nn_blocks: int = 10, architecture: str = "new_style"):
        if torch is None: raise ImportError("PyTorch required")

        self.game_adapter = C4Adapter_NewStyle(game_instance) 
        self.perspective_player = perspective_player
        self.device = torch.device(device)
        self.architecture = architecture

        if self.architecture == "old_sequential":
            self.net = AdvancedC4ZeroNet_OldSequential(ch=nn_channels, blocks=nn_blocks).to(self.device)
            print(f"Loading AlphaZeroC4Player with OLD SEQUENTIAL architecture for model: {model_path}")
        else: # Default to new style
            self.net = AdvancedC4ZeroNet_NewStyle(ch=nn_channels, blocks=nn_blocks).to(self.device)
            print(f"Loading AlphaZeroC4Player with NEW STYLE architecture for model: {model_path}")
        
        try:
            # General loading, hoping the state_dict matches the chosen architecture
            state_dict = torch.load(model_path, map_location=self.device)
            if "net" in state_dict and isinstance(state_dict["net"], dict): # From full training state
                self.net.load_state_dict(state_dict["net"])
            elif isinstance(state_dict, dict): # Just the model state_dict
                self.net.load_state_dict(state_dict)
            else:
                raise ValueError("Checkpoint format not recognized for chosen architecture.")
            print(f"Successfully loaded model weights: {model_path}")
        except Exception as e:
            raise IOError(f"Failed to load model {model_path} with {self.architecture} arch: {e}")
        self.net.eval()

        def mcts_model_fn(encoded_state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                policy_logits, value_estimates = self.net(encoded_state_batch)
            return policy_logits, value_estimates.unsqueeze(-1)

        self.mcts_instance = AlphaZeroMCTS(
            game_interface=self.game_adapter, model_fn=mcts_model_fn, device=self.device,
            c_puct=c_puct, dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon
        )
        self.mcts_simulations = mcts_simulations

    def search(self, state: dict) -> int: # Returns column index for C4
        int_action, _ = self.mcts_instance.get_action_policy(
            root_state=state, num_simulations=self.mcts_simulations, temperature=0.0)
        return int_action # C4 actions are already integers
# --- End AlphaZeroC4Player ---


class ZeroGuidedMCTS(MCTS):
    """MCTS variant that uses a Zero network to guide simulations."""

    def __init__(self, game: ConnectFour, net: C4ZeroNet, perspective_player: str, temperature: float = 0.0, **kwargs):
        super().__init__(game=game, perspective_player=perspective_player, **kwargs)
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for ZeroGuidedMCTS")
        self.net = net
        self.temperature = temperature

    def _guided_action(self, state: dict) -> int:
        tensor = encode_state(state, state["current_player"]).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(tensor)
            probs = logits.softmax(1)[0].tolist()
        legal = self.game.getLegalActions(state)
        masked = [probs[a] if a in legal else 0.0 for a in range(self.game.COLS)]
        if sum(masked) == 0:
            return random.choice(legal)
        if self.temperature > 0:
            dist = [p ** (1.0 / self.temperature) for p in masked]
            s = sum(dist)
            dist = [p / s for p in dist]
            return random.choices(range(self.game.COLS), dist)[0]
        return max(legal, key=lambda a: masked[a])

    def _simulate(self, state: dict) -> float:
        temp_state = self.game.copyState(state)
        depth = 0
        while not self.game.isTerminal(temp_state) and depth < self.max_depth:
            action = self._guided_action(temp_state)
            temp_state = self.game.applyAction(temp_state, action)
            depth += 1
        outcome = self.game.getGameOutcome(temp_state)
        if outcome == self.perspective_player:
            return 1.0
        if outcome == "Draw" or outcome is None:
            return 0.0
        return -1.0

try:
    from simple_games.c4_visualizer import (
        init_display,
        draw_board,
        draw_board_with_action_values,
    )
except Exception:  # pragma: no cover - pygame optional
    init_display = draw_board = draw_board_with_action_values = None

PLAYERS_DIR = Path("../c4_players")
RESULTS_FILE = Path("c4_results.json")
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
        raise FileNotFoundError("No player configs found in 'c4_players/'.")
    return players


def load_results(names) -> dict:
    """Load saved Elo data and keep it in sync with *names*.

    Adds default ratings for any new players and prunes ratings for
    players that no longer have a config file.  This prevents KeyError
    crashes when the set of configs changes between runs.
    """

    if RESULTS_FILE.exists():
        with RESULTS_FILE.open() as f:
            data = json.load(f)
    else:
        data = {
            "ratings": {},
            "pair_results": {},
            # legacy fields kept for backward-compatibility
            "pair_index": 0,
            "orientation": 0,
        }

    # Make sure every current player has a rating entry.
    for n in names:
        data.setdefault("ratings", {}).setdefault(n, 1500.0)

    # Remove ratings for configs that no longer exist.
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


def play_one_game(
    game: ConnectFour,
    params_x: dict,
    params_o: dict,
    seed: int,
    screen=None,
    display_moves: bool = False
) -> int:
    random.seed(seed)

    def make_player(cfg, role, game_instance):
        player_type = cfg.get("type", "mcts") 

        if player_type == "human":
            raise NotImplementedError("Human player for C4 tournament not implemented yet.")
        
        # Handle our new AlphaZero player and adapt old "zero" types to it
        elif player_type in {"alphazero_c4", "mcts_zero_adv", "zero_adv", "mcts_zero", "zero"}:
            model_path = cfg.get("model_path") or cfg.get("weights") # Accept old "weights" key
            if not model_path:
                raise ValueError(f"Player type {player_type} config must specify 'model_path' or 'weights'.")
            
            # Determine architecture: default to new, but allow override for old sequential or old basic C4ZeroNet
            architecture = cfg.get("architecture", "new_style") # For AlphaZeroC4Player
            if cfg.get("arch") == "advanced" and architecture == "new_style": # From old mcts_zero_adv
                architecture = "old_sequential" # Assume old advanced means old sequential for our player
            elif player_type in {"zero", "mcts_zero"} and not cfg.get("arch"):
                # This implies it might be the very basic C4ZeroNet from c4_zero.py
                # Our AlphaZeroC4Player is not designed for that one directly.
                # For now, let's assume if it's a zero type, it implies an AdvancedNet structure.
                # If a C4ZeroNet (basic) is to be used, it needs its own player class or different handling.
                # Defaulting to new_style for these if not specified, which might fail if state_dict differs too much.
                # A more robust solution would be specific player classes or more arch flags.
                print(f"Warning: player type {player_type} without specific 'arch' or 'architecture' might assume incompatible network for AlphaZeroC4Player.")

            return AlphaZeroC4Player(
                game_instance=game_instance, 
                perspective_player=role, 
                model_path=model_path,
                device=cfg.get("device", "cpu"),
                mcts_simulations=cfg.get("mcts_simulations") or cfg.get("num_iterations", 100), # Accept old num_iterations
                c_puct=cfg.get("c_puct", cfg.get("c_param", 1.41)), # Accept old c_param
                dirichlet_alpha=cfg.get("dirichlet_alpha", 0.3),
                dirichlet_epsilon=cfg.get("dirichlet_epsilon", 0.0), # Noise off for eval
                nn_channels=cfg.get("nn_channels", 128), 
                nn_blocks=cfg.get("nn_blocks", 10),
                architecture=architecture
            )
        elif player_type == "minimax": # Explicitly handle minimax
            depth = int(cfg.get("depth", 4))
            return MinimaxConnectFourPlayer(game_instance, perspective_player=role, depth=depth)
        
        else: # Fallback to original MCTS player from mcts.Mcts
            # Filter out keys specific to Zero/AlphaZero players before passing to old MCTS
            known_az_keys = {'type', 'model_path', 'weights', 'architecture', 'arch', 'device', 
                             'nn_channels', 'nn_blocks', 'mcts_simulations', 'dirichlet_alpha', 
                             'dirichlet_epsilon', 'c_puct', 'temperature'}
            
            # Also filter keys that might be in cfg but not used by old MCTS
            # The old MCTS directly takes: num_iterations, max_depth, c_param, forced_check_depth, minimax_depth etc.
            # We will pass all remaining keys from cfg. If a key is genuinely unexpected by old MCTS, it will error.
            # The goal is to avoid passing clearly incompatible keys like 'model_path'.
            mcts_params = {k: v for k, v in cfg.items() if k not in known_az_keys}
            print(f"Fallback to original MCTS for player type: {player_type} with params: {mcts_params}")
            return MCTS(game=game_instance, perspective_player=role, **mcts_params)

    mcts_x = make_player(params_x, "X", game)
    mcts_o = make_player(params_o, "O", game)
    state = game.getInitialState()
    if screen is not None:
        draw_board(screen, state["board"])
        pygame.event.pump()
    while not game.isTerminal(state):
        to_move = game.getCurrentPlayer(state)
        if to_move == "X":
            player = mcts_x
        else:
            player = mcts_o

        # --------------------------------------------------------------
        # Keep the GUI responsive: we need to call ``pygame.event.pump``
        # regularly during potentially long MCTS searches.  If the rich
        # visualiser is available we already pump inside the callback that
        # draws action values.  If not, we still provide a lightweight
        # callback that simply pumps events every few iterations.
        # --------------------------------------------------------------

        if screen is not None and isinstance(player, MCTS):
            if draw_board_with_action_values is not None:
                def cb(root, iter_count):
                    values = {
                        a: (child.average_value(), child.visit_count)
                        for a, child in root.children.items()
                    }
                    draw_board_with_action_values(screen, root.state["board"], values, iter_count)
                    pygame.event.pump()
            else:
                def cb(root, iter_count):
                    # Pump events every 250 iterations to prevent the window
                    # from being marked "unresponsive" by the window manager.
                    if iter_count % 250 == 0:
                        pygame.event.pump()

            action = player.search(state, draw_callback=cb)
        elif screen is not None and isinstance(player, MinimaxConnectFourPlayer) and draw_board_with_action_values is not None:
            def cb(values):
                draw_board_with_action_values(
                    screen,
                    state["board"],
                    {a: (v, 1) for a, v in values.items()},
                    None,
                )
                pygame.event.pump()

            action = player.search(state, value_callback=cb)
        else:
            action = player.search(state)
        state = game.applyAction(state, action)
        if screen is not None:
            draw_board(screen, state["board"])
            pygame.event.pump()
            pygame.time.delay(300)
        if display_moves: game.printState(state); print("-----")
    outcome = game.getGameOutcome(state)
    return 1 if outcome == "X" else -1 if outcome == "O" else 0


def run(display: bool = True) -> None:
    game = ConnectFour()
    players = load_players()
    names = list(players)
    data = load_results(names)
    if display and pygame is not None and init_display is not None:
        screen = init_display(game.ROWS, game.COLS)
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
            print(f"Game {g}: {x_name} (R) vs {o_name} (Y)")
            result = play_one_game(
                game,
                players[x_name],
                players[o_name],
                seed=random.randint(0, 2**32 - 1),
                screen=screen,
                display_moves=args.display_moves
            )

            # Record results with the X player first so that each orientation
            # has its own entry.  Using a conditional here caused all games to
            # be stored under the same key regardless of orientation.
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

            ra = data["ratings"][x_name]
            rb = data["ratings"][o_name]
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


def init_players() -> None:
    PLAYERS_DIR.mkdir(exist_ok=True)
    samples = {
        "iter20":  {"num_iterations": 20,  "max_depth": 42, "c_param": 1.4, "forced_check_depth": 0},
        "iter50":  {"num_iterations": 50,  "max_depth": 42, "c_param": 1.4, "forced_check_depth": 0},
        "iter100": {"num_iterations": 100, "max_depth": 42, "c_param": 1.4, "forced_check_depth": 0},
        "iter150": {"num_iterations": 150, "max_depth": 42, "c_param": 1.4, "forced_check_depth": 0},
        "iter200": {"num_iterations": 200, "max_depth": 42, "c_param": 1.4, "forced_check_depth": 0},
        "iter300": {"num_iterations": 300, "max_depth": 42, "c_param": 1.4, "forced_check_depth": 0},
        "iter400": {"num_iterations": 400, "max_depth": 42, "c_param": 1.4, "forced_check_depth": 0},
        "iter600": {"num_iterations": 600, "max_depth": 42, "c_param": 1.4, "forced_check_depth": 0},
        "iter6000": {"num_iterations": 6000, "max_depth": 42, "c_param": 1.4, "forced_check_depth": 0},
        # "iter100000": {"num_iterations": 100000, "max_depth": 42, "c_param": 1.4, "forced_check_depth": 0},
        "iter800": {"num_iterations": 800, "max_depth": 42, "c_param": 1.4, "forced_check_depth": 0},
        "c1":      {"num_iterations": 200, "max_depth": 42, "c_param": 1.0, "forced_check_depth": 0},
        "check2":  {"num_iterations": 200, "max_depth": 42, "c_param": 1.4, "forced_check_depth": 2},
        "c2":      {"num_iterations": 200, "max_depth": 42, "c_param": 2.0, "forced_check_depth": 0},
        "c3":      {"num_iterations": 200, "max_depth": 42, "c_param": 3.0, "forced_check_depth": 0},
        "search_4":      {"num_iterations": 200, "max_depth": 42, "c_param": 3.0, "forced_check_depth": 4},
        "search_4":      {"num_iterations": 200, "max_depth": 42, "c_param": 3.0, "forced_check_depth": 6},
        "minimax": {"type": "minimax", "depth": 4},
        "minimax_6": {"type": "minimax", "depth": 6},
        "hybrid_4": {"num_iterations": 200, "max_depth": 42, "c_param": 3.0, "forced_check_depth": 0, "minimax_depth": 4 },
        "hybrid_6": {"num_iterations": 200, "max_depth": 42, "c_param": 3.0, "forced_check_depth": 0, "minimax_depth": 6 },
        # "mcts_zero": { "type": "mcts_zero", "weights": "c4_weights/weights.pth", "num_iterations": 200, "max_depth": 42, "c_param": 1.4, "forced_check_depth": 0, "temperature": 0.0, },
        # Example player using the *advanced* Zero network – adjust the
        # weights path to point at your checkpoint from c4_zero_advanced.py
        "adv_mcts_zero_2": {
            "type": "mcts_zero_adv",
            "arch": "advanced",
            "weights": "c4_checkpoints_az/last_model.pt",
            "num_iterations": 200,
            "max_depth": 42,
            "c_param": 1.4,
            "forced_check_depth": 0,
            "temperature": 0.0,
        },
    }
    for name, cfg in samples.items():
        path = PLAYERS_DIR / f"{name}.json"
        if not path.exists():
            with path.open("w") as f:
                json.dump(cfg, f, indent=2)
            print("wrote", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connect Four MCTS Tournament")
    parser.add_argument("--no-display", action="store_true", help="Run tournament without pygame visualizer.")
    parser.add_argument("--display-moves", action="store_true", help="Print board state to console after each move.")
    parser.add_argument(
        "--init-players",
        action="store_true",
        help="create sample configs in c4_players/",
    )
    args = parser.parse_args()

    if args.init_players:
        init_players()
    else:
        run(display=True)
