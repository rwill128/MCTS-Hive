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
import os # Added for potential PID logging
from pathlib import Path
from typing import Dict, Tuple, List, Any
import concurrent.futures # Added for parallelism

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
        # Ensure keys exist before incrementing
        if a in counts:
            counts[a] = counts.get(a, 0) + games
        if b in counts:
            counts[b] = counts.get(b, 0) + games
    return counts


def choose_pair(names: List[str], players_configs: Dict[str, Dict], data: dict, az_vs_others_only: bool) -> Tuple[Tuple[int, int], int] | None:
    """Chooses a pair of players to play. 
    If az_vs_others_only is True, ensures AlphaZero players don't play each other.
    """
    num_players = len(names)
    if num_players < 2:
        # raise ValueError("Need at least two players to choose a pair.")
        print("Warning: Less than two players available, cannot choose a pair.")
        return None # Return None if no pair can be chosen

    possible_pairs = []
    for i in range(num_players):
        for j in range(i + 1, num_players):
            player_i_name = names[i]
            player_j_name = names[j]
            
            player_i_type = players_configs.get(player_i_name, {}).get("type", "mcts")
            player_j_type = players_configs.get(player_j_name, {}).get("type", "mcts")
            
            # Define what constitutes an "AlphaZero" type for this rule
            az_types = {"alphazero_c4", "mcts_zero_adv", "zero_adv", "mcts_zero", "zero"}

            is_player_i_az = player_i_type in az_types
            is_player_j_az = player_j_type in az_types

            if az_vs_others_only and is_player_i_az and is_player_j_az:
                continue # Skip AZ vs AZ if flag is set
            
            possible_pairs.append((i, j))
    
    if not possible_pairs:
        # This can happen if all remaining players are AZ and az_vs_others_only is True
        print("Warning: No valid pairs found based on current matchmaking rules (e.g., az_vs_others_only).")
        return None # Return None if no pair can be chosen

    i, j = random.choice(possible_pairs)

    key_ab = f"{names[i]}_vs_{names[j]}"
    key_ba = f"{names[j]}_vs_{names[i]}"
    pair_results = data.get("pair_results", {})
    record_ab = pair_results.get(key_ab, {})
    record_ba = pair_results.get(key_ba, {})
    count_ab = sum(record_ab.values())
    count_ba = sum(record_ba.values())

    orientation = random.randint(0, 1)
    if count_ab < count_ba: orientation = 0
    elif count_ba < count_ab: orientation = 1
    
    return (i, j), orientation


# Renamed original play_one_game for sequential execution
def play_one_game_sequential_version(
    game: ConnectFour,
    params_x: dict,
    params_o: dict,
    seed: int,
    screen=None,
    display_moves: bool = False
) -> int:
    random.seed(seed)
    # make_player is defined inside play_one_game_sequential_version and play_one_game_parallel_worker
    # We need to modify it in both places, or extract it to be a common helper if signatures align.
    # For now, let's modify it where it's used. This applies to the sequential version first.

    def make_player(cfg, role, game_instance):
        player_type = cfg.get("type", "mcts") 
        # Access cli_args, assuming it's global or passed appropriately. 
        # It's made global in if __name__ == "__main__"
        global_sim_override = cli_args.global_az_simulations if hasattr(cli_args, 'global_az_simulations') else -1

        if player_type == "human":
            raise NotImplementedError("Human player for C4 tournament not implemented yet.")
        elif player_type in {"alphazero_c4", "mcts_zero_adv", "zero_adv", "mcts_zero", "zero"}:
            model_path = cfg.get("model_path") or cfg.get("weights")
            if not model_path: raise ValueError(f"Player type {player_type} config for {cfg.get('name', role)} must specify 'model_path' or 'weights'.")
            architecture = cfg.get("architecture", "new_style")
            if cfg.get("arch") == "advanced" and architecture == "new_style": architecture = "old_sequential"
            
            current_mcts_sims = cfg.get("mcts_simulations") or cfg.get("num_iterations", 100)
            if global_sim_override > 0:
                mcts_sims_to_use = global_sim_override
                # print(f"DEBUG: Overriding sims for {cfg.get('name',role)} from {current_mcts_sims} to {mcts_sims_to_use}") # Optional debug
            else:
                mcts_sims_to_use = current_mcts_sims

            return AlphaZeroC4Player(
                game_instance=game_instance, perspective_player=role, model_path=model_path,
                device=cfg.get("device", "cpu"),
                mcts_simulations=mcts_sims_to_use, # Use the determined sim count
                c_puct=cfg.get("c_puct", cfg.get("c_param", 1.41)),
                dirichlet_alpha=cfg.get("dirichlet_alpha", 0.3),
                dirichlet_epsilon=cfg.get("dirichlet_epsilon", 0.0),
                nn_channels=cfg.get("nn_channels", 128), 
                nn_blocks=cfg.get("nn_blocks", 10),
                architecture=architecture
            )
        elif player_type == "minimax":
            depth = int(cfg.get("depth", 4))
            return MinimaxConnectFourPlayer(game_instance, perspective_player=role, depth=depth)
        else: 
            known_az_keys = {'type', 'model_path', 'weights', 'architecture', 'arch', 'device', 
                             'nn_channels', 'nn_blocks', 'mcts_simulations', 'num_iterations', # also num_iterations
                             'dirichlet_alpha', 'dirichlet_epsilon', 'c_puct', 'c_param', 'temperature'} # also c_param
            mcts_params = {k: v for k, v in cfg.items() if k not in known_az_keys}
            return MCTS(game=game_instance, perspective_player=role, **mcts_params)
    # ... (rest of play_one_game_sequential_version) ...
    player_x = make_player(params_x, "X", game)
    player_o = make_player(params_o, "O", game)
    state = game.getInitialState()
    if screen is not None and init_display is not None and draw_board is not None:
        draw_board(screen, state["board"])
        if pygame and hasattr(pygame, "event"): pygame.event.pump()
    
    move_count = 0
    while not game.isTerminal(state):
        to_move = game.getCurrentPlayer(state)
        active_player = player_x if to_move == "X" else player_o
        
        action_values_for_display = None 

        # Callbacks for MCTS/Minimax visualization - simplified for brevity here
        # Original callback logic for rich visualization needs to be here if used.
        action = active_player.search(state) # Simplified search call for this step

        if action not in game.getLegalActions(state):
            print(f"ILLEGAL ACTION CHOSEN by player {to_move} ({params_x.get('name', 'X') if to_move == 'X' else params_o.get('name', 'O')}): {action} from state:")
            if hasattr(game, 'printState'): game.printState(state) 
            else: print(state["board"])
            return -1 if to_move == "X" else 1 

        state = game.applyAction(state, action)
        move_count += 1
        if screen is not None and init_display is not None and draw_board is not None:
            draw_board(screen, state["board"])
            if pygame and hasattr(pygame, "event"): pygame.event.pump()
            if pygame and hasattr(pygame, "time"): pygame.time.delay(100)
        if display_moves: 
            if hasattr(game, 'printState'): game.printState(state) 
            else: print(state["board"])
            print("-----")

    outcome = game.getGameOutcome(state)
    if outcome == "X": return 1
    if outcome == "O": return -1
    return 0 # Placeholder for actual game result

# Global list to store game histories from parallel runs
completed_parallel_games: List[Dict[str, Any]] = []

# Worker function for parallel execution
def play_one_game_parallel_worker(game_type_str: str, params_x_dict: Dict, params_o_dict: Dict, 
                                  player_x_name: str, player_o_name: str, 
                                  seed: int, global_az_sims_override: int # Pass the override to worker
                                  ) -> Tuple[str, str, int, List[Dict], Dict, Dict]: 
    """Plays one game in a separate process. Player objects are created here. Returns history."""
    # print(f"WORKER {os.getpid()}: Starting game {player_x_name} vs {player_o_name} with seed {seed}")
    
    if game_type_str == "ConnectFour":
        game_instance = ConnectFour() # Each worker gets its own game instance
    else:
        raise ValueError(f"Unsupported game type for parallel worker: {game_type_str}")

    # Replicated make_player logic (crucial for models to be loaded in the worker process)
    # This local make_player will shadow the global one if play_one_game_sequential_version is also in this file
    def make_player_local(cfg: Dict, role: str, game_inst: ConnectFour):
        player_type = cfg.get("type", "mcts")
        # No direct access to cli_args here, so global_az_sims_override is passed in

        if player_type == "human":
            raise NotImplementedError("Human player cannot be used in parallel tournament.")
        elif player_type in {"alphazero_c4", "mcts_zero_adv", "zero_adv", "mcts_zero", "zero"}:
            model_path = cfg.get("model_path") or cfg.get("weights")
            if not model_path: raise ValueError(f"Player type {player_type} (role {role}, name {cfg.get('name', 'Unknown')}) specifies no model path.")
            architecture = cfg.get("architecture", "new_style")
            if cfg.get("arch") == "advanced" and architecture == "new_style": architecture = "old_sequential"
            
            current_mcts_sims = cfg.get("mcts_simulations") or cfg.get("num_iterations", 100)
            mcts_sims_to_use = global_az_sims_override if global_az_sims_override > 0 else current_mcts_sims

            return AlphaZeroC4Player(
                game_instance=game_inst, perspective_player=role, model_path=model_path,
                device=cfg.get("device", "cpu"), 
                mcts_simulations=mcts_sims_to_use,
                c_puct=cfg.get("c_puct", cfg.get("c_param", 1.41)),
                dirichlet_alpha=cfg.get("dirichlet_alpha", 0.3),
                dirichlet_epsilon=cfg.get("dirichlet_epsilon", 0.0),
                nn_channels=cfg.get("nn_channels", 128), 
                nn_blocks=cfg.get("nn_blocks", 10),
                architecture=architecture
            )
        elif player_type == "minimax":
            depth = int(cfg.get("depth", 4))
            return MinimaxConnectFourPlayer(game_inst, perspective_player=role, depth=depth)
        else: 
            known_az_keys = {'type', 'model_path', 'weights', 'architecture', 'arch', 'device', 
                             'nn_channels', 'nn_blocks', 'mcts_simulations', 'num_iterations', 
                             'dirichlet_alpha', 'dirichlet_epsilon', 'c_puct', 'c_param', 'temperature'}
            mcts_params = {k: v for k, v in cfg.items() if k not in known_az_keys}
            return MCTS(game=game_inst, perspective_player=role, **mcts_params)

    random.seed(seed)
    if torch: torch.manual_seed(seed)
    if np: np.random.seed(seed)

    player_x = make_player_local(params_x_dict, "X", game_instance)
    player_o = make_player_local(params_o_dict, "O", game_instance)
    
    state = game_instance.getInitialState()
    game_history_states: List[Dict] = [game_instance.copyState(state)] # Store initial state
    
    while not game_instance.isTerminal(state):
        to_move = game_instance.getCurrentPlayer(state)
        active_player = player_x if to_move == "X" else player_o
        
        try:
            action = active_player.search(state)
        except Exception as e:
            print(f"WORKER ERROR during search for player {to_move} ({player_x_name if to_move == 'X' else player_o_name}): {e}")
            return player_x_name, player_o_name, (-1 if to_move == "X" else 1), game_history_states, params_x_dict, params_o_dict

        if action not in game_instance.getLegalActions(state):
            print(f"WORKER ILLEGAL ACTION by {to_move} ({player_x_name if to_move == 'X' else player_o_name}): {action}. Legal: {game_instance.getLegalActions(state)}")
            return player_x_name, player_o_name, (-1 if to_move == "X" else 1), game_history_states, params_x_dict, params_o_dict
        
        state = game_instance.applyAction(state, action)
        game_history_states.append(game_instance.copyState(state)) # Store state after each move
        
    outcome = game_instance.getGameOutcome(state)
    result = 0 
    if outcome == "X": result = 1
    elif outcome == "O": result = -1
    
    return player_x_name, player_o_name, result, game_history_states, params_x_dict, params_o_dict

# Helper for Elo and stats update
def _update_elo_and_stats(elo_data: Dict, p_x_name: str, p_o_name: str, result: int):
    key = f"{p_x_name}_vs_{p_o_name}"
    record = elo_data["pair_results"].setdefault(key, {"w": 0, "d": 0, "l": 0})
    score_x = 0.0
    if result == 1: record["w"] += 1; score_x = 1.0; print(f"  {p_x_name} (X) wins against {p_o_name} (O)")
    elif result == 0: record["d"] += 1; score_x = 0.5; print(f"  Draw between {p_x_name} (X) and {p_o_name} (O)")
    else: record["l"] += 1; score_x = 0.0; print(f"  {p_o_name} (O) wins against {p_x_name} (X)")

    ra = elo_data["ratings"].setdefault(p_x_name, 1500.0)
    rb = elo_data["ratings"].setdefault(p_o_name, 1500.0)
    ea = expected(ra, rb)
    eb = expected(rb, ra)
    elo_data["ratings"][p_x_name] = update(ra, score_x, ea)
    elo_data["ratings"][p_o_name] = update(rb, 1.0 - score_x, eb)

def print_elo_standings(elo_data: Dict):
    print("Current Elo standings:")
    for n, r in sorted(elo_data["ratings"].items(), key=lambda x: -x[1]):
        print(f"  {n:<20} {r:6.1f}")
    print()

def replay_game_visual(game_info: Dict[str, Any], game_rows: int, game_cols: int):
    """Replays a single game from history in a Pygame window."""
    if pygame is None or init_display is None or draw_board is None:
        print("Pygame or visualizer components not available. Cannot replay game visually.")
        return

    history_states = game_info.get("history_states")
    if not history_states:
        print(f"No history found for game ID {game_info.get('id', 'N/A')}.")
        return

    print(f"\nReplaying Game ID: {game_info.get('id', 'N/A')}")
    print(f"{game_info.get('x_player', 'X')} (X) vs {game_info.get('o_player', 'O')} (O)")
    result_for_x = game_info.get('result_for_x')
    outcome_str = "Draw"
    if result_for_x == 1: outcome_str = f"{game_info.get('x_player', 'X')} Wins"
    elif result_for_x == -1: outcome_str = f"{game_info.get('o_player', 'O')} Wins"
    
    pygame.init() # Ensure pygame is initialized for this replay session
    screen = init_display(game_rows, game_cols) # Use game-specific dimensions
    pygame.display.set_caption(f"Replay: {game_info.get('x_player', 'X')} vs {game_info.get('o_player', 'O')} - {outcome_str}")

    running = True
    current_state_idx = 0
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    current_state_idx = min(len(history_states) - 1, current_state_idx + 1)
                elif event.key == pygame.K_LEFT:
                    current_state_idx = max(0, current_state_idx - 1)
                elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
        
        if not history_states: # Should not happen if checked before, but safeguard
            running = False
            continue
            
        current_board_state = history_states[current_state_idx]["board"]
        draw_board(screen, current_board_state)
        
        # Display move number and whose turn it was leading to this state
        font = pygame.font.Font(None, 24)
        move_text = f"Move: {current_state_idx}"
        # Player who made the move to get to this state (is current_player of *previous* state)
        # Or, if it's initial state (idx 0), it's first player to move.
        player_to_move_this_state = history_states[current_state_idx]["current_player"]
        turn_text_str = f"To Move: {player_to_move_this_state}"
        if current_state_idx > 0:
            player_who_moved = history_states[current_state_idx-1]["current_player"]
            turn_text_str = f"{player_who_moved} played to get here. Next: {player_to_move_this_state}"
        elif current_state_idx == 0:
             turn_text_str = f"Initial state. To Move: {player_to_move_this_state}"

        text_surface_move = font.render(move_text, True, (200,200,200))
        text_surface_turn = font.render(turn_text_str, True, (200,200,200))
        screen.blit(text_surface_move, (10, screen.get_height() - 50))
        screen.blit(text_surface_turn, (10, screen.get_height() - 25))
        
        if current_state_idx == len(history_states) - 1: # Last state
            outcome_font = pygame.font.Font(None, 30)
            outcome_surf = outcome_font.render(f"Game Over: {outcome_str}", True, (255,255,0))
            screen.blit(outcome_surf, (screen.get_width() // 2 - outcome_surf.get_width() // 2, 10))


        pygame.display.flip()
        clock.tick(5) # Slow down replay a bit, or use key presses

    if pygame.get_init(): # Check if pygame was initialized by this function
        pygame.quit()

def generate_player_configs_from_checkpoints(source_checkpoint_dir: Path, target_player_json_dir: Path):
    """Scans a directory for .pt model checkpoints and generates player JSON configs."""
    if not source_checkpoint_dir.is_dir():
        print(f"Error: Checkpoint directory not found: {source_checkpoint_dir}")
        return

    target_player_json_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scanning for checkpoints in: {source_checkpoint_dir}")
    print(f"Will generate player JSONs in: {target_player_json_dir}")

    checkpoints_found = list(source_checkpoint_dir.glob("c4_chkpt_ep*.pt"))
    checkpoints_found += list(source_checkpoint_dir.glob("last_c4_model.pt")) # Also include last model

    if not checkpoints_found:
        print(f"No .pt checkpoints found in {source_checkpoint_dir} matching expected patterns.")
        return

    generated_count = 0
    for ckpt_path in checkpoints_found:
        # Construct player name from checkpoint filename
        # e.g., c4_chkpt_ep002000.pt -> adv_mcts_zero_ep2000
        #       last_c4_model.pt -> adv_mcts_zero_last
        base_name = ckpt_path.stem
        if "_ep" in base_name:
            try:
                epoch_num = int(base_name.split("_ep")[-1])
                player_name = f"adv_mcts_zero_ep{epoch_num:06d}"
            except ValueError:
                player_name = f"adv_mcts_zero_{base_name.replace('c4_chkpt_','')}" # Fallback naming
        elif base_name == "last_c4_model":
            player_name = "adv_mcts_zero_last"
        else:
            player_name = f"adv_mcts_zero_{base_name}"

        # Construct relative path for JSON if possible, or absolute
        # For robustness, using absolute path might be safer if tournament is run from different CWD
        # However, player JSONs often use paths relative to project structure.
        # Let's try to make it relative to the parent of target_player_json_dir if source_checkpoint_dir is a sub-dir
        # For simplicity now, let's use a path that's relative to where the JSONs are expected to be read from
        # Assuming c4_players and examples/c4_checkpoints_az are siblings under MCTS-Hive/
        # The target_player_json_dir is ../c4_players relative to examples/ dir
        # The source_checkpoint_dir is examples/c4_checkpoints_az
        # So, weights path in JSON should be like "../examples/c4_checkpoints_az/filename.pt"
        try:
            # Path(ckpt_path).resolve() gives absolute
            # Path.cwd() is current working directory
            # target_player_json_dir.resolve().parent is the MCTS-Hive directory if PLAYERS_DIR = Path("../c4_players") from examples
            # We want path relative to MCTS-Hive directory
            weights_path_for_json = str(Path(os.path.relpath(ckpt_path.resolve(), target_player_json_dir.resolve().parent)).replace("\\", "/"))
        except ValueError: # Happens if paths are on different drives on Windows
            weights_path_for_json = str(ckpt_path.resolve()).replace("\\", "/") # Fallback to absolute
            print(f"Warning: Could not form relative path for {ckpt_path}, using absolute path in JSON.")

        player_config = {
            "type": "alphazero_c4", # Use the new unified type
            "architecture": "new_style", # Assuming these are from the new training script
            "model_path": weights_path_for_json,
            "mcts_simulations": 100, # Default for evaluation, can be overridden
            "c_puct": 1.41,
            "nn_channels": 128, # Default C4 new arch params
            "nn_blocks": 10
            # Add other default AZ player params if needed, like device (defaults to cpu in player class)
        }

        json_file_path = target_player_json_dir / f"{player_name}.json"
        
        if json_file_path.exists():
            print(f"Player config {json_file_path} already exists. Skipping.")
            continue

        with open(json_file_path, 'w') as f:
            json.dump(player_config, f, indent=2)
        print(f"Generated player config: {json_file_path}")
        generated_count += 1

    print(f"\nGenerated {generated_count} new player configuration files.")

def run(display: bool = True) -> None:
    global cli_args, completed_parallel_games # Ensure cli_args and completed_parallel_games are accessible
    # Argument parsing should happen here or cli_args passed in
    # For this step, assume cli_args is available globally after parsing in if __name__ == "__main__"

    game_type_for_tournament = "ConnectFour" 
    main_game_instance = ConnectFour() 
    players = load_players()
    player_names = list(players.keys())
    elo_data = load_results(player_names)

    if display and cli_args.num_parallel > 1:
        print("Display is disabled when running parallel games.")
        display = False # Override display flag

    screen = None
    if display and pygame is not None and init_display is not None:
        screen = init_display(main_game_instance.ROWS, main_game_instance.COLS)
    
    game_count = 0
    try:
        if cli_args.num_parallel <= 1:
            print("Running in single-process mode.")
            while True: # Main single-process loop
                pair_choice_result = choose_pair(player_names, players, elo_data, cli_args.az_vs_others_only)
                if pair_choice_result is None:
                    print("No suitable opponent pairs found. Tournament might be stalled or complete under current rules.")
                    break # Exit loop if no pairs can be chosen
                (i, j), orientation = pair_choice_result
                p_x_name, p_o_name = (player_names[i], player_names[j]) if orientation == 0 else (player_names[j], player_names[i])
                params_x = players[p_x_name]
                params_o = players[p_o_name]
                game_count += 1
                print(f"Game {game_count}: {p_x_name} (R) vs {p_o_name} (Y)")
                
                current_result = play_one_game_sequential_version(
                    main_game_instance, params_x, params_o,
                    seed=random.randint(0, 2**32 -1),
                    screen=screen, display_moves=cli_args.display_moves
                )
                _update_elo_and_stats(elo_data, p_x_name, p_o_name, current_result)
                save_results(elo_data)
                if game_count % 10 == 0: print_elo_standings(elo_data)
                # Add quit condition for single process mode if desired (e.g. from input)
        else:
            print(f"Running in parallel with {cli_args.num_parallel} workers.")
            if display and screen is not None: 
                 if pygame and hasattr(pygame, "quit"): pygame.quit()
                 print("Pygame display explicitly quit for parallel mode.")
                 display = False # Ensure display is off for parallel logic below

            with concurrent.futures.ProcessPoolExecutor(max_workers=cli_args.num_parallel) as executor:
                futures_map: Dict[concurrent.futures.Future, Tuple[str, str]] = {}
                
                for _ in range(cli_args.num_parallel * 2): 
                    if len(player_names) < 2: break
                    pair_choice_result = choose_pair(player_names, players, elo_data, cli_args.az_vs_others_only)
                    if pair_choice_result is None: break # Stop submitting if no more valid pairs
                    (idx_i, idx_j), orientation = pair_choice_result
                    p_x_name, p_o_name = (player_names[idx_i], player_names[idx_j]) if orientation == 0 else (player_names[idx_j], player_names[idx_i])
                    params_x_submit = players[p_x_name]
                    params_o_submit = players[p_o_name]
                    
                    future = executor.submit(play_one_game_parallel_worker, game_type_for_tournament, 
                                             params_x_submit, params_o_submit, p_x_name, p_o_name, 
                                             random.randint(0, 2**32 -1),
                                             cli_args.global_az_simulations # Pass the override here
                                             )
                    futures_map[future] = (p_x_name, p_o_name)

                while futures_map: 
                    done_futures, _ = concurrent.futures.wait(list(futures_map.keys()), timeout=0.1, return_when=concurrent.futures.FIRST_COMPLETED)

                    for future in done_futures:
                        p_x_name_fut, p_o_name_fut = futures_map.pop(future)
                        try:
                            _, _, game_res, history, _, _ = future.result()
                            _update_elo_and_stats(elo_data, p_x_name_fut, p_o_name_fut, game_res)
                            game_count += 1
                            completed_parallel_games.append({
                                "id": game_count,
                                "x_player": p_x_name_fut,
                                "o_player": p_o_name_fut,
                                "result_for_x": game_res,
                                "history_states": history
                            })
                            print(f"Game {game_count} (parallel) finished: {p_x_name_fut} vs {p_o_name_fut} -> Result for X: {game_res}. History saved.")
                            if game_count % cli_args.save_every == 0: 
                                save_results(elo_data)
                                print_elo_standings(elo_data)
                        except Exception as exc:
                            print(f"Game {p_x_name_fut} vs {p_o_name_fut} generated an exception in worker: {exc}")

                        # Submit a new game to keep the pool busy
                        # The executor itself will raise an error if we try to submit after shutdown has started.
                        # This try-except is to gracefully handle that during KeyboardInterrupt shutdown.
                        try:
                            if len(player_names) >= 2:
                                pair_choice_result = choose_pair(player_names, players, elo_data, cli_args.az_vs_others_only)
                                if pair_choice_result is None:
                                    # No more valid pairs to submit, let existing futures complete
                                    print("No more suitable pairs to submit for parallel execution.")
                                    continue # Continue processing other done futures, but don't submit new
                                (idx_i, idx_j), orientation = pair_choice_result
                                next_px_name, next_po_name = (player_names[idx_i], player_names[idx_j]) if orientation == 0 else (player_names[idx_j], player_names[idx_i])
                                next_params_x = players[next_px_name]
                                next_params_o = players[next_po_name]
                                new_future = executor.submit(play_one_game_parallel_worker, game_type_for_tournament,
                                                           next_params_x, next_params_o, next_px_name, next_po_name,
                                                           random.randint(0, 2**32-1),
                                                           cli_args.global_az_simulations # Pass the override here
                                                           )
                                futures_map[new_future] = (next_px_name, next_po_name)
                        except RuntimeError as e: # Catches "cannot schedule new futures after shutdown"
                            if "shutdown" in str(e).lower():
                                print("Executor shutting down, no more games will be submitted.")
                                break # Break from submitting new tasks if executor is shutting down
                            else:
                                raise # Re-raise other RuntimeErrors
                    if not futures_map and not done_futures : # If map became empty and no futures were processed, might be stuck or all done
                        # This condition might need more refinement to prevent busy-wait if all tasks are truly done
                        # but for continuous play with KeyboardInterrupt, it's okay.
                        pass 

    except KeyboardInterrupt: print("\nTournament interrupted. Saving final results...")
    finally:
        save_results(elo_data)
        print_elo_standings(elo_data)
        if pygame and hasattr(pygame, 'quit') and pygame.get_init(): # Quit main tournament display if it was running
            pygame.quit()

        if completed_parallel_games:
            print(f"\n{len(completed_parallel_games)} games were played and histories stored.")
            while True:
                try:
                    print("\nAvailable games to replay:")
                    for i, game_data in enumerate(completed_parallel_games):
                        res_str = "Draw"
                        if game_data['result_for_x'] == 1: res_str = f"{game_data['x_player']} Wins"
                        elif game_data['result_for_x'] == -1: res_str = f"{game_data['o_player']} Wins"
                        print(f"  {i+1}: {game_data['x_player']} (X) vs {game_data['o_player']} (O) - {res_str} ({len(game_data['history_states'])} moves)")
                    
                    choice = input("Enter game number to replay (e.g., 1), or 'q' to quit replay: ").strip().lower()
                    if choice == 'q' or not choice:
                        break
                    game_idx = int(choice) - 1
                    if 0 <= game_idx < len(completed_parallel_games):
                        # Determine game dimensions for replay - assumes ConnectFour for now
                        # This should be more generic if other games are used.
                        rows, cols = (ConnectFour.ROWS, ConnectFour.COLS) if game_type_for_tournament == "ConnectFour" else (3,3) # Default to 3x3 for unknown
                        replay_game_visual(completed_parallel_games[game_idx], rows, cols)
                    else:
                        print("Invalid game number.")
                except ValueError:
                    print("Invalid input. Please enter a number or 'q'.")
                except Exception as e:
                    print(f"Error during replay selection: {e}")
                    break # Exit replay loop on other errors
        elif cli_args.num_parallel > 1:
            print("No parallel games were completed or histories stored to replay.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connect Four MCTS Tournament")
    parser.add_argument("--no-display", action="store_true", help="Run tournament without pygame visualizer.")
    parser.add_argument("--display-moves", action="store_true", help="Print board state to console after each move.")
    parser.add_argument("--num-parallel", type=int, default=1, help="Number of games to run in parallel (display disabled if > 1).")
    parser.add_argument("--init-players",action="store_true",help="create sample configs in c4_players/")
    parser.add_argument("--save-every", type=int, default=10, help="Save results and print Elo every N games in parallel mode.")
    parser.add_argument("--az-vs-others-only", action="store_true", help="If set, AlphaZero players only play against non-AlphaZero players.")
    parser.add_argument("--generate-az-player-configs", type=str, metavar="CHECKPOINT_DIR", default=None, 
                        help="Scan CHECKPOINT_DIR for .pt files and generate player JSONs in c4_players/. Tournament won't run.")
    parser.add_argument("--global-az-simulations", type=int, default=-1, 
                        help="Override MCTS simulations for all AlphaZero-type players in the tournament. -1 means use JSON/default values.")

    cli_args = parser.parse_args() # Global cli_args

    if cli_args.init_players:
        # init_players() # Ensure init_players is defined if this option is used
        print("init_players function needs to be defined or restored.")
        pass
    elif cli_args.generate_az_player_configs:
        # Call a new function to generate configs
        source_checkpoint_dir = Path(cli_args.generate_az_player_configs)
        target_player_json_dir = PLAYERS_DIR # Defined globally as Path("../c4_players")
        generate_player_configs_from_checkpoints(source_checkpoint_dir, target_player_json_dir)
    else:
        run(display=not cli_args.no_display)
