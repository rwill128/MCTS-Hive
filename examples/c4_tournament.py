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
    If az_vs_others_only is True, ensures one player is AlphaZero type and the other is not.
    Otherwise, selects uniformly at random from all possible unique pairs.
    """
    num_players = len(names)
    if num_players < 2:
        print("Warning: Less than two players available, cannot choose a pair.")
        return None

    possible_pairs = []
    az_types = {"alphazero_c4", "mcts_zero_adv", "zero_adv", "mcts_zero", "zero"}

    for i in range(num_players):
        for j in range(i + 1, num_players):
            player_i_name = names[i]
            player_j_name = names[j]
            
            player_i_config = players_configs.get(player_i_name, {})
            player_j_config = players_configs.get(player_j_name, {})

            player_i_type = player_i_config.get("type", "mcts")
            player_j_type = player_j_config.get("type", "mcts")
            
            is_player_i_az = player_i_type in az_types
            is_player_j_az = player_j_type in az_types

            if az_vs_others_only:
                # Rule: Exactly one player must be an AZ type
                if is_player_i_az != is_player_j_az: # XOR condition: one is True, the other is False
                    possible_pairs.append((i, j))
            else:
                # Original behavior: allow any pair
                possible_pairs.append((i, j))
    
    if not possible_pairs:
        print("Warning: No valid pairs found based on current matchmaking rules.")
        return None

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
                                  seed: int, global_az_sims_override: int,
                                  # New arguments for saving history:
                                  history_dir_path_str: str | None, 
                                  game_id_for_filename: int
                                  ) -> Tuple[str, str, int, str | None]: # Returns x_name, o_name, result, saved_history_filepath
    """Plays one game, saves history to disk, returns outcome and filepath."""
    # ... (Worker setup: game_instance, make_player_local, seeding) ...
    # This setup is as defined in the previous accepted version of c4_tournament.py
    # For brevity, I'm not repeating the full make_player_local here.
    if game_type_str == "ConnectFour": game_instance = ConnectFour()
    else: raise ValueError(f"Unsupported game type: {game_type_str}")
    # make_player_local would be defined here, using global_az_sims_override
    # player_x = make_player_local(...); player_o = make_player_local(...)
    # Placeholder for make_player logic until it's fully integrated in the context
    def make_player_local(cfg, role, game_inst): return MCTS(game=game_inst, perspective_player=role, **cfg) 
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    player_x = make_player_local(params_x_dict, "X", game_instance)
    player_o = make_player_local(params_o_dict, "O", game_instance)

    state = game_instance.getInitialState()
    game_history_for_saving: List[Dict] = [game_instance.copyState(state)]
    
    while not game_instance.isTerminal(state):
        # ... (game playing logic as before, appending to game_history_for_saving) ...
        to_move = game_instance.getCurrentPlayer(state)
        active_player = player_x if to_move == "X" else player_o
        try: action = active_player.search(state)
        except Exception as e: return player_x_name, player_o_name, (-1 if to_move == "X" else 1), None
        if action not in game_instance.getLegalActions(state): return player_x_name, player_o_name, (-1 if to_move == "X" else 1), None
        state = game_instance.applyAction(state, action)
        game_history_for_saving.append(game_instance.copyState(state))
        
    outcome = game_instance.getGameOutcome(state)
    result = 0 
    if outcome == "X": result = 1
    elif outcome == "O": result = -1

    saved_filepath_str = None
    if history_dir_path_str:
        history_dir = Path(history_dir_path_str)
        history_dir.mkdir(parents=True, exist_ok=True)
        winner_char = "X" if result == 1 else ("O" if result == -1 else "D")
        # Sanitize player names for filename
        safe_px_name = "".join(c if c.isalnum() else '_' for c in player_x_name)
        safe_po_name = "".join(c if c.isalnum() else '_' for c in player_o_name)
        
        filename = f"game_{game_id_for_filename:06d}_X-{safe_px_name}_O-{safe_po_name}_W-{winner_char}.json"
        filepath = history_dir / filename
        try:
            # For JSON serialization, board (list of lists) is fine.
            # Player dicts are also fine.
            game_data_to_save = {
                "game_id": game_id_for_filename,
                "player_x": player_x_name,
                "player_o": player_o_name,
                "params_x": params_x_dict, # Save player configs too
                "params_o": params_o_dict,
                "result_for_x": result,
                "history_states": game_history_for_saving # List of state dicts
            }
            with open(filepath, 'w') as f_hist:
                json.dump(game_data_to_save, f_hist, indent=2)
            saved_filepath_str = str(filepath)
            # print(f"[WORKER {os.getpid()}] Saved history to {saved_filepath_str}")
        except Exception as e:
            print(f"[WORKER {os.getpid()}] Error saving history to {filepath}: {e}")

    return player_x_name, player_o_name, result, saved_filepath_str

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
    checkpoints_found += list(source_checkpoint_dir.glob("last_c4_model.pt"))

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
                player_name = f"adv_mcts_zero_{base_name.replace('c4_chkpt_','')}"
        elif base_name == "last_c4_model":
            player_name = "adv_mcts_zero_last"
        else:
            player_name = f"adv_mcts_zero_{base_name}"

        # Create path like "c4_checkpoints_az/c4_chkpt_ep000300.pt"
        # This assumes source_checkpoint_dir itself is the folder name we want in the path.
        # e.g., if source_checkpoint_dir is "examples/c4_checkpoints_az", its name is "c4_checkpoints_az".
        weights_path_for_json = str(Path(source_checkpoint_dir.name) / ckpt_path.name).replace("\\", "/")

        player_config = {
            "type": "alphazero_c4", 
            "architecture": "new_style", 
            "model_path": weights_path_for_json,
            "mcts_simulations": 100, 
            "c_puct": 1.41,
            "nn_channels": 128, 
            "nn_blocks": 10
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
    global cli_args #, completed_parallel_games # completed_parallel_games is now local to run if only used for replay prompt which was removed for file saving
    cli_args = cli_args # This line is redundant if cli_args is already global from __main__

    game_type_for_tournament = "ConnectFour" 
    main_game_instance = ConnectFour() 
    players = load_players()
    player_names = list(players.keys())
    elo_data = load_results(player_names)

    # This global list is for the removed replay feature. If only saving to disk, it's not strictly needed by run.
    # However, if any part of the loop still appends to it, it should be initialized here.
    completed_parallel_games_list: List[Dict[str, Any]] = [] 


    if display and cli_args.num_parallel > 1:
        print("Display is disabled when running parallel games.")
        display = False

    screen = None
    if display and pygame is not None and init_display is not None:
        screen = init_display(main_game_instance.ROWS, main_game_instance.COLS)
    
    game_id_counter = 0 # Simple counter for game IDs

    if cli_args.history_dir:
        Path(cli_args.history_dir).mkdir(parents=True, exist_ok=True)
        print(f"Game histories will be saved to: {cli_args.history_dir}")

    try:
        if cli_args.num_parallel <= 1:
            print("Running in single-process mode.")
            while True: 
                pair_choice_result = choose_pair(player_names, players, elo_data, cli_args.az_vs_others_only)
                if pair_choice_result is None:
                    print("No suitable opponent pairs found. Tournament might be stalled or complete under current rules.")
                    break 
                (i, j), orientation = pair_choice_result
                p_x_name, p_o_name = (player_names[i], player_names[j]) if orientation == 0 else (player_names[j], player_names[i])
                params_x = players[p_x_name]
                params_o = players[p_o_name]
                
                game_id_counter += 1
                current_game_id = game_id_counter
                print(f"Game {current_game_id}: {p_x_name} (R) vs {p_o_name} (Y)")
                
                # Ensure play_one_game_sequential_version is correctly handling history saving if desired
                # For now, assuming it doesn't interact with completed_parallel_games_list directly
                current_result = play_one_game_sequential_version(
                    main_game_instance, params_x, params_o,
                    seed=random.randint(0, 2**32 -1),
                    screen=screen, display_moves=cli_args.display_moves
                    # Add history_dir and game_id if sequential should also save history files:
                    # history_dir_path_str=cli_args.history_dir, 
                    # game_id_for_filename=current_game_id 
                )
                _update_elo_and_stats(elo_data, p_x_name, p_o_name, current_result)
                save_results(elo_data)
                if current_game_id % 10 == 0: print_elo_standings(elo_data)
        else: 
            print(f"Running in parallel with {cli_args.num_parallel} workers.")
            if display and screen is not None: 
                 if pygame and hasattr(pygame, "quit"): pygame.quit()
                 print("Pygame display explicitly quit for parallel mode.")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=cli_args.num_parallel) as executor:
                futures_map: Dict[concurrent.futures.Future, Tuple[str, str, int]] = {} # Store game_id with names
                
                # Submit initial batch of tasks
                for _ in range(cli_args.num_parallel * 2): 
                    if len(player_names) < 2: break
                    pair_choice_result = choose_pair(player_names, players, elo_data, cli_args.az_vs_others_only)
                    if pair_choice_result is None: break 
                    (idx_i, idx_j), orientation = pair_choice_result
                    p_x_name, p_o_name = (player_names[idx_i], player_names[idx_j]) if orientation == 0 else (player_names[idx_j], player_names[idx_i])
                    params_x_submit = players[p_x_name]
                    params_o_submit = players[p_o_name]
                    
                    game_id_counter += 1
                    current_game_id_for_worker = game_id_counter
                    
                    future = executor.submit(play_one_game_parallel_worker, 
                                             game_type_for_tournament, 
                                             params_x_submit, params_o_submit, 
                                             p_x_name, p_o_name, 
                                             random.randint(0, 2**32 -1),
                                             cli_args.global_az_simulations,
                                             cli_args.history_dir, 
                                             current_game_id_for_worker 
                                             )
                    futures_map[future] = (p_x_name, p_o_name, current_game_id_for_worker)

                if not futures_map and len(player_names) >=2 :
                    print("No suitable initial opponent pairs found for parallel execution. Check rules or player pool.")

                while futures_map: 
                    done_futures, _ = concurrent.futures.wait(list(futures_map.keys()), timeout=0.2, return_when=concurrent.futures.FIRST_COMPLETED)

                    for future in done_futures:
                        p_x_name_fut, p_o_name_fut, processed_game_id = futures_map.pop(future)
                        try:
                            _, _, game_res, saved_hist_path = future.result()
                            _update_elo_and_stats(elo_data, p_x_name_fut, p_o_name_fut, game_res)
                            print(f"Game {processed_game_id} (parallel) finished: {p_x_name_fut} vs {p_o_name_fut} -> Result for X: {game_res}. History at: {saved_hist_path}")
                            if processed_game_id % cli_args.save_every == 0: 
                                save_results(elo_data)
                                print_elo_standings(elo_data)
                        except Exception as exc:
                            print(f"Game {p_x_name_fut} vs {p_o_name_fut} (id: {processed_game_id}) generated an exception in worker: {exc}")
                        
                        try:
                            if len(player_names) >= 2:
                                pair_choice_result = choose_pair(player_names, players, elo_data, cli_args.az_vs_others_only)
                                if pair_choice_result is None: continue 
                                (idx_i, idx_j), orientation = pair_choice_result
                                next_px_name, next_po_name = (player_names[idx_i], player_names[idx_j]) if orientation == 0 else (player_names[idx_j], player_names[idx_i])
                                next_params_x = players[next_px_name]
                                next_params_o = players[next_po_name]
                                
                                game_id_counter += 1
                                next_game_id_for_worker = game_id_counter

                                new_future = executor.submit(play_one_game_parallel_worker, game_type_for_tournament,
                                                           next_params_x, next_params_o, next_px_name, next_po_name,
                                                           random.randint(0, 2**32-1),
                                                           cli_args.global_az_simulations,
                                                           cli_args.history_dir,
                                                           next_game_id_for_worker)
                                futures_map[new_future] = (next_px_name, next_po_name, next_game_id_for_worker)
                        except RuntimeError as e: 
                            if "shutdown" in str(e).lower():
                                print("Executor shutting down, no more games will be submitted.")
                                break 
                            else: raise 
                    if not futures_map: break # Exit while if map is empty (e.g. all tasks done and no new ones possible)

    except KeyboardInterrupt: print("\nTournament interrupted. Saving final results...")
    finally:
        save_results(elo_data)
        print_elo_standings(elo_data)
        if pygame and hasattr(pygame, 'quit') and pygame.get_init(): pygame.quit()
        print("Tournament finished. Game histories (if enabled) are in the specified history directory.")

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
    parser.add_argument("--history-dir", type=str, metavar="DIR", default=None, 
                        help="Directory to save game histories. If not specified, histories are not saved.")

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
