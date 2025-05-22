#!/usr/bin/env python3
"""Generic AlphaZero-style self-play training loop.

Key features include:
    - MCTS-driven self-play (potentially parallelized)
    - Neural network training (policy & value heads)
    - Prioritized Experience Replay (optional)
    - Learning rate scheduling
    - Data augmentation (game-specific)
    - League play (playing against past selves)
    - Checkpointing and Weights & Biases logging
"""

from __future__ import annotations
import argparse
import random
from collections import deque
from pathlib import Path
from typing import List, Tuple, Dict, Any, Type # Added Type for class references
import concurrent.futures
import os
import time
import pickle

try:
    import numpy as np
except Exception: np = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception: torch = nn = F = None

try:
    import wandb
except ImportError: wandb = None

# --- Game Specific Imports - These will be dynamically selected later ---
from simple_games.connect_four import ConnectFour
# from HivePocket.HivePocket import HiveGame # Example for future
# from simple_games.tic_tac_toe import TicTacToe # Example for future
from mcts.alpha_zero_mcts import AlphaZeroMCTS
from mcts.replay_buffer import PrioritizedReplayBuffer

# --- Game Specific Components (ConnectFour initially, to be generalized) ---
# These will eventually be loaded based on args.game_name

# ConnectFour Constants (Example)
C4_BOARD_H = ConnectFour.ROWS
C4_BOARD_W = ConnectFour.COLS
C4_ACTION_SIZE = C4_BOARD_W

# ConnectFour State Encoding
def encode_c4_state(state: dict, perspective: str) -> torch.Tensor:
    if torch is None: raise RuntimeError("PyTorch is required")
    t = torch.zeros(3, C4_BOARD_H, C4_BOARD_W)
    for r in range(C4_BOARD_H):
        for c in range(C4_BOARD_W):
            piece = state["board"][r][c]
            if piece == perspective: t[0, r, c] = 1.0
            elif piece is not None: t[1, r, c] = 1.0
    if state["current_player"] == perspective: t[2].fill_(1.0)
    return t

# ConnectFour Data Augmentation
def reflect_c4_state_policy(state_dict: Dict, policy_vector: np.ndarray) -> Tuple[Dict, np.ndarray]:
    if np is None: raise RuntimeError("NumPy is required")
    reflected_board = [row[::-1] for row in state_dict["board"]]
    reflected_state_dict = {"board": reflected_board, "current_player": state_dict["current_player"]}
    reflected_policy_vector = policy_vector[::-1].copy()
    return reflected_state_dict, reflected_policy_vector

# ConnectFour Adapter
class ConnectFourAdapter:
    def __init__(self, c4_game: ConnectFour):
        self.c4_game = c4_game
        self.action_size = self.c4_game.get_action_size()
    def getCurrentPlayer(self, state: Dict) -> str: return self.c4_game.getCurrentPlayer(state)
    def getLegalActions(self, state: Dict) -> List[int]: return self.c4_game.getLegalActions(state)
    def applyAction(self, state: Dict, action_int: int) -> Dict: return self.c4_game.applyAction(state, action_int)
    def isTerminal(self, state: Dict) -> bool: return self.c4_game.isTerminal(state)
    def getGameOutcome(self, state: Dict) -> str: return self.c4_game.getGameOutcome(state)
    def encode_state(self, state: Dict, player_perspective: str) -> torch.Tensor: return encode_c4_state(state, player_perspective)
    def copyState(self, state: Dict) -> Dict: return self.c4_game.copyState(state)
    def get_action_size(self) -> int: return self.action_size
    # Add a get_board_shape() method for NN input configuration
    def get_input_board_shape(self) -> Tuple[int, int, int]: return (3, C4_BOARD_H, C4_BOARD_W) # (Channels, H, W)

# ConnectFour Network
if torch is not None:
    class ResidualBlockC4(nn.Module):
        def __init__(self, ch: int): super().__init__(); self.c1 = nn.Conv2d(ch,ch,3,padding=1,bias=False); self.b1=nn.BatchNorm2d(ch); self.c2=nn.Conv2d(ch,ch,3,padding=1,bias=False); self.b2=nn.BatchNorm2d(ch)
        def forward(self, x: torch.Tensor) -> torch.Tensor: y=F.relu(self.b1(self.c1(x))); y=self.b2(self.c2(y)); return F.relu(x+y)
    class AdvancedC4ZeroNet(nn.Module):
        def __init__(self, input_shape: Tuple[int,int,int], action_size: int, ch: int = 128, blocks: int = 10):
            super().__init__()
            c, h, w = input_shape # Expect (channels, height, width)
            self.stem = nn.Sequential(nn.Conv2d(c, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU())
            self.res = nn.Sequential(*[ResidualBlockC4(ch) for _ in range(blocks)])
            self.policy_conv = nn.Conv2d(ch,2,kernel_size=1); self.policy_bn = nn.BatchNorm2d(2); self.policy_flatten = nn.Flatten()
            self.policy_linear = nn.Linear(2*h*w, action_size)
            self.value_conv = nn.Conv2d(ch,1,kernel_size=1); self.value_bn = nn.BatchNorm2d(1); self.value_flatten = nn.Flatten()
            self.value_linear1 = nn.Linear(1*h*w, 64); self.value_linear2 = nn.Linear(64,1)
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            x_stem=self.stem(x); x_res=self.res(x_stem); p=F.relu(self.policy_bn(self.policy_conv(x_res))); pol_logits=self.policy_linear(self.policy_flatten(p)); v=F.relu(self.value_bn(self.value_conv(x_res))); v=F.relu(self.value_linear1(self.value_flatten(v))); val=torch.tanh(self.value_linear2(v)); return pol_logits, val.squeeze(1)
else: ResidualBlockC4=object; AdvancedC4ZeroNet=object

# --- End Game Specific Components Placeholder ---

args_global = None

# --- Factory functions (to be expanded) ---
def get_game_specific_classes(game_name: str) -> Dict[str, Type]:
    if game_name.lower() == "connectfour" or game_name.lower() == "c4":
        return {
            "game_class": ConnectFour,
            "adapter_class": ConnectFourAdapter,
            "network_class": AdvancedC4ZeroNet,
            "encode_state_fn": encode_c4_state, # May not be needed if adapter handles it
            "reflect_fn": reflect_c4_state_policy,
            "board_constants": {"h": C4_BOARD_H, "w": C4_BOARD_W, "action_size": C4_ACTION_SIZE} # Example
        }
    # elif game_name.lower() == "hive":
    #     # return { "game_class": HiveGame, "adapter_class": HiveAdapter, ... }
    # elif game_name.lower() == "tictactoe" or game_name.lower() == "ttt":
    #     # return { "game_class": TicTacToe, "adapter_class": TicTacToeAdapter, ... }
    else:
        raise ValueError(f"Unsupported game: {game_name}")

# ... (self_play_actor_worker - will need to use these generic components)
# ... (play_one_game - becomes part of worker or sequential path, also needs to be generic)
# ... (train_step - mostly generic already, but takes net and adapter)
# ... (save/load buffer - generic)

def run(parsed_cli_args=None) -> None:
    global args_global 
    args_global = parsed_cli_args if parsed_cli_args is not None else parser().parse_args()

    if wandb is None and args_global.use_wandb:
        print("Warning: --use-wandb specified but wandb library not found. Disabling W&B.")
        args_global.use_wandb = False
    # ... (W&B Init if args_global.use_wandb) ...
    
    # --- Get Game Specific Components ---
    try:
        game_specifics = get_game_specific_classes(args_global.game_name)
        GameClass = game_specifics["game_class"]
        AdapterClass = game_specifics["adapter_class"]
        NetworkClass = game_specifics["network_class"]
        # reflect_fn = game_specifics["reflect_fn"] # To be used in train_step/worker
        # board_constants = game_specifics["board_constants"]
    except ValueError as e:
        print(f"Error setting up game: {e}"); return
    except KeyError as e:
        print(f"Error: Game specific class or function not found in get_game_specific_classes for {args_global.game_name}: Missing key {e}"); return

    # ... (debug_single_loop overrides as before)
    # ... (PER beta annealing logic, device setup, ckdir)
    
    game_instance = GameClass() # Instantiate selected game
    game_adapter = AdapterClass(game_instance) # Instantiate adapter for the game

    # Network initialization now needs input_shape and action_size from adapter
    input_shape_from_adapter = game_adapter.get_input_board_shape() # Adapter needs this method
    action_size_from_adapter = game_adapter.get_action_size()

    learning_net = NetworkClass(
        input_shape=input_shape_from_adapter, 
        action_size=action_size_from_adapter,
        ch=args_global.channels, 
        blocks=args_global.blocks
    ).to(dev) # dev needs to be defined
    
    # ... (opt, scheduler, MCTS instance using generic components) ...
    # ... (state_path, start_ep, resume logic) ...
    # ... (buffer init, load_experiences, temp_schedule) ...
    # ... (opponent pool management) ...
    # ... (main training loop: data generation with workers, training step)
    #       Worker function will also need to instantiate game/adapter/net based on config.
    # ... (finally block) ...
    pass # Placeholder for the rest of the run function


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generic AlphaZero-style Training Loop.")
    p.add_argument("--game-name", type=str, default="ConnectFour", help="Name of the game to train (e.g., ConnectFour, Hive, TicTacToe). Case-insensitive.")
    # ... (all other existing args from c4_zero_advanced.py parser, including W&B, debug, play, train, buffer, nn, mgmt groups) ...
    # Make sure defaults in parser are sensible generic starting points or game-specific defaults are handled later.
    return p

if __name__ == "__main__":
    # To make game-specific classes available to worker processes if using spawn start method:
    # This might be needed if classes are not top-level or easily pickled.
    # For now, assume direct import works or use fork if on Linux/macOS.
    # from multiprocessing import set_start_method
    # try:
    #     set_start_method('spawn') # or 'fork' or 'forkserver'
    # except RuntimeError:
    #     pass 
    run() 