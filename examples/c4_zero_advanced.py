#!/usr/bin/env python3
"""Advanced self-play training loop for Connect Four.

This script adapts the techniques used in AlphaZero to a Connect
Four environment. Key features include:
    - Residual convolutional network
    - MCTS-driven self-play
    - KL-divergence policy loss with entropy regularisation
    - Dirichlet noise on the initial policy
    - Temperature decay for action selection
    - Replay buffer (standard or Prioritized Experience Replay)
    - Optional learning rate scheduling and data augmentation
    - Periodic checkpoints and full training state resumption
"""

from __future__ import annotations
import os # Ensure os is imported early
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1' # Suppress pygame community message globally for this script and its children

import argparse
import random
from collections import deque
from pathlib import Path
from typing import List, Tuple, Dict, Any
import concurrent.futures
import time
import pickle

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy optional
    np = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch optional
    torch = None
    nn = None
    F = None

try:
    import wandb # Added for Weights & Biases
except ImportError:
    wandb = None # Allow running without wandb if not installed

from simple_games.connect_four import ConnectFour
from mcts.alpha_zero_mcts import AlphaZeroMCTS
from mcts.replay_buffer import PrioritizedReplayBuffer # Added for PER

BOARD_H = ConnectFour.ROWS
BOARD_W = ConnectFour.COLS
ACTION_SIZE = BOARD_W # In ConnectFour, action is the column index

# ---------------------------------------------------------------------------
# State encoding (ConnectFour specific)
# ---------------------------------------------------------------------------
def encode_c4_state(state: dict, perspective: str) -> torch.Tensor: # Renamed for clarity
    """Return a 3xHxW tensor representing C4 `state` from `perspective`."""
    if torch is None: raise RuntimeError("PyTorch is required")
    t = torch.zeros(3, BOARD_H, BOARD_W)
    for r in range(BOARD_H):
        for c in range(BOARD_W):
            piece = state["board"][r][c]
            if piece == perspective:
                t[0, r, c] = 1.0
            elif piece is not None: # Opponent's piece
                t[1, r, c] = 1.0
    # Channel 2: Player turn (all 1s if current player is perspective, else 0s)
    if state["current_player"] == perspective:
        t[2].fill_(1.0)
    return t

# ---------------------------------------------------------------------------
# Data Augmentation (ConnectFour specific - horizontal reflection)
# ---------------------------------------------------------------------------
def reflect_c4_state_policy(state_dict: Dict, policy_vector: np.ndarray) -> Tuple[Dict, np.ndarray]:
    if np is None: raise RuntimeError("NumPy is required")

    reflected_board = [row[::-1] for row in state_dict["board"]]
    reflected_state_dict = {
        "board": reflected_board,
        "current_player": state_dict["current_player"]
    }
    # Policy for C4 is BOARD_W wide
    reflected_policy_vector = policy_vector[::-1].copy() # Simple reversal for C4 policy
    return reflected_state_dict, reflected_policy_vector

# ---------------------------------------------------------------------------
# Game Adapter for ConnectFour MCTS
# ---------------------------------------------------------------------------
class ConnectFourAdapter: # Kept name, but logic mirrors TTT adapter
    def __init__(self, c4_game: ConnectFour):
        self.c4_game = c4_game
        self.action_size = self.c4_game.get_action_size() # This is BOARD_W

    def getCurrentPlayer(self, state: Dict) -> str:
        return self.c4_game.getCurrentPlayer(state)

    def getLegalActions(self, state: Dict) -> List[int]: # Actions are already ints (column indices)
        return self.c4_game.getLegalActions(state)

    def applyAction(self, state: Dict, action_int: int) -> Dict: # Action is already int
        return self.c4_game.applyAction(state, action_int)

    def isTerminal(self, state: Dict) -> bool:
        return self.c4_game.isTerminal(state)

    def getGameOutcome(self, state: Dict) -> str: 
        return self.c4_game.getGameOutcome(state)

    def encode_state(self, state: Dict, player_perspective: str) -> torch.Tensor:
        return encode_c4_state(state, player_perspective) # Use C4 specific encoding

    def copyState(self, state: Dict) -> Dict:
        return self.c4_game.copyState(state)
    
    def get_action_size(self) -> int:
        return self.action_size

# ---------------------------------------------------------------------------
# Neural network (ConnectFour specific)
# ---------------------------------------------------------------------------
if torch is not None:
    class ResidualBlockC4(nn.Module): # Renamed for clarity
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

    class AdvancedC4ZeroNet(nn.Module):
        def __init__(self, ch: int = 128, blocks: int = 10): # Default C4 size
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, ch, 3, padding=1, bias=False), # 3 input channels
                nn.BatchNorm2d(ch), nn.ReLU(),
            )
            self.res = nn.Sequential(*[ResidualBlockC4(ch) for _ in range(blocks)])
            
            self.policy_conv = nn.Conv2d(ch, 2, kernel_size=1)
            self.policy_bn = nn.BatchNorm2d(2)
            self.policy_flatten = nn.Flatten()
            # Input to policy_linear: 2 * BOARD_H * BOARD_W
            self.policy_linear = nn.Linear(2 * BOARD_H * BOARD_W, ACTION_SIZE)

            self.value_conv = nn.Conv2d(ch, 1, kernel_size=1)
            self.value_bn = nn.BatchNorm2d(1)
            self.value_flatten = nn.Flatten()
            # Input to value_linear1: 1 * BOARD_H * BOARD_W
            self.value_linear1 = nn.Linear(1 * BOARD_H * BOARD_W, 64) 
            self.value_linear2 = nn.Linear(64, 1)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            x_stem = self.stem(x)
            x_res = self.res(x_stem)
            
            p = F.relu(self.policy_bn(self.policy_conv(x_res)))
            policy_logits = self.policy_linear(self.policy_flatten(p))
            
            v = F.relu(self.value_bn(self.value_conv(x_res)))
            v = F.relu(self.value_linear1(self.value_flatten(v)))
            value = torch.tanh(self.value_linear2(v))
            
            return policy_logits, value.squeeze(1)
else: 
    class ResidualBlockC4: pass
    class AdvancedC4ZeroNet: pass

# ---------------------------------------------------------------------------
# Self-play helpers (aligned with ttt_zero_advanced.py)
# ---------------------------------------------------------------------------
# BUFFER_PATH defined in parser now

def play_one_game(
    learning_net: AdvancedC4ZeroNet, # The network currently being trained
    game_adapter: ConnectFourAdapter, 
    learning_mcts: AlphaZeroMCTS, # MCTS instance for the learning network
    
    opponent_type: str, # "self", "past_checkpoint"
    opponent_net: AdvancedC4ZeroNet | None, # Network for the opponent if not self-play
    opponent_mcts: AlphaZeroMCTS | None, # MCTS for the opponent if not self-play
    # Note: opponent_net and opponent_mcts are None if opponent_type is "self"

    temp_schedule: List[Tuple[int, float]], 
    mcts_simulations_learning: int, # Sims for the learning agent
    mcts_simulations_opponent: int, # Sims for the opponent agent (can be different)
    max_moves: int = BOARD_H * BOARD_W, 
    debug_mode: bool = True
) -> List[Tuple[dict, np.ndarray, int]]:
    if torch is None or np is None: raise RuntimeError("PyTorch and NumPy are required")
    
    st = game_adapter.c4_game.getInitialState()
    hist: List[Tuple[dict, np.ndarray, int]] = [] # Stores (state, policy_target_for_learning_net, final_value_for_P1)
    move_no = 0
    current_temp = 1.0

    if debug_mode: print(f"\n--- play_one_game START (Connect Four) ---")
    if debug_mode: print(f"[play_one_game C4] Opponent type: {opponent_type}")

    # Determine which player is the learning agent (always perspective 'X' for data collection)
    learning_agent_player_char = "X"
    opponent_player_char = "O"

    while not game_adapter.isTerminal(st) and move_no < max_moves:
        if debug_mode:
            print(f"\n[play_one_game C4] Move: {move_no}")
            board_str = "\n".join([str(row) for row in st["board"]])
            print(f"[play_one_game C4] Current state (player {st['current_player']}):\n{board_str}")

        for threshold_moves, temp_val in temp_schedule:
            if move_no < threshold_moves: current_temp = temp_val; break
        
        current_player_char = game_adapter.getCurrentPlayer(st)
        is_learning_agent_turn = (current_player_char == learning_agent_player_char)

        # Determine which network and MCTS to use
        active_net = learning_net if is_learning_agent_turn else opponent_net
        active_mcts = learning_mcts if is_learning_agent_turn else opponent_mcts
        active_sims = mcts_simulations_learning if is_learning_agent_turn else mcts_simulations_opponent
        
        # If opponent is "self" (i.e. opponent_net/mcts are None), use learning agent's components
        if opponent_type == "self" and not is_learning_agent_turn:
            active_net = learning_net
            active_mcts = learning_mcts # Use learning_mcts for both if pure self-play
            # active_sims remains mcts_simulations_opponent (could be same as learning)
        
        if active_mcts is None: # Should only happen if opponent_type was invalidly configured
            raise ValueError("Active MCTS is None. Opponent not configured correctly.")

        if debug_mode: print(f"[play_one_game C4] Turn: {current_player_char}, MCTS sims: {active_sims}, Temp: {current_temp}")

        chosen_action_int, mcts_policy_dict = active_mcts.get_action_policy(
            root_state=st, num_simulations=active_sims, temperature=current_temp,
            debug_mcts=debug_mode 
        )
        if debug_mode: print(f"[play_one_game C4] MCTS chosen_action_int: {chosen_action_int}")
        if debug_mode: print(f"[play_one_game C4] MCTS policy_dict: {mcts_policy_dict}")
        
        policy_target_vector = np.zeros(game_adapter.get_action_size(), dtype=np.float32)
        if mcts_policy_dict:
            for action_idx_int, prob in mcts_policy_dict.items():
                if 0 <= action_idx_int < game_adapter.get_action_size(): policy_target_vector[action_idx_int] = prob
        
        current_sum = policy_target_vector.sum()
        if abs(current_sum - 1.0) > 1e-5 and current_sum > 1e-5:
             policy_target_vector /= current_sum
        elif current_sum < 1e-5: # Fallback if sum is zero (e.g. no legal moves from MCTS policy for some reason)
            if not game_adapter.isTerminal(st):
                legal_actions = game_adapter.getLegalActions(st)
                if legal_actions: 
                    uniform_prob = 1.0 / len(legal_actions)
                    for la_int in legal_actions: policy_target_vector[la_int] = uniform_prob
        if debug_mode: print(f"[play_one_game C4] Policy target vector (sum={policy_target_vector.sum()}): {policy_target_vector}")

        current_state_copy = game_adapter.copyState(st)
        if is_learning_agent_turn:
            hist.append((current_state_copy, policy_target_vector, 0))
            if debug_mode: print(f"[play_one_game C4] LEARNING AGENT ({current_player_char}) played. Stored policy target.")
        elif debug_mode:
             print(f"[play_one_game C4] OPPONENT ({current_player_char}) played. Policy target not stored for this turn.")
        
        st = game_adapter.applyAction(st, chosen_action_int)
        move_no += 1

    winner = game_adapter.getGameOutcome(st)
    z_for_learning_agent = 0
    if winner == learning_agent_player_char: z_for_learning_agent = 1
    elif winner == opponent_player_char: z_for_learning_agent = -1
    
    if debug_mode:
        print(f"\n[play_one_game C4] Game Over. Winner: {winner}, z (for {learning_agent_player_char}): {z_for_learning_agent}")
        board_str_final = "\n".join([str(row) for row in st["board"]])
        print(f"[play_one_game C4] Final board state (player {st['current_player']}):\n{board_str_final}")

    final_history = []
    for idx, (recorded_state, policy, _) in enumerate(hist):
        player_at_state = game_adapter.getCurrentPlayer(recorded_state)
        value_for_state_player = z_for_learning_agent if player_at_state == "X" else -z_for_learning_agent
        final_history.append((recorded_state, policy, value_for_state_player))
        if debug_mode: print(f"[play_one_game C4] final_hist item {idx}: Player: {player_at_state}, Val: {value_for_state_player}, Policy sum: {np.sum(policy) if policy is not None else 'N/A'}")
    
    if debug_mode: print("--- play_one_game END (Connect Four) ---")
    return final_history

# ---------------------------------------------------------------------------
# Training helpers (aligned with ttt_zero_advanced.py)
# ---------------------------------------------------------------------------
def train_step(net: AdvancedC4ZeroNet, batch_experiences: list, is_weights: np.ndarray, 
               opt: torch.optim.Optimizer, dev: str, game_adapter: ConnectFourAdapter, 
               augment_prob: float, ent_beta_val: float, debug_mode: bool = False) -> Tuple[float, np.ndarray]:
    if debug_mode: print("\n--- train_step START (Connect Four) ---")
    if debug_mode: print(f"[train_step C4] Batch size: {len(batch_experiences)}, Augment: {augment_prob}, EntBeta: {ent_beta_val}")

    S_list, P_tgt_list, V_tgt_list = [], [], []
    if not batch_experiences: 
        if debug_mode: print("[train_step C4] Batch empty.")
        return 0.0, np.array([])

    for i, (s_dict_orig, p_tgt_orig, v_tgt_orig) in enumerate(batch_experiences):
        s_to_enc, p_to_use = s_dict_orig, p_tgt_orig
        augmented = False
        if random.random() < augment_prob:
            s_to_enc, p_to_use = reflect_c4_state_policy(s_dict_orig, p_tgt_orig) # Use C4 reflection
            augmented = True
        
        player_persp = game_adapter.getCurrentPlayer(s_to_enc)
        S_list.append(game_adapter.encode_state(s_to_enc, player_persp))
        P_tgt_list.append(p_to_use)
        V_tgt_list.append(v_tgt_orig)
        if debug_mode and i < 1: # Log first sample
             print(f"  [train_step C4] Sample {i} Augmented: {augmented}, Player: {player_persp}, ValueTgt: {v_tgt_orig}")
             print(f"  [train_step C4] PolicyTgt (sum {np.sum(p_to_use)}): {p_to_use[:4]}...")


    S = torch.stack(S_list).to(dev)
    P_tgt = torch.tensor(np.array(P_tgt_list), dtype=torch.float32, device=dev)
    V_tgt = torch.tensor(V_tgt_list, dtype=torch.float32, device=dev)
    is_weights_tensor = torch.tensor(is_weights, dtype=torch.float32, device=dev).unsqueeze(1)

    if debug_mode:
        print(f"[train_step C4] S: {S.shape}, P_tgt: {P_tgt.shape}, V_tgt: {V_tgt.shape}")
        if S.numel() > 0: print(f"[train_step C4] S[0] sum: {S[0].sum().item()}") # Basic check

    logits, V_pred = net(S)
    if debug_mode and logits.numel() > 0: print(f"[train_step C4] Logits[0]: {logits[0][:4].cpu().detach().numpy()}... V_pred[0]: {V_pred[0].item() if V_pred.numel()>0 else 'N/A'}")

    logP_pred = F.log_softmax(logits, dim=1)
    loss_p_per_sample = F.kl_div(logP_pred, P_tgt, reduction="none").sum(dim=1)
    loss_v_per_sample = F.mse_loss(V_pred.squeeze(), V_tgt, reduction="none")
    
    weighted_loss_p = (loss_p_per_sample * is_weights_tensor.squeeze()).mean()
    weighted_loss_v = (loss_v_per_sample * is_weights_tensor.squeeze()).mean()
    
    P_pred_dist = torch.exp(logP_pred)
    entropy = -(P_pred_dist * logP_pred).sum(dim=1).mean()
    total_loss = weighted_loss_p + weighted_loss_v - ent_beta_val * entropy

    if debug_mode:
        print(f"[train_step C4] Losses (p,v,ent,total): {weighted_loss_p.item():.4f}, {weighted_loss_v.item():.4f}, {entropy.item():.4f}, {total_loss.item():.4f}")

    opt.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    
    td_errors = np.abs(V_tgt.cpu().detach().numpy() - V_pred.squeeze().cpu().detach().numpy())
    if debug_mode and len(td_errors)>0: print(f"[train_step C4] TD errors (first 4): {td_errors[:4]}")
    if debug_mode: print("--- train_step END (Connect Four) ---")
    return float(total_loss.item()), td_errors

# Aligned with ttt_zero_advanced.py
def save_buffer_experiences(buf: PrioritizedReplayBuffer | deque, path: Path) -> None:
    if torch is None: raise RuntimeError("PyTorch is required")
    data_to_save = list(buf.data_buffer[:len(buf)]) if isinstance(buf, PrioritizedReplayBuffer) else list(buf)
    torch.save(data_to_save, path)
    print(f"Saved {len(data_to_save)} experiences from buffer to {path}")

# Aligned with ttt_zero_advanced.py
def load_experiences_to_buffer(target_buffer: PrioritizedReplayBuffer | deque, path: Path) -> None:
    if torch is None: raise RuntimeError("PyTorch is required")
    if not path.exists():
        print(f"Buffer file {path} not found. Starting empty.")
        return
    try:
        data = torch.load(path, weights_only=False) # Must be False for arbitrary buffer data
    except Exception as e:
        print(f"Error loading buffer from {path}: {e}. Starting empty.")
        return

    num_loaded = 0
    if isinstance(target_buffer, PrioritizedReplayBuffer):
        for exp in data:
            if len(target_buffer) < target_buffer.capacity: 
                target_buffer.add(exp)
                num_loaded +=1
        print(f"Loaded {num_loaded} exp into PER buffer. Size: {len(target_buffer)}")
    elif isinstance(target_buffer, deque):
        target_buffer.extend(data)
        print(f"Loaded {len(data)} exp into deque. Size: {len(target_buffer)}")

args_global = None # For global access to parsed args

# ---------------------------------------------------------------------------
# Training loop (aligned with ttt_zero_advanced.py structure)
# ---------------------------------------------------------------------------
def run(parsed_cli_args=None) -> None:
    global args_global 
    args_global = parsed_cli_args if parsed_cli_args is not None else parser().parse_args()

    # --- W&B Initialization ---
    if args_global.use_wandb:
        if wandb is None:
            print("Warning: wandb is not installed. Skipping W&B logging. `pip install wandb`")
        else:
            try:
                wandb.init(
                    project=args_global.wandb_project,
                    name=args_global.wandb_run_name, # Will be None if not set, W&B generates one
                    entity=args_global.wandb_entity,  # Will be None if not set, uses default entity
                    config=vars(args_global) # Log all CLI arguments
                )
                print(f"Weights & Biases logging enabled for project '{args_global.wandb_project}'. Run: {wandb.run.name}")
            except Exception as e:
                print(f"Error initializing Weights & Biases: {e}. Disabling W&B logging.")
                args_global.use_wandb = False # Disable if init fails
    # --- End W&B Initialization ---

    if args_global.debug_single_loop:
        print("!!!!!!!!!!!!!!!!! DEBUG SINGLE LOOP MODE ENABLED (Connect Four) !!!!!!!!!!!!!!!!!")
        args_global.epochs = 1
        args_global.bootstrap_games = 1 
        args_global.mcts_simulations = 10 # C4 MCTS is slower, keep debug sims lower than TTT's debug
        args_global.batch_size = 8 
        args_global.min_buffer_fill_standard = 1 
        args_global.min_buffer_fill_for_per_training = 1
        args_global.min_buffer_fill_for_per_bootstrap = 1
        args_global.log_every = 1
        args_global.ckpt_every = 1 
        args_global.save_buffer_every = 1
        args_global.skip_bootstrap = False
    
    if args_global.use_per and args_global.per_beta_epochs <= 0:
        args_global.per_beta_epochs = args_global.epochs
        if not args_global.debug_single_loop: 
            print(f"PER: Beta annealing epochs set to total epochs: {args_global.per_beta_epochs}")

    if torch is None or np is None: raise RuntimeError("PyTorch and NumPy are required")
    dev = "cuda" if args_global.gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev}")

    ckdir = Path(args_global.ckpt_dir)
    ckdir.mkdir(exist_ok=True, parents=True)
    
    c4_game_instance = ConnectFour()
    game_adapter = ConnectFourAdapter(c4_game_instance)

    learning_net = AdvancedC4ZeroNet(ch=args_global.channels, blocks=args_global.blocks).to(dev)
    opt = torch.optim.Adam(learning_net.parameters(), lr=args_global.lr)
    
    scheduler = None
    if args_global.lr_scheduler == "cosine":
        t_max_epochs = args_global.lr_t_max if args_global.lr_t_max > 0 else args_global.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max_epochs, eta_min=args_global.lr_eta_min)
        if not args_global.debug_single_loop: print(f"Using CosineAnnealingLR: T_max={t_max_epochs}, eta_min={args_global.lr_eta_min}")
    
    def learning_mcts_model_fn(batch_in): learning_net.eval(); lgs,v = learning_net(batch_in); learning_net.train(); return lgs, v.unsqueeze(-1)
    learning_mcts_instance = AlphaZeroMCTS(game_adapter, learning_mcts_model_fn, torch.device(dev), args_global.c_puct, args_global.dirichlet_alpha, args_global.dirichlet_epsilon)

    state_path = ckdir / "train_state_c4.pt" # Path for C4
    start_ep = 1
    if args_global.resume_full_state and state_path.exists() and not args_global.debug_single_loop:
        print("Resuming full training state from", state_path)
        st_checkpoint = torch.load(state_path, map_location=dev, weights_only=False) # Full state, not just weights
        learning_net.load_state_dict(st_checkpoint["net"])
        opt.load_state_dict(st_checkpoint["opt"])
        start_ep = int(st_checkpoint.get("epoch", 1)) + 1
        
        # Forcefully set the LR in the loaded optimizer to the new desired args_global.lr
        print(f"Optimizer LR before override: {opt.param_groups[0]['lr']}")
        for param_group in opt.param_groups:
            param_group['lr'] = args_global.lr
        print(f"Optimizer LR after override with new --lr: {opt.param_groups[0]['lr']}")
        
        # Re-initialize scheduler to use the new LR as base and new T_max for the remaining run
        # We do not load the old scheduler state to effectively restart its cycle with new params.
        if args_global.lr_scheduler == "cosine":
            # T_max for the *remaining* duration of training from start_ep, or user-defined lr_t_max
            t_max_for_resume = args_global.lr_t_max if args_global.lr_t_max > 0 else (args_global.epochs - start_ep + 1)
            if t_max_for_resume <= 0: t_max_for_resume = 1 # Ensure positive T_max
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max_for_resume, eta_min=args_global.lr_eta_min)
            print(f"Re-initialized CosineAnnealingLR upon resume: New Base LR {args_global.lr}, T_max for remaining run={t_max_for_resume}, eta_min={args_global.lr_eta_min}")
        # Add other scheduler types here if needed

    elif args_global.resume_weights and not args_global.debug_single_loop:
        print("Resuming weights only from", args_global.resume_weights)
        # For resume_weights, we are loading only the state_dict of the model
        state_dict_to_load = torch.load(args_global.resume_weights, map_location=dev, weights_only=True)
        learning_net.load_state_dict(state_dict_to_load)
        # Optimizer and scheduler are fresh, using current cli_args.lr
        if args_global.lr_scheduler == "cosine":
            t_max_epochs = args_global.lr_t_max if args_global.lr_t_max > 0 else args_global.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max_epochs, eta_min=args_global.lr_eta_min)
            print(f"Initialized CosineAnnealingLR (weights resume): Base LR {args_global.lr}, T_max={t_max_epochs}, eta_min={args_global.lr_eta_min}")
    else: # Fresh start (no resume flags or debug_single_loop which might skip resume)
        if not args_global.debug_single_loop: print("Starting fresh training run or debug run without resume.")
        # Optimizer already created with args_global.lr
        if args_global.lr_scheduler == "cosine":
            t_max_epochs = args_global.lr_t_max if args_global.lr_t_max > 0 else args_global.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max_epochs, eta_min=args_global.lr_eta_min)
            print(f"Initialized CosineAnnealingLR (fresh start): Base LR {args_global.lr}, T_max={t_max_epochs}, eta_min={args_global.lr_eta_min}")

    buf: PrioritizedReplayBuffer | deque
    if args_global.use_per:
        debug_capacity = args_global.batch_size if args_global.debug_single_loop else args_global.buffer_size
        if args_global.debug_single_loop and debug_capacity == 0: debug_capacity = 8 
        buf = PrioritizedReplayBuffer(
            capacity=debug_capacity, alpha=args_global.per_alpha,
            beta_start=args_global.per_beta_start, beta_epochs=args_global.per_beta_epochs,
            epsilon=args_global.per_epsilon )
        print(f"Using PrioritizedReplayBuffer (capacity: {buf.capacity}).")
    else:
        debug_maxlen = args_global.batch_size if args_global.debug_single_loop else args_global.buffer_size
        if args_global.debug_single_loop and debug_maxlen == 0: debug_maxlen = 8
        buf = deque(maxlen=debug_maxlen)
        print(f"Using standard deque replay buffer (maxlen: {buf.maxlen}).")

    buffer_file_path = Path(args_global.buffer_path)
    if not args_global.debug_single_loop:
        load_experiences_to_buffer(buf, buffer_file_path)
    elif args_global.debug_single_loop:
        print("[run DEBUG C4] Skipping buffer load for debug_single_loop.")
    
    temp_schedule = [(args_global.temp_decay_moves, 1.0), (float('inf'), args_global.final_temp)]
    
    if args_global.debug_single_loop:
        print(f"[run DEBUG C4] Initialized. MCTS sims: {args_global.mcts_simulations}, Batch: {args_global.batch_size}, Epochs: {args_global.epochs}")
        print(f"[run DEBUG C4] Buffer type: {'PER' if args_global.use_per else 'Deque'}, Initial len(buf): {len(buf)}")

    min_fill_for_bootstrap = args_global.min_buffer_fill_for_per_bootstrap if isinstance(buf, PrioritizedReplayBuffer) else args_global.min_buffer_fill_standard
    if args_global.debug_single_loop: min_fill_for_bootstrap = 1
    
    games_to_play_bootstrap = 0
    if not args_global.skip_bootstrap and len(buf) < min_fill_for_bootstrap:
        games_to_play_bootstrap = args_global.bootstrap_games 
        if len(buf) < min_fill_for_bootstrap and not args_global.debug_single_loop:
            avg_states_per_game = 30 # Connect four average states
            needed_states = min_fill_for_bootstrap - len(buf)
            needed_games = (needed_states + avg_states_per_game -1) // avg_states_per_game
            games_to_play_bootstrap = max(games_to_play_bootstrap, needed_games)
            print(f"Buffer below min fill ({len(buf)}/{min_fill_for_bootstrap}). Playing {games_to_play_bootstrap} bootstrap games.")
        elif args_global.debug_single_loop:
            print(f"[run DEBUG C4] Bootstrap: Will play {games_to_play_bootstrap} game(s) as len(buf)={len(buf)} < min_fill={min_fill_for_bootstrap}")
        
        if games_to_play_bootstrap > 0:
            if not args_global.debug_single_loop: print(f"Bootstrapping {games_to_play_bootstrap} games …", flush=True)
            for g in range(games_to_play_bootstrap):
                learning_net.eval() 
                game_hist = play_one_game(
                    learning_net, game_adapter, learning_mcts_instance, "self", None, None,
                    temp_schedule, args_global.mcts_simulations, args_global.mcts_simulations_opponent,
                    max_moves=args_global.max_game_moves, 
                    debug_mode=args_global.debug_single_loop
                )
                
                if isinstance(buf, PrioritizedReplayBuffer):
                    for exp in game_hist:
                        buf.add(exp)
                else: # It's a deque
                    buf.extend(game_hist)
                
                learning_net.train() 
                if args_global.debug_single_loop or (g+1) % args_global.save_buffer_every == 0:
                    print(f"  Bootstrap game {g+1}/{games_to_play_bootstrap} ({len(game_hist)} states) → buffer {len(buf)}", flush=True)
                    if not args_global.debug_single_loop : save_buffer_experiences(buf, buffer_file_path)

    if not args_global.debug_single_loop: print(f"Starting training from epoch {start_ep} for {args_global.epochs} epochs.")
    
    overall_training_steps = 0 # For W&B step logging
    games_collected_this_session = 0 # To track games generated in this run
    opponent_checkpoints_pool: deque[Path] = deque(maxlen=args_global.max_opponent_pool_size)

    try:
        for ep in range(start_ep, args_global.epochs + 1):
            if args_global.debug_single_loop:
                _debug_flag = True # Ensure block starts with an assignment
                print(f"\n--- [DEBUG] Epoch {ep} START (Connect Four) ---")
            
            # Training Phase: sample experiences and update network
            # Determine minimum buffer fill to start training
            current_min_train_fill = (args_global.min_buffer_fill_for_per_training
                                      if isinstance(buf, PrioritizedReplayBuffer)
                                      else args_global.min_buffer_fill_standard)
            if args_global.debug_single_loop:
                current_min_train_fill = 1
            # If buffer not ready, skip training and step scheduler
            if len(buf) < current_min_train_fill:
                if ep % args_global.log_every == 0:
                    print(f"Epoch {ep} | Buffer {len(buf)} < min {current_min_train_fill}. Skipping training. LR {opt.param_groups[0]['lr']:.2e}")
                if scheduler: scheduler.step()
            else:
                # Sample a batch
                actual_batch_size = min(args_global.batch_size, len(buf))
                if isinstance(buf, PrioritizedReplayBuffer):
                    sampled = buf.sample(actual_batch_size)
                    if sampled is None:
                        if ep % args_global.log_every == 0:
                            print(f"Epoch {ep} | PER sample failed. Skip train. LR {opt.param_groups[0]['lr']:.2e}")
                        if scheduler: scheduler.step()
                        continue
                    batch_experiences, is_weights, data_indices = sampled
                    if args_global.debug_single_loop:
                        print(f"[run DEBUG C4] PER Sampled {len(batch_experiences)} exps. Indices: {data_indices[:4]}, ISW: {is_weights[:4]}")
                else:
                    batch_experiences = random.sample(list(buf), actual_batch_size)
                    is_weights = np.ones(len(batch_experiences), dtype=np.float32)
                    if args_global.debug_single_loop:
                        print(f"[run DEBUG C4] Deque Sampled {len(batch_experiences)} exps.")
                # Perform training step
                loss, td_errors = train_step(
                    learning_net, batch_experiences, is_weights,
                    opt, dev, game_adapter,
                    args_global.augment_prob, args_global.ent_beta,
                    debug_mode=args_global.debug_single_loop
                )
                # Update PER priorities and anneal beta
                if isinstance(buf, PrioritizedReplayBuffer):
                    buf.update_priorities(data_indices, td_errors)
                    buf.advance_epoch_for_beta_anneal()
                # Logging
                if ep % args_global.log_every == 0:
                    print(f"Epoch {ep} | Loss {loss:.4f} | Buffer {len(buf)} | LR {opt.param_groups[0]['lr']:.2e}")
                # Scheduler step
                if scheduler: scheduler.step()
                # Periodic checkpoint and buffer save
                if ep % args_global.ckpt_every == 0 and not args_global.debug_single_loop:
                    ckpt_path = ckdir / f"c4_chkpt_ep{ep:06d}.pt"
                    torch.save(learning_net.state_dict(), ckpt_path)
                    full_state = {"net": learning_net.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                                  "scheduler": scheduler.state_dict() if scheduler else None}
                    torch.save(full_state, state_path)
                    print(f"Saved checkpoint: {ckpt_path} and state: {state_path}")
                    if opponent_checkpoints_pool.maxlen is not None and opponent_checkpoints_pool.maxlen > 0 : # Add to opponent pool
                        opponent_checkpoints_pool.append(ckpt_path)
                        if args_global.debug_single_loop:
                            print(f"[DEBUG C4] Added {ckpt_path} to opponent pool. Pool size: {len(opponent_checkpoints_pool)}")

                if ep % args_global.save_buffer_every == 0 and len(buf) > 0 and not args_global.debug_single_loop:
                    save_buffer_experiences(buf, buffer_file_path)
                    print(f"Saved replay buffer at epoch {ep}")
                # Self-play data generation after training
                if args_global.num_parallel_selfplay <= 1:
                    for g in range(args_global.games_per_epoch):
                        learning_net.eval()
                        game_hist = play_one_game(
                            learning_net, game_adapter, learning_mcts_instance,
                            "self", None, None,
                            temp_schedule, args_global.mcts_simulations,
                            args_global.mcts_simulations_opponent,
                            max_moves=args_global.max_game_moves,
                            debug_mode=args_global.debug_single_loop
                        )
                        if isinstance(buf, PrioritizedReplayBuffer):
                            for exp in game_hist:
                                buf.add(exp)
                        else:
                            buf.extend(game_hist)
                        learning_net.train()
                        games_collected_this_session += 1
                        if ep % args_global.save_buffer_every == 0 and not args_global.debug_single_loop:
                            save_buffer_experiences(buf, buffer_file_path)
                            print(f"Epoch {ep} | Added game {g+1}/{args_global.games_per_epoch}, Buffer {len(buf)}")
                else:
                    # Parallel self-play actors
                    sp_weights_path = ckdir / f'selfplay_weights_ep{ep:06d}.pt'
                    torch.save({'net': learning_net.state_dict()}, sp_weights_path)
                    nn_arch_config = {'channels': args_global.channels, 'blocks': args_global.blocks}
                    mcts_config_learning = {'c_puct': args_global.c_puct, 'dirichlet_alpha': args_global.dirichlet_alpha, 
                                            'dirichlet_epsilon': args_global.dirichlet_epsilon, 
                                            'mcts_simulations_learning': args_global.mcts_simulations, 
                                            'mcts_simulations_opponent': args_global.mcts_simulations_opponent}
                    # opponent_config = {'type': 'self'} # This will be set per game below

                    selfplay_dir = ckdir / 'selfplay_exps' # Directory for temporary experience files from workers
                    selfplay_dir.mkdir(exist_ok=True, parents=True) # Ensure it exists
                    
                    seed_base = ep * args_global.games_per_epoch # Ensure unique seeds across epochs for same worker game index

                    if args_global.debug_single_loop:
                        print(f"[DEBUG C4 ep {ep}] Parallel self-play. Games per epoch: {args_global.games_per_epoch}, Workers: {args_global.num_parallel_selfplay}")
                        print(f"[DEBUG C4 ep {ep}] Opponent pool size: {len(opponent_checkpoints_pool)}, Past self prob: {args_global.play_past_self_prob}")

                    with concurrent.futures.ProcessPoolExecutor(max_workers=args_global.num_parallel_selfplay) as executor:
                        futures = []
                        for game_idx_in_epoch in range(args_global.games_per_epoch):
                            current_opponent_config = {}
                            # Decide opponent for this specific game
                            if len(opponent_checkpoints_pool) > 0 and random.random() < args_global.play_past_self_prob:
                                chosen_opponent_path = random.choice(list(opponent_checkpoints_pool))
                                current_opponent_config = {
                                    "type": "past_checkpoint",
                                    "path": str(chosen_opponent_path),
                                    "mcts_config": { # Pass opponent MCTS params from CLI args
                                        "c_puct_opponent": args_global.c_puct_opponent,
                                        "mcts_simulations_opponent": args_global.mcts_simulations_opponent
                                    }
                                }
                                if args_global.debug_single_loop:
                                    print(f"[DEBUG C4 ep {ep}, game {game_idx_in_epoch}] Playing vs PAST: {chosen_opponent_path.name}")
                            else:
                                current_opponent_config = {"type": "self"}
                                if args_global.debug_single_loop:
                                    print(f"[DEBUG C4 ep {ep}, game {game_idx_in_epoch}] Playing vs SELF.")
                            
                            # Worker ID can cycle if games_per_epoch > num_parallel_selfplay
                            worker_id_for_game = game_idx_in_epoch % args_global.num_parallel_selfplay
                            game_seed = seed_base + game_idx_in_epoch # Unique seed for each game

                            futures.append(
                                executor.submit(
                                    self_play_actor_worker,
                                    worker_id_for_game, # Pass the cycling worker_id
                                    str(sp_weights_path),
                                    'ConnectFour', # Hardcoded for c4_zero_advanced
                                    nn_arch_config,
                                    mcts_config_learning, # Learning agent's MCTS config
                                    current_opponent_config, # Opponent's config for this game
                                    str(selfplay_dir),
                                    temp_schedule,
                                    args_global.max_game_moves,
                                    dev, # Device for worker (can be "cpu" if main is "cuda" for workers)
                                    game_seed, # Seed for this game
                                    args_global.debug_single_loop
                                )
                            )
                        
                        games_processed_this_epoch = 0
                        for fut in concurrent.futures.as_completed(futures):
                            saved_filepath, rec_count, completed_worker_id = fut.result()
                            if saved_filepath:
                                try:
                                    new_exps = pickle.load(open(saved_filepath, 'rb'))
                                    # Cleanup the temp file after loading
                                    try: Path(saved_filepath).unlink()
                                    except OSError as e_del: print(f"Warning: Could not delete temp exp file {saved_filepath}: {e_del}")

                                except Exception as e_load:
                                    print(f"Error loading experiences from {saved_filepath}: {e_load}")
                                    continue
                                
                                if isinstance(buf, PrioritizedReplayBuffer):
                                    for exp in new_exps:
                                        buf.add(exp)
                                else: # Deque
                                    buf.extend(new_exps)
                                
                                games_collected_this_session += 1
                                games_processed_this_epoch +=1
                                if not args_global.debug_single_loop:
                                    print(f"Epoch {ep} | Games {games_processed_this_epoch}/{args_global.games_per_epoch} collected | Worker {completed_worker_id} added {len(new_exps)} states | Buffer size {len(buf)}")
                                elif args_global.debug_single_loop:
                                    print(f"[DEBUG C4 ep {ep}] Worker {completed_worker_id} added {len(new_exps)} states from {Path(saved_filepath).name if saved_filepath else 'N/A'}. Buffer {len(buf)}")
                            else: # No saved filepath, likely an error in worker or no experiences generated
                                print(f"Warning: Worker {completed_worker_id} returned no experience file path for a game in epoch {ep}.")
                                
                    # Clean up the temporary weights file used by workers for this epoch
                    try:
                        sp_weights_path.unlink()
                        if args_global.debug_single_loop: print(f"[DEBUG C4 ep {ep}] Deleted self-play weights: {sp_weights_path.name}")
                    except OSError as e:
                        print(f"Warning: Could not delete self-play weights {sp_weights_path}: {e}")
            
            # W&B Logging at end of epoch
            if args_global.use_wandb and wandb.run:
                log_data = {
                    "epoch": ep,
                    "total_loss": loss if 'loss' in locals() else None, # Only if training happened
                    "buffer_size": len(buf),
                    "learning_rate": opt.param_groups[0]['lr'],
                    "games_collected_session": games_collected_this_session,
                }
                if isinstance(buf, PrioritizedReplayBuffer):
                    log_data["per_beta"] = buf.beta
                
                wandb.log(log_data, step=ep) # Use epoch as step

    except KeyboardInterrupt: print("\nTraining interrupted.")
    finally:
        print("Saving final model and buffer (Connect Four)...")
        final_model_path = ckdir / "last_c4_model.pt" # Path for C4
        torch.save(learning_net.state_dict(), final_model_path)
        if len(buf) > 0: save_buffer_experiences(buf, buffer_file_path)
        ep_to_save = ep if 'ep' in locals() and 'start_ep' in locals() and ep >= start_ep else start_ep - 1
        full_state_to_save = {
            "net": learning_net.state_dict(), "opt": opt.state_dict(), "epoch": ep_to_save,
            "scheduler": scheduler.state_dict() if scheduler else None, }
        torch.save(full_state_to_save, state_path)
        print(f"Final C4 model: {str(final_model_path)}, Buffer: {str(buffer_file_path)}, State: {str(state_path)}")
        if args_global.use_wandb and wandb.run:
            wandb.finish()
            print("Weights & Biases run finished.")

# ---------------------------------------------------------------------------
# CLI parser (aligned with ttt_zero_advanced.py, defaults adjusted for C4)
# ---------------------------------------------------------------------------
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AlphaZero-style training for Connect Four.")
    p.add_argument("--debug-single-loop", action="store_true", help="Run for one minimal loop with extensive debugging prints.")
    # No --eval-model-path for c4 script for now, can be added if needed

    p.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    
    g_play = p.add_argument_group("Self-Play & MCTS (Connect Four)")
    g_play.add_argument("--bootstrap-games", type=int, default=20, help="Initial games to fill buffer.") # C4 default
    g_play.add_argument("--skip-bootstrap", action="store_true", help="Skip bootstrap if buffer meets min fill.")
    g_play.add_argument("--mcts-simulations", type=int, default=50, help="MCTS simulations per move.") # C4 default
    g_play.add_argument("--c-puct", type=float, default=1.41, help="PUCT exploration constant.")
    g_play.add_argument("--dirichlet-alpha", type=float, default=0.3, help="Alpha for Dirichlet noise.")
    g_play.add_argument("--dirichlet-epsilon", type=float, default=0.25, help="Epsilon for Dirichlet noise.")
    g_play.add_argument("--temp-decay-moves", type=int, default=10, help="Moves to use T=1 for exploration.") # C4 default
    g_play.add_argument("--final-temp", type=float, default=0.1, help="Temp after decay (0 for deterministic).")
    g_play.add_argument("--play-past-self-prob", type=float, default=0.0, help="Probability to play against a past self (0.0 to disable).")
    g_play.add_argument("--max-opponent-pool-size", type=int, default=20, help="Max number of recent past checkpoints for opponent pool.")
    g_play.add_argument("--mcts-simulations-opponent", type=int, default=50, help="MCTS simulations for past self opponent.")
    g_play.add_argument("--c-puct-opponent", type=float, default=1.41, help="PUCT for past self opponent MCTS.")
    g_play.add_argument("--max-game-moves", type=int, default=BOARD_H * BOARD_W + 10, help="Max moves per game.") # Added
    g_play.add_argument("--num-parallel-selfplay", type=int, default=1, help="Number of parallel game generation workers (1 for sequential).")
    g_play.add_argument("--games-per-epoch", type=int, default=1, help="Number of self-play games to generate per epoch/training step.")

    g_train = p.add_argument_group("Training (Connect Four)")
    g_train.add_argument("--epochs", type=int, default=10000, help="Total training epochs.")
    g_train.add_argument("--lr", type=float, default=1e-5, help="Initial learning rate.") # C4 specific LR
    g_train.add_argument("--batch-size", type=int, default=256, help="Batch size for training.")
    g_train.add_argument("--ent-beta", type=float, default=1e-3, help="Entropy regularization.")
    g_train.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    g_train.add_argument("--lr-t-max", type=int, default=0, help="T_max for CosineLR (0 for args.epochs).")
    g_train.add_argument("--lr-eta-min", type=float, default=1e-6, help="Min LR for CosineLR.")
    g_train.add_argument("--augment-prob", type=float, default=0.5, help="Probability of reflection augmentation.")

    g_buffer = p.add_argument_group("Replay Buffer (Connect Four)")
    g_buffer.add_argument("--buffer-size", type=int, default=50000, help="Max replay buffer size.")
    g_buffer.add_argument("--min-buffer-fill-standard", type=int, default=1000, help="Min samples for standard buffer.")
    g_buffer.add_argument("--min-buffer-fill-for-per-training", type=int, default=400, help="Min samples for PER training.")
    g_buffer.add_argument("--min-buffer-fill-for-per-bootstrap", type=int, default=400, help="Min PER samples for bootstrap target.")
    g_buffer.add_argument("--buffer-path", type=str, default="c4_adv_mcts_buffer.pth", help="Path to save/load buffer.")
    g_buffer.add_argument("--save-buffer-every", type=int, default=50, help="Save buffer every N epochs.")
    g_buffer.add_argument("--use-per", action="store_true", help="Use Prioritized Experience Replay.")
    g_buffer.add_argument("--per-alpha", type=float, default=0.6, help="Alpha for PER.")
    g_buffer.add_argument("--per-beta-start", type=float, default=0.4, help="Initial beta for PER.")
    g_buffer.add_argument("--per-beta-epochs", type=int, default=0, help="Epochs to anneal beta (0 for const beta, or total epochs).")
    g_buffer.add_argument("--per-epsilon", type=float, default=1e-5, help="Epsilon for PER priorities.")

    g_nn = p.add_argument_group("Neural Network (Connect Four)")
    g_nn.add_argument("--channels", type=int, default=128, help="Channels per conv layer.")
    g_nn.add_argument("--blocks", type=int, default=10, help="Number of residual blocks.")

    g_mgmt = p.add_argument_group("Checkpointing & Logging (Connect Four)")
    g_mgmt.add_argument("--ckpt-dir", default="c4_checkpoints_az", help="Directory for model checkpoints.") # C4 specific
    g_mgmt.add_argument("--ckpt-every", type=int, default=100, help="Save checkpoint every N epochs.")
    g_mgmt.add_argument("--log-every", type=int, default=1, help="Log training stats every N epochs.")
    g_mgmt.add_argument("--resume-weights", metavar="PATH", help="Path to load network weights.")
    g_mgmt.add_argument("--resume-full-state", action="store_true", help="Resume full training state.")
    
    # W&B Arguments
    g_wandb = p.add_argument_group("Weights & Biases")
    g_wandb.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging.")
    g_wandb.add_argument("--wandb-project", type=str, default="c4_alphazero_advanced", help="W&B project name.")
    g_wandb.add_argument("--wandb-run-name", type=str, default=None, help="Custom W&B run name (defaults to W&B auto-generated name).")
    g_wandb.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (username or team) if not using default.")

    return p

# --- Worker function for parallel self-play data generation ---
def self_play_actor_worker(
    worker_id: int, 
    current_learning_net_state_dict_path: str, 
    game_class_name: str, 
    nn_arch_config: Dict, 
    mcts_config_learning: Dict, 
    opponent_config: Dict, # Should contain: {"type": str, "path": Optional[str], "mcts_config": Dict}
    experience_save_dir: str, 
    temp_schedule_list: List[Tuple[int, float]],
    max_game_moves: int,
    worker_dev_str: str,
    seed: int,
    debug_mode: bool = False
) -> Tuple[str | None, int, int]:
    """Plays one game in a worker process, saves history, returns filepath and stats."""
    # os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1' # Moved to top of script
    
    if debug_mode: print(f"[ACTOR {worker_id} PID {os.getpid()}] Started. Seed: {seed}. Opponent type: {opponent_config['type']}")
    
    # --- Initialization within Worker ---
    torch.manual_seed(seed + worker_id); np.random.seed(seed + worker_id); random.seed(seed + worker_id)
    worker_device = torch.device(worker_dev_str)

    if game_class_name == "ConnectFour":
        game_instance_worker = ConnectFour()
        game_adapter_worker = ConnectFourAdapter(game_instance_worker)
        NNClass = AdvancedC4ZeroNet # Assuming this is the correct network class for C4
    else:
        print(f"[ACTOR {worker_id}] Unsupported game: {game_class_name}"); return None, 0, worker_id

    learning_net = NNClass(ch=nn_arch_config["channels"], blocks=nn_arch_config["blocks"]).to(worker_device)
    try:
        state_d = torch.load(current_learning_net_state_dict_path, map_location=worker_device, weights_only=True)
        if "net" in state_d: learning_net.load_state_dict(state_d["net"])
        else: learning_net.load_state_dict(state_d)
    except Exception as e:
        print(f"[ACTOR {worker_id}] Error loading learning_net from {current_learning_net_state_dict_path}: {e}"); return None, 0, worker_id
    learning_net.eval()

    def ln_mcts_fn(b): 
        with torch.no_grad(): lgs,v = learning_net(b); return lgs, v.unsqueeze(-1)
    learning_mcts = AlphaZeroMCTS(game_adapter_worker, ln_mcts_fn, worker_device, 
                                  mcts_config_learning["c_puct"], mcts_config_learning["dirichlet_alpha"], 
                                  mcts_config_learning["dirichlet_epsilon"])

    opp_net = None; opp_mcts = None
    actual_opponent_type = opponent_config["type"]
    opponent_checkpoint_path_from_config = opponent_config.get("path") # This is the specific path chosen by main process

    if actual_opponent_type == "past_checkpoint":
        if opponent_checkpoint_path_from_config:
            opp_net = NNClass(ch=nn_arch_config["channels"], blocks=nn_arch_config["blocks"]).to(worker_device)
            try:
                opp_sd = torch.load(opponent_checkpoint_path_from_config, map_location=worker_device, weights_only=True)
                if "net" in opp_sd: opp_net.load_state_dict(opp_sd["net"])
                else: opp_net.load_state_dict(opp_sd)
                opp_net.eval()
                def opp_mcts_fn(b): 
                    with torch.no_grad(): lgs,v = opp_net(b); return lgs, v.unsqueeze(-1)
                opp_mcts_cfg = opponent_config.get("mcts_config", {})
                opp_mcts = AlphaZeroMCTS(game_adapter_worker, opp_mcts_fn, worker_device, 
                                         opp_mcts_cfg.get("c_puct_opponent", 1.41), 0, 0) 
                if debug_mode: print(f"[ACTOR {worker_id}] Opponent is past self: {opponent_checkpoint_path_from_config}")
            except Exception as e:
                if debug_mode: print(f"[ACTOR {worker_id}] Failed to load opponent {opponent_checkpoint_path_from_config}: {e}. Defaulting to self-play.")
                actual_opponent_type = "self"; opp_net = None; opp_mcts = None # Fallback
        else:
            # This case means main process said "past_checkpoint" but didn't provide a path.
            if debug_mode: print(f"[ACTOR {worker_id}] Opponent type was 'past_checkpoint' but no path provided by main. Defaulting to self-play.")
            actual_opponent_type = "self"
    
    st = game_adapter_worker.c4_game.getInitialState()
    game_history_tuples: List[Tuple[dict, np.ndarray, int]] = []
    move_no = 0
    current_temp = 1.0
    learning_agent_char_in_game = "X"; opponent_char_in_game = "O"

    while not game_adapter_worker.isTerminal(st) and move_no < max_game_moves:
        current_player = game_adapter_worker.getCurrentPlayer(st)
        is_learning_turn = (current_player == learning_agent_char_in_game)
        
        for tm, tv in temp_schedule_list: 
            if move_no < tm: current_temp = tv; break

        mcts_to_use = None
        sims_for_current_move = 0

        if is_learning_turn:
            mcts_to_use = learning_mcts
            sims_for_current_move = mcts_config_learning["mcts_simulations_learning"]
        else: # Opponent's turn
            if actual_opponent_type == "self":
                mcts_to_use = learning_mcts 
                sims_for_current_move = mcts_config_learning["mcts_simulations_learning"]
            else: # Past checkpoint opponent, opp_mcts should be initialized
                if opp_mcts is not None:
                    mcts_to_use = opp_mcts
                    # Sims for opponent already in mcts_config_learning["mcts_simulations_opponent"]
                    # Or from opponent_config["mcts_config"]["mcts_simulations_opponent"] if specifically set
                    # The MCTS for opponent (opp_mcts) was initialized with its own C_PUCT, so that's fine.
                    # The number of simulations for the opponent is generally passed via mcts_config_learning from main args.
                    sims_for_current_move = mcts_config_learning.get("mcts_simulations_opponent", 50) # Fallback if not in dict
                else: # Should not happen if configured correctly, fallback to self-play logic
                    if debug_mode: print(f"[ACTOR {worker_id}] Opponent MCTS was None for past player. Fallback to learning MCTS for opponent.")
                    mcts_to_use = learning_mcts
                    sims_for_current_move = mcts_config_learning["mcts_simulations_opponent"]

        if mcts_to_use is None: # Should ideally not be reached if logic above is sound
            if debug_mode: print(f"[ACTOR {worker_id}] mcts_to_use is None unexpectedly. Defaulting to learning_mcts.")
            mcts_to_use = learning_mcts 
            sims_for_current_move = mcts_config_learning["mcts_simulations_learning"] if is_learning_turn else mcts_config_learning["mcts_simulations_opponent"]
        
        legal_actions = game_adapter_worker.getLegalActions(st)
        if not legal_actions: 
            if debug_mode: print(f"[ACTOR {worker_id}] No legal actions for {current_player}. Move {move_no}.");
            break

        chosen_action, policy_dict = mcts_to_use.get_action_policy(st, sims_for_current_move, current_temp, debug_mcts=False)
        
        policy_vec = np.zeros(game_adapter_worker.get_action_size(), dtype=np.float32)
        if policy_dict: 
            for act_idx, prob in policy_dict.items(): 
                if 0 <= act_idx < game_adapter_worker.get_action_size(): policy_vec[act_idx] = prob
        s_sum = policy_vec.sum()
        if abs(s_sum - 1.0) > 1e-5 and s_sum > 1e-5: policy_vec /= s_sum
        elif s_sum < 1e-5 and legal_actions: 
            for la in legal_actions: policy_vec[la] = 1.0/len(legal_actions)

        if is_learning_turn: 
            game_history_tuples.append((game_adapter_worker.copyState(st), policy_vec, 0))
        
        st = game_adapter_worker.applyAction(st, chosen_action)
        move_no += 1

    winner = game_adapter_worker.getGameOutcome(st)
    if move_no >= max_game_moves and winner is None: winner = "Draw"
    
    z_outcome = 0 
    if winner == learning_agent_char_in_game: z_outcome = 1
    elif winner == opponent_char_in_game: z_outcome = -1
    
    final_history_for_learner: List[Tuple[dict, np.ndarray, int]] = []
    for recorded_s, recorded_pi, _ in game_history_tuples:
        final_history_for_learner.append((recorded_s, recorded_pi, z_outcome))

    # Debug log for very short games from learner's perspective
    if debug_mode and len(final_history_for_learner) < 7: # Arbitrary threshold for "short"
        print(f"[ACTOR {worker_id} SHORT GAME DEBUG] Game moves: {move_no}, Winner: {winner}, Learner ('{learning_agent_char_in_game}') Exps: {len(final_history_for_learner)}, Opponent: {actual_opponent_type}")

    saved_filepath = None
    if final_history_for_learner and experience_save_dir:
        Path(experience_save_dir).mkdir(parents=True, exist_ok=True)
        ts_save = int(time.time() * 1000000) 
        game_fname = f"exp_w{worker_id}_g{seed}_{ts_save}_m{move_no}_res{z_outcome}.pkl"
        filepath_obj = Path(experience_save_dir) / game_fname
        try:
            with open(filepath_obj, 'wb') as f_out_pkl:
                pickle.dump(final_history_for_learner, f_out_pkl)
            saved_filepath = str(filepath_obj)
            if debug_mode: print(f"[ACTOR {worker_id}] Saved {len(final_history_for_learner)} experiences to {saved_filepath}")
        except Exception as e:
            print(f"[ACTOR {worker_id}] Error saving experiences to {filepath_obj}: {e}")
            saved_filepath = None
            
    return saved_filepath, len(final_history_for_learner), worker_id

if __name__ == "__main__":
    run()
