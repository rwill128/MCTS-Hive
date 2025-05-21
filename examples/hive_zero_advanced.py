#!/usr/bin/env python3
"""Advanced self-play training loop for Hive (Pocket Edition).

Adapts AlphaZero-style techniques for Hive.
"""

from __future__ import annotations
import argparse
import random
from collections import deque
from pathlib import Path
from typing import List, Tuple, Dict, Any # Keep Any for now

try:
    import numpy as np
except Exception: 
    np = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception: 
    torch = None
    nn = None
    F = None

if torch:
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

from HivePocket.HivePocket import HiveGame # Import HiveGame
from mcts.alpha_zero_mcts import AlphaZeroMCTS
from mcts.replay_buffer import PrioritizedReplayBuffer

# --- Hive Specific Constants (from hive_zero.py) ---
BOARD_R = 6  # covers Pocket board positions, results in a 13x13 grid effectively
HEX_GRID_DIAMETER = BOARD_R * 2 + 1  # H and W in hive_zero.py
PIECE_TYPES = ["Q", "B", "S", "A", "G"] # Queen, Beetle, Spider, Ant, Grasshopper
NUM_PIECE_TYPES = len(PIECE_TYPES)
# Channels: player pieces (NUM_PIECE_TYPES), opponent pieces (NUM_PIECE_TYPES), side-to-move (1)
NUM_CHANNELS = NUM_PIECE_TYPES * 2 + 1 
COORD_SHIFT = BOARD_R # To map (q,r) to positive array indices

# Action space definition (from hive_zero.py, needs careful review for moves)
# This defines a flat space for (q, r, piece_type_to_place)
# This might only cover placements, or a specific type of move interpretation.
AXIAL_TO_IDX_HIVE: Dict[Tuple[int, int, str] | str, int] = {} # Allow str for PASS key
FLAT_ACTION_REPRESENTATION_HIVE: List[Tuple[int, int, str] | str] = [] # Allow str for PASS representation

# Add grid-based actions first
for q_coord in range(-BOARD_R, BOARD_R + 1):
    for r_coord in range(-BOARD_R, BOARD_R + 1):
        for piece_char in PIECE_TYPES:
            action_tuple = (q_coord, r_coord, piece_char)
            AXIAL_TO_IDX_HIVE[action_tuple] = len(FLAT_ACTION_REPRESENTATION_HIVE)
            FLAT_ACTION_REPRESENTATION_HIVE.append(action_tuple)

# Add PASS action - it will be the last index
PASS_ACTION_STR = "PASS"
AXIAL_TO_IDX_HIVE[PASS_ACTION_STR] = len(FLAT_ACTION_REPRESENTATION_HIVE)
FLAT_ACTION_REPRESENTATION_HIVE.append(PASS_ACTION_STR)

ACTION_SIZE = len(FLAT_ACTION_REPRESENTATION_HIVE) # Now includes PASS (e.g., 845 + 1 = 846)
PASS_ACTION_INDEX = AXIAL_TO_IDX_HIVE[PASS_ACTION_STR]

# This ACTION_SIZE is a CRITICAL parameter and its interpretation for Hive moves is complex.
# The original hive_zero.py uses this for its policy output size.
# We will need to ensure the adapter correctly maps legal Hive game actions to this flat space.

if np is None or torch is None: # Guard for headless/no-torch use after constants
    print("NumPy and PyTorch are required for the main script execution beyond constants.")
    # Potentially exit or define dummy classes if essential parts can't run

# ---------------------------------------------------------------------------
# State encoding (Hive specific - adapted from hive_zero.py)
# ---------------------------------------------------------------------------
def encode_hive_state(state: dict, perspective: str) -> torch.Tensor:
    if torch is None: raise RuntimeError("PyTorch is required")
    t = torch.zeros((NUM_CHANNELS, HEX_GRID_DIAMETER, HEX_GRID_DIAMETER))
    # Determine opponent perspective for piece encoding
    # Assuming perspective is 'Player1' or 'Player2' as used in HiveGame
    # The original hive_zero.py used `side = 0 if state["current_player"] == perspective else 1` for the last plane.
    # And for piece planes: `(0 if owner == perspective else NUM_PIECE_TYPES)`

    for (q, r), stack in state["board"].items():
        if not stack: continue
        owner, insect_type_full = stack[-1] # e.g., ('Player1', 'Q1')
        piece_char = insect_type_full[0] # Just 'Q', 'B', etc.

        if piece_char not in PIECE_TYPES: continue # Should not happen with valid Hive pieces

        plane_offset = 0
        if owner != perspective: # Opponent's piece
            plane_offset = NUM_PIECE_TYPES
        
        piece_plane_idx = PIECE_TYPES.index(piece_char) + plane_offset
        
        # Convert axial (q,r) to 0-indexed (y,x) for tensor
        y, x = r + COORD_SHIFT, q + COORD_SHIFT # Matches to_xy in hive_zero

        if 0 <= y < HEX_GRID_DIAMETER and 0 <= x < HEX_GRID_DIAMETER:
            t[piece_plane_idx, y, x] = 1.0 # Mark top piece
        # Consider representing stack depth or Beetle-on-top differently if needed later.

    # Last plane: side-to-move (1.0 if perspective player is to move, 0.0 otherwise)
    if state["current_player"] == perspective:
        t[-1].fill_(1.0)
    # else it remains 0.0, which means opponent is to move from perspective.
    return t

# ---------------------------------------------------------------------------
# Data Augmentation (Hive - Placeholder, complex due to hex grid)
# ---------------------------------------------------------------------------
def reflect_hive_state_policy(state_dict: Dict, policy_vector: np.ndarray) -> Tuple[Dict, np.ndarray]:
    # Hive symmetries are 6-fold rotations. Reflection is more complex.
    # For now, return original (no augmentation) to keep it simple.
    # print("Warning: Hive data augmentation (reflection/rotation) is not implemented.")
    return state_dict, policy_vector

# ---------------------------------------------------------------------------
# Game Adapter for Hive MCTS
# ---------------------------------------------------------------------------
class HiveAdapter:
    def __init__(self, hive_game: HiveGame):
        self.hive_game = hive_game
        self.action_size = ACTION_SIZE
        # Ensure PIECE_TYPES, AXIAL_TO_IDX_HIVE, FLAT_ACTION_REPRESENTATION_HIVE are accessible
        # (they are global in this file)

    def _hive_action_to_int(self, hive_action: Tuple) -> int | None:
        """Converts a native HiveGame action to our flat integer index."""
        action_type = hive_action[0]
        idx = None
        if action_type == "PLACE":
            _, piece_type_full, (q, r) = hive_action
            piece_char = piece_type_full[0]
            idx = AXIAL_TO_IDX_HIVE.get((q, r, piece_char))
        elif action_type == "MOVE":
            _, _, (q, r) = hive_action 
            idx = AXIAL_TO_IDX_HIVE.get((q, r, PIECE_TYPES[0]))
        elif action_type == "PASS":
            idx = PASS_ACTION_INDEX # Use the dedicated index
        return idx

    def _int_to_hive_action(self, int_action: int, legal_hive_actions: List[Tuple]) -> Tuple | None:
        """
        Converts a flat integer index back to a native HiveGame action.
        Must select from the provided legal_hive_actions.
        """
        if int_action == PASS_ACTION_INDEX:
            # If the policy chose PASS, check if it's legal. If so, return it.
            for legal_act in legal_hive_actions:
                if legal_act[0] == "PASS":
                    return legal_act
            # If PASS was chosen by policy but isn't legal, this is an issue.
            # Fallback to another legal move (or random if many non-PASS options exist).
            # This implies the policy should have learned not to output PASS if illegal.
            if legal_hive_actions: # Fallback if PASS not legal but other moves are
                # Avoid picking PASS again if it was the only one and illegal
                non_pass_actions = [act for act in legal_hive_actions if act[0] != "PASS"]
                if non_pass_actions:
                    return random.choice(non_pass_actions)
                elif legal_hive_actions[0][0] == "PASS": # Only PASS was legal, and int_action was not PASS_ACTION_INDEX
                     # This case is complex: int_action was not PASS, but PASS is only legal option. Should not happen if policy is good.
                     return legal_hive_actions[0] # Force PASS if it's the only option
            return None # No legal action to map to if PASS was illegal and no other options

        if not (0 <= int_action < PASS_ACTION_INDEX): # Grid-based actions
             # If int_action is out of bounds for grid actions (e.g. a bad policy output for PASS_ACTION_INDEX)
            if legal_hive_actions: return random.choice(legal_hive_actions)
            return None

        q_flat, r_flat, piece_char_flat = FLAT_ACTION_REPRESENTATION_HIVE[int_action]

        # Try PLACE
        for legal_act in legal_hive_actions:
            if legal_act[0] == "PLACE":
                _, piece_type_full, (q_game, r_game) = legal_act
                if (q_game, r_game) == (q_flat, r_flat) and piece_type_full[0] == piece_char_flat:
                    return legal_act
        
        # Try MOVE
        possible_moves_to_dest = []
        for legal_act in legal_hive_actions:
            if legal_act[0] == "MOVE":
                _, _, (q_dest, r_dest) = legal_act
                if (q_dest, r_dest) == (q_flat, r_flat):
                    possible_moves_to_dest.append(legal_act)
        if possible_moves_to_dest:
            return random.choice(possible_moves_to_dest)

        # Fallback if int_action (grid based) didn't map to a legal PLACE or MOVE
        if legal_hive_actions:
            # If only PASS is legal at this point, but int_action was not PASS_ACTION_INDEX, pick PASS.
            if len(legal_hive_actions) == 1 and legal_hive_actions[0][0] == "PASS":
                return legal_hive_actions[0]
            # Otherwise, pick a random non-PASS action if possible, or any random legal action.
            non_pass_actions = [act for act in legal_hive_actions if act[0] != "PASS"]
            if non_pass_actions: return random.choice(non_pass_actions)
            return random.choice(legal_hive_actions) 
        return None

    def getCurrentPlayer(self, state: Dict) -> str:
        return self.hive_game.getCurrentPlayer(state)

    def getLegalActions(self, state: Dict) -> List[int]:
        native_actions = self.hive_game.getLegalActions(state)
        int_actions = []
        seen_indices = set()
        has_pass_action = False
        for native_act in native_actions:
            idx = self._hive_action_to_int(native_act)
            if idx is not None:
                if idx == PASS_ACTION_INDEX:
                    has_pass_action = True # Ensure PASS is only added once
                elif idx not in seen_indices: # For grid actions
                    int_actions.append(idx)
                    seen_indices.add(idx)
        if has_pass_action:
            int_actions.append(PASS_ACTION_INDEX) # Add PASS if it was a legal option
        return int_actions

    def applyAction(self, state: Dict, int_action: int) -> Dict:
        legal_native_actions = self.hive_game.getLegalActions(state)
        if not legal_native_actions: # Should not happen if game is not terminal
            # Or could be a forced pass situation if only PASS was legal and filtered out
            if self.hive_game.isTerminal(state):
                 return state # No action to apply if terminal
            # This implies an issue or PASS was the only move
            # If only PASS was legal, and we didn't map PASS to an int_action,
            # then MCTS should not have picked any other action.
            # However, if it did due to policy error, we might need a fallback.
            # For now, assume int_action will correspond to a translatable action.
            pass # Fall through to _int_to_hive_action which might pick randomly if mapping fails

        hive_action_to_apply = self._int_to_hive_action(int_action, legal_native_actions)
        
        if hive_action_to_apply is None: 
            # This means int_action from policy didn't map to any legal native action
            # This is a problem: policy might be suggesting illegal things or unmappable things
            # Fallback: if legal_native_actions exist, pick one (e.g. PASS if it's the only one, or random)
            if legal_native_actions:
                print(f"Warning: int_action {int_action} could not be mapped to a specific legal Hive action. Choosing random legal action.")
                hive_action_to_apply = random.choice(legal_native_actions)
            else:
                # This state should be terminal or have had a PASS option. If no actions, it's an issue.
                print(f"Error: No legal native actions to pick from for int_action {int_action} in applyAction.")
                return state # No change
                
        return self.hive_game.applyAction(state, hive_action_to_apply)

    def isTerminal(self, state: Dict) -> bool:
        return self.hive_game.isTerminal(state)

    def getGameOutcome(self, state: Dict) -> str: 
        return self.hive_game.getGameOutcome(state)

    def encode_state(self, state: Dict, player_perspective: str) -> torch.Tensor:
        return encode_hive_state(state, player_perspective)

    def copyState(self, state: Dict) -> Dict:
        return self.hive_game.copyState(state)
    
    def get_action_size(self) -> int:
        return self.action_size # This is the flat ACTION_SIZE (e.g. 845)

# ---------------------------------------------------------------------------
# Neural network (Hive specific)
# ---------------------------------------------------------------------------
if torch is not None:
    class ResidualBlockHive(nn.Module):
        def __init__(self, ch: int):
            super().__init__()
            self.c1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
            self.b1 = nn.BatchNorm2d(ch)
            self.c2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
            self.b2 = nn.BatchNorm2d(ch)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = F.relu(self.b1(self.c1(x)))
            y = self.b2(self.c2(y))
            return F.relu(x + y)

    class AdvancedHiveZeroNet(nn.Module):
        def __init__(self, ch: int = 128, blocks: int = 10): # Defaults for Hive
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(NUM_CHANNELS, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(),
            )
            self.res = nn.Sequential(*[ResidualBlockHive(ch) for _ in range(blocks)])
            
            # Policy head outputs ACTION_SIZE logits (now includes PASS)
            self.policy_conv = nn.Conv2d(ch, 2, kernel_size=1)
            self.policy_bn = nn.BatchNorm2d(2)
            self.policy_flatten = nn.Flatten()
            self.policy_linear = nn.Linear(2 * HEX_GRID_DIAMETER * HEX_GRID_DIAMETER, ACTION_SIZE) 

            # Value head
            self.value_conv = nn.Conv2d(ch, 1, kernel_size=1)
            self.value_bn = nn.BatchNorm2d(1)
            self.value_flatten = nn.Flatten()
            self.value_linear1 = nn.Linear(1 * HEX_GRID_DIAMETER * HEX_GRID_DIAMETER, 256) 
            self.value_linear2 = nn.Linear(256, 1)

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
    class ResidualBlockHive:
        def __init__(self, *args, **kwargs):
            pass
    class AdvancedHiveZeroNet:
        def __init__(self, *args, **kwargs):
            pass

args_global = None 

# ---------------------------------------------------------------------------
# Self-play helpers (adapted for Hive)
# ---------------------------------------------------------------------------
def play_one_game(
    net: AdvancedHiveZeroNet, game_adapter: HiveAdapter, mcts_instance: AlphaZeroMCTS,     
    temp_schedule: List[Tuple[int, float]], mcts_simulations: int,
    max_moves: int = 100, debug_mode: bool = False
) -> List[Tuple[dict, np.ndarray, int]]:
    if torch is None or np is None: raise RuntimeError("PyTorch and NumPy are required")
    st = game_adapter.hive_game.getInitialState() 
    hist: List[Tuple[dict, np.ndarray, int]] = [] 
    move_no = 0
    current_temp = 1.0
    game_terminated_by_mcts = False # Flag to see if MCTS loop ended game

    if debug_mode: print("\n--- play_one_game START (Hive) ---")

    while not game_adapter.isTerminal(st) and move_no < max_moves:
        if debug_mode:
            print(f"\n[play_one_game Hive] Move: {move_no}, Player: {st['current_player']}")
        for th_moves, t_val in temp_schedule:
            if move_no < th_moves: current_temp = t_val; break
        player_perspective = game_adapter.getCurrentPlayer(st)
        legal_actions_for_mcts = game_adapter.getLegalActions(st)
        if not legal_actions_for_mcts:
            if debug_mode: print(f"[play_one_game Hive] No legal actions from adapter. Ending game.")
            game_terminated_by_mcts = True # Game effectively ends here due to no MCTS actions
            break
        if debug_mode: print(f"[play_one_game Hive] MCTS sims: {mcts_simulations}, Temp: {current_temp}, Perspective: {player_perspective}")
        
        chosen_action_int, mcts_policy_dict = mcts_instance.get_action_policy(
            root_state=st, num_simulations=mcts_simulations, temperature=current_temp,
            debug_mcts=debug_mode)
        if chosen_action_int is None and not legal_actions_for_mcts : # Should be caught by above legal_actions check
             if debug_mode: print("[play_one_game Hive] MCTS returned no action and no legal actions were found. Breaking.")
             game_terminated_by_mcts = True
             break
        elif chosen_action_int is None and legal_actions_for_mcts:
             if debug_mode: print("[play_one_game Hive] MCTS returned no action despite legal actions. Choosing random.")
             chosen_action_int = random.choice(legal_actions_for_mcts)
        
        policy_target_vector = np.zeros(game_adapter.get_action_size(), dtype=np.float32)
        if mcts_policy_dict:
            for action_idx, prob in mcts_policy_dict.items():
                if 0 <= action_idx < game_adapter.get_action_size(): policy_target_vector[action_idx] = prob
        current_sum = policy_target_vector.sum()
        if abs(current_sum - 1.0) > 1e-5 and current_sum > 1e-5: policy_target_vector /= current_sum
        elif current_sum < 1e-5 and legal_actions_for_mcts:
            uniform_prob = 1.0 / len(legal_actions_for_mcts)
            for la_int in legal_actions_for_mcts: policy_target_vector[la_int] = uniform_prob
        
        if debug_mode: print(f"[play_one_game Hive] MCTS chosen_action_int: {chosen_action_int}, Policy: {mcts_policy_dict}")
        hist.append((game_adapter.copyState(st), policy_target_vector, 0))
        st = game_adapter.applyAction(st, chosen_action_int)
        move_no += 1

    # Determine game outcome
    winner = game_adapter.getGameOutcome(st) 
    if not game_terminated_by_mcts and move_no >= max_moves and winner is None:
        if debug_mode: print(f"[play_one_game Hive] Game reached max_moves ({max_moves}) limit. Declaring DRAW.")
        winner = "Draw" # Explicitly a draw if max_moves reached and not otherwise terminal
    elif game_terminated_by_mcts and winner is None:
        # If MCTS found no moves but game wasn't declared terminal by HiveGame (e.g. only PASS was available and filtered)
        # This is a tricky state. Forcing a draw is safest.
        if debug_mode: print(f"[play_one_game Hive] Game ended due to no MCTS actions, but not terminal. Declaring DRAW.")
        winner = "Draw"

    z_p1 = 0
    if winner == "Player1": z_p1 = 1
    elif winner == "Player2": z_p1 = -1
    # For "Draw" or any other unhandled winner string, z_p1 remains 0.
    
    if debug_mode: print(f"\n[play_one_game Hive] Game Over. Actual Winner: {winner}, z_p1 (Player1's perspective): {z_p1}")

    final_history = []
    for rec_state, pol, _ in hist:
        p_at_state = game_adapter.getCurrentPlayer(rec_state)
        val_for_player = z_p1 if p_at_state == "Player1" else -z_p1
        final_history.append((rec_state, pol, val_for_player))
    if debug_mode: print("--- play_one_game END (Hive) ---")
    return final_history

# ---------------------------------------------------------------------------
# Training helpers (adapted for Hive)
# ---------------------------------------------------------------------------
def train_step(net: AdvancedHiveZeroNet, batch_experiences: list, is_weights: np.ndarray, 
               opt: torch.optim.Optimizer, dev: str, game_adapter: HiveAdapter, 
               augment_prob: float, ent_beta_val: float, debug_mode: bool = False) -> Tuple[float, np.ndarray]:
    if debug_mode: print("\n--- train_step START (Hive) ---")
    S_list, P_tgt_list, V_tgt_list = [], [], []
    if not batch_experiences: 
        if debug_mode: print("[train_step Hive] Batch empty."); 
        return 0.0, np.array([])
    for s_orig, p_orig, v_orig in batch_experiences:
        s_enc, p_use = s_orig, p_orig
        if random.random() < augment_prob: s_enc, p_use = reflect_hive_state_policy(s_orig, p_orig)
        player_persp = game_adapter.getCurrentPlayer(s_enc)
        S_list.append(game_adapter.encode_state(s_enc, player_persp))
        P_tgt_list.append(p_use); V_tgt_list.append(v_orig)
    S, P_tgt, V_tgt = torch.stack(S_list).to(dev), torch.tensor(np.array(P_tgt_list), dtype=torch.float32, device=dev), torch.tensor(V_tgt_list, dtype=torch.float32, device=dev)
    is_weights_t = torch.tensor(is_weights, dtype=torch.float32, device=dev).unsqueeze(1)
    logits, V_pred = net(S)
    logP = F.log_softmax(logits, dim=1)
    loss_p_per_sample = F.kl_div(logP, P_tgt, reduction="none").sum(dim=1)
    loss_v_per_sample = F.mse_loss(V_pred.squeeze(), V_tgt, reduction="none")
    weighted_loss_p = (loss_p_per_sample * is_weights_t.squeeze()).mean()
    weighted_loss_v = (loss_v_per_sample * is_weights_t.squeeze()).mean()
    entropy = -(torch.exp(logP) * logP).sum(dim=1).mean()
    total_loss = weighted_loss_p + weighted_loss_v - ent_beta_val * entropy
    opt.zero_grad(); total_loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    td_err = np.abs(V_tgt.cpu().detach().numpy() - V_pred.squeeze().cpu().detach().numpy())
    if debug_mode: print(f"[train_step Hive] Loss: {total_loss.item():.4f}")
    if debug_mode: print("--- train_step END (Hive) ---")
    return float(total_loss.item()), td_err

def save_buffer_experiences(buf: PrioritizedReplayBuffer | deque, path: Path) -> None:
    if torch is None: raise RuntimeError("PyTorch is required")
    data_to_save = list(buf.data_buffer[:len(buf)]) if isinstance(buf, PrioritizedReplayBuffer) else list(buf)
    torch.save(data_to_save, path); print(f"Saved {len(data_to_save)} exp to {path}")

def load_experiences_to_buffer(target_buffer: PrioritizedReplayBuffer | deque, path: Path) -> None:
    if torch is None: raise RuntimeError("PyTorch is required")
    if not path.exists(): print(f"Buffer {path} not found."); return
    try: data = torch.load(path, weights_only=False) 
    except: print(f"Error loading {path}."); return
    if isinstance(target_buffer, PrioritizedReplayBuffer):
        loaded_count = 0
        for exp in data:
            if len(target_buffer) < target_buffer.capacity: target_buffer.add(exp); loaded_count+=1
        print(f"Loaded {loaded_count} exp into PER buffer. Size: {len(target_buffer)}/{target_buffer.capacity}")
    elif isinstance(target_buffer, deque): target_buffer.extend(data); print(f"Loaded {len(data)} exp into deque. Size: {len(target_buffer)}/{target_buffer.maxlen}")

# ---------------------------------------------------------------------------
# Training loop (adapted for Hive)
# ---------------------------------------------------------------------------
def run(parsed_cli_args=None) -> None:
    global args_global 
    args_global = parsed_cli_args if parsed_cli_args is not None else parser().parse_args()

    if args_global.debug_single_loop:
        _debug_flag_main = True 
        print("!!!!!!!!!!!!!!!!! DEBUG SINGLE LOOP MODE ENABLED (Hive) !!!!!!!!!!!!!!!!!")
        args_global.epochs = 1; args_global.bootstrap_games = 1; args_global.mcts_simulations = 10; args_global.batch_size = 4
        args_global.min_buffer_fill_standard = 1; args_global.min_buffer_fill_for_per_training = 1; args_global.min_buffer_fill_for_per_bootstrap = 1
        args_global.log_every = 1; args_global.ckpt_every = 1; args_global.save_buffer_every = 1
        args_global.skip_bootstrap = False
        args_global.max_game_moves = 15 # Drastically reduce for debug to see if game loop finishes
        print(f"[DEBUG] Overridden max_game_moves to: {args_global.max_game_moves}")
    
    if args_global.use_per and args_global.per_beta_epochs <= 0:
        args_global.per_beta_epochs = args_global.epochs
        if not args_global.debug_single_loop: 
            print(f"PER: Beta annealing epochs set to total epochs: {args_global.per_beta_epochs}")

    if torch is None or np is None: raise RuntimeError("PyTorch and NumPy are required")
    dev = "cuda" if args_global.gpu and torch.cuda.is_available() else "cpu"; print(f"Using device: {dev}")
    ckdir = Path(args_global.ckpt_dir); ckdir.mkdir(exist_ok=True, parents=True)
    hive_game_instance = HiveGame(); game_adapter = HiveAdapter(hive_game_instance)
    net = AdvancedHiveZeroNet(ch=args_global.channels, blocks=args_global.blocks).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args_global.lr)
    scheduler = None
    if args_global.lr_scheduler == "cosine":
        t_max = args_global.lr_t_max if args_global.lr_t_max > 0 else args_global.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=args_global.lr_eta_min)
        if not args_global.debug_single_loop: print(f"Using CosineAnnealingLR: T_max={t_max}, eta_min={args_global.lr_eta_min}")
    
    def mcts_model_fn(batch_in): net.eval(); lgs,v = net(batch_in); net.train(); return lgs, v.unsqueeze(-1)
    mcts = AlphaZeroMCTS(game_adapter, mcts_model_fn, torch.device(dev), args_global.c_puct, args_global.dirichlet_alpha, args_global.dirichlet_epsilon)
    state_path = ckdir / "train_state_hive.pt"; start_ep = 1
    
    if args_global.resume_full_state and state_path.exists() and not args_global.debug_single_loop:
        st_ckpt = torch.load(state_path, map_location=dev); net.load_state_dict(st_ckpt["net"]); opt.load_state_dict(st_ckpt["opt"])
        start_ep = int(st_ckpt.get("epoch",1)) + 1
        if scheduler and "scheduler" in st_ckpt and st_ckpt["scheduler"]: scheduler.load_state_dict(st_ckpt["scheduler"])
    elif args_global.resume_weights and not args_global.debug_single_loop:
        net.load_state_dict(torch.load(args_global.resume_weights, map_location=dev))

    buf_capacity = args_global.batch_size if args_global.debug_single_loop and args_global.batch_size > 0 else (args_global.buffer_size if not args_global.debug_single_loop else 8) 
    if args_global.use_per: buf = PrioritizedReplayBuffer(buf_capacity, args_global.per_alpha, args_global.per_beta_start, args_global.per_beta_epochs, args_global.per_epsilon); print(f"Using PER (cap: {buf.capacity}).")
    else: buf = deque(maxlen=buf_capacity); print(f"Using deque (maxlen: {buf.maxlen}).")
    buffer_file_path = Path(args_global.buffer_path)
    if not args_global.debug_single_loop: load_experiences_to_buffer(buf, buffer_file_path)
    else: print("[run DEBUG Hive] Skipping buffer load.")
    temp_schedule = [(args_global.temp_decay_moves, 1.0), (float('inf'), args_global.final_temp)]
    MAX_HIVE_MOVES = args_global.max_game_moves

    min_fill_boot = args_global.min_buffer_fill_for_per_bootstrap if isinstance(buf, PrioritizedReplayBuffer) else args_global.min_buffer_fill_standard
    if args_global.debug_single_loop: min_fill_boot = 1
    games_to_boot = 0
    if not args_global.skip_bootstrap and len(buf) < min_fill_boot:
        games_to_boot = args_global.bootstrap_games
        if len(buf) < min_fill_boot and not args_global.debug_single_loop:
            avg_states_pg = 40; needed_s = min_fill_boot - len(buf); needed_g = (needed_s+avg_states_pg-1)//avg_states_pg
            games_to_boot = max(games_to_boot, needed_g)
        if games_to_boot > 0 and not args_global.debug_single_loop: print(f"Bootstrapping {games_to_boot} games…")
        for g in range(games_to_boot):
            game_hist = play_one_game(net, game_adapter, mcts, temp_schedule, args_global.mcts_simulations, MAX_HIVE_MOVES, args_global.debug_single_loop)
            print(f"Game {g+1}/{games_to_boot}")
            if isinstance(buf, PrioritizedReplayBuffer): [buf.add(exp) for exp in game_hist]
            else: buf.extend(game_hist)
            if args_global.debug_single_loop or (g+1)%args_global.save_buffer_every==0: print(f"  Bootstrap game {g+1}/{games_to_boot} ({len(game_hist)} states) → buf {len(buf)}")

    if not args_global.debug_single_loop: print(f"Starting training epoch {start_ep} for {args_global.epochs} epochs.")
    try:
        for ep in range(start_ep, args_global.epochs + 1):
            if args_global.debug_single_loop:
                _debug_flag_ep_loop = True # Linter fix
                print(f"\n--- [DEBUG] Epoch {ep} START (Hive) ---")
            net.eval()
            game_hist = play_one_game(net, game_adapter, mcts, temp_schedule, args_global.mcts_simulations, MAX_HIVE_MOVES, args_global.debug_single_loop)
            if isinstance(buf, PrioritizedReplayBuffer): [buf.add(exp) for exp in game_hist]
            else: buf.extend(game_hist)
            net.train()
            min_train_fill = args_global.min_buffer_fill_for_per_training if isinstance(buf, PrioritizedReplayBuffer) else args_global.min_buffer_fill_standard
            if args_global.debug_single_loop: min_train_fill = 1
            if len(buf) < min_train_fill:
                if ep % args_global.log_every == 0: print(f"Epoch {ep} | Buffer {len(buf)} < min {min_train_fill}. Skip train.")
            else:
                if args_global.debug_single_loop:
                    _debug_flag_train_block = True # Linter fix
                    print(f"\n[run DEBUG Hive] Buffer ready. len(buf): {len(buf)}, batch: {args_global.batch_size}")
                actual_bs = min(args_global.batch_size, len(buf))
                if actual_bs == 0: continue
                batch_exp, is_w, data_idx = (None,None,None)
                if isinstance(buf, PrioritizedReplayBuffer):
                    sampled = buf.sample(actual_bs)
                    if sampled is None: continue
                    batch_exp, is_w, data_idx = sampled
                else:
                    batch_exp = random.sample(list(buf), actual_bs); is_w = np.ones(len(batch_exp),dtype=np.float32)
                loss, td_errs = train_step(net,batch_exp,is_w,opt,dev,game_adapter,args_global.augment_prob,args_global.ent_beta,args_global.debug_single_loop)
                if isinstance(buf,PrioritizedReplayBuffer) and data_idx is not None and len(data_idx)>0: buf.update_priorities(data_idx, td_errs)
                if ep % args_global.log_every == 0: print(f"Epoch {ep} | Loss {loss:.4f} | Buf {len(buf)} | LR {opt.param_groups[0]['lr']:.2e}")
            if scheduler: scheduler.step()
            if ep % args_global.ckpt_every == 0: 
                chkpt_path = ckdir / f"hive_chkpt_ep{ep:06d}.pt"; full_state_path = state_path
                if not args_global.debug_single_loop: torch.save(net.state_dict(), chkpt_path); torch.save({"net":net.state_dict(),"opt":opt.state_dict(),"epoch":ep,"scheduler":scheduler.state_dict() if scheduler else None},full_state_path)
                if args_global.debug_single_loop or ep % args_global.log_every == 0: print(f"Saved: {str(chkpt_path)} & {str(full_state_path)}")
            if ep % args_global.save_buffer_every == 0 and len(buf) > 0 and not args_global.debug_single_loop: save_buffer_experiences(buf, buffer_file_path)
            if isinstance(buf, PrioritizedReplayBuffer): buf.advance_epoch_for_beta_anneal()
            if args_global.debug_single_loop: print(f"--- Epoch {ep} END (Hive) ---")
    except KeyboardInterrupt: print("\nTraining interrupted.")
    finally:
        print("Saving final model and buffer (Hive)...")
        final_model_path = ckdir / "last_hive_model.pt" 
        torch.save(net.state_dict(), final_model_path)
        if len(buf) > 0: save_buffer_experiences(buf, buffer_file_path)
        ep_s = ep if 'ep' in locals() and 'start_ep' in locals() and ep>=start_ep else start_ep-1
        torch.save({"net":net.state_dict(),"opt":opt.state_dict(),"epoch":ep_s,"scheduler":scheduler.state_dict() if scheduler else None}, state_path)
        print(f"Final Hive model: {str(final_model_path)}, Buffer: {str(buffer_file_path)}, State: {str(state_path)}")

# ---------------------------------------------------------------------------
# CLI parser (adapted for Hive)
# ---------------------------------------------------------------------------
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AlphaZero-style training for Hive (Pocket Edition).")
    p.add_argument("--debug-single-loop", action="store_true", help="Minimal loop for debugging.")
    p.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    g_play = p.add_argument_group("Self-Play & MCTS (Hive)")
    g_play.add_argument("--bootstrap-games", type=int, default=20)
    g_play.add_argument("--skip-bootstrap", action="store_true")
    g_play.add_argument("--mcts-simulations", type=int, default=100) 
    g_play.add_argument("--max-game-moves", type=int, default=150)
    g_play.add_argument("--c-puct", type=float, default=1.5)
    g_play.add_argument("--dirichlet-alpha", type=float, default=0.1)
    g_play.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    g_play.add_argument("--temp-decay-moves", type=int, default=15)
    g_play.add_argument("--final-temp", type=float, default=0.05)
    g_train = p.add_argument_group("Training (Hive)")
    g_train.add_argument("--epochs", type=int, default=20000)
    g_train.add_argument("--lr", type=float, default=1e-4)
    g_train.add_argument("--batch-size", type=int, default=128)
    g_train.add_argument("--ent-beta", type=float, default=1e-3)
    g_train.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    g_train.add_argument("--lr-t-max", type=int, default=0)
    g_train.add_argument("--lr-eta-min", type=float, default=1e-7)
    g_train.add_argument("--augment-prob", type=float, default=0.0) 
    g_buffer = p.add_argument_group("Replay Buffer (Hive)")
    g_buffer.add_argument("--buffer-size", type=int, default=100000)
    g_buffer.add_argument("--min-buffer-fill-standard", type=int, default=2000)
    g_buffer.add_argument("--min-buffer-fill-for-per-training", type=int, default=20000)
    g_buffer.add_argument("--min-buffer-fill-for-per-bootstrap", type=int, default=2000)
    g_buffer.add_argument("--buffer-path", type=str, default="hive_adv_mcts_buffer.pth")
    g_buffer.add_argument("--save-buffer-every", type=int, default=100)
    g_buffer.add_argument("--use-per", action="store_true")
    g_buffer.add_argument("--per-alpha", type=float, default=0.6)
    g_buffer.add_argument("--per-beta-start", type=float, default=0.4)
    g_buffer.add_argument("--per-beta-epochs", type=int, default=0)
    g_buffer.add_argument("--per-epsilon", type=float, default=1e-5)
    g_nn = p.add_argument_group("Neural Network (Hive)")
    g_nn.add_argument("--channels", type=int, default=128)
    g_nn.add_argument("--blocks", type=int, default=10)
    g_mgmt = p.add_argument_group("Checkpointing & Logging (Hive)")
    g_mgmt.add_argument("--ckpt-dir", default="hive_checkpoints_az")
    g_mgmt.add_argument("--ckpt-every", type=int, default=100)
    g_mgmt.add_argument("--log-every", type=int, default=10)
    g_mgmt.add_argument("--resume-weights", metavar="PATH")
    g_mgmt.add_argument("--resume-full-state", action="store_true")
    return p

if __name__ == "__main__":
    run() 