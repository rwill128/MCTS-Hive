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

# ... (AdvancedHiveZeroNet - will define next) ...
# ... (play_one_game, train_step, save/load buffer, run, parser - will adapt from C4 version) ... 

args_global = None # For global access to parsed args

# ---------------------------------------------------------------------------
# Self-play helpers (adapted for Hive)
# ---------------------------------------------------------------------------
def play_one_game(
    net: AdvancedHiveZeroNet, 
    game_adapter: HiveAdapter, 
    mcts_instance: AlphaZeroMCTS,     
    temp_schedule: List[Tuple[int, float]], 
    mcts_simulations: int,
    max_moves: int = 100, # Typical Hive games are shorter than C4, but can be long
    debug_mode: bool = False
) -> List[Tuple[dict, np.ndarray, int]]:
    if torch is None or np is None: raise RuntimeError("PyTorch and NumPy are required")
    st = game_adapter.hive_game.getInitialState() 
    hist: List[Tuple[dict, np.ndarray, int]] = [] 
    move_no = 0
    current_temp = 1.0

    if debug_mode: print("\n--- play_one_game START (Hive) ---")

    while not game_adapter.isTerminal(st) and move_no < max_moves:
        if debug_mode:
            print(f"\n[play_one_game Hive] Move: {move_no}")
            # Skipping detailed board print for Hive due to complexity, print player and turn
            print(f"[play_one_game Hive] Current state (player {st['current_player']}) to move.")

        for threshold_moves, temp_val in temp_schedule:
            if move_no < threshold_moves: current_temp = temp_val; break
        
        player_perspective = game_adapter.getCurrentPlayer(st)
        if debug_mode: print(f"[play_one_game Hive] MCTS sims: {mcts_simulations}, Temp: {current_temp}, Perspective: {player_perspective}")
        
        # Get legal actions from adapter (these are already flat ints)
        legal_actions_for_mcts = game_adapter.getLegalActions(st)
        if not legal_actions_for_mcts:
            # This can happen if the only legal move is PASS and we are filtering it,
            # or if the game is over but isTerminal missed it (unlikely for HiveGame).
            if debug_mode: print(f"[play_one_game Hive] No legal (non-PASS) actions from adapter. State player: {st['current_player']}. Ending game.")
            break # End game if no actions can be mapped/played by MCTS

        chosen_action_int, mcts_policy_dict = mcts_instance.get_action_policy(
            root_state=st, num_simulations=mcts_simulations, temperature=current_temp,
            debug_mcts=debug_mode 
        )
        if debug_mode: print(f"[play_one_game Hive] MCTS chosen_action_int: {chosen_action_int}")
        if debug_mode: print(f"[play_one_game Hive] MCTS policy_dict: {mcts_policy_dict}") # keys are int actions
        
        policy_target_vector = np.zeros(game_adapter.get_action_size(), dtype=np.float32)
        if mcts_policy_dict:
            for action_idx_int, prob in mcts_policy_dict.items():
                if 0 <= action_idx_int < game_adapter.get_action_size(): 
                    policy_target_vector[action_idx_int] = prob
        
        current_sum = policy_target_vector.sum()
        if abs(current_sum - 1.0) > 1e-5 and current_sum > 1e-5:
             policy_target_vector /= current_sum
        elif current_sum < 1e-5 and not game_adapter.isTerminal(st):
            # Fallback if MCTS policy sum is zero (e.g. all illegal moves in policy or no legal actions mapped)
            # Use uniform over the adapter's legal integer actions
            if legal_actions_for_mcts: 
                uniform_prob = 1.0 / len(legal_actions_for_mcts)
                for la_int in legal_actions_for_mcts: policy_target_vector[la_int] = uniform_prob
            elif debug_mode:
                print("[play_one_game Hive] Warning: MCTS policy sum is zero and no legal adapter actions for fallback.")
        
        if debug_mode: print(f"[play_one_game Hive] Policy target (sum={policy_target_vector.sum()}): {policy_target_vector[:10]}...")

        current_state_copy = game_adapter.copyState(st)
        hist.append((current_state_copy, policy_target_vector, 0))
        if debug_mode: print(f"[play_one_game Hive] Appended to history. Player: {current_state_copy['current_player']}")
        
        # chosen_action_int from MCTS is already a flat int, adapter.applyAction expects this.
        st = game_adapter.applyAction(st, chosen_action_int)
        move_no += 1

    winner = game_adapter.getGameOutcome(st) # Player1, Player2, or Draw
    z = 0 # perspective of Player1
    if winner == "Draw": z = 0
    elif winner == "Player1": z = 1
    elif winner == "Player2": z = -1
    
    if debug_mode: print(f"\n[play_one_game Hive] Game Over. Winner: {winner}, z (Player1's perspective): {z}")

    final_history = []
    for idx, (recorded_state, policy, _) in enumerate(hist):
        player_at_state = game_adapter.getCurrentPlayer(recorded_state) # Player1 or Player2
        # Value target should be from the perspective of player_at_state
        value_for_state_player = z if player_at_state == "Player1" else -z
        final_history.append((recorded_state, policy, value_for_state_player))
        if debug_mode: print(f"[play_one_game Hive] final_hist item {idx}: Player: {player_at_state}, Val: {value_for_state_player}, Policy sum: {np.sum(policy) if policy is not None else 'N/A'}")
    
    if debug_mode: print("--- play_one_game END (Hive) ---")
    return final_history

# ---------------------------------------------------------------------------
# Training helpers (adapted for Hive)
# ---------------------------------------------------------------------------
def train_step(net: AdvancedHiveZeroNet, batch_experiences: list, is_weights: np.ndarray, 
               opt: torch.optim.Optimizer, dev: str, game_adapter: HiveAdapter, 
               augment_prob: float, ent_beta_val: float, debug_mode: bool = False) -> Tuple[float, np.ndarray]:
    if debug_mode: print("\n--- train_step START (Hive) ---")
    # ... (rest of train_step logic from c4_zero_advanced, using reflect_hive_state_policy) ...
    S_list, P_tgt_list, V_tgt_list = [], [], []
    if not batch_experiences: 
        if debug_mode: print("[train_step Hive] Batch empty.")
        return 0.0, np.array([])

    for i, (s_dict_orig, p_tgt_orig, v_tgt_orig) in enumerate(batch_experiences):
        s_to_enc, p_to_use = s_dict_orig, p_tgt_orig
        augmented = False
        if random.random() < augment_prob:
            s_to_enc, p_to_use = reflect_hive_state_policy(s_dict_orig, p_tgt_orig) 
            augmented = True
        
        # For Hive, perspective is always 'Player1' or 'Player2' from game state
        player_persp = game_adapter.getCurrentPlayer(s_to_enc) 
        S_list.append(game_adapter.encode_state(s_to_enc, player_persp))
        P_tgt_list.append(p_to_use)
        V_tgt_list.append(v_tgt_orig)
        if debug_mode and i < 1:
             print(f"  [train_step Hive] Sample {i} Augmented: {augmented}, Player: {player_persp}, ValueTgt: {v_tgt_orig}")
             p_to_use_np = np.array(p_to_use)
             print(f"  [train_step Hive] PolicyTgt (sum {p_to_use_np.sum()}): {p_to_use_np[:min(10,len(p_to_use_np))]}...")

    S = torch.stack(S_list).to(dev)
    P_tgt = torch.tensor(np.array(P_tgt_list), dtype=torch.float32, device=dev)
    V_tgt = torch.tensor(V_tgt_list, dtype=torch.float32, device=dev)
    is_weights_tensor = torch.tensor(is_weights, dtype=torch.float32, device=dev).unsqueeze(1)

    logits, V_pred = net(S)
    logP_pred = F.log_softmax(logits, dim=1)
    loss_p_per_sample = F.kl_div(logP_pred, P_tgt, reduction="none").sum(dim=1)
    loss_v_per_sample = F.mse_loss(V_pred.squeeze(), V_tgt, reduction="none")
    weighted_loss_p = (loss_p_per_sample * is_weights_tensor.squeeze()).mean()
    weighted_loss_v = (loss_v_per_sample * is_weights_tensor.squeeze()).mean()
    P_pred_dist = torch.exp(logP_pred)
    entropy = -(P_pred_dist * logP_pred).sum(dim=1).mean()
    total_loss = weighted_loss_p + weighted_loss_v - ent_beta_val * entropy

    if debug_mode: print(f"[train_step Hive] Losses (p,v,ent,total): {weighted_loss_p.item():.4f}, {weighted_loss_v.item():.4f}, {entropy.item():.4f}, {total_loss.item():.4f}")

    opt.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    td_errors = np.abs(V_tgt.cpu().detach().numpy() - V_pred.squeeze().cpu().detach().numpy())
    if debug_mode: print("--- train_step END (Hive) ---")
    return float(total_loss.item()), td_errors

# ... (save_buffer_experiences, load_experiences_to_buffer - should be generic enough) ...

# ---------------------------------------------------------------------------
# Training loop (adapted for Hive)
# ---------------------------------------------------------------------------
def run(parsed_cli_args=None) -> None:
    global args_global 
    args_global = parsed_cli_args if parsed_cli_args is not None else parser().parse_args()

    # ... (debug_single_loop overrides, PER beta annealing, device setup as in C4/TTT)
    if args_global.debug_single_loop:
        print("!!!!!!!!!!!!!!!!! DEBUG SINGLE LOOP MODE ENABLED (Hive) !!!!!!!!!!!!!!!!!")
        args_global.epochs = 1
        args_global.bootstrap_games = 1 
        args_global.mcts_simulations = 10 
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
    
    hive_game_instance = HiveGame()
    game_adapter = HiveAdapter(hive_game_instance)

    net = AdvancedHiveZeroNet(ch=args_global.channels, blocks=args_global.blocks).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args_global.lr)
    
    scheduler = None
    if args_global.lr_scheduler == "cosine":
        t_max_epochs = args_global.lr_t_max if args_global.lr_t_max > 0 else args_global.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max_epochs, eta_min=args_global.lr_eta_min)
        if not args_global.debug_single_loop: print(f"Using CosineAnnealingLR: T_max={t_max_epochs}, eta_min={args_global.lr_eta_min}")
    
    def mcts_model_fn_wrapper(encoded_state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        net.eval() 
        with torch.no_grad(): policy_logits, value_estimates = net(encoded_state_batch)
        net.train() 
        return policy_logits, value_estimates.unsqueeze(-1)

    mcts = AlphaZeroMCTS(
        game_interface=game_adapter, model_fn=mcts_model_fn_wrapper, device=torch.device(dev),
        c_puct=args_global.c_puct, dirichlet_alpha=args_global.dirichlet_alpha,
        dirichlet_epsilon=args_global.dirichlet_epsilon
    )

    state_path = ckdir / "train_state_hive.pt" # Path for Hive
    start_ep = 1
    # ... (resume logic as in C4/TTT, checking state_path)
    if args_global.resume_full_state and state_path.exists() and not args_global.debug_single_loop:
        print("Resuming full Hive training state from", state_path)
        st_checkpoint = torch.load(state_path, map_location=dev) 
        net.load_state_dict(st_checkpoint["net"])
        opt.load_state_dict(st_checkpoint["opt"])
        start_ep = int(st_checkpoint.get("epoch", 1)) + 1
        if scheduler and "scheduler" in st_checkpoint and st_checkpoint["scheduler"]:
            try: scheduler.load_state_dict(st_checkpoint["scheduler"]); print("Resumed LR scheduler.")
            except: print("Could not load scheduler state.")
        if args_global.lr_scheduler == "cosine" and scheduler and start_ep > scheduler.T_max:
             print(f"Warning: Resumed epoch {start_ep} > scheduler T_max {scheduler.T_max}.")
    elif args_global.resume_weights and not args_global.debug_single_loop:
        print("Resuming Hive weights from", args_global.resume_weights)
        net.load_state_dict(torch.load(args_global.resume_weights, map_location=dev))


    buf: PrioritizedReplayBuffer | deque
    # ... (buffer initialization logic as in C4/TTT, using args_global for PER/deque)
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
    if not args_global.debug_single_loop: load_experiences_to_buffer(buf, buffer_file_path)
    elif args_global.debug_single_loop: print("[run DEBUG Hive] Skipping buffer load.")
    
    temp_schedule = [(args_global.temp_decay_moves, 1.0), (float('inf'), args_global.final_temp)]
    
    # ... (bootstrap logic as in C4/TTT, calling play_one_game with Hive specifics, ACTION_SIZE for Hive is large, so max_moves might be different)
    # For Hive, max_moves in play_one_game can be higher, e.g., 100-150, or from args.
    MAX_HIVE_MOVES = args_global.max_game_moves # Add this to parser

    # Bootstrap games
    min_fill_for_bootstrap = args_global.min_buffer_fill_for_per_bootstrap if isinstance(buf, PrioritizedReplayBuffer) else args_global.min_buffer_fill_standard
    if args_global.debug_single_loop: min_fill_for_bootstrap = 1
    
    games_to_play_bootstrap = 0
    if not args_global.skip_bootstrap and len(buf) < min_fill_for_bootstrap:
        games_to_play_bootstrap = args_global.bootstrap_games 
        if len(buf) < min_fill_for_bootstrap and not args_global.debug_single_loop:
            avg_states_per_game = 40 # Estimate for Hive
            needed_states = min_fill_for_bootstrap - len(buf)
            needed_games = (needed_states + avg_states_per_game -1) // avg_states_per_game
            games_to_play_bootstrap = max(games_to_play_bootstrap, needed_games)
            print(f"Buffer below min fill ({len(buf)}/{min_fill_for_bootstrap}). Playing {games_to_play_bootstrap} bootstrap games.")
        elif args_global.debug_single_loop:
            print(f"[run DEBUG Hive] Bootstrap: Will play {games_to_play_bootstrap} game(s)")
        
        if games_to_play_bootstrap > 0:
            if not args_global.debug_single_loop: print(f"Bootstrapping {games_to_play_bootstrap} games …", flush=True)
            for g in range(games_to_play_bootstrap):
                net.eval() 
                game_hist = play_one_game(
                    net, game_adapter, mcts, temp_schedule, 
                    mcts_simulations=args_global.mcts_simulations, max_moves=MAX_HIVE_MOVES,
                    debug_mode=args_global.debug_single_loop )
                
                if isinstance(buf, PrioritizedReplayBuffer): 
                    for exp in game_hist: buf.add(exp)
                else: buf.extend(game_hist)
                net.train() 
                if args_global.debug_single_loop or (g+1) % args_global.save_buffer_every == 0:
                    print(f"  Bootstrap game {g+1}/{games_to_play_bootstrap} ({len(game_hist)} states) → buffer {len(buf)}", flush=True)
                    if not args_global.debug_single_loop : save_buffer_experiences(buf, buffer_file_path)

    if not args_global.debug_single_loop: print(f"Starting training from epoch {start_ep} for {args_global.epochs} epochs.")
    try:
        for ep in range(start_ep, args_global.epochs + 1):
            if args_global.debug_single_loop: print(f"\n--- Epoch {ep} START (Hive) ---")
            net.eval() 
            game_hist = play_one_game(
                net, game_adapter, mcts, temp_schedule, 
                mcts_simulations=args_global.mcts_simulations, max_moves=MAX_HIVE_MOVES, 
                debug_mode=args_global.debug_single_loop )
            
            if isinstance(buf, PrioritizedReplayBuffer): 
                for exp in game_hist: buf.add(exp)
            else: buf.extend(game_hist)
            net.train() 

            current_min_train_fill = args_global.min_buffer_fill_for_per_training if isinstance(buf, PrioritizedReplayBuffer) else args_global.min_buffer_fill_standard
            if args_global.debug_single_loop: current_min_train_fill = 1

            if len(buf) < current_min_train_fill:
                # ... (skip training print logic) ...
                if ep % args_global.log_every == 0: print(f"Epoch {ep} | Buffer {len(buf)} < min {current_min_train_fill}. Skip train. LR {opt.param_groups[0]['lr']:.2e}", flush=True)
            else:
                # ... (sampling, train_step call, PER update, logging as in C4/TTT) ...
                actual_batch_size = min(args_global.batch_size, len(buf))
                if actual_batch_size == 0: # Should be caught by min_buffer_fill generally
                    if ep % args_global.log_every == 0: print(f"Epoch {ep} | Buffer empty or batch zero. Skip train.")
                    if scheduler: scheduler.step()
                    continue
                
                batch_experiences, is_weights, data_indices = None, None, None
                if isinstance(buf, PrioritizedReplayBuffer):
                    sampled_data = buf.sample(actual_batch_size)
                    if sampled_data is None: # Not enough samples or zero sum in PER
                        if ep % args_global.log_every == 0: print(f"Epoch {ep} | PER sample failed. Skip train.")
                        if scheduler: scheduler.step()
                        continue
                    batch_experiences, is_weights, data_indices = sampled_data
                else:
                    batch_experiences = random.sample(list(buf), actual_batch_size)
                    is_weights = np.ones(len(batch_experiences), dtype=np.float32)

                loss, td_errors = train_step(net, batch_experiences, is_weights, opt, dev, game_adapter, 
                                             args_global.augment_prob, args_global.ent_beta, 
                                             debug_mode=args_global.debug_single_loop)
                if isinstance(buf, PrioritizedReplayBuffer) and data_indices is not None and len(data_indices)>0:
                    buf.update_priorities(data_indices, td_errors)
                if ep % args_global.log_every == 0:
                    print(f"Epoch {ep} | Loss {loss:.4f} | Buffer {len(buf)} | LR {opt.param_groups[0]['lr']:.2e}", flush=True)
            
            if scheduler: scheduler.step()
            # ... (checkpointing, buffer saving, PER beta anneal as in C4/TTT, adjusting paths/names for Hive) ...
            if ep % args_global.ckpt_every == 0:
                chkpt_path = ckdir / f"hive_chkpt_ep{ep:06d}.pt"
                if not args_global.debug_single_loop: torch.save(net.state_dict(), chkpt_path)
                full_state_to_save = {"net": net.state_dict(), "opt": opt.state_dict(), "epoch": ep, "scheduler": scheduler.state_dict() if scheduler else None,}
                if not args_global.debug_single_loop: torch.save(full_state_to_save, state_path)
                print(f"Saved: {str(chkpt_path)} and {str(state_path)}", flush=True)
            if ep % args_global.save_buffer_every == 0 and len(buf) > 0 and not args_global.debug_single_loop:
                 save_buffer_experiences(buf, buffer_file_path)
                 print(f"Saved replay buffer at epoch {ep}")
            if isinstance(buf, PrioritizedReplayBuffer): buf.advance_epoch_for_beta_anneal()
            if args_global.debug_single_loop: print(f"--- Epoch {ep} END (Hive) ---")

    except KeyboardInterrupt: print("\nTraining interrupted.")
    finally:
        # ... (final save logic as in C4/TTT, adjusting paths/names for Hive) ...
        print("Saving final model and buffer (Hive)...")
        final_model_path = ckdir / "last_hive_model.pt" 
        torch.save(net.state_dict(), final_model_path)
        if len(buf) > 0: save_buffer_experiences(buf, buffer_file_path)
        ep_to_save = ep if 'ep' in locals() and 'start_ep' in locals() and ep >= start_ep else start_ep - 1
        full_state_to_save = {"net": net.state_dict(), "opt": opt.state_dict(), "epoch": ep_to_save, "scheduler": scheduler.state_dict() if scheduler else None, }
        torch.save(full_state_to_save, state_path)
        print(f"Final Hive model: {str(final_model_path)}, Buffer: {str(buffer_file_path)}, State: {str(state_path)}")

# ---------------------------------------------------------------------------
# CLI parser (adapted for Hive)
# ---------------------------------------------------------------------------
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AlphaZero-style training for Hive (Pocket Edition).")
    # Keep most args from c4_zero_advanced, adjust defaults for Hive
    p.add_argument("--debug-single-loop", action="store_true", help="Minimal loop for debugging.")
    p.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    
    g_play = p.add_argument_group("Self-Play & MCTS (Hive)")
    g_play.add_argument("--bootstrap-games", type=int, default=20, help="Initial games.")
    g_play.add_argument("--skip-bootstrap", action="store_true", help="Skip bootstrap.")
    g_play.add_argument("--mcts-simulations", type=int, default=100, help="MCTS simulations per move.") # Higher default for Hive
    g_play.add_argument("--max-game-moves", type=int, default=150, help="Max moves per game before forced draw.")
    g_play.add_argument("--c-puct", type=float, default=1.5, help="PUCT constant (higher for larger action spaces often).")
    g_play.add_argument("--dirichlet-alpha", type=float, default=0.1, help="Dirichlet noise alpha (smaller for larger policy). ") # Adjusted
    g_play.add_argument("--dirichlet-epsilon", type=float, default=0.25, help="Dirichlet noise epsilon.")
    g_play.add_argument("--temp-decay-moves", type=int, default=15, help="Moves to use T=1 for exploration.") # Adjusted for Hive
    g_play.add_argument("--final-temp", type=float, default=0.05, help="Temperature after decay.")

    g_train = p.add_argument_group("Training (Hive)")
    g_train.add_argument("--epochs", type=int, default=20000, help="Total training epochs.")
    g_train.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.") # Might need tuning for Hive
    g_train.add_argument("--batch-size", type=int, default=128, help="Batch size.") # Smaller for larger network/data
    g_train.add_argument("--ent-beta", type=float, default=1e-3, help="Entropy regularization.")
    g_train.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    g_train.add_argument("--lr-t-max", type=int, default=0, help="T_max for CosineLR (0 for args.epochs).")
    g_train.add_argument("--lr-eta-min", type=float, default=1e-7, help="Min LR for CosineLR.")
    g_train.add_argument("--augment-prob", type=float, default=0.0, help="Augmentation (curr. no-op for Hive).") # Default 0 as it's no-op

    g_buffer = p.add_argument_group("Replay Buffer (Hive)")
    g_buffer.add_argument("--buffer-size", type=int, default=100000, help="Max replay buffer size.") # Larger for Hive
    g_buffer.add_argument("--min-buffer-fill-standard", type=int, default=2000, help="Min samples for standard buffer.")
    g_buffer.add_argument("--min-buffer-fill-for-per-training", type=int, default=20000, help="Min samples for PER training.")
    g_buffer.add_argument("--min-buffer-fill-for-per-bootstrap", type=int, default=2000, help="Min PER samples for bootstrap.")
    g_buffer.add_argument("--buffer-path", type=str, default="hive_adv_mcts_buffer.pth", help="Path for buffer.")
    g_buffer.add_argument("--save-buffer-every", type=int, default=100, help="Save buffer every N epochs.")
    g_buffer.add_argument("--use-per", action="store_true", help="Use Prioritized Replay.")
    g_buffer.add_argument("--per-alpha", type=float, default=0.6, help="Alpha for PER.")
    g_buffer.add_argument("--per-beta-start", type=float, default=0.4, help="Initial beta for PER.")
    g_buffer.add_argument("--per-beta-epochs", type=int, default=0, help="Epochs to anneal beta.")
    g_buffer.add_argument("--per-epsilon", type=float, default=1e-5, help="Epsilon for PER priorities.")

    g_nn = p.add_argument_group("Neural Network (Hive)")
    g_nn.add_argument("--channels", type=int, default=128, help="Channels per conv layer.") # Starting point
    g_nn.add_argument("--blocks", type=int, default=10, help="Number of residual blocks.") # Starting point

    g_mgmt = p.add_argument_group("Checkpointing & Logging (Hive)")
    g_mgmt.add_argument("--ckpt-dir", default="hive_checkpoints_az", help="Checkpoint directory.")
    g_mgmt.add_argument("--ckpt-every", type=int, default=100, help="Save checkpoint every N epochs.")
    g_mgmt.add_argument("--log-every", type=int, default=10, help="Log training stats every N epochs.")
    g_mgmt.add_argument("--resume-weights", metavar="PATH", help="Path to load network weights.")
    g_mgmt.add_argument("--resume-full-state", action="store_true", help="Resume full training state.")
    
    return p

if __name__ == "__main__":
    run() 