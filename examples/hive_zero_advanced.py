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
AXIAL_TO_IDX_HIVE: Dict[Tuple[int, int, str], int] = {}
FLAT_ACTION_REPRESENTATION_HIVE: List[Tuple[int, int, str]] = []
for q_coord in range(-BOARD_R, BOARD_R + 1):
    for r_coord in range(-BOARD_R, BOARD_R + 1):
        # Further filter to valid hexes in a hexagonal region if needed, though square is fine for tensor
        for piece_char in PIECE_TYPES:
            AXIAL_TO_IDX_HIVE[(q_coord, r_coord, piece_char)] = len(FLAT_ACTION_REPRESENTATION_HIVE)
            FLAT_ACTION_REPRESENTATION_HIVE.append((q_coord, r_coord, piece_char))

ACTION_SIZE = len(FLAT_ACTION_REPRESENTATION_HIVE) # e.g., 13*13*5 = 845 for a 13x13 grid if all hexes considered
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
        self.action_size = ACTION_SIZE # Global ACTION_SIZE = 845
        # Ensure PIECE_TYPES, AXIAL_TO_IDX_HIVE, FLAT_ACTION_REPRESENTATION_HIVE are accessible
        # (they are global in this file)

    def _hive_action_to_int(self, hive_action: Tuple) -> int | None:
        """Converts a native HiveGame action to our flat integer index."""
        action_type = hive_action[0]
        idx = None
        if action_type == "PLACE":
            # e.g., ("PLACE", "Q1", (0,0)) -> piece_char 'Q', (q,r) = (0,0)
            _, piece_type_full, (q, r) = hive_action
            piece_char = piece_type_full[0]
            idx = AXIAL_TO_IDX_HIVE.get((q, r, piece_char))
        elif action_type == "MOVE":
            # e.g., ("MOVE", (0,0), (0,1)) -> (q,r) = (0,1) (destination)
            # The original hive_zero.py used PIECE_TYPES[0] (i.e., 'Q') as a placeholder piece type
            # for MOVE actions when mapping to the flat index. This means the policy for 
            # (dest_q, dest_r, 'Q') covers all moves to (dest_q, dest_r).
            _, _, (q, r) = hive_action # Destination coordinates
            # We need to decide which piece_char to use for the lookup. 
            # The original hive_zero.py used PIECE_TYPES[0] ('Q') for all moves to a destination.
            # This is a simplification/ambiguity. Let's stick to it for now for consistency.
            # A better approach might involve a more complex action space if this fails.
            idx = AXIAL_TO_IDX_HIVE.get((q, r, PIECE_TYPES[0])) # Use 'Q' as placeholder type for move destination
        elif action_type == "PASS":
            # Pass actions don't map to this grid-based flat action space.
            # They need special handling. For now, MCTS might not see PASS via this mapping.
            # AlphaZero MCTS typically requires all actions to map to the policy vector.
            # This will be an issue. The flat action space needs to include PASS, or PASS is handled outside.
            # For now, let's return a dedicated index for PASS if we add one, or None.
            # The original hive_zero.py's mask_illegal returns pri*0 if only PASS is legal,
            # and flat_to_action defaults to random choice if idx isn't found, which could pick PASS.
            # This interaction is tricky. Let's assume for now PASS actions are filtered out by MCTS if they don't map.
            # A proper solution is to add a dedicated PASS action to ACTION_SIZE if HiveGame can return it.
            # For now, we'll ignore PASS in this mapping, MCTS will only get PLACE/MOVE actions.
            # This matches original hive_zero.py's mask_illegal not assigning an index to PASS.
            pass 
        return idx

    def _int_to_hive_action(self, int_action: int, legal_hive_actions: List[Tuple]) -> Tuple | None:
        """
        Converts a flat integer index back to a native HiveGame action.
        Must select from the provided legal_hive_actions.
        """
        if int_action < 0 or int_action >= ACTION_SIZE:
            # Fallback if int_action is out of bounds (e.g. from a buggy policy)
            return random.choice(legal_hive_actions) if legal_hive_actions else None

        # Get the (q, r, piece_char_for_flat_idx) corresponding to the flat int_action
        q_flat, r_flat, piece_char_flat = FLAT_ACTION_REPRESENTATION_HIVE[int_action]

        # Try to find a legal PLACE action that matches
        for legal_act in legal_hive_actions:
            if legal_act[0] == "PLACE":
                _, piece_type_full, (q_game, r_game) = legal_act
                if (q_game, r_game) == (q_flat, r_flat) and piece_type_full[0] == piece_char_flat:
                    return legal_act
        
        # If no PLACE action matched, try to find a MOVE action whose DESTINATION matches (q_flat, r_flat).
        # The piece_char_flat is ignored for selecting a move, as per original hive_zero logic.
        # This means the policy for (q,r,'Q'), (q,r,'A') etc. all map to moving *any* piece to (q,r).
        # We must pick *a* legal move that goes to (q_flat, r_flat).
        # If multiple pieces can move to the same (q_flat, r_flat), this is ambiguous.
        # The original flat_to_action would just return *any* legal move to that destination.
        possible_moves_to_dest = []
        for legal_act in legal_hive_actions:
            if legal_act[0] == "MOVE":
                _, _, (q_dest, r_dest) = legal_act
                if (q_dest, r_dest) == (q_flat, r_flat):
                    possible_moves_to_dest.append(legal_act)
        if possible_moves_to_dest:
            return random.choice(possible_moves_to_dest) # Pick one if multiple match destination

        # If no PLACE or MOVE action matches the flat int_action (e.g., flat action is illegal)
        # or if only PASS is available.
        if legal_hive_actions: 
            # Check if only PASS is legal
            if len(legal_hive_actions) == 1 and legal_hive_actions[0][0] == "PASS":
                return legal_hive_actions[0]
            return random.choice(legal_hive_actions) # Fallback to a random legal move
        return None # Should not happen if legal_hive_actions has items

    def getCurrentPlayer(self, state: Dict) -> str:
        return self.hive_game.getCurrentPlayer(state)

    def getLegalActions(self, state: Dict) -> List[int]:
        native_actions = self.hive_game.getLegalActions(state)
        int_actions = []
        seen_indices = set() # To handle multiple native moves mapping to same flat idx for MOVEs
        for native_act in native_actions:
            if native_act[0] == "PASS": # How to handle PASS? Policy vector doesn't have a slot for it.
                # Option 1: Add a dedicated index for PASS (e.g., ACTION_SIZE itself).
                # Option 2: MCTS handles it if all other actions have 0 probability.
                # Option 3: Ignore PASS for now, assuming it's rare or game ends before forced pass.
                # Current flat action space does not account for PASS. Let's filter it out.
                # This means the NN won't learn to pass. This is a limitation.
                continue 
            idx = self._hive_action_to_int(native_act)
            if idx is not None and idx not in seen_indices:
                int_actions.append(idx)
                seen_indices.add(idx) # Only add unique flat indices
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