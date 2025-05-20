#!/usr/bin/env python3
"""Advanced self-play training loop for Tic-Tac-Toe.

Adapts AlphaZero-style techniques for Tic-Tac-Toe.
"""

from __future__ import annotations
import argparse
import random
from collections import deque
from pathlib import Path
from typing import List, Tuple, Dict, Any 

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

if torch: # Print GPU info only if torch is available
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

from simple_games.tic_tac_toe import TicTacToe # Changed import
from mcts.alpha_zero_mcts import AlphaZeroMCTS
from mcts.replay_buffer import PrioritizedReplayBuffer

BOARD_SIZE = 3 # For Tic-Tac-Toe
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE # 9 actions

# ---------------------------------------------------------------------------
# State encoding for Tic-Tac-Toe
# ---------------------------------------------------------------------------
def encode_ttt_state(state: dict, perspective: str) -> torch.Tensor:
    """Return a 3x3x3 tensor representing TTT `state` from `perspective`."""
    if torch is None:
        raise RuntimeError("PyTorch is required for encode_ttt_state")
    # Channels: 0=player pieces, 1=opponent pieces, 2=player turn (all 1s if current turn)
    t = torch.zeros(3, BOARD_SIZE, BOARD_SIZE) 
    opponent = 'O' if perspective == 'X' else 'X'
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece = state["board"][r][c]
            if piece == perspective:
                t[0, r, c] = 1.0
            elif piece == opponent:
                t[1, r, c] = 1.0
    if state["current_player"] == perspective:
        t[2].fill_(1.0)
    return t

# ---------------------------------------------------------------------------
# Data Augmentation for Tic-Tac-Toe
# ---------------------------------------------------------------------------
def reflect_ttt_state_policy(state_dict: Dict, policy_vector: np.ndarray) -> Tuple[Dict, np.ndarray]:
    """Reflects a Tic-Tac-Toe board state and its policy vector horizontally."""
    if np is None:
        raise RuntimeError("NumPy is required for reflect_ttt_state_policy")

    reflected_board = [row[::-1] for row in state_dict["board"]]
    reflected_state_dict = {
        "board": reflected_board,
        "current_player": state_dict["current_player"]
    }

    reflected_policy_vector = np.zeros_like(policy_vector) # policy_vector is size 9
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            original_idx = r * BOARD_SIZE + c
            reflected_c = BOARD_SIZE - 1 - c
            reflected_idx = r * BOARD_SIZE + reflected_c
            reflected_policy_vector[reflected_idx] = policy_vector[original_idx]
    
    return reflected_state_dict, reflected_policy_vector

# ---------------------------------------------------------------------------
# Game Adapter for Tic-Tac-Toe MCTS
# ---------------------------------------------------------------------------
class TicTacToeAdapter:
    def __init__(self, ttt_game: TicTacToe):
        self.ttt_game = ttt_game
        self.action_size = self.ttt_game.get_action_size() # Should be 9

    def _action_to_int(self, rc_action: Tuple[int, int]) -> int:
        r, c = rc_action
        return r * BOARD_SIZE + c

    def _int_to_action(self, int_action: int) -> Tuple[int, int]:
        r = int_action // BOARD_SIZE
        c = int_action % BOARD_SIZE
        return (r, c)

    def getCurrentPlayer(self, state: Dict) -> str:
        return self.ttt_game.getCurrentPlayer(state)

    def getLegalActions(self, state: Dict) -> List[int]: # Returns list of int actions
        rc_actions = self.ttt_game.getLegalActions(state)
        return [self._action_to_int(action) for action in rc_actions]

    def applyAction(self, state: Dict, int_action: int) -> Dict: # Takes int action
        rc_action = self._int_to_action(int_action)
        return self.ttt_game.applyAction(state, rc_action)

    def isTerminal(self, state: Dict) -> bool:
        return self.ttt_game.isTerminal(state)

    def getGameOutcome(self, state: Dict) -> str: 
        return self.ttt_game.getGameOutcome(state)

    def encode_state(self, state: Dict, player_perspective: str) -> torch.Tensor:
        return encode_ttt_state(state, player_perspective)

    def copyState(self, state: Dict) -> Dict:
        return self.ttt_game.copyState(state)
    
    def get_action_size(self) -> int:
        return self.action_size


# ---------------------------------------------------------------------------
# Neural network for Tic-Tac-Toe (Simpler)
# ---------------------------------------------------------------------------
if torch is not None:
    class ResidualBlockTTT(nn.Module): # Simplified
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

    class AdvancedTTTZeroNet(nn.Module):
        def __init__(self, ch: int = 32, blocks: int = 2): # Reduced defaults
            super().__init__()
            # Input is 3x3x3 for TTT
            self.stem = nn.Sequential(
                nn.Conv2d(3, ch, kernel_size=3, padding=1, bias=False), # 3 input channels
                nn.BatchNorm2d(ch), 
                nn.ReLU(),
            )
            self.res = nn.Sequential(*[ResidualBlockTTT(ch) for _ in range(blocks)])
            
            # Policy head
            # After stem and res, if input is 3x3, output is ch x 3 x 3
            self.policy_conv = nn.Conv2d(ch, 2, kernel_size=1) # 2 intermediate policy channels
            self.policy_bn = nn.BatchNorm2d(2)
            self.policy_flatten = nn.Flatten()
            self.policy_linear = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, ACTION_SIZE) # Output 9 for TTT

            # Value head
            self.value_conv = nn.Conv2d(ch, 1, kernel_size=1) # 1 intermediate value channel
            self.value_bn = nn.BatchNorm2d(1)
            self.value_flatten = nn.Flatten()
            # Input to linear: 1 * BOARD_SIZE * BOARD_SIZE
            self.value_linear1 = nn.Linear(1 * BOARD_SIZE * BOARD_SIZE, 32) # Smaller hidden layer
            self.value_linear2 = nn.Linear(32, 1)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            x_stem = self.stem(x)
            x_res = self.res(x_stem)
            
            # Policy path
            p = self.policy_conv(x_res)
            p = self.policy_bn(p)
            p = F.relu(p)
            p = self.policy_flatten(p)
            policy_logits = self.policy_linear(p)
            
            # Value path
            v = self.value_conv(x_res)
            v = self.value_bn(v)
            v = F.relu(v)
            v = self.value_flatten(v)
            v = F.relu(self.value_linear1(v))
            value = torch.tanh(self.value_linear2(v)) # Tanh for value in [-1, 1]
            
            return policy_logits, value.squeeze(1)
else: 
    class ResidualBlockTTT: pass
    class AdvancedTTTZeroNet: pass


# ---------------------------------------------------------------------------
# Self-play helpers
# ---------------------------------------------------------------------------
BUFFER_PATH = Path("ttt_adv_mcts_buffer.pth") # Changed default path

def play_one_game(
    net: AdvancedTTTZeroNet, 
    game_adapter: TicTacToeAdapter, 
    mcts_instance: AlphaZeroMCTS,     
    temp_schedule: List[Tuple[int, float]], 
    mcts_simulations: int,
    max_moves: int = ACTION_SIZE,
    debug_mode: bool = False
) -> List[Tuple[dict, np.ndarray, int]]:
    if torch is None or np is None: raise RuntimeError("PyTorch and NumPy are required")
    st = game_adapter.ttt_game.getInitialState() 
    hist: List[Tuple[dict, np.ndarray, int]] = [] 
    move_no = 0
    current_temp = 1.0

    if debug_mode: print("\n--- play_one_game START ---")

    while not game_adapter.isTerminal(st) and move_no < max_moves:
        if debug_mode:
            print(f"\n[play_one_game] Move: {move_no}")
            board_str = "\n".join([str(row) for row in st["board"]])
            print(f"[play_one_game] Current state (player {st['current_player']}):\n{board_str}")

        for threshold_moves, temp_val in temp_schedule:
            if move_no < threshold_moves: current_temp = temp_val; break
        
        player_perspective = game_adapter.getCurrentPlayer(st)
        if debug_mode: print(f"[play_one_game] MCTS simulations: {mcts_simulations}, Temp: {current_temp}, Perspective: {player_perspective}")
        
        chosen_action_int, mcts_policy_dict = mcts_instance.get_action_policy(
            root_state=st, num_simulations=mcts_simulations, temperature=current_temp,
            debug_mcts=debug_mode 
        )
        if debug_mode: print(f"[play_one_game] MCTS chosen_action_int: {chosen_action_int}")
        if debug_mode: print(f"[play_one_game] MCTS policy_dict: {mcts_policy_dict}")
        
        policy_target_vector = np.zeros(game_adapter.get_action_size(), dtype=np.float32)
        if mcts_policy_dict: 
            for action_idx_int, prob in mcts_policy_dict.items():
                if 0 <= action_idx_int < game_adapter.get_action_size(): policy_target_vector[action_idx_int] = prob
        
        current_sum = policy_target_vector.sum()
        if abs(current_sum - 1.0) > 1e-5 and current_sum > 1e-5:
             policy_target_vector /= current_sum
        elif current_sum < 1e-5:
            if not game_adapter.isTerminal(st):
                legal_actions = game_adapter.getLegalActions(st)
                if legal_actions: 
                    uniform_prob = 1.0 / len(legal_actions)
                    for la_int in legal_actions: policy_target_vector[la_int] = uniform_prob
        if debug_mode: print(f"[play_one_game] Policy target vector for history (sum={policy_target_vector.sum()}): {policy_target_vector}")

        current_state_copy = game_adapter.copyState(st)
        hist.append((current_state_copy, policy_target_vector, 0))
        if debug_mode: print(f"[play_one_game] Appended to history. State player: {current_state_copy['current_player']}")
        
        st = game_adapter.applyAction(st, chosen_action_int)
        move_no += 1

    winner = game_adapter.getGameOutcome(st)
    z = 0
    if winner == "Draw": z = 0
    elif winner == "X": z = 1
    elif winner == "O": z = -1
    if debug_mode:
        print(f"\n[play_one_game] Game Over. Winner: {winner}, z (X's perspective): {z}")
        board_str_final = "\n".join([str(row) for row in st["board"]])
        print(f"[play_one_game] Final board state (player {st['current_player']} to move if not terminal):\n{board_str_final}")

    final_history = []
    for idx, (recorded_state, policy, _) in enumerate(hist):
        player_at_state = game_adapter.getCurrentPlayer(recorded_state)
        value_for_state_player = z if player_at_state == "X" else -z
        final_history.append((recorded_state, policy, value_for_state_player))
        if debug_mode: print(f"[play_one_game] final_history item {idx}: Player_at_state: {player_at_state}, Value_for_player: {value_for_state_player}, Policy sum: {np.sum(policy) if policy is not None else 'N/A'}")
    
    if debug_mode: print("--- play_one_game END ---")
    return final_history


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_step(net: AdvancedTTTZeroNet, batch_experiences: list, is_weights: np.ndarray, 
               opt: torch.optim.Optimizer, dev: str, game_adapter: TicTacToeAdapter, 
               augment_prob: float, ent_beta_val: float, debug_mode: bool = False) -> Tuple[float, np.ndarray]:
    if debug_mode: print("\n--- train_step START ---")
    if debug_mode: print(f"[train_step] Batch size: {len(batch_experiences)}, Augment prob: {augment_prob}, Ent_beta: {ent_beta_val}")

    S_list, P_tgt_list, V_tgt_list = [], [], []
    if not batch_experiences: # Handle empty batch case
        if debug_mode: print("[train_step] Batch is empty, skipping processing.")
        return 0.0, np.array([])

    for i, (s_dict_orig, p_tgt_orig, v_tgt_orig) in enumerate(batch_experiences):
        s_to_enc, p_to_use = s_dict_orig, p_tgt_orig
        augmented_this_sample = False
        if random.random() < augment_prob:
            s_to_enc, p_to_use = reflect_ttt_state_policy(s_dict_orig, p_tgt_orig)
            augmented_this_sample = True
        
        player_persp = game_adapter.getCurrentPlayer(s_to_enc)
        S_list.append(game_adapter.encode_state(s_to_enc, player_persp))
        P_tgt_list.append(p_to_use)
        V_tgt_list.append(v_tgt_orig)
        if debug_mode and i < 2: 
            board_str_orig = "\n".join([str(row) for row in s_dict_orig["board"]])
            board_str_enc = "\n".join([str(row) for row in s_to_enc["board"]])
            print(f"  [train_step] Sample {i} Original State (player {s_dict_orig['current_player']}):\n{board_str_orig}")
            print(f"  [train_step] Sample {i} Original Policy: {p_tgt_orig}")
            print(f"  [train_step] Sample {i} Augmented: {augmented_this_sample}")
            print(f"  [train_step] Sample {i} To Encode State (player {s_to_enc['current_player']}):\n{board_str_enc}")
            print(f"  [train_step] Sample {i} To Use Policy: {p_to_use}")
            print(f"  [train_step] Sample {i} Value Target (for this state player): {v_tgt_orig}")

    S = torch.stack(S_list).to(dev)
    P_tgt = torch.tensor(np.array(P_tgt_list), dtype=torch.float32, device=dev)
    V_tgt = torch.tensor(V_tgt_list, dtype=torch.float32, device=dev)
    is_weights_tensor = torch.tensor(is_weights, dtype=torch.float32, device=dev).unsqueeze(1)

    if debug_mode:
        print(f"[train_step] S shape: {S.shape}, P_tgt shape: {P_tgt.shape}, V_tgt shape: {V_tgt.shape}")
        if S.numel() > 0:
            s_example_str = "\n".join([str(plane.cpu().numpy()) for plane in S[0]])
            print(f"[train_step] S example (first item):\n{s_example_str}")
        else: print("[train_step] S is empty.")
        print(f"[train_step] P_tgt example: {P_tgt[0].cpu().numpy() if P_tgt.numel() > 0 else '[]'}")
        print(f"[train_step] V_tgt example: {V_tgt[0].item() if V_tgt.numel() > 0 else '[]'}")
        print(f"[train_step] IS weights example: {is_weights_tensor.squeeze()[:min(4, len(is_weights_tensor))].cpu().numpy() if is_weights_tensor.numel() > 0 else '[]'}")

    logits, V_pred = net(S)
    if debug_mode:
        print(f"[train_step] Logits shape: {logits.shape}, V_pred shape: {V_pred.shape}")
        print(f"[train_step] Logits example: {logits[0].cpu().detach().numpy() if logits.numel() > 0 else '[]'}")
        print(f"[train_step] V_pred example (squeezed): {V_pred.squeeze()[0].item() if V_pred.numel() > 0 else '[]'}")

    logP_pred = F.log_softmax(logits, dim=1)
    loss_p_per_sample = F.kl_div(logP_pred, P_tgt, reduction="none").sum(dim=1)
    loss_v_per_sample = F.mse_loss(V_pred.squeeze(), V_tgt, reduction="none")
    
    weighted_loss_p = (loss_p_per_sample * is_weights_tensor.squeeze()).mean()
    weighted_loss_v = (loss_v_per_sample * is_weights_tensor.squeeze()).mean()
    
    P_pred_dist = torch.exp(logP_pred)
    entropy = -(P_pred_dist * logP_pred).sum(dim=1).mean()
    total_loss = weighted_loss_p + weighted_loss_v - ent_beta_val * entropy

    if debug_mode:
        print(f"[train_step] loss_p_per_sample (example): {loss_p_per_sample[0].item() if loss_p_per_sample.numel() > 0 else 'N/A'}")
        print(f"[train_step] loss_v_per_sample (example): {loss_v_per_sample[0].item() if loss_v_per_sample.numel() > 0 else 'N/A'}")
        print(f"[train_step] weighted_loss_p: {weighted_loss_p.item()}, weighted_loss_v: {weighted_loss_v.item()}")
        print(f"[train_step] entropy: {entropy.item()}")
        print(f"[train_step] TOTAL LOSS: {total_loss.item()}")

    opt.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    
    td_errors = np.abs(V_tgt.cpu().detach().numpy() - V_pred.squeeze().cpu().detach().numpy())
    if debug_mode: print(f"[train_step] TD errors (example): {td_errors[:min(4, len(td_errors))] if len(td_errors)>0 else '[]'}")
    if debug_mode: print("--- train_step END ---")
    return float(total_loss.item()), td_errors

def save_buffer_experiences(buf: PrioritizedReplayBuffer | deque, path: Path) -> None:
    if torch is None: raise RuntimeError("PyTorch is required")
    data_to_save = list(buf.data_buffer[:len(buf)]) if isinstance(buf, PrioritizedReplayBuffer) else list(buf)
    torch.save(data_to_save, path)
    print(f"Saved {len(data_to_save)} experiences from buffer to {path}")

def load_experiences_to_buffer(target_buffer: PrioritizedReplayBuffer | deque, path: Path) -> None: # Removed maxlen
    if torch is None: raise RuntimeError("PyTorch is required")
    if not path.exists():
        print(f"Buffer file {path} not found. Starting empty.")
        return
    try:
        data = torch.load(path, weights_only=False) 
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
        target_buffer.extend(data) # Deque handles its own maxlen
        print(f"Loaded {len(data)} exp into deque. Size: {len(target_buffer)}")


args_global = None # To store parsed args

# --- Definition of evaluate_model_policy needs to be before run() ---
def evaluate_model_policy(model_path: str, game_adapter: TicTacToeAdapter, device_str: str, num_mcts_sims_for_eval: int = 100):
    """Loads a trained model and prints its policy for a few sample states."""
    if torch is None or np is None: raise RuntimeError("PyTorch/NumPy needed")
    print(f"\n--- Evaluating Model Policy from: {model_path} ---")
    device = torch.device(device_str)

    nn_ch = args_global.channels if args_global else 32
    nn_bl = args_global.blocks if args_global else 2
    net = AdvancedTTTZeroNet(ch=nn_ch, blocks=nn_bl).to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        if "net" in state_dict and isinstance(state_dict["net"], dict):
            net.load_state_dict(state_dict["net"])
            print(f"Loaded model weights from 'net' key in checkpoint: {model_path}")
        elif isinstance(state_dict, dict):
            net.load_state_dict(state_dict)
            print(f"Loaded model weights directly from checkpoint: {model_path}")
        else:
            raise ValueError("Checkpoint format not recognized.")
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return
    net.eval()

    def mcts_model_fn(encoded_state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad(): policy_logits, value_estimates = net(encoded_state_batch)
        return policy_logits, value_estimates.unsqueeze(-1)

    mcts_eval = AlphaZeroMCTS(
        game_interface=game_adapter, model_fn=mcts_model_fn, device=device,
        c_puct=args_global.c_puct if args_global else 1.0, 
        dirichlet_alpha=0, dirichlet_epsilon=0 
    )

    sample_states = [
        (game_adapter.ttt_game.getInitialState(), "Empty Board (X to play)"),
        ({"board": [['X',None,None],[None,None,None],[None,None,None]], "current_player": "O"}, "X in corner (0,0), O to play"),
        ({"board": [[None,None,None],[None,'X',None],[None,None,None]], "current_player": "O"}, "X in center (1,1), O to play"),
        ({"board": [['X','O','X'],['O','X',None],[None,None,'O']], "current_player": "X"}, "Complex state, X to play for win/draw"),
    ]

    for state_dict, desc in sample_states:
        print(f"\n--- Evaluating State: {desc} ---")
        current_player = game_adapter.getCurrentPlayer(state_dict)
        board_str = "\n".join([str(row) for row in state_dict["board"]])
        print(f"Board:\n{board_str}\nPlayer to move: {current_player}")

        encoded_s = game_adapter.encode_state(state_dict, current_player).unsqueeze(0).to(device)
        with torch.no_grad():
            raw_logits, raw_value = net(encoded_s)
        raw_policy_probs = F.softmax(raw_logits.squeeze(0), dim=0).cpu().numpy()
        print(f"  Direct Net Value: {raw_value.item():.3f}")
        print(f"  Direct Net Policy (probs for actions 0-8):\n  {[f'{p:.3f}' for p in raw_policy_probs]}")
        legal_actions_int = game_adapter.getLegalActions(state_dict)
        print(f"  Legal actions (int): {legal_actions_int}")
        print(f"  Direct Net Policy for Legal Actions:")
        for la in legal_actions_int:
            r,c = game_adapter._int_to_action(la)
            print(f"    Action ({r},{c}) [{la}]: {raw_policy_probs[la]:.3f}")

        if not game_adapter.isTerminal(state_dict):
            _, mcts_policy = mcts_eval.get_action_policy(state_dict, num_mcts_sims_for_eval, temperature=0.0)
            print(f"  MCTS ({num_mcts_sims_for_eval} sims) Policy (probs for actions 0-8):\n  {[f'{mcts_policy.get(i, 0.0):.3f}' for i in range(ACTION_SIZE)]}")
            print(f"  MCTS Policy for Legal Actions:")
            if mcts_policy: # Check if mcts_policy is not empty
                for la_int in sorted(mcts_policy.keys()):
                    r,c = game_adapter._int_to_action(la_int)
                    print(f"    Action ({r},{c}) [{la_int}]: {mcts_policy[la_int]:.3f}")
            else:
                print("    MCTS policy dictionary is empty.")
        else:
            print("  State is terminal, skipping MCTS.")
    print("\n--- Model Policy Evaluation END ---")
# --- End definition of evaluate_model_policy ---

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run(parsed_cli_args=None) -> None:
    global args_global 
    args_global = parsed_cli_args if parsed_cli_args is not None else parser().parse_args()

    if args_global.eval_model_path:
        if torch is None: raise RuntimeError("PyTorch needed for model evaluation.")
        # Ensure game_adapter is created before being passed to evaluate_model_policy
        ttt_game_eval = TicTacToe()
        game_adapter_eval = TicTacToeAdapter(ttt_game_eval)
        evaluate_model_policy(args_global.eval_model_path, game_adapter_eval, 
                              "cuda" if args_global.gpu and torch.cuda.is_available() else "cpu",
                              args_global.mcts_simulations)
        return

    if args_global.debug_single_loop:
        print("!!!!!!!!!!!!!!!!! DEBUG SINGLE LOOP MODE ENABLED !!!!!!!!!!!!!!!!!")
        args_global.epochs = 1
        args_global.bootstrap_games = 1 
        args_global.mcts_simulations = 5 
        args_global.batch_size = 4 
        args_global.min_buffer_fill_standard = 1 
        args_global.min_buffer_fill_for_per_training = 1
        args_global.min_buffer_fill_for_per_bootstrap = 1
        args_global.log_every = 1
        args_global.ckpt_every = 1 
        args_global.save_buffer_every = 1
        args_global.skip_bootstrap = False # Ensure at least one game is played for the buffer

    if args_global.use_per and args_global.per_beta_epochs <= 0:
        args_global.per_beta_epochs = args_global.epochs
        if not args_global.debug_single_loop: 
            print(f"PER: Beta annealing epochs set to total epochs: {args_global.per_beta_epochs}")

    if torch is None or np is None: raise RuntimeError("PyTorch and NumPy are required")
    dev = "cuda" if args_global.gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev}")

    ckdir = Path(args_global.ckpt_dir)
    ckdir.mkdir(exist_ok=True, parents=True)
    
    ttt_game_instance = TicTacToe()
    game_adapter = TicTacToeAdapter(ttt_game_instance)

    net = AdvancedTTTZeroNet(ch=args_global.channels, blocks=args_global.blocks).to(dev)
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

    state_path = ckdir / "train_state_ttt.pt"
    start_ep = 1
    if args_global.resume_full_state and state_path.exists() and not args_global.debug_single_loop:
        print("Resuming full training state from", state_path)
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
        print("Resuming weights from", args_global.resume_weights)
        net.load_state_dict(torch.load(args_global.resume_weights, map_location=dev))

    buf: PrioritizedReplayBuffer | deque
    if args_global.use_per:
        # For debug_single_loop, ensure capacity is at least batch_size for PER to sample
        debug_capacity = args_global.batch_size if args_global.debug_single_loop else args_global.buffer_size
        if args_global.debug_single_loop and debug_capacity == 0: debug_capacity = 4 # Ensure PER can be created
        buf = PrioritizedReplayBuffer(
            capacity=debug_capacity,
            alpha=args_global.per_alpha,
            beta_start=args_global.per_beta_start, beta_epochs=args_global.per_beta_epochs,
            epsilon=args_global.per_epsilon )
        print(f"Using PrioritizedReplayBuffer (capacity: {buf.capacity}).")
    else:
        debug_maxlen = args_global.batch_size if args_global.debug_single_loop else args_global.buffer_size
        if args_global.debug_single_loop and debug_maxlen == 0: debug_maxlen = 4
        buf = deque(maxlen=debug_maxlen)
        print(f"Using standard deque replay buffer (maxlen: {buf.maxlen}).")

    buffer_file_path = Path(args_global.buffer_path)
    if not args_global.debug_single_loop:
        load_experiences_to_buffer(buf, buffer_file_path)
    elif args_global.debug_single_loop:
        print("[run DEBUG] Skipping buffer load for debug_single_loop.")

    temp_schedule = [
        (args_global.temp_decay_moves, 1.0), (float('inf'), args_global.final_temp)
    ]
    
    if args_global.debug_single_loop:
        print(f"[run DEBUG] Initialized. MCTS sims: {args_global.mcts_simulations}, Batch: {args_global.batch_size}, Epochs: {args_global.epochs}")
        print(f"[run DEBUG] Buffer type: {'PER' if args_global.use_per else 'Deque'}, Initial len(buf): {len(buf)}")
        print(f"[run DEBUG] Min fill bootstrap: {args_global.min_buffer_fill_for_per_bootstrap if args_global.use_per else args_global.min_buffer_fill_standard}")

    min_fill_for_bootstrap = args_global.min_buffer_fill_for_per_bootstrap if isinstance(buf, PrioritizedReplayBuffer) else args_global.min_buffer_fill_standard
    if args_global.debug_single_loop: min_fill_for_bootstrap = 1
    
    games_to_play_bootstrap = 0
    if not args_global.skip_bootstrap and len(buf) < min_fill_for_bootstrap:
        games_to_play_bootstrap = args_global.bootstrap_games 
        if len(buf) < min_fill_for_bootstrap and not args_global.debug_single_loop:
            avg_states_per_game = 5 
            needed_states = min_fill_for_bootstrap - len(buf)
            needed_games = (needed_states + avg_states_per_game -1) // avg_states_per_game
            games_to_play_bootstrap = max(games_to_play_bootstrap, needed_games)
            print(f"Buffer below min fill ({len(buf)}/{min_fill_for_bootstrap}). Playing {games_to_play_bootstrap} bootstrap games.")
        elif args_global.debug_single_loop:
            print(f"[run DEBUG] Bootstrap: Will play {games_to_play_bootstrap} game(s) as len(buf)={len(buf)} < min_fill={min_fill_for_bootstrap}")
        
        if games_to_play_bootstrap > 0:
            if not args_global.debug_single_loop: print(f"Bootstrapping {games_to_play_bootstrap} games …", flush=True)
            for g in range(games_to_play_bootstrap):
                net.eval() 
                game_hist = play_one_game(
                    net, game_adapter, mcts, temp_schedule, 
                    mcts_simulations=args_global.mcts_simulations, max_moves=ACTION_SIZE,
                    debug_mode=args_global.debug_single_loop )
                
                add_method = buf.add if isinstance(buf, PrioritizedReplayBuffer) else buf.extend
                if isinstance(buf, PrioritizedReplayBuffer): 
                    for exp in game_hist: add_method(exp)
                else: 
                    add_method(game_hist) # Deque uses extend
                net.train() 
                if args_global.debug_single_loop or (g+1) % args_global.save_buffer_every == 0:
                    print(f"  Bootstrap game {g+1}/{games_to_play_bootstrap} ({len(game_hist)} states) → buffer {len(buf)}", flush=True)
                    if not args_global.debug_single_loop : save_buffer_experiences(buf, buffer_file_path) # Avoid saving in debug loop unless needed

    if not args_global.debug_single_loop: print(f"Starting training from epoch {start_ep} for {args_global.epochs} epochs.")
    try:
        for ep in range(start_ep, args_global.epochs + 1):
            if args_global.debug_single_loop: print(f"\n--- Epoch {ep} START ---")
            net.eval() 
            game_hist = play_one_game(
                net, game_adapter, mcts, temp_schedule, 
                mcts_simulations=args_global.mcts_simulations, max_moves=ACTION_SIZE,
                debug_mode=args_global.debug_single_loop )
            
            add_method = buf.add if isinstance(buf, PrioritizedReplayBuffer) else buf.extend
            if isinstance(buf, PrioritizedReplayBuffer): 
                for exp in game_hist: add_method(exp)
            else: 
                add_method(game_hist)
            net.train() 

            current_min_train_fill = args_global.min_buffer_fill_for_per_training if isinstance(buf, PrioritizedReplayBuffer) else args_global.min_buffer_fill_standard
            if args_global.debug_single_loop: current_min_train_fill = 1

            if len(buf) < current_min_train_fill:
                if ep % args_global.log_every == 0: 
                    print(f"Epoch {ep} | Buffer {len(buf)} < min {current_min_train_fill}. Skip train. LR {opt.param_groups[0]['lr']:.2e}", flush=True)
                if ep % args_global.ckpt_every == 0 and not args_global.debug_single_loop: save_buffer_experiences(buf, buffer_file_path) 
            else:
                if args_global.debug_single_loop: print(f"\n[run DEBUG] Buffer ready for sampling. len(buf): {len(buf)}, batch_size: {args_global.batch_size}")
                
                batch_experiences, is_weights, data_indices = None, None, None
                actual_batch_size = min(args_global.batch_size, len(buf))

                if actual_batch_size == 0: # Cannot sample if buffer is empty or batch size is 0
                    if ep % args_global.log_every == 0: print(f"Epoch {ep} | Buffer empty or batch size zero. Skipping training step.", flush=True)
                    if scheduler: scheduler.step()
                    continue

                if isinstance(buf, PrioritizedReplayBuffer):
                    sampled_data = buf.sample(actual_batch_size)
                    if sampled_data is None:
                        if ep % args_global.log_every == 0: print(f"Epoch {ep} | PER sample failed. Skip train. LR {opt.param_groups[0]['lr']:.2e}", flush=True)
                        if scheduler: scheduler.step() 
                        continue 
                    batch_experiences, is_weights, data_indices = sampled_data
                    if args_global.debug_single_loop and sampled_data:
                        print(f"[run DEBUG] PER Sampled {len(batch_experiences)} experiences.")
                        if batch_experiences and len(batch_experiences) > 0: print(f"[run DEBUG] First PER exp state board: {batch_experiences[0][0]['board']}")
                        print(f"[run DEBUG] PER IS weights (first 4): {is_weights[:min(4, len(is_weights))] if len(is_weights) > 0 else '[]'}")
                        print(f"[run DEBUG] PER data_indices (first 4): {data_indices[:min(4, len(data_indices))] if len(data_indices) > 0 else '[]'}")
                else: 
                    batch_experiences = random.sample(list(buf), actual_batch_size)
                    is_weights = np.ones(len(batch_experiences), dtype=np.float32)
                    if args_global.debug_single_loop:
                        print(f"[run DEBUG] Standard Sampled {len(batch_experiences)} experiences.")
                        if batch_experiences and len(batch_experiences) > 0: print(f"[run DEBUG] First standard exp state board: {batch_experiences[0][0]['board']}")
                
                loss, td_errors = train_step(net, batch_experiences, is_weights, opt, dev, game_adapter, 
                                             args_global.augment_prob, args_global.ent_beta, 
                                             debug_mode=args_global.debug_single_loop)
                
                if isinstance(buf, PrioritizedReplayBuffer) and data_indices is not None and len(data_indices) > 0:
                    if args_global.debug_single_loop: print(f"[run DEBUG] Updating PER priorities for indices: {data_indices[:min(4, len(data_indices))]} with TD_errors: {td_errors[:min(4, len(td_errors))]}")
                    buf.update_priorities(data_indices, td_errors)

                if ep % args_global.log_every == 0:
                    print(f"Epoch {ep} | Loss {loss:.4f} | Buffer {len(buf)} | LR {opt.param_groups[0]['lr']:.2e}", flush=True)
            
            if scheduler: scheduler.step()
            
            if ep % args_global.ckpt_every == 0:
                chkpt_path = ckdir / f"ttt_chkpt_ep{ep:06d}.pt"
                if not args_global.debug_single_loop: torch.save(net.state_dict(), chkpt_path) # Avoid saving checkpoints in debug loop if not needed
                current_epoch_to_save = ep 
                full_state_to_save = {
                    "net": net.state_dict(), "opt": opt.state_dict(),
                    "epoch": current_epoch_to_save, 
                    "scheduler": scheduler.state_dict() if scheduler else None,
                }
                if not args_global.debug_single_loop: torch.save(full_state_to_save, state_path)
                print(f"Saved: {chkpt_path} and {state_path}", flush=True)

            if ep % args_global.save_buffer_every == 0 and len(buf) > 0 and not args_global.debug_single_loop:
                 save_buffer_experiences(buf, buffer_file_path)
                 print(f"Saved replay buffer at epoch {ep}")
            
            if isinstance(buf, PrioritizedReplayBuffer):
                buf.advance_epoch_for_beta_anneal()
            if args_global.debug_single_loop: print(f"--- Epoch {ep} END ---")
    except KeyboardInterrupt: print("\nTraining interrupted.") # Added newline for clarity
    finally:
        print("Saving final model and buffer...")
        final_model_path = ckdir / "last_ttt_model.pt" 
        torch.save(net.state_dict(), final_model_path)
        if len(buf) > 0: save_buffer_experiences(buf, buffer_file_path)
        
        # Ensure 'ep' is defined for final save, use start_ep-1 if loop didn't run
        ep_to_save = ep if 'ep' in locals() and 'start_ep' in locals() and ep >= start_ep else start_ep - 1
        full_state_to_save = {
            "net": net.state_dict(), "opt": opt.state_dict(),
            "epoch": ep_to_save,
            "scheduler": scheduler.state_dict() if scheduler else None,
        }
        torch.save(full_state_to_save, state_path)
        print(f"Final model: {final_model_path}, Buffer: {buffer_file_path}, State: {state_path}")

# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AlphaZero-style training for Tic-Tac-Toe.")
    p.add_argument("--debug-single-loop", action="store_true", help="Run for one minimal loop with extensive debugging prints.")
    p.add_argument("--eval-model-path", type=str, default=None, help="Path to a .pt model file to evaluate its policy on sample states.")
    p.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    p.add_argument("--bootstrap-games", type=int, default=50, help="Initial games to fill buffer.")
    p.add_argument("--skip-bootstrap", action="store_true", help="Skip bootstrap if buffer meets min fill.")
    p.add_argument("--mcts-simulations", type=int, default=25, help="MCTS simulations per move.")
    p.add_argument("--c-puct", type=float, default=1.0, help="PUCT exploration constant.")
    p.add_argument("--dirichlet-alpha", type=float, default=0.5, help="Alpha for Dirichlet noise.")
    p.add_argument("--dirichlet-epsilon", type=float, default=0.25, help="Epsilon for Dirichlet noise.")
    p.add_argument("--temp-decay-moves", type=int, default=4, help="Moves to use T=1 for exploration.")
    p.add_argument("--final-temp", type=float, default=0.05, help="Temp after decay (0 for deterministic).")
    p.add_argument("--epochs", type=int, default=1000, help="Total training epochs.")
    p.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate.")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    p.add_argument("--ent-beta", type=float, default=1e-2, help="Entropy regularization coefficient.")
    p.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    p.add_argument("--lr-t-max", type=int, default=0, help="T_max for CosineAnnealingLR (0 for args.epochs).")
    p.add_argument("--lr-eta-min", type=float, default=1e-7, help="Minimum LR for CosineAnnealingLR.")
    p.add_argument("--augment-prob", type=float, default=0.5, help="Probability of reflection augmentation.")
    p.add_argument("--buffer-size", type=int, default=5000, help="Max replay buffer size.")
    p.add_argument("--min-buffer-fill-standard", type=int, default=100, help="Min samples for standard buffer.")
    p.add_argument("--min-buffer-fill-for-per-training", type=int, default=500, help="Min samples for PER training.")
    p.add_argument("--min-buffer-fill-for-per-bootstrap", type=int, default=100, help="Min PER samples for bootstrap target.")
    p.add_argument("--buffer-path", type=str, default="ttt_adv_mcts_buffer.pth", help="Path to save/load buffer.")
    p.add_argument("--save-buffer-every", type=int, default=20, help="Save buffer every N epochs.")
    p.add_argument("--use-per", action="store_true", help="Use Prioritized Experience Replay.")
    p.add_argument("--per-alpha", type=float, default=0.7, help="Alpha for PER.")
    p.add_argument("--per-beta-start", type=float, default=0.5, help="Initial beta for PER.")
    p.add_argument("--per-beta-epochs", type=int, default=0, help="Epochs to anneal beta (0 for const beta_start, set to total epochs if 0 and PER is used).")
    p.add_argument("--per-epsilon", type=float, default=1e-4, help="Epsilon for PER priorities.")
    p.add_argument("--channels", type=int, default=32, help="Channels per conv layer.")
    p.add_argument("--blocks", type=int, default=2, help="Number of residual blocks.")
    p.add_argument("--ckpt-dir", default="ttt_checkpoints_az", help="Directory for model checkpoints.")
    p.add_argument("--ckpt-every", type=int, default=50, help="Save checkpoint every N epochs.")
    p.add_argument("--log-every", type=int, default=1, help="Log training stats every N epochs.")
    p.add_argument("--resume-weights", metavar="PATH", help="Path to load network weights.")
    p.add_argument("--resume-full-state", action="store_true", help="Resume full training state.")
    return p

if __name__ == "__main__":
    run() 