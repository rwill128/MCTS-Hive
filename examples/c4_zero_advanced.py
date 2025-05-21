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

if torch:
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

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
    debug_mode: bool = False
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
        target_buffer.extend(data)
        print(f"Loaded {len(data)} exp into deque. Size: {len(target_buffer)}")

args_global = None # For global access to parsed args

# ---------------------------------------------------------------------------
# Training loop (aligned with ttt_zero_advanced.py structure)
# ---------------------------------------------------------------------------
def run(parsed_cli_args=None) -> None:
    global args_global 
    args_global = parsed_cli_args if parsed_cli_args is not None else parser().parse_args()

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
        st_checkpoint = torch.load(state_path, map_location=dev) 
        learning_net.load_state_dict(st_checkpoint["net"])
        opt.load_state_dict(st_checkpoint["opt"])
        start_ep = int(st_checkpoint.get("epoch", 1)) + 1
        if scheduler and "scheduler" in st_checkpoint and st_checkpoint["scheduler"]:
            try: scheduler.load_state_dict(st_checkpoint["scheduler"]); print("Resumed LR scheduler.")
            except: print("Could not load scheduler state.")
        if args_global.lr_scheduler == "cosine" and scheduler and start_ep > scheduler.T_max: # Fixed quote
             print(f"Warning: Resumed epoch {start_ep} > scheduler T_max {scheduler.T_max}.")
    elif args_global.resume_weights and not args_global.debug_single_loop:
        print("Resuming weights from", args_global.resume_weights)
        learning_net.load_state_dict(torch.load(args_global.resume_weights, map_location=dev))

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
                    temp_schedule, args_global.mcts_simulations, args_global.mcts_simulations, max_moves=BOARD_H * BOARD_W, # C4 max moves
                    debug_mode=args_global.debug_single_loop )
                
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
    try:
        for ep in range(start_ep, args_global.epochs + 1):
            if args_global.debug_single_loop:
                _debug_flag = True # Ensure block starts with an assignment
                print(f"\n--- [DEBUG] Epoch {ep} START (Connect Four) ---")
            
            learning_net.eval() 
            game_hist = play_one_game(
                learning_net, game_adapter, learning_mcts_instance, "self", None, None,
                temp_schedule, args_global.mcts_simulations, args_global.mcts_simulations, max_moves=BOARD_H * BOARD_W, # C4 max moves
                debug_mode=args_global.debug_single_loop )
            
            if isinstance(buf, PrioritizedReplayBuffer):
                for exp in game_hist:
                    buf.add(exp)
            else: # It's a deque
                buf.extend(game_hist)
            learning_net.train() 

            current_min_train_fill = args_global.min_buffer_fill_for_per_training if isinstance(buf, PrioritizedReplayBuffer) else args_global.min_buffer_fill_standard
            if args_global.debug_single_loop: current_min_train_fill = 1

            if len(buf) < current_min_train_fill:
                if ep % args_global.log_every == 0: 
                    print(f"Epoch {ep} | Buffer {len(buf)} < min {current_min_train_fill}. Skip train. LR {opt.param_groups[0]['lr']:.2e}", flush=True)
                if ep % args_global.ckpt_every == 0 and not args_global.debug_single_loop: save_buffer_experiences(buf, buffer_file_path) 
            else:
                if args_global.debug_single_loop:
                    _debug_flag = True # Ensure block starts with an assignment
                    print(f"\n[run DEBUG C4] Buffer ready. len(buf): {len(buf)}, batch: {args_global.batch_size}")
                
                batch_experiences, is_weights, data_indices = None, None, None
                actual_batch_size = min(args_global.batch_size, len(buf))

                if actual_batch_size == 0:
                    if ep % args_global.log_every == 0: print(f"Epoch {ep} | Buffer empty or batch zero. Skip train.", flush=True)
                    if scheduler: scheduler.step()
                    continue

                if isinstance(buf, PrioritizedReplayBuffer):
                    sampled_data = buf.sample(actual_batch_size)
                    if sampled_data is None:
                        if ep % args_global.log_every == 0: print(f"Epoch {ep} | PER sample failed. Skip train. LR {opt.param_groups[0]['lr']:.2e}", flush=True)
                        if scheduler: scheduler.step() 
                        continue 
                    batch_experiences, is_weights, data_indices = sampled_data
                    if args_global.debug_single_loop and batch_experiences is not None:
                        print(f"[run DEBUG C4] PER Sampled {len(batch_experiences)} experiences.")
                        if len(batch_experiences) > 0 and batch_experiences[0] and batch_experiences[0][0] and 'board' in batch_experiences[0][0] and batch_experiences[0][0]['board']:
                            print(f"[run DEBUG C4] First PER exp state board: {batch_experiences[0][0]['board'][0]}...")
                        print(f"[run DEBUG C4] PER IS weights (first 4): {is_weights[:min(4, len(is_weights))] if len(is_weights) > 0 else '[]'}")
                        print(f"[run DEBUG C4] PER data_indices (first 4): {data_indices[:min(4, len(data_indices))] if len(data_indices) > 0 else '[]'}")
                else: 
                    batch_experiences = random.sample(list(buf), actual_batch_size)
                    is_weights = np.ones(len(batch_experiences), dtype=np.float32)
                    if args_global.debug_single_loop and batch_experiences is not None:
                        print(f"[run DEBUG C4] Standard Sampled {len(batch_experiences)} experiences.")
                        if len(batch_experiences) > 0 and batch_experiences[0] and batch_experiences[0][0] and 'board' in batch_experiences[0][0] and batch_experiences[0][0]['board']:
                            print(f"[run DEBUG C4] First standard exp state board: {batch_experiences[0][0]['board'][0]}...")
                
                loss, td_errors = train_step(learning_net, batch_experiences, is_weights, opt, dev, game_adapter, 
                                             args_global.augment_prob, args_global.ent_beta, 
                                             debug_mode=args_global.debug_single_loop)
                
                if isinstance(buf, PrioritizedReplayBuffer) and data_indices is not None and len(data_indices) > 0:
                    if args_global.debug_single_loop: print(f"[run DEBUG C4] Updating PER priorities for {len(data_indices)} indices.")
                    buf.update_priorities(data_indices, td_errors)

                if ep % args_global.log_every == 0:
                    print(f"Epoch {ep} | Loss {loss:.4f} | Buffer {len(buf)} | LR {opt.param_groups[0]['lr']:.2e}", flush=True)
            
            if scheduler: scheduler.step()
            
            if ep % args_global.ckpt_every == 0:
                chkpt_path = ckdir / f"c4_chkpt_ep{ep:06d}.pt" # Path for C4
                if not args_global.debug_single_loop: torch.save(learning_net.state_dict(), chkpt_path)
                full_state_to_save = {
                    "net": learning_net.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "scheduler": scheduler.state_dict() if scheduler else None,
                }
                if not args_global.debug_single_loop: torch.save(full_state_to_save, state_path)
                print(f"Saved: {str(chkpt_path)} and {str(state_path)}", flush=True)

            if ep % args_global.save_buffer_every == 0 and len(buf) > 0 and not args_global.debug_single_loop:
                 save_buffer_experiences(buf, buffer_file_path)
                 print(f"Saved replay buffer at epoch {ep}")
            
            if isinstance(buf, PrioritizedReplayBuffer):
                buf.advance_epoch_for_beta_anneal()
            if args_global.debug_single_loop: 
                _debug_flag = True # Ensure block starts with an assignment
                print(f"--- [DEBUG] Epoch {ep} END (Connect Four) ---") 
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
    g_buffer.add_argument("--min-buffer-fill-for-per-training", type=int, default=10000, help="Min samples for PER training.")
    g_buffer.add_argument("--min-buffer-fill-for-per-bootstrap", type=int, default=1000, help="Min PER samples for bootstrap target.")
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
    
    return p

if __name__ == "__main__":
    run()
