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
    max_moves: int = ACTION_SIZE # Max moves in TTT is 9
) -> List[Tuple[dict, np.ndarray, int]]:
    if torch is None or np is None:
        raise RuntimeError("PyTorch and NumPy are required for play_one_game")

    st = game_adapter.ttt_game.getInitialState() 
    hist: List[Tuple[dict, np.ndarray, int]] = [] 
    move_no = 0
    current_temp = 1.0

    while not game_adapter.isTerminal(st) and move_no < max_moves:
        for threshold_moves, temp_val in temp_schedule:
            if move_no < threshold_moves:
                current_temp = temp_val
                break
        
        player_perspective = game_adapter.getCurrentPlayer(st)
        chosen_action_int, mcts_policy_dict = mcts_instance.get_action_policy(
            root_state=st,
            num_simulations=mcts_simulations, 
            temperature=current_temp
        )
        
        policy_target_vector = np.zeros(game_adapter.get_action_size(), dtype=np.float32)
        for action_idx_int, prob in mcts_policy_dict.items(): # action_idx_int is 0-8
            if 0 <= action_idx_int < game_adapter.get_action_size():
                policy_target_vector[action_idx_int] = prob
            else:
                print(f"Warning: MCTS returned action {action_idx_int} out of bounds for TTT policy vector.")

        if policy_target_vector.sum() > 1e-5 : # Ensure sum is substantial before normalizing
             policy_target_vector /= policy_target_vector.sum()
        else:
            if not game_adapter.isTerminal(st):
                legal_actions_for_fallback = game_adapter.getLegalActions(st) # These are ints 0-8
                if legal_actions_for_fallback:
                    uniform_prob = 1.0 / len(legal_actions_for_fallback)
                    for la_int in legal_actions_for_fallback:
                        policy_target_vector[la_int] = uniform_prob
        
        hist.append((game_adapter.copyState(st), policy_target_vector, 0)) 
        st = game_adapter.applyAction(st, chosen_action_int)
        move_no += 1

    winner = game_adapter.getGameOutcome(st)
    z = 0
    if winner == "Draw": z = 0
    elif winner == "X": z = 1
    elif winner == "O": z = -1
    
    final_history = []
    for recorded_state, policy, _ in hist:
        player_at_state = game_adapter.getCurrentPlayer(recorded_state)
        value_for_state_player = z if player_at_state == "X" else -z
        final_history.append((recorded_state, policy, value_for_state_player))
        
    return final_history


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_step(net: AdvancedTTTZeroNet, batch_experiences: list, is_weights: np.ndarray, 
               opt: torch.optim.Optimizer, dev: str, game_adapter: TicTacToeAdapter, 
               augment_prob: float, ent_beta_val: float) -> Tuple[float, np.ndarray]: # Added ent_beta_val
    S_list = []
    P_tgt_list = []
    V_tgt_list = []

    for s_dict_orig, p_tgt_orig, v_tgt_orig in batch_experiences:
        s_dict_to_encode = s_dict_orig
        p_tgt_to_use = p_tgt_orig

        if random.random() < augment_prob:
            s_dict_reflected, p_tgt_reflected = reflect_ttt_state_policy(s_dict_orig, p_tgt_orig) # Use TTT reflection
            s_dict_to_encode = s_dict_reflected
            p_tgt_to_use = p_tgt_reflected
        
        player_perspective = game_adapter.getCurrentPlayer(s_dict_to_encode)
        S_list.append(game_adapter.encode_state(s_dict_to_encode, player_perspective))
        P_tgt_list.append(p_tgt_to_use)
        V_tgt_list.append(v_tgt_orig)

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
    opt.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    
    td_errors = np.abs(V_tgt.cpu().detach().numpy() - V_pred.squeeze().cpu().detach().numpy())
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

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run(parsed_cli_args=None) -> None:
    global args_global 
    args_global = parsed_cli_args if parsed_cli_args is not None else parser().parse_args()

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
        print(f"Using CosineAnnealingLR: T_max={t_max_epochs}, eta_min={args_global.lr_eta_min}")
    
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

    state_path = ckdir / "train_state_ttt.pt" # Changed path
    start_ep = 1
    if args_global.resume_full_state and state_path.exists():
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
    elif args_global.resume_weights:
        print("Resuming weights from", args_global.resume_weights)
        net.load_state_dict(torch.load(args_global.resume_weights, map_location=dev))

    buf: PrioritizedReplayBuffer | deque
    if args_global.use_per:
        buf = PrioritizedReplayBuffer(
            capacity=args_global.buffer_size, alpha=args_global.per_alpha,
            beta_start=args_global.per_beta_start, beta_epochs=args_global.per_beta_epochs,
            epsilon=args_global.per_epsilon )
        print(f"Using PrioritizedReplayBuffer.")
    else:
        buf = deque(maxlen=args_global.buffer_size)
        print(f"Using standard deque replay buffer.")

    buffer_file_path = Path(args_global.buffer_path)
    load_experiences_to_buffer(buf, buffer_file_path) # Removed maxlen arg

    temp_schedule = [
        (args_global.temp_decay_moves, 1.0), (float('inf'), args_global.final_temp)
    ]
    
    # Bootstrap games
    # Determine min_fill for bootstrap based on buffer type
    min_fill_for_bootstrap = args_global.min_buffer_fill_for_per_bootstrap if isinstance(buf, PrioritizedReplayBuffer) else args_global.min_buffer_fill_standard
    
    if not args_global.skip_bootstrap and len(buf) < min_fill_for_bootstrap:
        games_to_play_bootstrap = args_global.bootstrap_games
        if len(buf) < min_fill_for_bootstrap:
            avg_states_per_game = 5 # TTT has fewer states
            needed_states = min_fill_for_bootstrap - len(buf)
            needed_games = (needed_states + avg_states_per_game -1) // avg_states_per_game
            games_to_play_bootstrap = max(games_to_play_bootstrap, needed_games)
            print(f"Buffer below min fill ({len(buf)}/{min_fill_for_bootstrap}). Playing at least {needed_games} bootstrap games.")
        
        print(f"Bootstrapping {games_to_play_bootstrap} games …", flush=True)
        for g in range(games_to_play_bootstrap):
            net.eval() 
            game_hist = play_one_game(
                net, game_adapter, mcts, temp_schedule, 
                mcts_simulations=args_global.mcts_simulations, max_moves=ACTION_SIZE )
            
            add_method = buf.add if isinstance(buf, PrioritizedReplayBuffer) else buf.extend
            if isinstance(buf, PrioritizedReplayBuffer): # PER adds one by one
                for exp in game_hist: add_method(exp)
            else: # Deque can extend
                add_method(game_hist)
            
            net.train() 
            print(f"  Bootstrap game {g+1}/{games_to_play_bootstrap} ({len(game_hist)} states) → buffer {len(buf)}", flush=True)
            if len(buf) > 0 and (g + 1) % args_global.save_buffer_every == 0 : 
                 save_buffer_experiences(buf, buffer_file_path)
                 print(f"Saved replay buffer during bootstrap at game {g+1}")

    print(f"Starting training from epoch {start_ep} for {args_global.epochs} epochs.")
    try:
        for ep in range(start_ep, args_global.epochs + 1):
            net.eval() 
            game_hist = play_one_game(
                net, game_adapter, mcts, temp_schedule, 
                mcts_simulations=args_global.mcts_simulations, max_moves=ACTION_SIZE )
            
            add_method = buf.add if isinstance(buf, PrioritizedReplayBuffer) else buf.extend
            if isinstance(buf, PrioritizedReplayBuffer):
                for exp in game_hist: add_method(exp)
            else:
                add_method(game_hist)
            net.train() 

            current_min_train_fill = args_global.min_buffer_fill_for_per_training if isinstance(buf, PrioritizedReplayBuffer) else args_global.min_buffer_fill_standard
            if len(buf) < current_min_train_fill:
                if ep % args_global.log_every == 0: # Still log if skipping
                    print(f"Epoch {ep} | Buffer {len(buf)} < min {current_min_train_fill}. Skip train. LR {opt.param_groups[0]['lr']:.2e}", flush=True)
                if ep % args_global.ckpt_every == 0: save_buffer_experiences(buf, buffer_file_path) 
            else:
                batch_experiences, is_weights, data_indices = None, None, None
                if isinstance(buf, PrioritizedReplayBuffer):
                    sampled_data = buf.sample(args_global.batch_size)
                    if sampled_data is None:
                        if ep % args_global.log_every == 0: print(f"Epoch {ep} | PER sample failed. Skip train. LR {opt.param_groups[0]['lr']:.2e}", flush=True)
                        continue 
                    batch_experiences, is_weights, data_indices = sampled_data
                else: 
                    batch_experiences = random.sample(list(buf), min(args_global.batch_size, len(buf)))
                    is_weights = np.ones(len(batch_experiences), dtype=np.float32)
                
                loss, td_errors = train_step(net, batch_experiences, is_weights, opt, dev, game_adapter, args_global.augment_prob, args_global.ent_beta)
                
                if isinstance(buf, PrioritizedReplayBuffer) and data_indices is not None:
                    buf.update_priorities(data_indices, td_errors)

                if ep % args_global.log_every == 0:
                    print(f"Epoch {ep} | Loss {loss:.4f} | Buffer {len(buf)} | LR {opt.param_groups[0]['lr']:.2e}", flush=True)
            
            if scheduler: scheduler.step()
            
            if ep % args_global.ckpt_every == 0:
                chkpt_path = ckdir / f"ttt_chkpt_ep{ep:06d}.pt" # Changed path
                torch.save(net.state_dict(), chkpt_path)
                current_epoch_to_save = ep 
                full_state_to_save = {
                    "net": net.state_dict(), "opt": opt.state_dict(),
                    "epoch": current_epoch_to_save, 
                    "scheduler": scheduler.state_dict() if scheduler else None,
                }
                torch.save(full_state_to_save, state_path)
                print(f"Saved: {chkpt_path} and {state_path}", flush=True)

            if ep % args_global.save_buffer_every == 0 and len(buf) > 0:
                 save_buffer_experiences(buf, buffer_file_path)
                 print(f"Saved replay buffer at epoch {ep}")
            
            if isinstance(buf, PrioritizedReplayBuffer):
                buf.advance_epoch_for_beta_anneal()

    except KeyboardInterrupt: print("Training interrupted.")
    finally:
        print("Saving final model and buffer...")
        final_model_path = ckdir / "last_ttt_model.pt" # Changed path
        torch.save(net.state_dict(), final_model_path)
        if len(buf) > 0: save_buffer_experiences(buf, buffer_file_path)
        
        current_epoch_to_save = ep if 'ep' in locals() and 'start_ep' in locals() and ep >= start_ep else start_ep -1
        full_state_to_save = {
            "net": net.state_dict(), "opt": opt.state_dict(),
            "epoch": current_epoch_to_save,
            "scheduler": scheduler.state_dict() if scheduler else None,
        }
        torch.save(full_state_to_save, state_path)
        print(f"Final model: {final_model_path}, Buffer: {buffer_file_path}, State: {state_path}")

# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AlphaZero-style training for Tic-Tac-Toe.")
    p.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    
    g_play = p.add_argument_group("Self-Play & MCTS")
    g_play.add_argument("--bootstrap-games", type=int, default=50, help="Initial games to fill buffer.") # Reduced
    g_play.add_argument("--skip-bootstrap", action="store_true", help="Skip bootstrap if buffer meets min fill.")
    g_play.add_argument("--mcts-simulations", type=int, default=25, help="MCTS simulations per move.") # Reduced
    g_play.add_argument("--c-puct", type=float, default=1.0, help="PUCT exploration constant.") # Adjusted
    g_play.add_argument("--dirichlet-alpha", type=float, default=0.5, help="Alpha for Dirichlet noise.") # Adjusted
    g_play.add_argument("--dirichlet-epsilon", type=float, default=0.25, help="Epsilon for Dirichlet noise.")
    g_play.add_argument("--temp-decay-moves", type=int, default=4, help="Moves to use T=1 for exploration.") # Adjusted
    g_play.add_argument("--final-temp", type=float, default=0.05, help="Temp after decay (0 for deterministic).") # Adjusted

    g_train = p.add_argument_group("Training")
    g_train.add_argument("--epochs", type=int, default=1000, help="Total training epochs.") # Reduced
    g_train.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate.") # Slightly higher for small net
    g_train.add_argument("--batch-size", type=int, default=64, help="Batch size for training.") # Reduced
    g_train.add_argument("--ent-beta", type=float, default=1e-2, help="Entropy regularization coefficient.") # Potentially higher
    g_train.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    g_train.add_argument("--lr-t-max", type=int, default=0, help="T_max for CosineAnnealingLR (0 for args.epochs).") # Default to total epochs
    g_train.add_argument("--lr-eta-min", type=float, default=1e-7, help="Minimum LR for CosineAnnealingLR.")
    g_train.add_argument("--augment-prob", type=float, default=0.5, help="Probability of reflection augmentation.")

    g_buffer = p.add_argument_group("Replay Buffer")
    g_buffer.add_argument("--buffer-size", type=int, default=5000, help="Max replay buffer size.") # Reduced
    g_buffer.add_argument("--min-buffer-fill-standard", type=int, default=100, help="Min samples for standard buffer.") # Reduced
    g_buffer.add_argument("--min-buffer-fill-for-per-training", type=int, default=500, help="Min samples for PER training.") # Reduced
    g_buffer.add_argument("--min-buffer-fill-for-per-bootstrap", type=int, default=100, help="Min PER samples for bootstrap target.") # Reduced
    g_buffer.add_argument("--buffer-path", type=str, default="ttt_adv_mcts_buffer.pth", help="Path to save/load buffer.")
    g_buffer.add_argument("--save-buffer-every", type=int, default=20, help="Save buffer every N epochs.") # More frequent
    g_buffer.add_argument("--use-per", action="store_true", help="Use Prioritized Experience Replay.")
    g_buffer.add_argument("--per-alpha", type=float, default=0.7, help="Alpha for PER.") # Adjusted
    g_buffer.add_argument("--per-beta-start", type=float, default=0.5, help="Initial beta for PER.") # Adjusted
    g_buffer.add_argument("--per-beta-epochs", type=int, default=0, help="Epochs to anneal beta (0 for const beta_start).")
    g_buffer.add_argument("--per-epsilon", type=float, default=1e-4, help="Epsilon for PER priorities.")

    g_nn = p.add_argument_group("Neural Network (TTT)")
    g_nn.add_argument("--channels", type=int, default=32, help="Channels per conv layer.") # Reduced
    g_nn.add_argument("--blocks", type=int, default=2, help="Number of residual blocks.") # Reduced

    g_mgmt = p.add_argument_group("Checkpointing & Logging")
    g_mgmt.add_argument("--ckpt-dir", default="ttt_checkpoints_az", help="Directory for model checkpoints.")
    g_mgmt.add_argument("--ckpt-every", type=int, default=50, help="Save checkpoint every N epochs.") # More frequent
    g_mgmt.add_argument("--log-every", type=int, default=1, help="Log training stats every N epochs.")
    g_mgmt.add_argument("--resume-weights", metavar="PATH", help="Path to load network weights.")
    g_mgmt.add_argument("--resume-full-state", action="store_true", help="Resume full training state.")
    
    return p

if __name__ == "__main__":
    run() 