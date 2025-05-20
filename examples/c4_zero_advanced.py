#!/usr/bin/env python3
"""Advanced self-play training loop for Connect Four.

This script adapts the techniques used in ``hive_zero.py`` to a Connect
Four environment.  Key features include

    • residual convolutional network
    • KL-divergence policy loss with entropy regularisation
    • Dirichlet noise on the initial policy
    • temperature decay after move 10
    • replay-buffer training and periodic checkpoints
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


print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

from simple_games.connect_four import ConnectFour
from mcts.alpha_zero_mcts import AlphaZeroMCTS

BOARD_H = ConnectFour.ROWS
BOARD_W = ConnectFour.COLS


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------
def encode_state(state: dict, perspective: str) -> torch.Tensor:
    """Return a 3×6×7 tensor representing ``state`` from ``perspective``."""
    if torch is None:
        raise RuntimeError("PyTorch is required for encode_state")
    t = torch.zeros(3, BOARD_H, BOARD_W)
    for r in range(BOARD_H):
        for c in range(BOARD_W):
            piece = state["board"][r][c]
            if piece == perspective:
                t[0, r, c] = 1.0
            elif piece is not None:
                t[1, r, c] = 1.0
    t[2].fill_(1.0 if state["current_player"] == perspective else 0.0)
    return t


# ---------------------------------------------------------------------------
# Game Adapter for MCTS
# ---------------------------------------------------------------------------
class ConnectFourAdapter:
    def __init__(self, c4_game: ConnectFour):
        self.c4_game = c4_game
        self.action_size = self.c4_game.get_action_size()

    def getCurrentPlayer(self, state: Dict) -> str:
        return self.c4_game.getCurrentPlayer(state)

    def getLegalActions(self, state: Dict) -> List[int]:
        return self.c4_game.getLegalActions(state)

    def applyAction(self, state: Dict, action: int) -> Dict:
        return self.c4_game.applyAction(state, action)

    def isTerminal(self, state: Dict) -> bool:
        return self.c4_game.isTerminal(state)

    def getGameOutcome(self, state: Dict) -> str:
        return self.c4_game.getGameOutcome(state)

    def encode_state(self, state: Dict, player_perspective: str) -> torch.Tensor:
        return encode_state(state, player_perspective)

    def copyState(self, state: Dict) -> Dict:
        return self.c4_game.copyState(state)
    
    def get_action_size(self) -> int:
        return self.action_size


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------
if torch is not None:
    class ResidualBlock(nn.Module):
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
        """Residual policy/value network for Connect Four.

        Parameters
        ----------
        ch : int, optional
            Number of channels in each convolutional layer (default 128).
        blocks : int, optional
            How many residual blocks to stack (default 10).
        """

        def __init__(self, ch: int = 128, blocks: int = 10):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(),
            )
            self.res = nn.Sequential(*[ResidualBlock(ch) for _ in range(blocks)])
            self.policy = nn.Sequential(
                nn.Conv2d(ch, 2, 1), nn.BatchNorm2d(2), nn.ReLU(),
                nn.Flatten(), nn.Linear(2 * BOARD_H * BOARD_W, BOARD_W)
            )
            self.value = nn.Sequential(
                nn.Conv2d(ch, 1, 1), nn.BatchNorm2d(1), nn.ReLU(),
                nn.Flatten(), nn.Linear(BOARD_H * BOARD_W, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.Tanh()
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            x = self.res(self.stem(x))
            return self.policy(x), self.value(x).squeeze(1)
else:  # pragma: no cover - torch not installed
    class ResidualBlock:
        def __init__(self, *a, **k) -> None:
            raise RuntimeError("PyTorch not available")

    class AdvancedC4ZeroNet:
        def __init__(self, *a, **k) -> None:
            raise RuntimeError("PyTorch not available")


# ---------------------------------------------------------------------------
# Self-play helpers
# ---------------------------------------------------------------------------
ROOT_NOISE_FRAC = 0.25
DIR_ALPHA = 0.3
ENT_BETA = 1e-3

# Default file used to persist the replay buffer between runs
BUFFER_PATH = Path("c4_adv_buffer.pth")


def softmax_T(x: np.ndarray, T: float) -> np.ndarray:
    if np is None:
        raise RuntimeError("NumPy is required for softmax_T")
    z = np.exp((x - x.max()) / T)
    return z / z.sum()


def mask_illegal(pri: np.ndarray, legal: List[int]) -> np.ndarray:
    if np is None:
        raise RuntimeError("NumPy is required for mask_illegal")
    mask = np.zeros_like(pri)
    for a in legal:
        mask[a] = 1.0
    if mask.sum() == 0:
        return pri * 0
    pri = pri * mask
    pri /= pri.sum()
    return pri


def play_one_game(
    net: AdvancedC4ZeroNet,
    game_adapter: ConnectFourAdapter,
    mcts_instance: AlphaZeroMCTS,
    temp_schedule: List[Tuple[int, float]],
    mcts_simulations: int,
    max_moves: int = 42
) -> List[Tuple[dict, np.ndarray, int]]:
    if torch is None or np is None:
        raise RuntimeError("PyTorch and NumPy are required for play_one_game")

    st = game_adapter.c4_game.getInitialState()
    hist: List[Tuple[dict, np.ndarray, int]] = []
    move_no = 0
    
    current_temp = 1.0

    while not game_adapter.isTerminal(st) and move_no < max_moves:
        for threshold_moves, temp_val in temp_schedule:
            if move_no < threshold_moves:
                current_temp = temp_val
                break
        
        player_perspective = game_adapter.getCurrentPlayer(st)

        chosen_action, mcts_policy_dict = mcts_instance.get_action_policy(
            root_state=st,
            num_simulations=mcts_simulations,
            temperature=current_temp
        )
        
        policy_target_vector = np.zeros(game_adapter.get_action_size(), dtype=np.float32)
        for action_idx, prob in mcts_policy_dict.items():
            if 0 <= action_idx < game_adapter.get_action_size():
                policy_target_vector[action_idx] = prob
            else:
                print(f"Warning: MCTS returned action {action_idx} out of bounds for policy vector.")

        if policy_target_vector.sum() > 0 :
             policy_target_vector /= policy_target_vector.sum()
        else:
            if not game_adapter.isTerminal(st):
                legal_actions_for_fallback = game_adapter.getLegalActions(st)
                if legal_actions_for_fallback:
                    uniform_prob = 1.0 / len(legal_actions_for_fallback)
                    for la in legal_actions_for_fallback:
                        policy_target_vector[la] = uniform_prob


        hist.append((game_adapter.copyState(st), policy_target_vector, 0))

        st = game_adapter.applyAction(st, chosen_action)
        move_no += 1

    winner = game_adapter.getGameOutcome(st)
    z = 0
    if winner == "Draw":
        z = 0
    elif winner == "X":
        z = 1
    elif winner == "O":
        z = -1

    final_history = []
    for recorded_state, policy, _ in hist:
        player_at_state = game_adapter.getCurrentPlayer(recorded_state)
        value_for_state_player = z if player_at_state == "X" else -z
        final_history.append((recorded_state, policy, value_for_state_player))
        
    return final_history


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def batch_tensors(batch: List[Tuple[dict, np.ndarray, int]], dev: str, game_adapter: ConnectFourAdapter):
    if np is None or torch is None:
        raise RuntimeError("NumPy and PyTorch are required for batch_tensors")
    
    S_list = []
    for s_dict, _, _ in batch:
        player_perspective = game_adapter.getCurrentPlayer(s_dict)
        S_list.append(game_adapter.encode_state(s_dict, player_perspective))
        
    S = torch.stack(S_list).to(dev)
    P_tgt = torch.tensor(np.array([p for _, p, _ in batch]), dtype=torch.float32, device=dev)
    V_tgt = torch.tensor([v for _, _, v in batch], dtype=torch.float32, device=dev)
    return S, P_tgt, V_tgt


def train_step(net: AdvancedC4ZeroNet, batch, opt, dev: str, game_adapter: ConnectFourAdapter) -> float:
    S, P_tgt, V_tgt = batch_tensors(batch, dev, game_adapter)
    logits, V_pred = net(S)
    
    logP_pred = F.log_softmax(logits, dim=1)
    loss_p = F.kl_div(logP_pred, P_tgt, reduction="batchmean")
    
    loss_v = F.mse_loss(V_pred.squeeze(), V_tgt)
    
    P_pred_dist = torch.exp(logP_pred)
    entropy = -(P_pred_dist * logP_pred).sum(dim=1).mean()

    loss = loss_p + loss_v - ENT_BETA * entropy
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    return float(loss.item())


def save_buffer(buf: deque, path: Path) -> None:
    """Persist the replay buffer to ``path``."""
    if torch is None:
        raise RuntimeError("PyTorch is required for save_buffer")
    torch.save(list(buf), path)


def load_buffer(path: Path, maxlen: int) -> deque:
    """Load a replay buffer from ``path`` if it exists."""
    if torch is None:
        raise RuntimeError("PyTorch is required for load_buffer")
    try:
        data = torch.load(path, weights_only=False)
    except TypeError:
        data = torch.load(path)
    return deque(data, maxlen=maxlen)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run(args=None) -> None:
    """Run the advanced self-play loop with optional ``args``."""
    if args is None:
        args = parser().parse_args()

    if torch is None or np is None:
        raise RuntimeError("PyTorch and NumPy are required for training")
    dev = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev}")

    ckdir = Path(args.ckpt_dir)
    ckdir.mkdir(exist_ok=True)
    
    connect_four_game = ConnectFour()
    game_adapter = ConnectFourAdapter(connect_four_game)

    net = AdvancedC4ZeroNet(ch=args.channels, blocks=args.blocks).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def mcts_model_fn(encoded_state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        net.eval()
        with torch.no_grad():
            policy_logits, value_estimates = net(encoded_state_batch)
        net.train()
        return policy_logits, value_estimates.unsqueeze(-1)

    mcts = AlphaZeroMCTS(
        game_interface=game_adapter,
        model_fn=mcts_model_fn,
        device=torch.device(dev),
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon
    )

    state_path = ckdir / "train_state.pt"
    start_ep = 1

    if args.resume_full_state and state_path.exists():
        print("Resuming full training state from", state_path)
        st_checkpoint = torch.load(state_path, map_location=dev) 
        net.load_state_dict(st_checkpoint["net"])
        opt.load_state_dict(st_checkpoint["opt"])
        start_ep = int(st_checkpoint.get("epoch", 1)) + 1
    elif args.resume_weights:
        print("Resuming weights from", args.resume_weights)
        net.load_state_dict(torch.load(args.resume_weights, map_location=dev))


    buf: deque
    buffer_actual_path = Path(args.buffer_path)
    if buffer_actual_path.exists():
        buf = load_buffer(buffer_actual_path, args.buffer_size)
        print(f"Loaded buffer with {len(buf)} samples from {buffer_actual_path}")
    else:
        buf = deque(maxlen=args.buffer_size)
        print(f"No buffer found at {buffer_actual_path}, starting new.")

    temp_schedule = [
        (args.temp_decay_moves, 1.0),
        (float('inf'), args.final_temp)
    ]

    if not args.skip_bootstrap:
        print(f"Bootstrapping {args.bootstrap_games} games …", flush=True)
        for g in range(args.bootstrap_games):
            net.eval()
            game_history = play_one_game(
                net, game_adapter, mcts, temp_schedule, 
                mcts_simulations=args.mcts_simulations,
                max_moves=BOARD_H * BOARD_W
            )
            buf.extend(game_history)
            net.train()
            print(f"  bootstrap game {g+1}/{args.bootstrap_games} ({len(game_history)} states) → buffer {len(buf)}", flush=True)
            if len(buf) > 0 and (g + 1) % args.save_buffer_every == 0 :
                 save_buffer(buf, buffer_actual_path)
                 print(f"Saved replay buffer during bootstrap at game {g+1}")


    print(f"Starting training from epoch {start_ep} for {args.epochs} epochs.")
    try:
        for ep in range(start_ep, args.epochs + 1):
            net.eval()
            game_history = play_one_game(
                net, game_adapter, mcts, temp_schedule, 
                mcts_simulations=args.mcts_simulations,
                max_moves=BOARD_H * BOARD_W
            )
            buf.extend(game_history)
            net.train()

            if len(buf) < args.min_buffer_fill:
                print(f"Epoch {ep} | Buffer size {len(buf)} < min fill {args.min_buffer_fill}. Skipping training. Game states: {len(game_history)}", flush=True)
                if ep % args.ckpt_every == 0:
                    save_buffer(buf, buffer_actual_path)
            else:
                batch = random.sample(list(buf), min(args.batch_size, len(buf)))
                loss = train_step(net, batch, opt, dev, game_adapter)
                if ep % args.log_every == 0:
                    print(f"Epoch {ep} | Loss {loss:.4f} | Buffer {len(buf)} | LR {opt.param_groups[0]['lr']:.2e}", flush=True)
            
            if ep % args.ckpt_every == 0:
                chkpt_path = ckdir / f"chkpt_ep{ep:06d}.pt"
                torch.save(net.state_dict(), chkpt_path)
                
                current_epoch_to_save = ep if 'ep' in locals() else start_ep -1
                full_state_to_save = {
                    "net": net.state_dict(),
                    "opt": opt.state_dict(),
                    "epoch": current_epoch_to_save,
                }
                torch.save(full_state_to_save, state_path)
                print(f"Saved checkpoint: {chkpt_path} and full state: {state_path}", flush=True)

            if ep % args.save_buffer_every == 0 and len(buf) > 0:
                 save_buffer(buf, buffer_actual_path)
                 print(f"Saved replay buffer at epoch {ep}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final state...")
    finally:
        print("Saving final model and buffer...")
        final_model_path = ckdir / "last_model.pt"
        torch.save(net.state_dict(), final_model_path)
        if buf:
            save_buffer(buf, buffer_actual_path)
        
        current_epoch_to_save = ep if 'ep' in locals() and ep > start_ep else start_ep -1
        if 'ep' not in locals() or ep < start_ep :
             current_epoch_to_save = start_ep -1
        
        full_state_to_save = {
            "net": net.state_dict(),
            "opt": opt.state_dict(),
            "epoch": current_epoch_to_save,
        }
        torch.save(full_state_to_save, state_path)
        print(f"Final model saved to {final_model_path}, buffer to {buffer_actual_path}, and full state to {state_path}")


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AlphaZero-style training for Connect Four with MCTS.")
    p.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    
    g_play = p.add_argument_group("Self-Play & MCTS")
    g_play.add_argument("--bootstrap-games", type=int, default=100, help="Number of initial games to fill buffer before training.")
    g_play.add_argument("--skip-bootstrap", action="store_true", help="Skip initial bootstrap games if buffer exists and is sufficient.")
    g_play.add_argument("--mcts-simulations", type=int, default=50, help="Number of MCTS simulations per move.")
    g_play.add_argument("--c-puct", type=float, default=1.41, help="PUCT exploration constant for MCTS.")
    g_play.add_argument("--dirichlet-alpha", type=float, default=0.3, help="Alpha for Dirichlet noise at MCTS root.")
    g_play.add_argument("--dirichlet-epsilon", type=float, default=0.25, help="Epsilon for mixing Dirichlet noise.")
    g_play.add_argument("--temp-decay-moves", type=int, default=10, help="Number of moves to use temperature=1.0 for exploration.")
    g_play.add_argument("--final-temp", type=float, default=0.1, help="Temperature for action selection after temp_decay_moves (0 for deterministic).")

    g_train = p.add_argument_group("Training")
    g_train.add_argument("--epochs", type=int, default=10000, help="Total training epochs (1 epoch = 1 game + 1 train step).")
    g_train.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate for Adam optimizer.")
    g_train.add_argument("--batch-size", type=int, default=256, help="Batch size for training.")
    g_train.add_argument("--ent-beta", type=float, default=1e-3, help="Entropy regularization coefficient.")


    g_buffer = p.add_argument_group("Replay Buffer")
    g_buffer.add_argument("--buffer-size", type=int, default=50000, help="Maximum size of the replay buffer.")
    g_buffer.add_argument("--min-buffer-fill", type=int, default=1000, help="Minimum samples in buffer before starting training updates.")
    g_buffer.add_argument("--buffer-path", type=str, default="c4_adv_mcts_buffer.pth", help="Path to save/load the replay buffer.")
    g_buffer.add_argument("--save-buffer-every", type=int, default=50, help="Save replay buffer every N epochs/bootstrap games.")


    g_nn = p.add_argument_group("Neural Network")
    g_nn.add_argument("--channels", type=int, default=128, help="Channels per conv layer in ResNet.")
    g_nn.add_argument("--blocks", type=int, default=10, help="Number of residual blocks in ResNet.")

    g_mgmt = p.add_argument_group("Checkpointing & Logging")
    g_mgmt.add_argument("--ckpt-dir", default="c4_checkpoints_az", help="Directory to save model checkpoints.")
    g_mgmt.add_argument("--ckpt-every", type=int, default=100, help="Save model checkpoint every N epochs.")
    g_mgmt.add_argument("--log-every", type=int, default=10, help="Log training stats every N epochs.")
    g_mgmt.add_argument("--resume-weights", metavar="PATH", help="Path to checkpoint file to load network weights before training.")
    g_mgmt.add_argument("--resume-full-state", action="store_true", help="Resume full training state (net, optimizer, epoch) from ckpt_dir/train_state.pt.")
    
    return p


if __name__ == "__main__":
    run()
