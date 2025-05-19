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
from typing import List, Tuple

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

from simple_games.connect_four import ConnectFour

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
        """Residual policy/value network for Connect Four."""

        def __init__(self, ch: int = 64, blocks: int = 4):
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


def play_one_game(net: AdvancedC4ZeroNet, T: float = 1.0, max_moves: int = 42) -> List[Tuple[dict, np.ndarray, int]]:
    if torch is None or np is None:
        raise RuntimeError("PyTorch and NumPy are required for play_one_game")
    game = ConnectFour()
    st = game.getInitialState()
    hist: List[Tuple[dict, np.ndarray, int]] = []
    move_no = 0

    while not game.isTerminal(st) and move_no < max_moves:
        x = encode_state(st, st["current_player"]).unsqueeze(0)
        with torch.no_grad():
            logits, _ = net(x)
            logits = logits.squeeze(0).cpu().numpy()
        temp = T if move_no < 10 else 0.3
        pri = softmax_T(logits, temp)
        legal = game.getLegalActions(st)
        pri = mask_illegal(pri, legal)

        if move_no == 0:
            pri = (1 - ROOT_NOISE_FRAC) * pri + ROOT_NOISE_FRAC * np.random.dirichlet(DIR_ALPHA * np.ones_like(pri))
            # Ensure numerical stability – the mixture can drift from exact unity
            pri /= pri.sum()

        hist.append((game.copyState(st), pri, 0))

        if pri.sum() == 0:
            act = random.choice(legal)
        else:
            idx = np.random.choice(BOARD_W, p=pri)
            act = idx

        st = game.applyAction(st, act)
        move_no += 1

    winner = game.getGameOutcome(st)
    z = 0 if winner == "Draw" else (1 if winner == "X" else -1)
    if st["current_player"] == "O":  # from X perspective
        z = -z

    return [(s, p, z) for s, p, _ in hist]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def batch_tensors(batch, dev: str):
    if np is None:
        raise RuntimeError("NumPy is required for batch_tensors")
    S = torch.stack([encode_state(s, "X") for s, _, _ in batch]).to(dev)
    P = torch.tensor(np.array([p for _, p, _ in batch]), dtype=torch.float32, device=dev)
    V = torch.tensor([v for _, _, v in batch], dtype=torch.float32, device=dev)
    return S, P, V


def train_step(net: AdvancedC4ZeroNet, batch, opt, dev: str) -> float:
    S, P_tgt, V_tgt = batch_tensors(batch, dev)
    logits, V_pred = net(S)
    logP = F.log_softmax(logits, dim=1)
    P_pred = logP.exp()
    loss_p = F.kl_div(logP, P_tgt, reduction="batchmean")
    loss_v = F.mse_loss(V_pred.squeeze(), V_tgt)
    entropy = -(P_pred * logP).sum(1).mean()
    loss = loss_p + loss_v - ENT_BETA * entropy
    opt.zero_grad(); loss.backward(); opt.step()
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
    # PyTorch ≥ 2.6 defaults to ``weights_only=True`` which blocks ordinary
    # pickled Python objects (our replay-buffer entries).  We explicitly set
    # ``weights_only=False`` when the argument is supported to restore the
    # pre-2.6 behaviour, while maintaining compatibility with older versions.
    try:
        data = torch.load(path, weights_only=False)  # PyTorch 2.6+
    except TypeError:
        # Older PyTorch without the ``weights_only`` parameter
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

    ckdir = Path(args.ckpt_dir)
    ckdir.mkdir(exist_ok=True)
    if args.resume is None:
        ckpts = sorted(ckdir.glob("chkpt_*.pt"))
        if ckpts:
            args.resume = str(ckpts[-1])

    # --------------------------------------------------------------
    # Build network & optimiser, then attempt to restore a *full* training
    # state (network weights + optimiser parameters + last epoch counter).
    # If no such state exists we fall back to the legacy behaviour of
    # restoring only the network weights (via --resume or latest checkpoint).
    # --------------------------------------------------------------
    net = AdvancedC4ZeroNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    state_path = ckdir / "train_state.pt"
    start_ep = 1

    if state_path.exists():
        print("Resuming full training state from", state_path)
        st = torch.load(state_path, map_location=dev)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        start_ep = int(st.get("epoch", 1))
    elif args.resume:
        print("Resuming weights from", args.resume)
        net.load_state_dict(torch.load(args.resume, map_location=dev))

    buf: deque
    if BUFFER_PATH.exists():
        buf = load_buffer(BUFFER_PATH, args.buffer)
        print(f"Loaded buffer with {len(buf)} samples")
    else:
        buf = deque(maxlen=args.buffer)

    if not args.skip_bootstrap:
        print(f"Bootstrapping {args.games} games …", flush=True)
        for g in range(args.games):
            buf.extend(play_one_game(net, T=args.temp))
            print(f"  game {g+1}/{args.games} → buffer {len(buf)}", flush=True)

    try:
        for ep in range(start_ep, args.epochs + 1):
            buf.extend(play_one_game(net, T=args.temp))
            batch = random.sample(buf, args.batch) if len(buf) >= args.batch else list(buf)
            loss = train_step(net, batch, opt, dev)
            if ep % args.log_every == 0:
                print(f"epoch {ep} | loss {loss:.4f} | buf {len(buf)}", flush=True)
            if ep % args.ckpt_every == 0:
                path = ckdir / f"chkpt_{ep:05d}.pt"
                torch.save(net.state_dict(), path)
                save_buffer(buf, BUFFER_PATH)
                print("saved", path, flush=True)

            # Save full training state so interruptions can resume seamlessly
            torch.save({
                "net": net.state_dict(),
                "opt": opt.state_dict(),
                "epoch": ep if 'ep' in locals() else start_ep,
            }, state_path)
    except KeyboardInterrupt:
        print("Stopping training …")
    finally:
        torch.save(net.state_dict(), ckdir / "last.pt")
        save_buffer(buf, BUFFER_PATH)
        # Ensure the most recent state is flushed to disk
        torch.save({
            "net": net.state_dict(),
            "opt": opt.state_dict(),
            "epoch": ep if 'ep' in locals() else start_ep,
        }, state_path)


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--buffer", type=int, default=50000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--ckpt-dir", default="c4_checkpoints")
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--resume", metavar="PATH", help="checkpoint to load before training")
    p.add_argument("--skip-bootstrap", action="store_true",
                   help="start training immediately (no fresh bootstrap games)")
    return p


if __name__ == "__main__":
    run()
