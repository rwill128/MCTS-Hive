#!/usr/bin/env python3
"""Hive‑Zero: end‑to‑end self‑play + training runner
====================================================
Run **one command** to launch continuous self‑play, online training and
checkpointing:

```
python hive_zero.py --gpu              # uses CUDA if available
```

Key points
----------
* Uses your existing *HiveGame* logic – no rules duplicated.
* 6‑layer residual CNN (~200 K params) with **policy + value** heads.
* **Replay buffer** (50 k positions) fed by self‑play in the background.
* Online SGD after every new game; checkpoint every N epochs.
* `ZeroEvaluator` plugs into your MCTS once you’re ready.

All previous smoke‑test functions are still here; they’re just integrated
into a proper loop.
"""

from __future__ import annotations

import argparse, json, math, random, time, pathlib, os
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from HivePocket.HivePocket import HiveGame, find_queen_position, hex_distance

# ------------------------------------------------------------
# Board constants --------------------------------------------
# ------------------------------------------------------------
BOARD_R = 6  # half‑diameter ⇒ 13×13 square big enough for Hive Pocket
H = W = BOARD_R * 2 + 1
PIECE_TYPES = ["Q", "B", "S", "A", "G"]
C = len(PIECE_TYPES) * 2 + 1  # ← channels incl. side‑to‑move

shift = BOARD_R

def to_xy(q: int, r: int):
    return r + shift, q + shift  # row,col indices

# ------------------------------------------------------------
# Encoder -----------------------------------------------------
# ------------------------------------------------------------

def encode_state(state: dict, perspective: str) -> torch.Tensor:
    t = torch.zeros((C, H, W), dtype=torch.float32)
    board = state["board"]
    side  = 0 if state["current_player"] == perspective else 1
    for (q, r), stack in board.items():
        if not stack: continue
        owner, insect = stack[-1]
        try:
            idx = PIECE_TYPES.index(insect[0])
        except ValueError:
            continue
        plane = idx if owner == perspective else idx + len(PIECE_TYPES)
        y, x = to_xy(q, r)
        if 0 <= y < H and 0 <= x < W:
            t[plane, y, x] = 1.0
    t[-1].fill_(side)
    return t

# ------------------------------------------------------------
# CNN ---------------------------------------------------------
# ------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(out + x)

class HiveZeroNet(nn.Module):
    def __init__(self, ch=64, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(C, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU())
        self.res = nn.Sequential(*[ResidualBlock(ch) for _ in range(blocks)])
        # policy
        self.pol = nn.Sequential(
            nn.Conv2d(ch, 2, 1), nn.BatchNorm2d(2), nn.ReLU(),
            nn.Flatten(), nn.Linear(2*H*W, H*W*len(PIECE_TYPES)))
        # value
        self.val = nn.Sequential(
            nn.Conv2d(ch, 1, 1), nn.BatchNorm2d(1), nn.ReLU(),
            nn.Flatten(), nn.Linear(H*W, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())
    def forward(self, x):
        x = self.res(self.stem(x))
        return self.pol(x), self.val(x).squeeze(1)

# ------------------------------------------------------------
# Action index mapping ---------------------------------------
# ------------------------------------------------------------
AXIAL_TO_INDEX, FLAT = {}, []
for q in range(-BOARD_R, BOARD_R + 1):
    for r in range(-BOARD_R, BOARD_R + 1):
        for t in PIECE_TYPES:
            AXIAL_TO_INDEX[(q, r, t)] = len(FLAT)
            FLAT.append((q, r, t))
POLICY_SIZE = len(FLAT)  # 845

# ------------------------------------------------------------
# Evaluator (plug into MCTS later) ---------------------------
# ------------------------------------------------------------
class ZeroEvaluator:
    def __init__(self, model: HiveZeroNet, device="cpu"):
        self.model, self.dev = model.to(device), device
        self.game = HiveGame()
    @torch.no_grad()
    def evaluate(self, state: dict, perspective: str):
        x = encode_state(state, perspective).unsqueeze(0).to(self.dev)
        logits, v = self.model(x)
        return logits.cpu().numpy()[0], v.item()

# ------------------------------------------------------------
# Self‑play ---------------------------------------------------
# ------------------------------------------------------------

def softmax_T(x, T):
    z = np.exp((x - x.max()) / T); return z / z.sum()

def mask_illegal(priors, legal):
    mask = np.zeros_like(priors)
    for a in legal:
        try:
            if a[0] == "PLACE":
                _, tp, (q, r) = a
                idx = AXIAL_TO_INDEX[(q, r, tp[0])]
            else:
                dest = a[2] if len(a) == 3 else a[2]
                q, r = dest
                idx = AXIAL_TO_INDEX[(q, r, PIECE_TYPES[0])]
            mask[idx] = 1.0
        except KeyError:
            # destination outside 13×13 bounding box – ignore for policy head
            continue
    priors = priors * mask
    if mask.sum() == 0:
        # should not happen but guard division‑by‑zero
        return priors
    priors /= priors.sum()
    return priors

def map_flat_to_action(flat_idx, legal):
    q, r, tchar = FLAT[flat_idx]
    for a in legal:
        if a[0] == "PLACE":
            _, tp, (q0, r0) = a
            if (q0, r0) == (q, r) and tp[0] == tchar:
                return a
        else:
            dest = a[2] if len(a) == 3 else a[2]
            if dest == (q, r):
                return a
    return random.choice(legal)

def play_one_game(model, temperature=1.0, max_moves=300):
    game, state = HiveGame(), HiveGame().getInitialState()
    evalzr = ZeroEvaluator(model)
    hist = []
    while not game.isTerminal(state) and len(hist) < max_moves:
        logits, _ = evalzr.evaluate(state, state["current_player"])
        priors = softmax_T(logits, temperature)
        priors = mask_illegal(priors, game.getLegalActions(state))
        hist.append((game.copyState(state), priors))
        flat = np.random.choice(POLICY_SIZE, p=priors)
        action = map_flat_to_action(flat, game.getLegalActions(state))
        state = game.applyAction(state, action)
    winner = game.getGameOutcome(state)
    z = 0 if winner == "Draw" else (+1 if winner == "Player1" else -1)
    if state["current_player"] == "Player2": z = -z
    return [(s, p, z) for s, p in hist]

# ------------------------------------------------------------
# Training ----------------------------------------------------
# ------------------------------------------------------------

def batch_to_tensor(batch, device):
    st = torch.stack([encode_state(s, "Player1") for s,_,_ in batch]).to(device)
    targ_p = torch.tensor([p for _,p,_ in batch], dtype=torch.float32).to(device)
    targ_v = torch.tensor([v for _,_,v in batch], dtype=torch.float32).to(device)
    return st, targ_p, targ_v

def train_step(model, batch, opt, device):
    st, targ_p, targ_v = batch_to_tensor(batch, device)
    logits, v_pred = model(st)
    loss_p = F.cross_entropy(logits, targ_p.argmax(1))
    loss_v = F.mse_loss(v_pred.squeeze(), targ_v)
    loss = loss_p + loss_v
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

# ------------------------------------------------------------
# Main loop ---------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------
# Main loop ---------------------------------------------------
# ------------------------------------------------------------

def run_selfplay_training(args):
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    net = HiveZeroNet().to(device)
    opt = optim.Adam(net.parameters(), lr=1e-3)

    # replay buffer -------------------------------------------------
    buffer: deque = deque(maxlen=args.buffer)

    # bootstrap games ----------------------------------------------
    print("Bootstrapping", args.games, "self‑play games …")
    for g in range(args.games):
        buffer.extend(play_one_game(net, temperature=1.0))
        print(f"  game {g+1}/{args.games} → buffer size {len(buffer)}")

    # continuous loop ----------------------------------------------
    ckpt_dir = pathlib.Path(args.ckpt_dir); ckpt_dir.mkdir(exist_ok=True)
    epoch = 0
    while epoch < args.epochs:
        epoch += 1
        # self‑play
        buffer.extend(play_one_game(net, temperature=args.temp))
        # training step
        if len(buffer) >= args.batch:
            batch = random.sample(buffer, args.batch)
            loss = train_step(net, batch, opt, device)
            if epoch % args.log_every == 0:
                print(f"epoch {epoch:5d} | buffer {len(buffer):5d} | loss {loss:.4f}")
        # checkpoint
        if epoch % args.ckpt_every == 0:
            p = ckpt_dir / f"chkpt_{epoch:05d}.pt"
            torch.save(net.state_dict(), p)
            print("saved", p)


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Hive‑Zero self‑play trainer")
    ap.add_argument("--gpu", action="store_true", help="use CUDA if available")
    ap.add_argument("--games", type=int, default=20, help="bootstrap games")
    ap.add_argument("--epochs", type=int, default=1000, help="training epochs (self‑play cycles)")
    ap.add_argument("--buffer", type=int, default=50000, help="replay buffer size")
    ap.add_argument("--batch", type=int, default=256, help="SGD batch size")
    ap.add_argument("--temp", type=float, default=1.0, help="self‑play softmax temperature")
    ap.add_argument("--ckpt-dir", default="checkpoints", help="where to store checkpoints")
    ap.add_argument("--ckpt-every", type=int, default=100, help="epochs between checkpoints")
    ap.add_argument("--log-every", type=int, default=10, help="print loss every N epochs")
    return ap


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_selfplay_training(args)
