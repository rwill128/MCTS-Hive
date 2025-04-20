#!/usr/bin/env python3
"""
Hive‑Zero  –  minimal AlphaZero‑style self‑play loop for Hive (Pocket).

Key features
────────────
✓ KL‑divergence policy loss with soft targets
✓ entropy regulariser to keep the policy spread
✓ Dirichlet noise on the root priors
✓ temperature decay after move 10
✓ replay‑buffer training, checkpointing, resume, skip‑bootstrap
"""

from __future__ import annotations
import argparse, random, pathlib, time
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from HivePocket.HivePocket import HiveGame

# ───────────────────── Board constants ─────────────────────
BOARD_R   = 6                       # covers Pocket board positions
H = W     = BOARD_R * 2 + 1         # 13×13 tensor
PIECE_TYPES = ["Q", "B", "S", "A", "G"]
C         = len(PIECE_TYPES) * 2 + 1   # +1 side‑to‑move plane
shift     = BOARD_R

def to_xy(q: int, r: int) -> tuple[int, int]:
    return r + shift, q + shift

def encode_state(state: dict, perspective: str) -> torch.Tensor:
    t = torch.zeros((C, H, W))
    side = 0 if state["current_player"] == perspective else 1
    for (q, r), stack in state["board"].items():
        if not stack:
            continue
        owner, insect = stack[-1]
        if insect[0] not in PIECE_TYPES:
            continue
        plane = PIECE_TYPES.index(insect[0]) + (0 if owner == perspective else len(PIECE_TYPES))
        y, x = to_xy(q, r)
        if 0 <= y < H and 0 <= x < W:
            t[plane, y, x] = 1.0
    t[-1].fill_(side)
    return t

# ───────────────────────── Network ─────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)

class HiveZeroNet(nn.Module):
    def __init__(self, ch=64, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(C, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU())
        self.res = nn.Sequential(*[ResidualBlock(ch) for _ in range(blocks)])
        self.policy = nn.Sequential(
            nn.Conv2d(ch, 2, 1), nn.BatchNorm2d(2), nn.ReLU(),
            nn.Flatten(), nn.Linear(2 * H * W, H * W * len(PIECE_TYPES)))
        self.value = nn.Sequential(
            nn.Conv2d(ch, 1, 1), nn.BatchNorm2d(1), nn.ReLU(),
            nn.Flatten(), nn.Linear(H * W, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh())
    def forward(self, x):
        x = self.res(self.stem(x))
        return self.policy(x), self.value(x).squeeze(1)

# ───────────────────── Evaluator wrapper ───────────────────
class ZeroEvaluator:
    def __init__(self, net: HiveZeroNet, dev="cpu"):
        self.net = net.to(dev); self.dev = dev
    @torch.no_grad()
    def evaluate(self, state: dict, perspective: str):
        x = encode_state(state, perspective).unsqueeze(0).to(self.dev)
        logits, v = self.net(x)
        return logits.cpu().numpy()[0], v.item()

# ─────────────── Flat action index (845) ───────────────────
AXIAL_TO_IDX, FLAT = {}, []
for q in range(-BOARD_R, BOARD_R + 1):
    for r in range(-BOARD_R, BOARD_R + 1):
        for t in PIECE_TYPES:
            AXIAL_TO_IDX[(q, r, t)] = len(FLAT)
            FLAT.append((q, r, t))
POLICY_SIZE = len(FLAT)

# ───────────────────── Self‑play helpers ───────────────────
def softmax_T(x: np.ndarray, T: float):
    z = np.exp((x - x.max()) / T)
    return z / z.sum()

def mask_illegal(pri, legal):
    mask = np.zeros_like(pri)
    for a in legal:
        try:
            if a[0] == "PLACE":
                _, tp, (q, r) = a
                idx = AXIAL_TO_IDX.get((q, r, tp[0]))
            elif a[0] == "MOVE":
                q, r = a[2]
                idx = AXIAL_TO_IDX.get((q, r, PIECE_TYPES[0]))
            else:                 # PASS
                idx = None
            if idx is not None:
                mask[idx] = 1.0
        except KeyError:
            # off‑board coordinate → just skip
            continue

    if mask.sum() == 0:          # only PASS legal
        return pri * 0
    pri *= mask
    pri /= pri.sum()
    return pri

def flat_to_action(idx, legal):
    q, r, t = FLAT[idx]
    for a in legal:
        if a[0] == "PLACE":
            _, tp, (q0, r0) = a
            if (q0, r0) == (q, r) and tp[0] == t:
                return a
        elif a[0] == "MOVE" and a[2] == (q, r):
            return a
    return random.choice(legal)

ROOT_NOISE_FRAC = 0.25
DIR_ALPHA        = 0.3

def play_one_game(net, T=1.0, max_moves=300):
    game = HiveGame()
    st   = game.getInitialState()
    ev   = ZeroEvaluator(net)
    hist: List[Tuple[dict, np.ndarray, int]] = []
    move_no = 0

    while not game.isTerminal(st) and move_no < max_moves:
        logits, _ = ev.evaluate(st, st["current_player"])
        temp = 0.3 if move_no >= 10 else T
        pri  = softmax_T(logits, temp)

        legal = game.getLegalActions(st)
        pri   = mask_illegal(pri, legal)

        # Dirichlet noise on the very first move
        if move_no == 0:
            pri = (1 - ROOT_NOISE_FRAC) * pri + ROOT_NOISE_FRAC * \
                  np.random.dirichlet(DIR_ALPHA * np.ones_like(pri))

        hist.append((game.copyState(st), pri, 0))

        if pri.sum() == 0:                # only PASS is legal
            act = ("PASS",)
        else:
            pri = pri / pri.sum()         # <── force ∑p = 1 (float64)
            idx = np.random.choice(POLICY_SIZE, p=pri)
            act = flat_to_action(idx, legal)

        st = game.applyAction(st, act)
        move_no += 1

    winner = game.getGameOutcome(st)
    z = 0 if winner == "Draw" else (1 if winner == "Player1" else -1)
    if st["current_player"] == "Player2":   # from P1 perspective
        z = -z

    return [(s, p, z) for s, p, _ in hist]

# ───────────────────── Training helpers ────────────────────
ENT_BETA = 1e-3

def batch_tensors(batch, dev):
    S = torch.stack([encode_state(s, "Player1") for s, _, _ in batch]).to(dev)
    P = torch.tensor(np.array([p for _, p, _ in batch]), dtype=torch.float32, device=dev)
    V = torch.tensor([v for _, _, v in batch], dtype=torch.float32, device=dev)
    return S, P, V

def train_step(net, batch, opt, dev):
    S, P_tgt, V_tgt = batch_tensors(batch, dev)
    logits, V_pred  = net(S)
    logP_pred = F.log_softmax(logits, dim=1)
    P_pred    = logP_pred.exp()
    loss_p = F.kl_div(logP_pred, P_tgt, reduction="batchmean")
    loss_v = F.mse_loss(V_pred.squeeze(), V_tgt)
    entropy = -(P_pred * logP_pred).sum(1).mean()
    loss = loss_p + loss_v - ENT_BETA * entropy
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

# ───────────────────────── Runner ───────────────────────────
def run(args):
    dev = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    net = HiveZeroNet().to(dev)
    if args.resume:
        print("Resuming from", args.resume)
        net.load_state_dict(torch.load(args.resume, map_location=dev))
    opt = optim.Adam(net.parameters(), lr=1e-3)
    buf: deque = deque(maxlen=args.buffer)

    if not args.skip_bootstrap:
        print(f"Bootstrapping {args.games} games …", flush=True)
        for g in range(args.games):
            buf.extend(play_one_game(net, T=args.temp))
            print(f"  game {g+1}/{args.games} → buffer {len(buf)}", flush=True)

    ckdir = pathlib.Path(args.ckpt_dir); ckdir.mkdir(exist_ok=True)
    for ep in range(1, args.epochs + 1):
        buf.extend(play_one_game(net, T=args.temp))
        batch = random.sample(buf, args.batch) if len(buf) >= args.batch else list(buf)
        loss = train_step(net, batch, opt, dev)
        if ep % args.log_every == 0:
            print(f"epoch {ep} | loss {loss:.4f} | buf {len(buf)}", flush=True)
        if ep % args.ckpt_every == 0:
            path = ckdir / f"chkpt_{ep:05d}.pt"
            torch.save(net.state_dict(), path)
            print("saved", path, flush=True)

# ────────────────────── CLI parser ─────────────────────────
def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--buffer", type=int, default=50000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--log-every",  type=int, default=10)
    p.add_argument("--resume",     metavar="PATH", help="checkpoint to load before training")
    p.add_argument("--skip-bootstrap", action="store_true",
                   help="start training immediately (no fresh bootstrap games)")
    return p

# ────────────────────────── main ───────────────────────────
if __name__ == "__main__":
    run(parser().parse_args())
