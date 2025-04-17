"""Minimal Alpha‑Zero style scaffold for Hive Pocket.
-------------------------------------------------------------------
 – Board encoder → (C,H,W) tensor
 – Small residual CNN with policy + value heads
 – Integrates with existing MCTS via ZeroEvaluator

 This is NOT a full training script; it gives you runnable stubs you can
 extend: play_self_games() produces (state, π, z) tuples; train() runs a
 single epoch of SGD.  You can iterate without touching your game logic.
"""

from __future__ import annotations

import random
import math
import numpy as np
from collections import deque
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from HivePocket.HivePocket import HiveGame, find_queen_position, hex_distance

# ------------------------------------------------------------
# Constants – board size (axial coords) and piece indexing
# ------------------------------------------------------------
BOARD_R = 6   # half‑diameter we will embed (covers Hive Pocket)
H = W = BOARD_R * 2 + 1   # 13×13 tensor

PIECE_TYPES = ["Q", "B", "S", "A", "G"]  # queen beetle spider ant grasshopper

# plane order: [my_Q,B,S,A,G,  opp_Q,B,S,A,G,  side_to_move]
C = len(PIECE_TYPES) * 2 + 1

# Helper to map axial -> matrix indices (shift to positive)
shift = BOARD_R

def to_xy(q, r):
    return r + shift, q + shift  # row,col

# ------------------------------------------------------------
# Encoder -----------------------------------------------------
# ------------------------------------------------------------

def encode_state(state: dict, perspective: str) -> torch.Tensor:
    """Return float tensor (C,H,W) with 0/1 occupancy."""
    board = state["board"]
    side  = 0 if state["current_player"] == perspective else 1
    tensor = torch.zeros((C, H, W), dtype=torch.float32)

    for (q, r), stack in board.items():
        if not stack:
            continue
        owner, insect = stack[-1]
        try:
            idx = PIECE_TYPES.index(insect[0])
        except ValueError:
            continue
        plane = idx if owner == perspective else idx + len(PIECE_TYPES)
        y, x = to_xy(q, r)
        if 0 <= y < H and 0 <= x < W:
            tensor[plane, y, x] = 1.0
    tensor[-1].fill_(side)  # side‑to‑move plane
    return tensor

# ------------------------------------------------------------
# Model -------------------------------------------------------
# ------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)

class HiveZeroNet(nn.Module):
    """6‑layer residual CNN – ~200 K params."""
    def __init__(self, channels=64, blocks=4):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(C, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU())
        self.res = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        # policy head
        self.policy = nn.Sequential(
            nn.Conv2d(channels, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * H * W, H * W * len(PIECE_TYPES)))
        # value head
        self.value = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(H * W, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh())

    def forward(self, x):
        x = self.input(x)
        x = self.res(x)
        p = self.policy(x)
        v = self.value(x).squeeze(1)
        return p, v

# ------------------------------------------------------------
# Move‑index mapping  (axial q,r, piece_type)  ->  flat index
# ------------------------------------------------------------

AXIAL_TO_INDEX = {}
FLAT = []
for q in range(-BOARD_R, BOARD_R + 1):
    for r in range(-BOARD_R, BOARD_R + 1):
        # we include the full bounding box (13×13) so policy head and index map match
        for t in PIECE_TYPES:
            AXIAL_TO_INDEX[(q, r, t)] = len(FLAT)
            FLAT.append((q, r, t))

POLICY_SIZE = len(FLAT)  # now 845 = 13*13*5

# ------------------------------------------------------------
# Zero‑style evaluator plugged into your MCTS ----------------
# ------------------------------------------------------------

class ZeroEvaluator:
    def __init__(self, model: HiveZeroNet, device="cpu"):
        self.model = model.to(device)
        self.dev = device
        self.game = HiveGame()

    @torch.no_grad()
    def evaluate(self, state: dict, perspective: str):
        """Return (policy_logits, value) as numpy arrays."""
        x = encode_state(state, perspective).unsqueeze(0).to(self.dev)
        logits, value = self.model(x)
        return logits.cpu().numpy()[0], value.item()

# ------------------------------------------------------------
# Self‑play skeleton -----------------------------------------
# ------------------------------------------------------------
def softmax_temperature(logits, T=1.0):
    z = np.exp((logits - logits.max()) / T)
    return z / z.sum()


def pick_action_from_flat(policy: np.ndarray, legal, q, r, typ):
    """Return first legal action matching the flat index; fallback random."""
    for a in legal:
        if a[0] == "PLACE":
            _, tp, (aq, ar) = a
            if (aq, ar) == (q, r) and tp[0] == typ:
                return a
        elif a[0] == "MOVE":
            dest = a[2] if len(a) == 3 else a[2]
            if dest == (q, r):
                return a
    return random.choice(legal)


def play_one_game(model: HiveZeroNet, self_play_T=1.0, max_moves=300):
    game   = HiveGame()
    state  = game.getInitialState()
    evalzr = ZeroEvaluator(model)
    history: List[Tuple[dict, np.ndarray]] = []

    while not game.isTerminal(state) and len(history) < max_moves:
        logits, _ = evalzr.evaluate(state, state["current_player"])
        priors    = softmax_temperature(logits, T=self_play_T)

        # --- mask illegal moves ----------------------------------
        mask  = np.zeros(POLICY_SIZE, dtype=np.float32)
        legal = game.getLegalActions(state)
        for a in legal:
            if a[0] == "PLACE":
                _, tp, (q0, r0) = a
                idx = AXIAL_TO_INDEX.get((q0, r0, tp[0]))
            else:                          # MOVE
                dest = a[2] if len(a) == 3 else a[2]
                q0, r0 = dest
                idx = AXIAL_TO_INDEX.get((q0, r0, PIECE_TYPES[0]))  # placeholder type
            if idx is not None:
                mask[idx] = 1.0
        priors = priors * mask
        if priors.sum() == 0:
            priors = mask / mask.sum()
        priors /= priors.sum()
        history.append((game.copyState(state), priors))

        # sample flat index and map back to action ----------------
        flat_idx = np.random.choice(POLICY_SIZE, p=priors)
        q, r, tchar = FLAT[flat_idx]
        action = pick_action_from_flat(priors, legal, q, r, tchar)

        state = game.applyAction(state, action)

    winner = game.getGameOutcome(state)
    z = 0 if winner == "Draw" else (+1 if winner == "Player1" else -1)
    if state["current_player"] == "Player2":
        z = -z
    return [(s, p, z) for s, p in history]

# ------------------------------------------------------------
# Train loop (unchanged except optimiser line fixed) ----------
# ------------------------------------------------------------

def train(model: HiveZeroNet, data, batch_size=32, epochs=1, device="cpu"):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        random.shuffle(data)
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            states = torch.stack([encode_state(s, "Player1") for s,_,_ in batch]).to(device)
            t_p = torch.tensor([p for _,p,_ in batch], dtype=torch.float32).to(device)
            t_v = torch.tensor([v for _,_,v in batch], dtype=torch.float32).to(device)
            logits, v_pred = model(states)
            loss_p = F.cross_entropy(logits, t_p.argmax(1))
            loss_v = F.mse_loss(v_pred.squeeze(), t_v)
            loss = loss_p + loss_v
            opt.zero_grad(); loss.backward(); opt.step()

# ------------------------------------------------------------
# Demo -------------------------------------------------------
# ------------------------------------------------------------
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = HiveZeroNet().to(dev)
    data = play_one_game(net)
    print("collected", len(data), "samples")
    train(net, data, device=dev)
    print("OK – network updated once")
