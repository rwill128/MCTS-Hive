#!/usr/bin/env python3
"""Toy MuZero-style training loop for Connect Four."""

from __future__ import annotations
import argparse
import math
import os
import random
from collections import deque
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


def encode_state(state: dict, perspective: str) -> torch.Tensor:
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


if torch is not None:
    class MuZeroNet(nn.Module):
        def __init__(self, ch: int = 32):
            super().__init__()
            self.repr = nn.Sequential(
                nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(),
                nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            )
            self.dyn = nn.Sequential(
                nn.Conv2d(ch + BOARD_W, ch, 3, padding=1), nn.ReLU(),
                nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            )
            self.reward = nn.Sequential(
                nn.Flatten(), nn.Linear(ch * BOARD_H * BOARD_W, 1), nn.Tanh()
            )
            self.policy = nn.Sequential(
                nn.Conv2d(ch, 2, 1), nn.ReLU(),
                nn.Flatten(), nn.Linear(2 * BOARD_H * BOARD_W, BOARD_W)
            )
            self.value = nn.Sequential(
                nn.Conv2d(ch, 1, 1), nn.ReLU(),
                nn.Flatten(), nn.Linear(BOARD_H * BOARD_W, 1), nn.Tanh()
            )

        def initial_inference(self, obs: torch.Tensor):
            h = self.repr(obs)
            p = self.policy(h)
            v = self.value(h).squeeze(1)
            return h, p, v

        def recurrent_inference(self, h: torch.Tensor, action: int):
            a = torch.zeros(h.size(0), BOARD_W, 1, 1, device=h.device)
            a[:, action] = 1.0
            a = a.expand(-1, -1, BOARD_H, BOARD_W)
            x = torch.cat([h, a], dim=1)
            h2 = self.dyn(x)
            r = self.reward(h2).squeeze(1)
            p = self.policy(h2)
            v = self.value(h2).squeeze(1)
            return h2, r, p, v
else:  # pragma: no cover - torch not installed
    class MuZeroNet:
        def __init__(self, *a, **k) -> None:
            raise RuntimeError("PyTorch not available")


class Node:
    def __init__(self, prior: float, state: dict, hidden: torch.Tensor | None):
        self.prior = prior
        self.state = state
        self.hidden = hidden
        self.reward = 0.0
        self.value_sum = 0.0
        self.visit_count = 0
        self.children = {}

    def expanded(self) -> bool:
        return bool(self.children)

    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


def select_child(node: Node) -> Tuple[int, Node]:
    pb_c_base = 19652
    pb_c_init = 1.25
    best_score = -float("inf")
    best_action = None
    best_child = None
    for action, child in node.children.items():
        pb_c = math.log((node.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
        pb_c *= math.sqrt(node.visit_count) / (child.visit_count + 1)
        score = child.value() + pb_c * child.prior
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    assert best_child is not None
    return best_action, best_child


def mask_illegal(pri: np.ndarray, legal: List[int]) -> np.ndarray:
    if np is None:
        raise RuntimeError("NumPy is required for mask_illegal")
    mask = np.zeros_like(pri)
    for a in legal:
        mask[a] = 1.0
    pri = pri * mask
    if pri.sum() == 0:
        pri = mask
    pri /= pri.sum()
    return pri


def expand(node: Node, pri: np.ndarray, hidden: torch.Tensor, game: ConnectFour, perspective: str) -> None:
    node.hidden = hidden
    legal = game.getLegalActions(node.state)
    pri = mask_illegal(pri, legal)
    for a in legal:
        node.children[a] = Node(float(pri[a]), game.applyAction(node.state, a), None)


def mcts(game: ConnectFour, net: MuZeroNet, root_state: dict, sims: int, dev: str, perspective: str) -> Tuple[int, np.ndarray]:
    x = encode_state(root_state, perspective).unsqueeze(0).to(dev)
    with torch.no_grad():
        root_h, root_logits, _ = net.initial_inference(x)
    root_pri = F.softmax(root_logits, dim=1)[0].cpu().numpy()
    root = Node(1.0, root_state, root_h)
    expand(root, root_pri, root_h, game, perspective)

    for _ in range(sims):
        node = root
        state = root_state
        hidden = root_h
        search_path = [node]
        value = 0.0
        while node.expanded():
            action, node = select_child(node)
            state = game.applyAction(state, action)
            with torch.no_grad():
                hidden, reward, logits, value = net.recurrent_inference(hidden, action)
            node.reward = float(reward.item())
            if not node.expanded():
                expand(node, F.softmax(logits, dim=1)[0].cpu().numpy(), hidden, game, perspective)
                break
            search_path.append(node)
        if game.isTerminal(state):
            outcome = game.getGameOutcome(state)
            value = 0 if outcome == "Draw" else (1 if outcome == perspective else -1)
        else:
            with torch.no_grad():
                _, _, value = net.initial_inference(encode_state(state, perspective).unsqueeze(0).to(dev))
                value = float(value.item())
        for n in reversed(search_path):
            value = n.reward + value
            n.value_sum += value
            n.visit_count += 1

    visit_counts = np.zeros(game.COLS, dtype=np.float32)
    for a, child in root.children.items():
        visit_counts[a] = child.visit_count

    if visit_counts.sum() > 0:
        probs = visit_counts / visit_counts.sum()
        best_action = int(visit_counts.argmax())
    else:
        probs = visit_counts
        best_action = random.choice(game.getLegalActions(root_state))

    return best_action, probs


ROOT_NOISE_FRAC = 0.25
DIR_ALPHA = 0.3

def play_one_game(net: MuZeroNet, sims: int, dev: str) -> List[Tuple[dict, np.ndarray, int]]:
    if torch is None or np is None:
        raise RuntimeError("PyTorch and NumPy are required for play_one_game")
    game = ConnectFour()
    st = game.getInitialState()
    hist: List[Tuple[dict, np.ndarray, str]] = []
    while not game.isTerminal(st):
        action, pri = mcts(game, net, st, sims, dev, st["current_player"])
        if hist:
            pri = (1 - ROOT_NOISE_FRAC) * pri + ROOT_NOISE_FRAC * np.random.dirichlet(DIR_ALPHA * np.ones_like(pri))
            pri /= pri.sum()
        hist.append((game.copyState(st), pri, st["current_player"]))
        st = game.applyAction(st, action)
    winner = game.getGameOutcome(st)
    out: List[Tuple[dict, np.ndarray, int]] = []
    for s, p, pl in hist:
        if winner == "Draw":
            z = 0
        elif winner == pl:
            z = 1
        else:
            z = -1
        out.append((s, p, z))
    return out


def batch_tensors(batch, dev: str):
    S = torch.stack([encode_state(s, s["current_player"]) for s, _, _ in batch]).to(dev)
    P = torch.tensor(np.array([p for _, p, _ in batch]), dtype=torch.float32, device=dev)
    V = torch.tensor([v for _, _, v in batch], dtype=torch.float32, device=dev)
    return S, P, V


def train_step(net: MuZeroNet, batch, opt, dev: str) -> float:
    S, P_tgt, V_tgt = batch_tensors(batch, dev)
    _, logits, V_pred = net.initial_inference(S)
    logP = F.log_softmax(logits, dim=1)
    loss_p = -(P_tgt * logP).sum(1).mean()
    loss_v = F.mse_loss(V_pred.squeeze(), V_tgt)
    loss = loss_p + loss_v
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.item())


def run(args=None) -> None:
    if args is None:
        args = parser().parse_args()
    if torch is None or np is None:
        raise RuntimeError("PyTorch and NumPy are required for training")
    dev = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    net = MuZeroNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    buf: deque = deque(maxlen=args.buffer)
    for g in range(args.games):
        buf.extend(play_one_game(net, args.sims, dev))
    for ep in range(1, args.epochs + 1):
        buf.extend(play_one_game(net, args.sims, dev))
        batch = random.sample(buf, min(args.batch, len(buf)))
        loss = train_step(net, batch, opt, dev)
        if ep % args.log_every == 0:
            print(f"epoch {ep} | loss {loss:.4f} | buf {len(buf)}", flush=True)
        if ep % args.ckpt_every == 0:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save(net.state_dict(), os.path.join(args.ckpt_dir, f"chkpt_{ep:04d}.pt"))
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(args.ckpt_dir, "last.pt"))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--buffer", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--sims", type=int, default=5, help="MCTS simulations per move")
    p.add_argument("--ckpt-dir", default="muz_ckpts")
    p.add_argument("--ckpt-every", type=int, default=1)
    p.add_argument("--log-every", type=int, default=1)
    return p


if __name__ == "__main__":
    run()
