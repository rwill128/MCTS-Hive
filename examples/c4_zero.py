#!/usr/bin/env python3
"""Minimal AlphaGo-style self-play training for Connect Four.

This script implements a small convolutional network and a short
self-play loop so unit tests can run quickly. Training data will be
stored in ``c4_data/`` and network weights in ``c4_weights/`` relative to
the project root. The :class:`ZeroC4Player` class wraps a trained network
for use in ``c4_tournament.py``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch optional
    torch = None
    nn = None
    F = None

HAS_TORCH = torch is not None

from simple_games.connect_four import ConnectFour

# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
BOARD_H = ConnectFour.ROWS
BOARD_W = ConnectFour.COLS


def encode_state(state: dict, perspective: str) -> torch.Tensor:
    """Return a 3×6×7 tensor representing ``state`` from ``perspective``."""
    if not HAS_TORCH:
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
if HAS_TORCH:
    class C4ZeroNet(nn.Module):
        """Tiny policy/value network for Connect Four."""

        def __init__(self, channels: int = 32):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            )
            self.policy = nn.Sequential(
                nn.Flatten(),
                nn.Linear(channels * BOARD_H * BOARD_W, 64),
                nn.ReLU(),
                nn.Linear(64, BOARD_W),
            )
            self.value = nn.Sequential(
                nn.Flatten(),
                nn.Linear(channels * BOARD_H * BOARD_W, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Tanh(),
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            x = self.conv(x)
            return self.policy(x), self.value(x).squeeze(1)
else:  # pragma: no cover - torch not installed
    class C4ZeroNet:
        def __init__(self, *a, **k):
            raise RuntimeError("PyTorch not available")


# ---------------------------------------------------------------------------
# Self-play generation and training
# ---------------------------------------------------------------------------

def play_one_game(net: C4ZeroNet, eps: float = 0.1) -> List[Tuple[torch.Tensor, int, float]]:
    """Play one self-play game and return training tuples."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for play_one_game")
    game = ConnectFour()
    state = game.getInitialState()
    hist: List[Tuple[torch.Tensor, int]] = []
    while not game.isTerminal(state):
        tensor = encode_state(state, state["current_player"])  # type: ignore[arg-type]
        with torch.no_grad():
            logits, _ = net(tensor.unsqueeze(0))
            probs = logits.softmax(1)[0].tolist()
        legal = game.getLegalActions(state)
        masked = [probs[a] if a in legal else 0.0 for a in range(BOARD_W)]
        if sum(masked) == 0:
            action = random.choice(legal)
        else:
            if random.random() < eps:
                action = random.choice(legal)
            else:
                action = max(legal, key=lambda a: masked[a])
        hist.append((tensor, action))
        state = game.applyAction(state, action)

    outcome = game.getGameOutcome(state)
    if outcome == "Draw":
        z = 0.0
    else:
        z = 1.0 if outcome == "X" else -1.0
    data = []
    player = "X"
    for tensor, action in hist:
        reward = z if player == "X" else -z
        data.append((tensor, action, reward))
        player = "O" if player == "X" else "X"
    return data


def train_step(net: C4ZeroNet, batch, opt) -> float:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for train_step")
    states = torch.stack([s for s, _, _ in batch])
    actions = torch.tensor([a for _, a, _ in batch], dtype=torch.long)
    values = torch.tensor([v for _, _, v in batch], dtype=torch.float32)
    logits, v_pred = net(states)
    loss_p = F.cross_entropy(logits, actions)
    loss_v = F.mse_loss(v_pred.squeeze(), values)
    loss = loss_p + loss_v
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.item())


def evaluate_loss(net: C4ZeroNet, data) -> float:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for evaluate_loss")
    with torch.no_grad():
        states = torch.stack([s for s, _, _ in data])
        actions = torch.tensor([a for _, a, _ in data], dtype=torch.long)
        values = torch.tensor([v for _, _, v in data], dtype=torch.float32)
        logits, v_pred = net(states)
        loss_p = F.cross_entropy(logits, actions)
        loss_v = F.mse_loss(v_pred.squeeze(), values)
        return float((loss_p + loss_v).item())


def save_dataset(data, path: Path) -> None:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for save_dataset")
    torch.save([(s.numpy(), a, v) for s, a, v in data], path)


def load_dataset(path: Path):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for load_dataset")
    raw = torch.load(path)
    return [(torch.tensor(s), a, v) for s, a, v in raw]


def save_weights(net: C4ZeroNet, path: Path) -> None:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for save_weights")
    torch.save(net.state_dict(), path)


def load_weights(net: C4ZeroNet, path: Path) -> None:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for load_weights")
    net.load_state_dict(torch.load(path))


# ---------------------------------------------------------------------------
# Network-backed player
# ---------------------------------------------------------------------------
class ZeroC4Player:
    """Connect Four player that selects moves using a trained network."""

    def __init__(self, net: C4ZeroNet, temperature: float = 0.0):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for ZeroC4Player")
        self.net = net
        self.temperature = temperature
        self.game = ConnectFour()

    def search(self, state: dict) -> int:
        tensor = encode_state(state, state["current_player"]).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(tensor)
            probs = logits.softmax(1)[0].tolist()
        legal = self.game.getLegalActions(state)
        masked = [probs[a] if a in legal else 0.0 for a in range(BOARD_W)]
        if sum(masked) == 0:
            return random.choice(legal)
        if self.temperature > 0:
            dist = [p ** (1 / self.temperature) for p in masked]
            s = sum(dist)
            dist = [p / s for p in dist]
            return random.choices(range(BOARD_W), dist)[0]
        return max(legal, key=lambda a: masked[a])


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

DATA_DIR = Path("c4_data")
WEIGHTS_DIR = Path("c4_weights")


def run(
    games: int = 2,
    epochs: int = 10,
    batch: int = 32,
    forever: bool = False,
    data_path: Path = DATA_DIR / "data.pth",
    weights_path: Path = WEIGHTS_DIR / "weights.pth",
) -> None:
    """Run self-play training and persist data/weights."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for training")

    DATA_DIR.mkdir(exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)

    net = C4ZeroNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    buffer = []
    loop = 0
    try:
        while True:
            loop += 1
            for _ in range(games):
                buffer.extend(play_one_game(net))
            print(f"Generated {len(buffer)} samples")
            loss_before = evaluate_loss(net, buffer)
            for _ in range(epochs):
                batch_data = random.sample(buffer, min(len(buffer), batch))
                train_step(net, batch_data, opt)
            loss_after = evaluate_loss(net, buffer)
            print(f"Loop {loop}: Loss {loss_before:.3f} -> {loss_after:.3f}")
            save_dataset(buffer, Path(data_path))
            save_weights(net, Path(weights_path))
            if not forever:
                break
    except KeyboardInterrupt:
        print("Stopping training ...")
    finally:
        save_weights(net, Path(weights_path))


if __name__ == "__main__":
    run()
