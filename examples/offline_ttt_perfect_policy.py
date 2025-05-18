#!/usr/bin/env python3
"""Precompute the perfect Tic-Tac-Toe policy offline.

This script exhaustively enumerates every reachable Tic-Tac-Toe state.
For each position it computes the minimax value and records the optimal
moves for the side to play. The resulting dictionary is saved to
``ttt_perfect_policy.json`` and can be used by any agent that needs a
lookup table of perfect play.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, List, Tuple

from simple_games.tic_tac_toe import TicTacToe


GameState = Tuple[Tuple[Tuple[str | None, ...], ...], str]

game = TicTacToe()

def serialize(state: dict) -> GameState:
    board = tuple(tuple(cell for cell in row) for row in state["board"])
    return board, state["current_player"]

@lru_cache(maxsize=None)
def minimax(board_serialized: GameState, to_move: str) -> int:
    board, _ = board_serialized
    state = {"board": [list(row) for row in board], "current_player": to_move}
    outcome = game.getGameOutcome(state)
    if outcome == "X":
        return 1
    if outcome == "O":
        return -1
    if outcome == "Draw":
        return 0
    scores = []
    for action in game.getLegalActions(state):
        ns = game.applyAction(state, action)
        score = minimax(serialize(ns), ns["current_player"])
        scores.append(score)
    return max(scores) if to_move == "X" else min(scores)

def build_policy(state: dict, policy: Dict[GameState, List[Tuple[int, int]]]) -> None:
    key = serialize(state)
    if key in policy:
        return
    if game.isTerminal(state):
        policy[key] = []
        return
    action_scores = []
    for action in game.getLegalActions(state):
        ns = game.applyAction(state, action)
        score = minimax(serialize(ns), ns["current_player"])
        if state["current_player"] == "O":
            score = -score
        action_scores.append((score, action))
        build_policy(ns, policy)
    best = max(s for s, _ in action_scores)
    best_actions = [a for s, a in action_scores if s == best]
    policy[key] = best_actions

def main() -> None:
    policy: Dict[GameState, List[Tuple[int, int]]] = {}
    initial = game.getInitialState()
    build_policy(initial, policy)
    with open("ttt_perfect_policy.json", "w") as f:
        json.dump({str(k): v for k, v in policy.items()}, f)
    print(f"Enumerated {len(policy)} states.")

if __name__ == "__main__":
    main()
