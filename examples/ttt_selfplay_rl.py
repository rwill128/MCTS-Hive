#!/usr/bin/env python3
"""Minimal self-play RL for Tic-Tac-Toe with Tkinter visualization.

This script demonstrates a tiny AlphaZero-style training loop without
neural networks. It maintains a value table for every board state and
updates it through self-play episodes. The Tkinter UI lets you step
through each update with play, pause, back and forward controls.
"""

import json
import os
import random
import time
import tkinter as tk
from ast import literal_eval
from tkinter import ttk
from typing import Dict, List, Tuple

from simple_games.tic_tac_toe import TicTacToe


class RLAgent:
    """Simplistic value-based agent for Tic-Tac-Toe with persistence."""

    def __init__(self, lr: float = 0.1, epsilon: float = 0.1,
                 storage_path: str = "ttt_rl_values.json"):
        self.game = TicTacToe()
        self.lr = lr
        self.epsilon = epsilon
        self.storage_path = storage_path
        self.values: Dict[Tuple, float] = {}
        self._load_values()

    def _load_values(self) -> None:
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                raw = json.load(f)
            self.values = {literal_eval(k): v for k, v in raw.items()}
        else:
            self.values = {}

    def save_values(self) -> None:
        with open(self.storage_path, "w") as f:
            json.dump({str(k): v for k, v in self.values.items()}, f)

    def reset(self) -> None:
        self.values.clear()
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)

    def _key(self, state: dict) -> Tuple:
        board = tuple(tuple(cell for cell in row) for row in state["board"])
        return board, state["current_player"]

    def move_values(self, state: dict) -> Dict[Tuple[int, int], float]:
        """Return value estimate for each legal action from this state."""
        actions = self.game.getLegalActions(state)
        values: Dict[Tuple[int, int], float] = {}
        for a in actions:
            ns = self.game.applyAction(state, a)
            key = self._key(ns)
            val = self.values.get(key, 0.0)
            if state["current_player"] == "O":
                val = -val
            values[a] = val
        return values

    def _policy_action(self, state: dict):
        actions = self.game.getLegalActions(state)
        if random.random() < self.epsilon:
            return random.choice(actions)
        vals = []
        for a in actions:
            ns = self.game.applyAction(state, a)
            key = self._key(ns)
            val = self.values.get(key, 0.0)
            if state["current_player"] == "O":
                val = -val
            vals.append(val)
        best = max(vals)
        return actions[vals.index(best)]

    def play_episode(self) -> Tuple[List[Tuple[Tuple, float, float]], str]:
        state = self.game.getInitialState()
        hist: List[Tuple] = []
        while not self.game.isTerminal(state):
            hist.append(self._key(state))
            action = self._policy_action(state)
            state = self.game.applyAction(state, action)
        outcome = self.game.getGameOutcome(state)
        reward = 1 if outcome == "X" else -1 if outcome == "O" else 0
        updates: List[Tuple[Tuple, float, float]] = []
        for key in reversed(hist):
            old = self.values.get(key, 0.0)
            new = old + self.lr * (reward - old)
            self.values[key] = new
            updates.append((key, old, new))
            reward = -reward
        updates.reverse()
        self.save_values()
        return updates, outcome


class TrainerUI:
    """Tkinter front-end to visualise training updates."""

    def __init__(self, agent: RLAgent, episodes: int = 50):
        self.agent = agent
        self.episodes = episodes
        self.steps: List[Tuple[Tuple, float, float]] = []
        self.step_idx = -1
        self.playing = False
        self.step_delay = 500
        self.fast_mode = False
        self.root = tk.Tk()
        self.root.title("TicTacToe RL Trainer")

        # full screen window
        try:
            self.root.state("zoomed")  # type: ignore[attr-defined]
        except tk.TclError:
            self.root.attributes("-zoomed", True)

        screen = min(self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        self.board_size = int(screen * 0.6)

        self.left_text = tk.Text(self.root, width=40, height=20)
        self.left_text.grid(row=0, column=0, padx=5, sticky="n")
        algo_msg = (
            "Algorithm: Temporal-Difference value learning\n"
            "We store a value V(s) for each board state from X's perspective.\n"
            "After each episode the final reward is backed up through all\n"
            "visited states using:\n"
            "    V(s) <- V(s) + lr * (R - V(s))\n"
            "The sign of R flips after every move because players alternate."
        )
        self.left_text.insert(tk.END, algo_msg)
        self.left_text.config(state=tk.DISABLED)

        self.canvas = tk.Canvas(self.root, width=self.board_size, height=self.board_size, bg="white")
        self.canvas.grid(row=0, column=1, pady=5)

        self.right_text = tk.Text(self.root, width=40, height=20)
        self.right_text.grid(row=0, column=2, padx=5, sticky="n")
        self.right_text.insert(tk.END, "Step explanation will appear here.")
        self.right_text.config(state=tk.DISABLED)

        self.table = tk.Text(self.root, width=60, height=10)
        self.table.grid(row=1, column=0, columnspan=3)

        self.prev_btn = ttk.Button(self.root, text="<<", command=self.prev_step)
        self.play_btn = ttk.Button(self.root, text="Play", command=self.toggle_play)
        self.next_btn = ttk.Button(self.root, text=">>", command=self.next_step)
        self.speed_btn = ttk.Button(self.root, text="Fast", command=self.toggle_speed)
        self.clear_btn = ttk.Button(self.root, text="Clear", command=self.clear_values)
        self.prev_btn.grid(row=2, column=0)
        self.play_btn.grid(row=2, column=1)
        self.next_btn.grid(row=2, column=2)
        self.speed_btn.grid(row=2, column=3)
        self.clear_btn.grid(row=2, column=4)

        self._generate_updates(self.episodes)

    def _generate_updates(self, episodes: int):
        for _ in range(episodes):
            upd, _ = self.agent.play_episode()
            self.steps.extend(upd)

    def _draw_board(self, key: Tuple):
        board, player = key
        self.canvas.delete("all")
        size = self.board_size // 3
        off = (self.board_size - 3 * size) // 2

        state = {"board": [list(row) for row in board], "current_player": player}
        move_vals = self.agent.move_values(state)

        def val_color(v: float) -> str:
            v = max(-1.0, min(1.0, v))
            r = int(max(0, -v) * 255)
            g = int(max(0, v) * 255)
            return f"#{r:02x}{g:02x}00"

        for r in range(3):
            for c in range(3):
                x1 = off + c * size
                y1 = off + r * size
                x2 = x1 + size
                y2 = y1 + size
                piece = board[r][c]
                if piece is None and (r, c) in move_vals:
                    color = val_color(move_vals[(r, c)])
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, width=0)
                    self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2,
                                            text=f"{move_vals[(r,c)]:.2f}")

        for i in range(4):
            self.canvas.create_line(off, off + i * size, off + 3 * size, off + i * size)
            self.canvas.create_line(off + i * size, off, off + i * size, off + 3 * size)

        for r in range(3):
            for c in range(3):
                piece = board[r][c]
                if piece is None:
                    continue
                x = off + c * size + size / 2
                y = off + r * size + size / 2
                if piece == "X":
                    self.canvas.create_line(x - size * 0.3, y - size * 0.3,
                                            x + size * 0.3, y + size * 0.3,
                                            width=3, fill="red")
                    self.canvas.create_line(x + size * 0.3, y - size * 0.3,
                                            x - size * 0.3, y + size * 0.3,
                                            width=3, fill="red")
                elif piece == "O":
                    self.canvas.create_oval(x - size * 0.3, y - size * 0.3,
                                            x + size * 0.3, y + size * 0.3,
                                            width=3, outline="blue")

    def _update_table(self):
        self.table.delete("1.0", tk.END)
        for idx, (k, v) in enumerate(list(self.agent.values.items())[:10], 1):
            self.table.insert(tk.END, f"{idx}. {k} -> {v:.2f}\n")

    def _animate_value_change(self, key: Tuple, old: float, new: float) -> None:
        steps = 10 if not self.fast_mode else 5
        diff = new - old
        for i in range(steps + 1):
            cur = old + diff * i / steps
            self.agent.values[key] = cur
            self._draw_board(key)
            self._update_table()
            self.right_text.config(state=tk.NORMAL)
            self.right_text.delete("1.0", tk.END)
            self.right_text.insert(
                tk.END,
                "V <- V + lr * (R - V)\n"
                f"old={old:.2f} target={new:.2f} current={cur:.2f}"
            )
            self.right_text.config(state=tk.DISABLED)
            self.root.update()
            time.sleep(self.step_delay / 1000 / steps)

    def _apply_step(self, idx: int, forward: bool):
        key, old, new = self.steps[idx]
        if forward:
            self._animate_value_change(key, old, new)
        else:
            self.agent.values[key] = old
            self._draw_board(key)
            self._update_table()

    def next_step(self):
        if self.step_idx + 1 >= len(self.steps):
            return
        self.step_idx += 1
        self._apply_step(self.step_idx, True)

    def prev_step(self):
        if self.step_idx < 0:
            return
        self._apply_step(self.step_idx, False)
        self.step_idx -= 1

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.config(text="Pause" if self.playing else "Play")
        if self.playing:
            self._auto_step()

    def toggle_speed(self):
        self.fast_mode = not self.fast_mode
        self.step_delay = 100 if self.fast_mode else 500
        self.speed_btn.config(text="Slow" if self.fast_mode else "Fast")

    def clear_values(self):
        self.agent.reset()
        self.steps = []
        self.step_idx = -1
        self.table.delete("1.0", tk.END)
        self.canvas.delete("all")
        self._generate_updates(self.episodes)
        if self.steps:
            self.next_step()

    def _auto_step(self):
        if not self.playing:
            return
        if self.step_idx + 1 >= len(self.steps):
            # stop playback when no more steps remain
            self.playing = False
            self.play_btn.config(text="Play")
            return
        self.next_step()
        self.root.after(0, self._auto_step)

    def run(self):
        if self.steps:
            self.next_step()
        self.root.mainloop()


if __name__ == "__main__":
    agent = RLAgent(lr=0.2, epsilon=0.2)
    ui = TrainerUI(agent, episodes=50)
    ui.run()
