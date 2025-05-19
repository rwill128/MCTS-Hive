#!/usr/bin/env python3
"""Interactive Connect Four debugging GUI.

This tool lets you edit a board position, choose the side to move,
provide MCTS or minimax parameters as JSON and then run a search.
Instead of applying the best move it simply reports the column the
agent chose. While MCTS or minimax is running a heatmap overlay shows
the current value estimate for each column and updates as the search
progresses.
"""

import json
import random
import threading
import tkinter as tk
from pathlib import Path
import pygame

from simple_games.connect_four import ConnectFour
from simple_games.minimax_connect_four import MinimaxConnectFourPlayer
from simple_games.c4_visualizer import (
    draw_board,
    draw_board_with_action_values,
    highlight_move,
    CELL_SIZE,
    MARGIN,
)
from mcts.Mcts import MCTS
try:
    from examples.c4_zero import (
        ZeroC4Player,
        C4ZeroNet,
        load_weights,
        encode_state,
        HAS_TORCH,
    )
    if HAS_TORCH:
        import torch
except Exception:  # pragma: no cover - optional dependency missing
    HAS_TORCH = False

UI_HEIGHT = 60


class ZeroGuidedMCTS(MCTS):
    """MCTS variant that uses a Zero network to guide simulations."""

    def __init__(self, game: ConnectFour, net: C4ZeroNet, perspective_player: str, temperature: float = 0.0, **kwargs):
        super().__init__(game=game, perspective_player=perspective_player, **kwargs)
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for ZeroGuidedMCTS")
        self.net = net
        self.temperature = temperature

    def _guided_action(self, state: dict) -> int:
        tensor = encode_state(state, state["current_player"]).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(tensor)
            probs = logits.softmax(1)[0].tolist()
        legal = self.game.getLegalActions(state)
        masked = [probs[a] if a in legal else 0.0 for a in range(self.game.COLS)]
        if sum(masked) == 0:
            return random.choice(legal)
        if self.temperature > 0:
            dist = [p ** (1.0 / self.temperature) for p in masked]
            s = sum(dist)
            dist = [p / s for p in dist]
            return random.choices(range(self.game.COLS), dist)[0]
        return max(legal, key=lambda a: masked[a])

    def _simulate(self, state: dict) -> float:
        temp_state = self.game.copyState(state)
        depth = 0
        while not self.game.isTerminal(temp_state) and depth < self.max_depth:
            action = self._guided_action(temp_state)
            temp_state = self.game.applyAction(temp_state, action)
            depth += 1
        outcome = self.game.getGameOutcome(temp_state)
        if outcome == self.perspective_player:
            return 1.0
        if outcome == "Draw" or outcome is None:
            return 0.0
        return -1.0


def board_pos_from_pixel(pos, rows, cols):
    """Return (row, col) or None for screen pixel *pos*."""
    x, y = pos
    y -= UI_HEIGHT
    if not (MARGIN <= x < MARGIN + cols * CELL_SIZE and
            MARGIN <= y < MARGIN + rows * CELL_SIZE):
        return None
    col = (x - MARGIN) // CELL_SIZE
    row_from_top = (y - MARGIN) // CELL_SIZE
    row = rows - 1 - row_from_top
    return int(row), int(col)


def cycle_cell(board, row, col):
    val = board[row][col]
    if val is None:
        board[row][col] = "X"
    elif val == "X":
        board[row][col] = "O"
    else:
        board[row][col] = None


def run_search(game, state, cfg_json, screen, board_surface):
    """Run an MCTS or minimax search from *state* using *cfg_json*."""
    try:
        cfg = json.loads(cfg_json)
    except json.JSONDecodeError:
        print("Invalid JSON config")
        return None

    if cfg.get("type") == "minimax":
        depth = int(cfg.get("depth", 4))
        player = MinimaxConnectFourPlayer(game, state["current_player"], depth)

        def cb(values):
            draw_board_with_action_values(
                board_surface,
                state["board"],
                {a: (v, 1) for a, v in values.items()},
                None,
            )
            screen.blit(board_surface, (0, UI_HEIGHT))
            pygame.display.flip()
            # Keep the window responsive and make updates visible
            pygame.event.pump()
            pygame.time.delay(50)

        move = player.search(state, value_callback=cb)
        return move
    elif cfg.get("type") == "zero":
        if not HAS_TORCH:
            print("PyTorch not available")
            return None
        net = C4ZeroNet()
        load_weights(net, Path(cfg["weights"]))
        temp = float(cfg.get("temperature", 0.0))
        player = ZeroC4Player(net, temperature=temp)

        tensor = encode_state(state, state["current_player"]).unsqueeze(0)
        with torch.no_grad():
            logits, _ = net(tensor)
            probs = logits.softmax(1)[0].tolist()
        values = {i: (p, 1) for i, p in enumerate(probs)}
        draw_board_with_action_values(
            board_surface,
            state["board"],
            values,
            None,
        )
        screen.blit(board_surface, (0, UI_HEIGHT))
        pygame.display.flip()

        move = player.search(state)
        return move
    elif cfg.get("type") == "mcts_zero":
        if not HAS_TORCH:
            print("PyTorch not available")
            return None
        net = C4ZeroNet()
        load_weights(net, Path(cfg["weights"]))
        temp = float(cfg.get("temperature", 0.0))
        params = {k: v for k, v in cfg.items() if k not in {"type", "weights", "temperature"}}
        mcts = ZeroGuidedMCTS(
            game=game,
            net=net,
            perspective_player=state["current_player"],
            temperature=temp,
            **params,
        )

        def draw_cb(root, iter_count):
            values = {
                a: (child.average_value(), child.visit_count)
                for a, child in root.children.items()
            }
            draw_board_with_action_values(
                board_surface,
                root.state["board"],
                values,
                iter_count,
            )
            screen.blit(board_surface, (0, UI_HEIGHT))
            pygame.display.flip()

        move = mcts.search(state, draw_callback=draw_cb)
        return move
    else:
        cfg.setdefault("perspective_player", state["current_player"])
        mcts = MCTS(game=game, cache=None, **cfg)

        def draw_cb(root, iter_count):
            values = {
                a: (child.average_value(), child.visit_count)
                for a, child in root.children.items()
            }
            draw_board_with_action_values(
                board_surface,
                root.state["board"],
                values,
                iter_count,
            )
            screen.blit(board_surface, (0, UI_HEIGHT))
            pygame.display.flip()

        move = mcts.search(state, draw_callback=draw_cb)
        return move


def main() -> None:
    game = ConnectFour()
    state = game.getInitialState()

    width = game.COLS * CELL_SIZE + 2 * MARGIN
    height = game.ROWS * CELL_SIZE + 2 * MARGIN + UI_HEIGHT
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("C4 Debugger")

    board_surface = pygame.Surface((width, height - UI_HEIGHT))
    font = pygame.font.SysFont(None, 24)
    json_holder = {"text": '{"num_iterations": 200, "max_depth": 42, "c_param": 1.4}'}
    run_flag = {"go": False}
    selected_move = None

    def request_run():
        run_flag["go"] = True

    def start_editor():
        """Launch the Tk window in a dedicated thread."""

        def tk_thread():
            root = tk.Tk()
            root.title("Search Config")
            text = tk.Text(root, width=50, height=5)
            text.insert("1.0", json_holder["text"])
            text.pack()

            def update(_=None):
                json_holder["text"] = text.get("1.0", tk.END).strip()

            text.bind("<KeyRelease>", update)
            tk.Button(
                root,
                text="Run Search",
                command=lambda: [update(), request_run()],
            ).pack()
            root.mainloop()

        threading.Thread(target=tk_thread, daemon=True).start()

    start_editor()

    clock = pygame.time.Clock()
    draw_board(board_surface, state["board"])

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.pos[1] < UI_HEIGHT:
                    if 10 <= event.pos[0] <= 90:
                        state["current_player"] = (
                            "O" if state["current_player"] == "X" else "X"
                        )
                    elif width - 90 <= event.pos[0] <= width - 10:
                        request_run()
                else:
                    rc = board_pos_from_pixel(event.pos, game.ROWS, game.COLS)
                    if rc:
                        cycle_cell(state["board"], *rc)
                        draw_board(board_surface, state["board"])

        if run_flag["go"]:
            run_flag["go"] = False
            selected_move = run_search(
                game,
                game.copyState(state),
                json_holder["text"],
                screen,
                board_surface,
            )
            highlight_move(board_surface, state["board"], selected_move)

        screen.fill((200, 200, 200))
        screen.blit(board_surface, (0, UI_HEIGHT))

        # UI panel
        pygame.draw.rect(screen, (230, 230, 230), (0, 0, width, UI_HEIGHT))
        turn_label = font.render(f"Turn: {state['current_player']}", True, (0, 0, 0))
        screen.blit(turn_label, (15, 20))
        info = font.render("Edit JSON in TK window", True, (0, 0, 0))
        screen.blit(info, (120, 20))
        pygame.draw.rect(screen, (180, 180, 180), (width - 80, 15, 70, 30))
        run_lbl = font.render("Run", True, (0, 0, 0))
        screen.blit(run_lbl, (width - 65, 20))

        if selected_move is not None:
            msg = font.render(f"Best move: {selected_move}", True, (0, 0, 0))
            screen.blit(msg, (10, height - 30))

        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()
