import pygame

from typing import Dict, Tuple

CELL_SIZE = 80
MARGIN = 10
BACKGROUND_COLOR = (0, 0, 255)
EMPTY_COLOR = (255, 255, 255)
X_COLOR = (255, 0, 0)
O_COLOR = (255, 255, 0)

def init_display(rows=6, cols=7):
    width = cols * CELL_SIZE + 2 * MARGIN
    height = rows * CELL_SIZE + 2 * MARGIN
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Connect Four")
    return screen

def draw_board(screen, board):
    rows = len(board)
    cols = len(board[0]) if rows else 0
    screen.fill(BACKGROUND_COLOR)
    for c in range(cols):
        for r in range(rows):
            x = MARGIN + c * CELL_SIZE + CELL_SIZE // 2
            y = MARGIN + (rows - 1 - r) * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(screen, (0, 0, 0), (x, y), CELL_SIZE // 2 - 2)
            piece = board[r][c]
            if piece == "X":
                color = X_COLOR
            elif piece == "O":
                color = O_COLOR
            else:
                color = EMPTY_COLOR
            pygame.draw.circle(screen, color, (x, y), CELL_SIZE // 2 - 6)
    pygame.display.flip()


def _value_to_color(value: float) -> Tuple[int, int, int]:
    """Map a value in [-1, 1] to an RGB color on a red-green gradient."""
    v = max(-1.0, min(1.0, value))
    norm = (v + 1.0) / 2.0
    if norm > 0.5:
        red = int(255 * (1 - norm) * 2)
        green = 255
    else:
        red = 255
        green = int(255 * norm * 2)
    return red, green, 0


def draw_board_with_action_values(
    screen,
    board,
    values: Dict[int, Tuple[float, int]],
    iteration: int | None = None,
) -> None:
    """Draw the board and overlay a heatmap showing per-column values."""
    draw_board(screen, board)
    if not values:
        return
    rows = len(board)
    cols = len(board[0]) if rows else 0
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    max_visits = max(v[1] for v in values.values()) if values else 1
    for col in range(cols):
        if col not in values:
            continue
        val, visits = values[col]
        # find landing row
        row = None
        for r in range(rows):
            if board[r][col] is None:
                row = r
                break
        if row is None:
            continue
        x = MARGIN + col * CELL_SIZE + CELL_SIZE // 2
        y = MARGIN + (rows - 1 - row) * CELL_SIZE + CELL_SIZE // 2
        base = _value_to_color(val)
        alpha = int(50 + 205 * (visits / max_visits))
        pygame.draw.circle(overlay, (*base, alpha), (x, y), CELL_SIZE // 2 - 6)
    screen.blit(overlay, (0, 0))
    if iteration is not None:
        font = pygame.font.SysFont(None, 24)
        txt = font.render(f"Iter: {iteration}", True, (0, 0, 0))
        screen.blit(txt, (10, 10))
    pygame.display.flip()


def highlight_move(screen, board, move: int) -> None:
    """Draw the board and highlight the landing spot for *move*."""
    draw_board(screen, board)
    if move is None:
        pygame.display.flip()
        return
    rows = len(board)
    if rows == 0 or move < 0 or move >= len(board[0]):
        pygame.display.flip()
        return
    landing_row = None
    for r in range(rows):
        if board[r][move] is None:
            landing_row = r
            break
    if landing_row is None:
        landing_row = rows - 1
    x = MARGIN + move * CELL_SIZE + CELL_SIZE // 2
    y = MARGIN + (rows - 1 - landing_row) * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.circle(screen, (0, 255, 0), (x, y), CELL_SIZE // 2 - 4, 5)
    pygame.display.flip()
