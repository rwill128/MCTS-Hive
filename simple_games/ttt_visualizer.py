import pygame

CELL_SIZE = 80
MARGIN = 10
BACKGROUND_COLOR = (30, 30, 30)
LINE_COLOR = (200, 200, 200)
X_COLOR = (255, 0, 0)
O_COLOR = (0, 0, 255)
LINE_WIDTH = 5


def init_display(rows: int = 3, cols: int = 3):
    width = cols * CELL_SIZE + 2 * MARGIN
    height = rows * CELL_SIZE + 2 * MARGIN
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Tic Tac Toe")
    return screen


def draw_board(screen, board):
    rows = len(board)
    cols = len(board[0]) if rows else 0
    screen.fill(BACKGROUND_COLOR)

    # grid lines
    for i in range(1, rows):
        y = MARGIN + i * CELL_SIZE
        pygame.draw.line(screen, LINE_COLOR, (MARGIN, y), (MARGIN + cols * CELL_SIZE, y), LINE_WIDTH)
    for i in range(1, cols):
        x = MARGIN + i * CELL_SIZE
        pygame.draw.line(screen, LINE_COLOR, (x, MARGIN), (x, MARGIN + rows * CELL_SIZE), LINE_WIDTH)

    # pieces
    for r in range(rows):
        for c in range(cols):
            piece = board[r][c]
            cx = MARGIN + c * CELL_SIZE + CELL_SIZE // 2
            cy = MARGIN + r * CELL_SIZE + CELL_SIZE // 2
            if piece == "X":
                off = CELL_SIZE // 3
                pygame.draw.line(screen, X_COLOR, (cx - off, cy - off), (cx + off, cy + off), LINE_WIDTH)
                pygame.draw.line(screen, X_COLOR, (cx + off, cy - off), (cx - off, cy + off), LINE_WIDTH)
            elif piece == "O":
                pygame.draw.circle(screen, O_COLOR, (cx, cy), CELL_SIZE // 3, LINE_WIDTH)

    pygame.display.flip()
