import pygame

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
