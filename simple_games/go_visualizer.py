import pygame

CELL_SIZE = 40
MARGIN = 20
BOARD_COLOR = (240, 217, 181)
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)


def init_display(size=5):
    width = (size - 1) * CELL_SIZE + 2 * MARGIN
    height = (size - 1) * CELL_SIZE + 2 * MARGIN
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Go")
    return screen


def draw_board(screen, board):
    size = len(board)
    screen.fill(BOARD_COLOR)
    for i in range(size):
        start = (MARGIN, MARGIN + i * CELL_SIZE)
        end = (MARGIN + (size - 1) * CELL_SIZE, MARGIN + i * CELL_SIZE)
        pygame.draw.line(screen, BLACK_COLOR, start, end, 1)
        start = (MARGIN + i * CELL_SIZE, MARGIN)
        end = (MARGIN + i * CELL_SIZE, MARGIN + (size - 1) * CELL_SIZE)
        pygame.draw.line(screen, BLACK_COLOR, start, end, 1)
    radius = CELL_SIZE // 2 - 2
    for r in range(size):
        for c in range(size):
            stone = board[r][c]
            if stone:
                x = MARGIN + c * CELL_SIZE
                y = MARGIN + r * CELL_SIZE
                color = BLACK_COLOR if stone == "B" else WHITE_COLOR
                pygame.draw.circle(screen, color, (x, y), radius)
    pygame.display.flip()
