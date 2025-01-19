import pygame
import sys
import math

from HivePocket.HivePocket import HiveGame


def axial_to_pixel(q, r, hex_size):
    """
    Convert axial coords (q,r) to pixel (x,y).
    This version uses a pointy-top layout:

         / \
        |   |
         \ /

    For a hex of size `hex_size`.
    """
    # For a pointy top, one common axial->pixel is:
    x = hex_size * math.sqrt(3) * (q + r/2.0)
    y = hex_size * (3.0/2.0) * r
    return (x, y)

def hex_corners(x, y, size):
    """
    Return the six corners of a regular hex
    with center at (x, y) and 'radius' = size.
    Pointy-top hex. The corners go around in a circle.
    """
    corners = []
    for i in range(6):
        angle_deg = 60 * i - 30  # -30° to make the top corner pointy
        angle_rad = math.radians(angle_deg)
        cx = x + size * math.cos(angle_rad)
        cy = y + size * math.sin(angle_rad)
        corners.append((cx, cy))
    return corners

def draw_hex(surface, x, y, size, color=(200, 200, 200), width=2):
    """
    Draw a hex outline (or filled hex) on surface.
    `width=0` => filled hex. Nonzero => just outline.
    """
    corners = hex_corners(x, y, size)
    pygame.draw.polygon(surface, color, corners, width)

def draw_piece_label(surface, x, y, piece, font):
    """
    Draw the piece label (e.g. "P1-Q" for Player1 Queen) at hex center.
    """
    label_surf = font.render(piece, True, (0,0,0))
    label_rect = label_surf.get_rect(center=(x, y))
    surface.blit(label_surf, label_rect)

def get_board_bounds(board):
    """
    Given board dict {(q,r): [(player, insectType), ...], ...},
    return minQ, maxQ, minR, maxR to help us know how large the grid is.
    """
    if not board:
        return (0, 0, 0, 0)

    all_q = [q for (q, r) in board.keys()]
    all_r = [r for (q, r) in board.keys()]
    return min(all_q), max(all_q), min(all_r), max(all_r)

def drawStatePygame(state, hex_size=40, window_padding=50):
    """
    Opens a pygame window and draws the board in 2D.
    Closes when you press the window's close button.
    """
    board = state["board"]

    # Determine board bounds
    minQ, maxQ, minR, maxR = get_board_bounds(board)
    width_range  = maxQ - minQ + 1
    height_range = maxR - minR + 1

    # Estimate a window size
    # Each hex ~ (sqrt(3)*hex_size) wide, (1.5*hex_size) tall in pointy-top layout
    est_width  = int(width_range  * math.sqrt(3) * hex_size + 2 * window_padding)
    est_height = int(height_range * 1.5         * hex_size + 2 * window_padding)

    pygame.init()
    screen = pygame.display.set_mode((est_width, est_height))
    pygame.display.set_caption("Hive Board Visualization")

    # For drawing text onto each hex
    font = pygame.font.SysFont(None, 20)

    # Fill background
    screen.fill((255, 255, 255))

    # Draw all cells
    for (q, r), stack in board.items():
        # Convert axial to pixel
        px, py = axial_to_pixel(q - minQ, r - minR, hex_size)
        # Shift by window_padding so everything is visible
        px += window_padding
        py += window_padding

        # Draw the hex (just an outline here)
        draw_hex(screen, px, py, hex_size, color=(0, 0, 0), width=2)

        if stack:
            # If there's a stack, draw top piece or show a stack indicator
            # For simplicity, just label the top piece with "P1-Q", etc.
            top_piece = stack[-1]
            player, insectType = top_piece
            piece_label = f"{player[-1]}-{insectType[0]}"  # e.g. "1-Q" or "2-S" (for Spider)
            if len(stack) > 1:
                piece_label += f"(+{len(stack)-1})"  # e.g. "1-Q(+2)" if more pieces in stack
            draw_piece_label(screen, px, py, piece_label, font)

    # Flip the display to update
    pygame.display.flip()

    # Main loop: wait until user closes
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # Could add more interactive features here if you like

    pygame.quit()


# -------------- Example usage: --------------
if __name__ == "__main__":
    import random

    game = HiveGame()
    state = game.getInitialState()

    # Make a few random placements so there's something to see
    # (You can replace this logic with real game logic)
    for _ in range(5):
        actions = game.getLegalActions(state)
        place_actions = [a for a in actions if a[0] == "PLACE"]
        if not place_actions:
            break
        action = random.choice(place_actions)
        state = game.applyAction(state, action)

    drawStatePygame(state)
