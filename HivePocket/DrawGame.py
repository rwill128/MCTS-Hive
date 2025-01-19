import pygame
import sys
import math

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
    Draw a hex on the surface with center at (x,y).
    - color: fill color if width=0, or line color if width>0.
    - width=0 => filled hex. Nonzero => just outline.
    """
    corners = hex_corners(x, y, size)
    pygame.draw.polygon(surface, color, corners, width)

def draw_piece_label(surface, x, y, piece, font, text_color=(0,0,0)):
    """
    Draw the piece label (e.g. "P1-Q" for Player1 Queen) at hex center in text_color.
    """
    label_surf = font.render(piece, True, text_color)
    label_rect = label_surf.get_rect(center=(x, y))
    surface.blit(label_surf, label_rect)

def get_board_bounds(board):
    """
    Given a board dict {(q,r): [(player, insectType), ...], ...},
    return minQ, maxQ, minR, maxR for bounding the displayed region.
    """
    if not board:
        return (0, 0, 0, 0)

    all_q = [q for (q, r) in board.keys()]
    all_r = [r for (q, r) in board.keys()]
    return min(all_q), max(all_q), min(all_r), max(all_r)

def drawStatePygame(state, hex_size=40, window_padding=50):
    """
    Opens a pygame window and draws the board in 2D, color-coded:
      - Player1 => Yellow hex
      - Player2 => Red hex
      - Empty cell => White hex
    Shows top piece label in the center (black text).
    Closes when you press the window's close button.
    """
    board = state["board"]

    # Determine board bounds
    minQ, maxQ, minR, maxR = get_board_bounds(board)
    width_range  = maxQ - minQ + 2
    height_range = maxR - minR + 2

    # Estimate window size:
    # Each hex ~ (sqrt(3)*hex_size) wide and (1.5*hex_size) tall in pointy-top layout
    est_width  = int(width_range  * math.sqrt(3) * hex_size + 2 * window_padding)
    est_height = int(height_range * 1.5         * hex_size + 2 * window_padding)

    pygame.init()
    screen = pygame.display.set_mode((est_width, est_height))
    pygame.display.set_caption("Hive Board Visualization")

    font = pygame.font.SysFont(None, 20)

    # Fill background
    screen.fill((255, 255, 255))

    # Draw each cell in board
    for (q, r), stack in board.items():
        # Convert (q, r) to pixel, offset by minQ/minR so we start at 0,0
        px, py = axial_to_pixel(q - minQ, r - minR, hex_size)
        px += window_padding
        py += window_padding

        if stack:
            # Color for top piece
            top_piece = stack[-1]
            player, insectType = top_piece

            if player == "Player1":
                fill_color = (255, 255, 0)   # Yellow
            else:
                fill_color = (255, 0, 0)     # Red

            # First draw a filled hex in player's color
            draw_hex(screen, px, py, hex_size, color=fill_color, width=0)
            # Then draw an outline
            draw_hex(screen, px, py, hex_size, color=(0, 0, 0), width=2)

            # Build label, e.g. "1-Q(+2)" if there's a stack
            piece_label = f"{player[-1]}-{insectType[0]}"
            if len(stack) > 1:
                piece_label += f"(+{len(stack)-1})"
            draw_piece_label(screen, px, py, piece_label, font)

        else:
            # Empty cell => white fill
            draw_hex(screen, px, py, hex_size, color=(255, 255, 255), width=0)
            draw_hex(screen, px, py, hex_size, color=(0, 0, 0), width=2)

    # Flip the display to update
    pygame.display.flip()

    # Wait until user closes window
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()