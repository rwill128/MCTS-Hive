import pygame
import sys
import math
import random
from HivePocket.HivePocket import HiveGame  # Your Hive game implementation.
from mcts.Mcts import MCTS                # Your MCTS class.

# ---------------------- Hex Grid Helpers -------------------------
HEX_SIZE = 40      # Radius of each hexagon.
OFFSET_X = 400     # Initial X offset so the grid is centered.
OFFSET_Y = 300     # Initial Y offset so the grid is centered.

def hex_to_pixel(q, r):
    x = HEX_SIZE * math.sqrt(3) * (q + r / 2)
    y = HEX_SIZE * (3 / 2) * r
    return (int(x + OFFSET_X), int(y + OFFSET_Y))

def polygon_corners(center, size):
    cx, cy = center
    corners = []
    for i in range(6):
        angle_deg = 60 * i - 30
        angle_rad = math.radians(angle_deg)
        x = cx + size * math.cos(angle_rad)
        y = cy + size * math.sin(angle_rad)
        corners.append((x, y))
    return corners

def pixel_to_hex(pos):
    x, y = pos
    x -= OFFSET_X
    y -= OFFSET_Y
    q = (math.sqrt(3)/3 * x - 1/3 * y) / HEX_SIZE
    r = (2/3 * y) / HEX_SIZE

    cube_x = q
    cube_z = r
    cube_y = -cube_x - cube_z

    rx = round(cube_x)
    ry = round(cube_y)
    rz = round(cube_z)

    x_diff = abs(rx - cube_x)
    y_diff = abs(ry - cube_y)
    z_diff = abs(rz - cube_z)

    if x_diff > y_diff and x_diff > z_diff:
        rx = -ry - rz
    elif y_diff > z_diff:
        ry = -rx - rz
    else:
        rz = -rx - ry

    return (rx, rz)

# ---------------------- Board Drawing -------------------------
def draw_hive_board(state, screen):
    """
    Draws a visualization of the Hive board onto the provided Pygame surface.
    - Computes a bounding region based on state["board"] (or uses a default region if empty).
    - Draws each hex cell with a light-gray outline.
    - If a cell is occupied, fills it with blue (Player1) or red (Player2) and renders the
      first letter of the insect type in white.
    """
    screen.fill((255, 255, 255))

    if state["board"]:
        qs = [q for (q, r) in state["board"].keys()]
        rs = [r for (q, r) in state["board"].keys()]
        q_min = min(qs) - 2
        q_max = max(qs) + 2
        r_min = min(rs) - 2
        r_max = max(rs) + 2
    else:
        q_min, q_max, r_min, r_max = -3, 3, -3, 3

    for q in range(q_min, q_max + 1):
        for r in range(r_min, r_max + 1):
            center = hex_to_pixel(q, r)
            corners = polygon_corners(center, HEX_SIZE)
            pygame.draw.polygon(screen, (200, 200, 200), corners, 1)
            if (q, r) in state["board"]:
                stack = state["board"][(q, r)]
                if stack:
                    top_piece = stack[-1]
                    owner, insect = top_piece
                    color = (0, 0, 255) if owner == "Player1" else (255, 0, 0)
                    pygame.draw.polygon(screen, color, corners, 0)
                    font = pygame.font.SysFont(None, 24)
                    text = font.render(insect[0], True, (255, 255, 255))
                    text_rect = text.get_rect(center=center)
                    screen.blit(text, text_rect)
    pygame.display.flip()

# ---------------------- Human Move Handling -------------------------
def get_human_move(state, game, screen):
    """
    Waits for a human (Player1) move.

    Behavior:
    - If the queen is not yet placed, only PLACE actions are allowed.
      (The user should press Q, A, S, B or G to select a piece type,
      and the piece will be placed at the hex cell under the mouse.)
    - Once the queen is placed, both PLACE and MOVE actions are allowed.
      The user may:
        • Press one of Q, A, S, B, G to attempt a placement move at the current mouse position.
        • Or click on one of their pieces to select it for moving. When a piece is selected,
          its legal destination cells are highlighted. Then, a click on one of the highlighted cells
          executes the move.
        • The user can cancel a move selection by pressing Escape.
    """
    global OFFSET_X, OFFSET_Y

    # Always refresh legal actions.
    legal_actions = game.getLegalActions(state)
    # Draw the board initially.
    draw_hive_board(state, screen)

    # Check if the queen is placed.
    from HivePocket.HivePocket import find_queen_position
    queen_placed = (find_queen_position(state["board"], state["current_player"]) is not None)

    # Define a mapping for placement keys.
    mapping = {"Q": "Queen", "A": "Ant", "S": "Spider", "B": "Beetle", "G": "Grasshopper"}

    # If the queen is not yet placed, force placement-only mode.
    if not queen_placed:
        print("Queen not placed yet. Press Q, A, S, B or G for Queen, Ant, Spider, Beetle, or Grasshopper.")
        while True:
            draw_hive_board(state, screen)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    OFFSET_X = event.w // 2
                    OFFSET_Y = event.h // 2
                    draw_hive_board(state, screen)
                elif event.type == pygame.KEYDOWN:
                    key = pygame.key.name(event.key).upper()
                    if key in mapping:
                        insect = mapping[key]
                        pos = pygame.mouse.get_pos()
                        hex_coords = pixel_to_hex(pos)
                        legal_actions = game.getLegalActions(state)
                        candidate = [a for a in legal_actions if a[0] == "PLACE" and a[1] == insect and a[2] == hex_coords]
                        if candidate:
                            print("Selected placement:", candidate[0])
                            return candidate[0]
                        else:
                            print(f"Illegal placement attempt: {insect} at {hex_coords}.")
            pygame.time.wait(100)

    # Otherwise, if queen is placed, allow both placement and move actions.
    else:
        # Variables to hold move-selection state.
        selected_origin = None
        highlighted_destinations = []
        print("Queen placed. You may either place a piece (press Q, A, S, B or G) or move a piece by clicking it.")
        while True:
            # Redraw board (if a move selection is active, redraw highlights).
            draw_hive_board(state, screen)
            if selected_origin is not None:
                for dest in highlighted_destinations:
                    center = hex_to_pixel(*dest)
                    pygame.draw.circle(screen, (255, 255, 0), center, HEX_SIZE // 2, 3)
                pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    OFFSET_X = event.w // 2
                    OFFSET_Y = event.h // 2
                    draw_hive_board(state, screen)
                elif event.type == pygame.KEYDOWN:
                    # Allow cancellation of move selection with Escape.
                    if event.key == pygame.K_ESCAPE:
                        selected_origin = None
                        highlighted_destinations = []
                        print("Move selection canceled.")
                        draw_hive_board(state, screen)
                    else:
                        key = pygame.key.name(event.key).upper()
                        if key in mapping:
                            insect = mapping[key]
                            pos = pygame.mouse.get_pos()
                            hex_coords = pixel_to_hex(pos)
                            legal_actions = game.getLegalActions(state)
                            candidate = [a for a in legal_actions if a[0] == "PLACE" and a[1] == insect and a[2] == hex_coords]
                            if candidate:
                                print("Selected placement:", candidate[0])
                                return candidate[0]
                            else:
                                print(f"Illegal placement attempt: {insect} at {hex_coords}.")
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    clicked_hex = pixel_to_hex(pos)
                    # If no piece is selected yet and the clicked cell contains one of our pieces, select it.
                    if selected_origin is None and clicked_hex in state["board"]:
                        top_piece = state["board"][clicked_hex][-1]
                        if top_piece[0] == state["current_player"]:
                            move_actions = [a for a in legal_actions if a[0] == "MOVE" and a[1] == clicked_hex]
                            if move_actions:
                                selected_origin = clicked_hex
                                highlighted_destinations = [a[2] for a in move_actions]
                                print("Selected piece for moving:", selected_origin, "possible destinations:", highlighted_destinations)
                                draw_hive_board(state, screen)
                                for dest in highlighted_destinations:
                                    center = hex_to_pixel(*dest)
                                    pygame.draw.circle(screen, (255, 255, 0), center, HEX_SIZE // 2, 3)
                                pygame.display.flip()
                    # If a piece is already selected, accept only clicks on a highlighted destination.
                    elif selected_origin is not None:
                        if clicked_hex in highlighted_destinations:
                            candidate = [a for a in legal_actions if a[0] == "MOVE" and a[1] == selected_origin and a[2] == clicked_hex]
                            if candidate:
                                return candidate[0]
                        else:
                            print("Click not on a highlighted destination. Press Escape to cancel selection.")
            pygame.time.wait(100)

# ---------------------- Main Game Loop -------------------------
def play_with_mcts():
    global OFFSET_X, OFFSET_Y
    pygame.init()
    screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
    pygame.display.set_caption("Hive Game (Human vs. Bot)")

    game = HiveGame()
    mcts = MCTS(game,
                draw_reward=0.1,
                win_reward=1,
                lose_reward=-1,
                num_iterations=2,
                c_param=1.4,
                forced_check_depth=0)
    state = game.getInitialState()
    print("Initial board:")
    game.printState(state)

    # Draw the board immediately so it's not blank.
    draw_hive_board(state, screen)

    clock = pygame.time.Clock()
    running = True

    while not game.isTerminal(state) and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                OFFSET_X = event.w // 2
                OFFSET_Y = event.h // 2

        if state["current_player"] == "Player1":
            print("\nHuman turn.")
            action = get_human_move(state, game, screen)
        else:
            print("\nBot turn. Thinking...")
            action = mcts.search(state)
            pygame.time.wait(500)
        state = game.applyAction(state, action)
        print("Applied move:", action)
        game.printState(state)
        draw_hive_board(state, screen)
        clock.tick(1)

    print("Game Over. Outcome:", game.getGameOutcome(state))
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    play_with_mcts()
