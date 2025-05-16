import pygame
import sys
import math
import random
from HivePocket.HivePocket import HiveGame  # Your Hive game implementation.
from mcts.Mcts import MCTS                # Your MCTS class.
from mcts.eval_cache import EvalCache

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
def draw_hive_board(state, surface):
    """
    Draws the Hive board onto the given surface.
    This function only draws the static board (grid and pieces).
    """
    surface.fill((255, 255, 255))

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
            pygame.draw.polygon(surface, (200, 200, 200), corners, 1)
            if (q, r) in state["board"]:
                stack = state["board"][(q, r)]
                if stack:
                    top_piece = stack[-1]
                    owner, insect = top_piece
                    color = (0, 0, 255) if owner == "Player1" else (255, 0, 0)
                    pygame.draw.polygon(surface, color, corners, 0)
                    font = pygame.font.SysFont(None, 24)
                    text = font.render(insect[0], True, (255, 255, 255))
                    text_rect = text.get_rect(center=center)
                    surface.blit(text, text_rect)

# ---------------------- Human Move Handling -------------------------
def get_human_move(state, game, screen, background):
    global OFFSET_X, OFFSET_Y
    # We'll always re-fetch legal actions inside the main loop
    # so it stays up-to-date if your code re-checks them.
    legal_actions = game.getLegalActions(state)

    # We'll still check if the queen is placed for user feedback,
    # but we won’t forcibly lock out movement actions.
    from HivePocket.HivePocket import find_queen_position
    queen_placed = (find_queen_position(state["board"], state["current_player"]) is not None)

    # For user feedback only:
    if not queen_placed:
        print("Your queen is not on the board yet. You have until your 4th move to place it.")

    # This dict is still used to map key presses to piece names
    mapping = {"Q": "Queen", "A": "Ant", "S": "Spider", "B": "Beetle", "G": "Grasshopper"}

    # We'll track whether the player is in the “middle of moving a piece”
    selected_origin = None
    highlighted_destinations = []

    while True:
        draw_hive_board(state, background)
        screen.blit(background, (0, 0))

        # If a piece is selected, highlight possible destinations
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
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE | pygame.DOUBLEBUF)
                OFFSET_X = event.w // 2
                OFFSET_Y = event.h // 2
                # Redraw after resizing
                draw_hive_board(state, background)
                screen.blit(background, (0, 0))
                pygame.display.flip()

            elif event.type == pygame.KEYDOWN:
                # ESC to cancel a partial move
                if event.key == pygame.K_ESCAPE:
                    selected_origin = None
                    highlighted_destinations = []
                    print("Move selection canceled.")
                    continue

                # Check if user pressed Q/A/S/B/G for placement
                key = pygame.key.name(event.key).upper()
                if key in mapping:
                    insect = mapping[key]
                    # We interpret the mouse position as the hex to place on
                    pos = pygame.mouse.get_pos()
                    hex_coords = pixel_to_hex(pos)

                    # Re-fetch legal actions each time
                    legal_actions = game.getLegalActions(state)

                    # Look for a matching legal "PLACE" action
                    candidates = [
                        a for a in legal_actions
                        if a[0] == "PLACE" and a[1] == insect and a[2] == hex_coords
                    ]
                    if candidates:
                        print("Selected placement:", candidates[0])
                        return candidates[0]
                    else:
                        print(f"Illegal placement attempt: {insect} at {hex_coords}.")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check if user is selecting a piece or a destination
                pos = event.pos
                clicked_hex = pixel_to_hex(pos)

                if selected_origin is None:
                    # Attempt to select a piece to move
                    if clicked_hex in state["board"]:
                        top_piece = state["board"][clicked_hex][-1]
                        if top_piece[0] == state["current_player"]:
                            # Show possible destinations for that piece
                            legal_actions = game.getLegalActions(state)
                            move_actions = [
                                a for a in legal_actions
                                if a[0] == "MOVE" and a[1] == clicked_hex
                            ]
                            if move_actions:
                                selected_origin = clicked_hex
                                highlighted_destinations = [a[2] for a in move_actions]
                                print("Selected piece for moving:", selected_origin,
                                      "destinations:", highlighted_destinations)
                            else:
                                print("No moves available for that piece.")
                else:
                    # We have selected a piece, so attempt a move to the new hex
                    if clicked_hex in highlighted_destinations:
                        legal_actions = game.getLegalActions(state)
                        candidates = [
                            a for a in legal_actions
                            if a[0] == "MOVE"
                               and a[1] == selected_origin
                               and a[2] == clicked_hex
                        ]
                        if candidates:
                            print("Selected move:", candidates[0])
                            return candidates[0]
                        else:
                            print("No matching move action found (shouldn't happen).")
                    else:
                        print("That hex is not one of the highlighted destinations.")
                        print("Press ESC to cancel move selection if you want to pick another piece.")

        # Slight delay so we’re not spinning at 100% CPU
        pygame.time.wait(100)

# ---------------------- Heatmap Drawing (Smooth Transition) -------------------------
def blend_color(current, target, alpha=0.1):
    """Blend each channel of the current color toward the target color."""
    return tuple(int(c + alpha * (t - c)) for c, t in zip(current, target))

def compute_target_color(child, heuristic_min=-500, heuristic_max=500):
    """Compute the target color (red-green gradient) from the child's average value."""
    avg_value = child.average_value()
    normalized_value = (avg_value - heuristic_min) / (heuristic_max - heuristic_min)
    normalized_value = max(0.0, min(1.0, normalized_value))
    # Map normalized_value to a red-green gradient:
    if normalized_value > 0.5:
        red = int(255 * (1 - normalized_value) * 2)
        green = 255
    else:
        red = 255
        green = int(255 * normalized_value * 2)
    blue = 0
    # Return with fixed alpha (transparency)
    return (red, green, blue, 128)

def update_heatmap_overlay(root_node, heatmap_overlay, heatmap_colors, max_visits, alpha=0.1):
    """
    Update the heatmap overlay surface with smooth color transitions.
    The color for each cell is blended from its previous value toward the new target value.
    """
    # Clear the overlay (fully transparent)
    heatmap_overlay.fill((0, 0, 0, 0))
    heuristic_min = -500
    heuristic_max = 500

    for action, child in root_node.children.items():
        target_hex = action[2]  # Destination hex cell.
        target_color = compute_target_color(child, heuristic_min, heuristic_max)
        # Normalize visit count using a logarithmic scale
        normalized_visits = (math.log(child.visit_count + 1) / math.log(max_visits + 1)) if max_visits > 0 else 0
        # Adjust brightness based on visits
        red = int(target_color[0] * normalized_visits)
        green = int(target_color[1] * normalized_visits)
        blue = int(target_color[2] * normalized_visits)
        final_target_color = (red, green, blue, 128)

        # Get previous color; if none, start at the target color.
        current_color = heatmap_colors.get(target_hex, final_target_color)
        new_color = blend_color(current_color, final_target_color, alpha)
        heatmap_colors[target_hex] = new_color

        center = hex_to_pixel(*target_hex)
        corners = polygon_corners(center, HEX_SIZE)
        pygame.draw.polygon(heatmap_overlay, new_color, corners, 0)

def draw_heatmap(root_node, iteration, screen, background, heatmap_overlay, heatmap_colors):
    """
    Composite the static board (from background) and the dynamic heatmap overlay,
    and display the iteration count.
    """
    # Start with the board background.
    screen.blit(background, (0, 0))
    # Render iteration count.
    font_big = pygame.font.SysFont(None, 24)
    iter_text = font_big.render(f"Iterations: {iteration}", True, (0, 0, 0))
    screen.blit(iter_text, (10, 10))
    # Determine maximum visit count for normalization.
    if root_node.children:
        max_visits = max(child.visit_count for child in root_node.children.values())
    else:
        max_visits = 1
    update_heatmap_overlay(root_node, heatmap_overlay, heatmap_colors, max_visits, alpha=0.1)
    # Overlay the heatmap.
    screen.blit(heatmap_overlay, (0, 0))
    pygame.display.flip()

# ---------------------- Main Game Loop -------------------------
def play_with_mcts():
    global OFFSET_X, OFFSET_Y
    pygame.init()
    screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Hive Game (Human vs. Bot)")

    # Create a background surface for the board and a heatmap overlay surface.
    background = pygame.Surface(screen.get_size())
    heatmap_overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    # Dictionary to store per-cell heatmap colors (for blending)
    heatmap_colors = {}

    game = HiveGame()
    cache = EvalCache()

    weights = {
        "winning_score": 10000,
        "queen_factor": 50,       # factor for queen's adjacency/surrounding
        "liberties_factor": 10,   # factor for queen's liberties
        "mobility_factor": 3,     # factor for movable pieces
        "early_factor": 10         # factor for early-game placement bonus
    }

    mcts = MCTS(game,
                num_iterations=100,
                max_depth=10,
                c_param=1.4,
                forced_check_depth=0,
                weights=weights,
                perspective_player="Player2",
                cache=cache)
    state = game.getInitialState()
    print("Initial board:")
    game.printState(state)
    draw_hive_board(state, background)
    screen.blit(background, (0, 0))
    pygame.display.flip()

    clock = pygame.time.Clock()
    running = True

    while not game.isTerminal(state) and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE | pygame.DOUBLEBUF)
                OFFSET_X = event.w // 2
                OFFSET_Y = event.h // 2
                # Recreate the background and overlay surfaces to match new size.
                background = pygame.Surface(screen.get_size())
                heatmap_overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
                draw_hive_board(state, background)
        if state["current_player"] == "Player1":
            print("\nHuman turn.")
            action = get_human_move(state, game, screen, background)
        else:
            print("\nBot turn. Thinking...")
            # Use the MCTS search with the smooth heatmap draw callback.
            action = mcts.search(
                state,
                draw_callback=lambda root, iter_count: draw_heatmap(root, iter_count, screen, background, heatmap_overlay, heatmap_colors)
            )
            pygame.time.wait(500)
        state = game.applyAction(state, action)
        print("Applied move:", action)
        game.printState(state)
        # Update the background board after the move.
        draw_hive_board(state, background)
        screen.blit(background, (0, 0))
        pygame.display.flip()
        clock.tick(1)
    print("Game Over. Outcome:", game.getGameOutcome(state))
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    play_with_mcts()
