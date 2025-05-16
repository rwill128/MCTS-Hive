import pygame
import sys
import math
import json
import datetime
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
        # action is typically (MOVE/PLACE, origin, destination...) or similar.
        if len(action) > 2:
            target_hex = action[2]  # Destination hex cell.
        else:
            continue

        target_color = compute_target_color(child, heuristic_min, heuristic_max)
        # Normalize visit count using a logarithmic scale.
        normalized_visits = (math.log(child.visit_count + 1) / math.log(max_visits + 1)) if max_visits > 0 else 0
        # Adjust brightness based on visits.
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

def draw_heatmap(root_node, iteration, screen, background, heatmap_overlay, heatmap_colors, current_player):
    """
    Composite the static board (from background) and the dynamic heatmap overlay,
    and display the current player's turn and iteration count.
    """
    # Start with the board background.
    screen.blit(background, (0, 0))
    font_big = pygame.font.SysFont(None, 24)

    # Render the current player's turn label.
    player_text = font_big.render(f"Turn: {current_player}", True, (0, 0, 0))
    screen.blit(player_text, (10, 10))

    # Render iteration count below the player's turn.
    iter_text = font_big.render(f"Iterations: {iteration}", True, (0, 0, 0))
    screen.blit(iter_text, (10, 40))

    # Determine maximum visit count for normalization.
    if root_node.children:
        max_visits = max(child.visit_count for child in root_node.children.values())
    else:
        max_visits = 1
    update_heatmap_overlay(root_node, heatmap_overlay, heatmap_colors, max_visits, alpha=0.1)
    # Overlay the heatmap.
    screen.blit(heatmap_overlay, (0, 0))
    pygame.display.flip()

# ---------------------- Main Game Loop: MCTS vs. MCTS -------------------------
def play_mcts_vs_mcts():
    global OFFSET_X, OFFSET_Y
    pygame.init()
    screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Hive Game (MCTS vs. MCTS)")

    # Create a background surface for the board and a heatmap overlay surface.
    background = pygame.Surface(screen.get_size())
    heatmap_overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    # Dictionary to store per-cell heatmap colors (for blending)
    heatmap_colors = {}

    game = HiveGame()
    cache = EvalCache()

    # ----------------- Define Agent Parameters Once -----------------
    params1 = {
        "num_iterations": 1000,
        "max_depth": 20,
        "c_param": 1.4,
        "forced_check_depth": 0,
        "weights": {
            "winning_score": 10000,
            "queen_factor": 5000,
            "liberties_factor": 5000,
            "mobility_factor": 300,
            "early_factor": 2
        },
        "perspective_player": "Player1"
    }
    params2 = {
        "num_iterations": 1000,
        "max_depth": 20,
        "c_param": 1.4,
        "forced_check_depth": 0,
        "weights": {
            "winning_score": 10000,
            "queen_factor": 5000,
            "liberties_factor": 5000,
            "mobility_factor": 300,
            "early_factor": 2
        },
        "perspective_player": "Player2"
    }

    mcts_player1 = MCTS(game, cache=cache, **params1)
    mcts_player2 = MCTS(game, cache=cache, **params2)

    state = game.getInitialState()
    print("Initial board:")
    game.printState(state)

    # Draw the initial board.
    draw_hive_board(state, background)
    screen.blit(background, (0, 0))
    pygame.display.flip()

    # Create a list to record each applied action for game history.
    game_history = []
    paused = False

    clock = pygame.time.Clock()

    # ----------------- Custom Draw Callback that Honors Pause -----------------
    def agent_draw_callback(root, iter_count):
        nonlocal paused
        # Draw heatmap and board.
        draw_heatmap(root, iter_count, screen, background, heatmap_overlay, heatmap_colors, state["current_player"])
        # While paused, process events and wait until unpaused.
        while paused:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    paused = False
                    print("Game unpaused.")
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.time.wait(50)

    running = True
    while not game.isTerminal(state) and running:
        # Process events. (Note that if an agent is in search, the callback will also process events.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE | pygame.DOUBLEBUF)
                OFFSET_X = event.w // 2
                OFFSET_Y = event.h // 2
                background = pygame.Surface(screen.get_size())
                heatmap_overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
                draw_hive_board(state, background)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                paused = not paused
                if paused:
                    print("Game paused.")
                else:
                    print("Game unpaused.")

        if not running:
            break

        # If paused outside of agent search, display a paused overlay.
        if paused:
            screen.blit(background, (0, 0))
            font_big = pygame.font.SysFont(None, 48)
            paused_text = font_big.render("Paused", True, (0, 0, 0))
            screen.blit(paused_text, ((screen.get_width() - paused_text.get_width()) // 2, 10))
            pygame.display.flip()
            clock.tick(30)
            continue

        # Decide which MCTS to use based on the current player.
        if state["current_player"] == "Player1":
            print("\nPlayer1 (MCTS) turn. Thinking...")
            action = mcts_player1.search(state, draw_callback=agent_draw_callback)
        else:
            print("\nPlayer2 (MCTS) turn. Thinking...")
            action = mcts_player2.search(state, draw_callback=agent_draw_callback)

        # Optional pause to observe the heatmap changes.
        pygame.time.wait(500)

        # Record the move before applying it.
        game_history.append(action)

        state = game.applyAction(state, action)
        print("Applied move:", action)
        game.printState(state)

        # Update the board background after the move.
        draw_hive_board(state, background)
        screen.blit(background, (0, 0))
        pygame.display.flip()
        clock.tick(1)

    outcome = game.getGameOutcome(state)
    print("Game Over. Outcome:", outcome)

    # Prepare the game log with agent parameters and the move history.
    history_data = {
        "winner": outcome,
        "agent1": params1,
        "agent2": params2,
        "moves": game_history
    }

    # Save the game log to a timestamped JSON file.
    log_filename = f"game_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_filename, "w") as f:
        json.dump(history_data, f, indent=4)
    print(f"Game history saved to {log_filename}")

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    play_mcts_vs_mcts()
