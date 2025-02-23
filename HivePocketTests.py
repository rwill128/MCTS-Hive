import pygame
import math
import unittest

from HivePocket.HivePocket import HiveGame

# Constants
HEX_SIZE = 40      # Radius of each hexagon
OFFSET_X = 400     # X offset to center the board
OFFSET_Y = 300     # Y offset to center the board

# Helper Functions (unchanged)
def hex_to_pixel(q, r):
    """
    Convert axial hex coordinates (q, r) to pixel coordinates.
    """
    x = HEX_SIZE * math.sqrt(3) * (q + r / 2)
    y = HEX_SIZE * (3 / 2) * r
    return (int(x + OFFSET_X), int(y + OFFSET_Y))

def polygon_corners(center, size):
    """
    Calculate the six corners of a hexagon at 'center' with given 'size'.
    """
    cx, cy = center
    corners = []
    for i in range(6):
        angle_deg = 60 * i - 30
        angle_rad = math.radians(angle_deg)
        x = cx + size * math.cos(angle_rad)
        y = cy + size * math.sin(angle_rad)
        corners.append((x, y))
    return corners

def draw_hive_board(state, surface):
    """
    Draw the Hive board onto the given surface based on the game state.
    This includes the grid and all pieces currently on the board.
    """
    surface.fill((255, 255, 255))  # White background

    # Determine the grid bounds based on occupied hexes, with padding
    if state["board"]:
        qs = [q for (q, r) in state["board"].keys()]
        rs = [r for (q, r) in state["board"].keys()]
        q_min = min(qs) - 2
        q_max = max(qs) + 2
        r_min = min(rs) - 2
        r_max = max(rs) + 2
    else:
        q_min, q_max, r_min, r_max = -3, 3, -3, 3  # Default empty board size

    # Draw the hexagonal grid
    for q in range(q_min, q_max + 1):
        for r in range(r_min, r_max + 1):
            center = hex_to_pixel(q, r)
            corners = polygon_corners(center, HEX_SIZE)
            pygame.draw.polygon(surface, (200, 200, 200), corners, 1)  # Gray outline
            # Draw pieces if they exist at this hex
            if (q, r) in state["board"]:
                stack = state["board"][(q, r)]
                if stack:
                    top_piece = stack[-1]
                    owner, insect = top_piece
                    color = (0, 0, 255) if owner == "Player1" else (255, 0, 0)  # Blue for P1, Red for P2
                    pygame.draw.polygon(surface, color, corners, 0)  # Filled hex
                    font = pygame.font.SysFont(None, 24)
                    text = font.render(insect[0], True, (255, 255, 255))  # First letter of insect
                    text_rect = text.get_rect(center=center)
                    surface.blit(text, text_rect)

def draw_board_with_highlights(state, selected_hex, actual_highlights, expected_highlights, surface):
    """
    Draw the board with highlights for the selected piece, actual moves, and expected moves.
    - Selected piece: Blue border
    - Correct moves (overlap of actual and expected): Green borders
    - Extra moves (actual but not expected): Red borders
    - Missing moves (expected but not actual): Yellow borders
    """
    # Draw the base board (assuming this function exists)
    draw_hive_board(state, surface)

    # Highlight the selected hex with a blue border
    if selected_hex in state["board"]:
        center = hex_to_pixel(*selected_hex)  # Convert hex coordinates to pixel coordinates
        corners = polygon_corners(center, HEX_SIZE)  # Get the hexagon's corners
        pygame.draw.polygon(surface, (0, 0, 255), corners, 3)  # Blue border, 3px thick

    # Determine the move sets
    overlapping = set(actual_highlights) & set(expected_highlights)  # Moves in both
    actual_only = set(actual_highlights) - set(expected_highlights)  # Moves only in actual
    expected_only = set(expected_highlights) - set(actual_highlights)  # Moves only in expected

    # Highlight overlapping moves with green borders (correct moves)
    for hex in overlapping:
        center = hex_to_pixel(*hex)
        corners = polygon_corners(center, HEX_SIZE)
        pygame.draw.polygon(surface, (0, 255, 0), corners, 3)  # Green border, 3px thick

    # Highlight actual-only moves with red borders (generated but wrong)
    for hex in actual_only:
        center = hex_to_pixel(*hex)
        corners = polygon_corners(center, HEX_SIZE)
        pygame.draw.polygon(surface, (255, 0, 0), corners, 3)  # Red border, 3px thick

    # Highlight expected-only moves with yellow borders (assertions not met)
    for hex in expected_only:
        center = hex_to_pixel(*hex)
        corners = polygon_corners(center, HEX_SIZE)
        pygame.draw.polygon(surface, (255, 255, 0), corners, 3)  # Yellow border, 3px thick

    # Add instructional text at the top
    font = pygame.font.SysFont(None, 24)
    text = font.render(
        f"Correct Moves (Green), Extra Moves (Red), Missing Moves (Yellow) for piece at {selected_hex}",
        True,
        (0, 0, 0)
    )
    surface.blit(text, (10, 10))  # Display text at position (10, 10)

def visualize_piece_moves(game, state, hex_location, expected_destinations):
    """
    Visualize the actual vs. expected moves for the piece at 'hex_location' in the given 'state'.
    Opens a Pygame window and displays the board with highlighted moves.
    """
    # Validate the hex location
    if hex_location not in state["board"] or not state["board"][hex_location]:
        print(f"No piece at {hex_location}.")
        return
    top_piece = state["board"][hex_location][-1]
    assert top_piece[0] == state["current_player"], (
        f"Piece at {hex_location} belongs to {top_piece[0]}, not the current player {state['current_player']}."
    )

    # Get all legal actions and filter for moves from this hex
    legal_actions = game.getLegalActions(state)
    assert legal_actions, "No legal actions available for the current player."
    piece_moves = [action for action in legal_actions if action[0] == "MOVE" and action[1] == hex_location]
    if not piece_moves:
        print(f"No moves available for the piece at {hex_location}.")
        return
    actual_destinations = [action[2] for action in piece_moves]
    assert actual_destinations, f"No destinations found for piece moves from {hex_location}."

    # Provide user feedback
    print(f"Visualizing moves for piece at {hex_location}.")  # Modified message to be generic
    print(f"Actual destinations: {actual_destinations}")
    print(f"Expected destinations: {expected_destinations}")

    # Initialize Pygame and set up the window
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Move Visualization")

    # Main visualization loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Draw the board with highlights for actual and expected destinations
        draw_board_with_highlights(state, hex_location, actual_destinations, expected_destinations, screen)
        pygame.display.flip()

    # Clean up and exit Pygame
    pygame.quit()

# Unit Test Class
class TestHiveGameMoves(unittest.TestCase):
    # Add a class attribute to control visualization
    always_visualize = True  # Default to False; set to True to visualize all tests

    def test_spider_moves(self):
        """
        Test the possible moves for a Player1 Spider at position (0, -1) in a specific board setup.
        Asserts that the legal moves match the expected destinations.
        If assertions fail or self.always_visualize is True, visualizes the actual vs. expected moves.
        """
        game = HiveGame()
        state = game.getInitialState()

        piece_in_question = (0, 0)

        # Set up the board with Player1's Queen at (0, 0) and Player2's Queen at (0, 1)
        state["board"][(0, 0)] = [("Player1", "Queen")]
        # state["board"][(0, -1)] = [("Player1", "Spider")]
        state["board"][(0, 1)] = [("Player2", "Queen")]
        state["current_player"] = "Player1"

        # Get legal actions and filter for moves from (0, -1)
        legal_actions = game.getLegalActions(state)
        piece_moves = [action for action in legal_actions if action[0] == "MOVE" and action[1] == piece_in_question]
        destinations = [action[2] for action in piece_moves]

        # Define the expected moves
        expected_destinations = [(1, 0), (-1, 1)]  # Assuming these are the correct moves for the Spider

        # Wrap assertions in a try-except-else block to handle visualization
        try:
            self.assertEqual(len(destinations), len(expected_destinations),
                             f"Expected {len(expected_destinations)} moves, but got {len(destinations)}")
            self.assertEqual(set(destinations), set(expected_destinations),
                             f"Expected destinations {expected_destinations}, but got {destinations}")
        except AssertionError as e:
            print(f"Assertion failed: {e}")
            visualize_piece_moves(game, state, piece_in_question, expected_destinations)
            raise  # Re-raise the assertion error to mark the test as failed
        else:
            # If the test passes and always_visualize is True, visualize the board
            if self.always_visualize:
                print("Test passed. Visualizing for verification...")
                visualize_piece_moves(game, state, piece_in_question, expected_destinations)

if __name__ == "__main__":
    # To visualize all tests, set always_visualize to True before running
    # Uncomment the following line to enable visualization for all tests:
    # TestHiveGameMoves.always_visualize = True
    unittest.main()