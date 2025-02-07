import unittest

from HivePocket import HiveRules
from HivePocket.GameState import GameState


class TestHiveRules(unittest.TestCase):

    def test_get_adjacent_cells(self):
        # Test a few different cells
        neighbors = list(HiveRules.HiveRules.get_adjacent_cells(0, 0))
        self.assertEqual(len(neighbors), 6)
        self.assertIn((1, 0), neighbors)
        self.assertIn((-1, 0), neighbors)
        self.assertIn((0, 1), neighbors)
        self.assertIn((0, -1), neighbors)
        self.assertIn((1, -1), neighbors)
        self.assertIn((-1, 1), neighbors)

        neighbors = list(HiveRules.HiveRules.get_adjacent_cells(2, -1))
        self.assertIn((3, -1), neighbors)
        self.assertIn((1, 0), neighbors)

    def test_is_board_connected(self):
        # Empty board is connected
        self.assertTrue(HiveRules.HiveRules.is_board_connected({}))

        # Single piece is connected
        board1 = {(0, 0): [("Player1", "Queen")]}
        self.assertTrue(HiveRules.HiveRules.is_board_connected(board1))

        # Two connected pieces
        board2 = {
            (0, 0): [("Player1", "Queen")],
            (1, 0): [("Player2", "Ant")],
        }
        self.assertTrue(HiveRules.HiveRules.is_board_connected(board2))

        # Two disconnected pieces
        board3 = {
            (0, 0): [("Player1", "Queen")],
            (2, 2): [("Player2", "Ant")],
        }
        self.assertFalse(HiveRules.HiveRules.is_board_connected(board3))

        # A more complex connected board
        board4 = {
            (0, 0): [("Player1", "Queen")],
            (1, 0): [("Player2", "Ant")],
            (0, 1): [("Player1", "Beetle")],
            (-1, 1): [("Player2", "Spider")],
        }
        self.assertTrue(HiveRules.HiveRules.is_board_connected(board4))

        # A disconnected board with a gap
        board5 = {
            (0, 0): [("Player1", "Queen")],
            (1, 0): [("Player2", "Ant")],
            (-1, 0): [("Player1", "Beetle")],
        }
        self.assertFalse(HiveRules.HiveRules.is_board_connected(board5))

    def test_can_slide(self):
        # Test cases for canSlide
        board = {
            (0, 0): [("Player1", "Ant")],
            (1, 0): [("Player2", "Ant")],
        }
        # Basic slide
        self.assertTrue(HiveRules.HiveRules.can_slide(board, 0, 0, 0, 1))

        # Blocked slide
        board[(0, 1)] = [("Player1", "Beetle")]
        board[(1,-1)] = [("Player1", "Beetle")]
        self.assertFalse(HiveRules.HiveRules.can_slide(board, 0, 0, 0, 1))

        # Slide around a corner (allowed)
        board = {
            (0,0): [('Player1', 'Ant')],
            (1,0): [('Player2', 'Ant')],
            (1,-1): [('Player1', 'Ant')],
        }
        self.assertTrue(HiveRules.HiveRules.can_slide(board, 0, 0, -1, 0))

        board = {
            (0, 0): [("Player1", "Ant")],
            (1, -1): [("Player2", "Beetle")],
        }
        self.assertTrue(HiveRules.HiveRules.can_slide(board, 0, 0, 1, 0))  # Can slide next to opponent

        # Test cases with no pieces on board - should always be false, it's not adjacent.
        empty_board = {}
        self.assertFalse(HiveRules.HiveRules.can_slide(empty_board, 0, 0, 1, 1))
        self.assertFalse(HiveRules.HiveRules.can_slide(empty_board, 0, 0, 2, 2))

    def test_get_ant_destinations(self):
        board = {
            (0, 0): [("Player1", "Ant")],
            (1, 0): [("Player2", "Queen")],
        }
        destinations = HiveRules.HiveRules.get_ant_destinations(board, (0, 0))
        self.assertIn((0, 1), destinations)
        self.assertIn((0, -1), destinations)
        self.assertNotIn((1, 0), destinations)  # Blocked by opponent
        self.assertNotIn((0, 0), destinations)

        # Test with a more complex board, including blocked paths
        board2 = {
            (0, 0): [("Player1", "Ant")],
            (1, 0): [("Player2", "Queen")],
            (0, 1): [("Player1", "Beetle")],
            (-1, 1): [("Player2", "Spider")],
        }
        destinations2 = HiveRules.HiveRules.get_ant_destinations(board2, (0, 0))
        self.assertIn((-1, 0), destinations2)
        self.assertIn((0, -1), destinations2)
        self.assertIn((1, -1), destinations2)

        self.assertNotIn((1,0), destinations2) # Verify blocked
        self.assertNotIn((0, 1), destinations2) # Verify blocked
        self.assertNotIn((-1, 1), destinations2) # Verify blocked

    def test_get_spider_destinations(self):
        board = {
            (0, 0): [('Player1', 'Spider')],
            (1, 0): [('Player2', 'Queen')]
        }
        destinations = HiveRules.HiveRules.get_spider_destinations(board, (0,0))
        self.assertIn((2, -1), destinations)

    def test_get_grasshopper_jumps(self):
        board = {
            (0, 0): [('Player1', 'Grasshopper')],
            (1, 0): [('Player2', 'Queen')],
            (2, 0): [('Player1', 'Ant')]
        }
        destinations = HiveRules.HiveRules.get_grasshopper_jumps(board, (0,0))
        self.assertIn((3, 0), destinations)

    def test_place_piece_actions_initial(self):
        state = GameState() # Initial state
        actions = HiveRules.HiveRules.get_legal_actions(state)

        self.assertEqual(len(actions), 5) #One of each piece
        self.assertTrue(all(a[0] == "PLACE" for a in actions))

    def test_place_piece_actions_second_turn(self):
        state = GameState()
        state.apply_action(('PLACE', 'Ant', (0,0))) # Place for P1

        actions = HiveRules.HiveRules.get_legal_actions(state)
        self.assertEqual(len(actions), 30) # 6 neighbors * 5 pieces.
        self.assertTrue(all(a[0] == "PLACE" for a in actions))

    def test_place_piece_actions_queen_required(self):
        state = GameState()
        state.apply_action(("PLACE", "Ant", (0, 0)))  # P1
        state.apply_action(("PLACE", "Ant", (0, 1)))  # P2
        state.apply_action(("PLACE", "Spider", (1, -1))) # P1
        state.apply_action(("PLACE", "Spider", (0, 2))) # P2
        state.apply_action(("PLACE", "Grasshopper", (2, -2))) # P1
        state.apply_action(("PLACE", "Grasshopper", (0, 3))) # P2
        #Now it's P1 turn again, and queen is required.
        actions = HiveRules.HiveRules.get_legal_actions(state)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0][1], "Queen") # Must be a queen placement
        self.assertEqual(actions[0][0], "PLACE")


    def test_move_piece_actions_basic(self):
        state = GameState(board={(0, 0): [("Player1", "Ant")], (1, 0): [("Player2", "Queen")]},
                          current_player="Player1")
        actions = HiveRules.HiveRules.get_legal_actions(state)
        self.assertTrue(len(actions) > 0)
        self