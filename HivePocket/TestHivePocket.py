import unittest
from HivePocket.HivePocket import HiveGame, hex_distance, find_queen_position
from HivePocketGlobalCache.ImmutableBoard import ImmutableBoard

class TestHiveGame(unittest.TestCase):

    def test_initial_state(self):
        game = HiveGame()
        state = game.getInitialState()
        self.assertEqual(state["current_player"], "Player1")
        self.assertEqual(state["move_number"], 0)
        self.assertEqual(state["board"], {})
        self.assertEqual(state["pieces_in_hand"]["Player1"]["Queen"], 1)
        self.assertEqual(state["pieces_in_hand"]["Player2"]["Queen"], 1)

    def test_hex_distance(self):
        self.assertEqual(hex_distance(0, 0, 1, 1), 1)
        self.assertEqual(hex_distance(0, 0, 2, 2), 2)
        self.assertEqual(hex_distance(0, 0, -1, -1), 1)
        self.assertEqual(hex_distance(0, 0, 0, 0), 0)

    def test_find_queen_position(self):
        board = {
            (0, 0): [("Player1", "Queen")],
            (1, 1): [("Player2", "Ant")]
        }
        self.assertEqual(find_queen_position(board, "Player1"), (0, 0))
        self.assertEqual(find_queen_position(board, "Player2"), None)

    def test_is_terminal(self):
        game = HiveGame()
        state = game.getInitialState()
        self.assertFalse(game.isTerminal(state))
        state["board"][(0, 0)] = [("Player1", "Queen")]
        state["board"][(1, 0)] = [("Player2", "Ant")]
        state["board"][(1, 1)] = [("Player2", "Ant")]
        state["board"][(0, 1)] = [("Player2", "Ant")]
        state["board"][(-1, 1)] = [("Player2", "Ant")]
        state["board"][(-1, 0)] = [("Player2", "Ant")]
        state["board"][(0, -1)] = [("Player2", "Ant")]
        self.assertTrue(game.isTerminal(state))

    def test_apply_action_place(self):
        game = HiveGame()
        state = game.getInitialState()
        action = ("PLACE", "Queen", (0, 0))
        new_state = game.applyAction(state, action)
        self.assertEqual(new_state["board"][(0, 0)], [("Player1", "Queen")])
        self.assertEqual(new_state["current_player"], "Player2")
        self.assertEqual(new_state["move_number"], 1)

    def test_apply_action_move(self):
        game = HiveGame()
        state = game.getInitialState()
        state["board"][(0, 0)] = [("Player1", "Queen")]
        state["board"][(1, 0)] = [("Player2", "Ant")]
        action = ("MOVE", (0, 0), (1, 1))
        new_state = game.applyAction(state, action)
        self.assertEqual(new_state["board"][(1, 1)], [("Player1", "Queen")])
        self.assertEqual(new_state["current_player"], "Player2")
        self.assertEqual(new_state["move_number"], 1)

    def test_immutable_board_with_piece_added(self):
        board = ImmutableBoard.empty()
        new_board = board.withPieceAdded(0, 0, ("Player1", "Queen"))
        self.assertEqual(new_board.getStack(0, 0), (("Player1", "Queen"),))
        self.assertEqual(board.getStack(0, 0), ())

    def test_immutable_board_with_piece_removed(self):
        board = ImmutableBoard({(0, 0): [("Player1", "Queen")]})
        new_board = board.withPieceRemoved(0, 0)
        self.assertEqual(new_board.getStack(0, 0), ())
        self.assertEqual(board.getStack(0, 0), (("Player1", "Queen"),))

if __name__ == "__main__":
    unittest.main()