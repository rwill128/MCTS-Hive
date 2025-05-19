import unittest
from simple_games.connect_four import ConnectFour


class TestConnectFourWinDetection(unittest.TestCase):
    def test_diagonal_win_detection(self):
        game = ConnectFour()
        state = game.getInitialState()
        board = state["board"]
        # Arrange a board where X has a diagonal threat. Three X pieces form
        # a diagonal from (0,0) to (2,2).
        board[0][0] = "X"
        board[1][1] = "X"
        board[2][2] = "X"
        # Fill column 3 so X's next move lands on row 3, completing the diagonal.
        board[0][3] = "O"
        board[1][3] = "O"
        board[2][3] = "O"
        state["current_player"] = "X"
        # The game should not yet be over.
        self.assertIsNone(game.getGameOutcome(state))
        # Dropping in column 3 should give X a diagonal win.
        next_state = game.applyAction(state, 3)
        self.assertEqual(next_state["board"][3][3], "X")
        self.assertEqual(game.getGameOutcome(next_state), "X")


if __name__ == "__main__":
    unittest.main()
