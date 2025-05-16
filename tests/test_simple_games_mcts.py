import sys
from types import SimpleNamespace
import unittest

# Provide a minimal pygame stub so MCTS can import
sys.modules['pygame'] = SimpleNamespace(
    event=SimpleNamespace(pump=lambda: None),
    time=SimpleNamespace(delay=lambda x: None)
)

from mcts.Mcts import MCTS
from simple_games.tic_tac_toe import TicTacToe
from simple_games.connect_four import ConnectFour

class TestTicTacToeMCTS(unittest.TestCase):
    def test_finds_winning_move(self):
        game = TicTacToe()
        state = {
            "board": [
                ["X", "O", None],
                ["X", "O", None],
                [None, None, None],
            ],
            "current_player": "X",
        }
        mcts = MCTS(game=game, perspective_player="X", forced_check_depth=0,
                    num_iterations=200, max_depth=9, c_param=1.4)
        best = mcts.search(state)
        self.assertEqual(best, (2, 0))

class TestConnectFourMCTS(unittest.TestCase):
    def test_finds_vertical_win(self):
        game = ConnectFour()
        state = game.getInitialState()
        board = state["board"]
        board[0][0] = "X"
        board[1][0] = "X"
        board[2][0] = "X"
        state["current_player"] = "X"
        mcts = MCTS(game=game, perspective_player="X", forced_check_depth=0,
                    num_iterations=200, max_depth=42, c_param=1.4)
        best = mcts.search(state)
        self.assertEqual(best, 0)

if __name__ == "__main__":
    unittest.main()
