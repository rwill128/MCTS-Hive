import sys
from types import SimpleNamespace
import unittest
import random

# Stub pygame so tests can run headless
sys.modules['pygame'] = SimpleNamespace(
    event=SimpleNamespace(pump=lambda: None),
    time=SimpleNamespace(delay=lambda x: None)
)

from mcts.Mcts import MCTS
from simple_games.tic_tac_toe import TicTacToe

class TestHybridMinimax(unittest.TestCase):
    def test_minimax_overrides_bad_mcts(self):
        random.seed(42)
        game = TicTacToe()
        state = {
            "board": [
                ["X", "O", None],
                ["X", "O", None],
                [None, None, None],
            ],
            "current_player": "X",
        }
        mcts = MCTS(
            game=game,
            perspective_player="X",
            forced_check_depth=0,
            num_iterations=1,
            max_depth=9,
            c_param=1.4,
            minimax_depth=2,
        )
        best = mcts.search(state)
        self.assertEqual(best, (2, 0))

if __name__ == "__main__":
    unittest.main()
