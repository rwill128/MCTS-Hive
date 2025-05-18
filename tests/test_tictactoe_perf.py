import time
import unittest
from simple_games.tic_tac_toe import TicTacToe


class TestTicTacToePrecompute(unittest.TestCase):
    def test_precompute_lines_speed(self):
        game_fast = TicTacToe(use_precomputed_lines=True)
        game_slow = TicTacToe(use_precomputed_lines=False)
        state = game_fast.getInitialState()
        # Make a partial board so outcome isn't trivial
        state["board"][0][0] = "X"
        state["board"][0][1] = "O"
        state["board"][1][0] = "X"
        state["board"][1][1] = "O"
        state["current_player"] = "X"

        def run(g):
            start = time.time()
            for _ in range(50000):
                g.getGameOutcome(state)
            return time.time() - start

        slow_times = [run(game_slow) for _ in range(3)]
        fast_times = [run(game_fast) for _ in range(3)]
        avg_slow = sum(slow_times) / len(slow_times)
        avg_fast = sum(fast_times) / len(fast_times)
        self.assertLess(avg_fast, avg_slow)


if __name__ == "__main__":
    unittest.main()
