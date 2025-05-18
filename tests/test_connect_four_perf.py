import time
import unittest
from simple_games.connect_four import ConnectFour


class TestConnectFourPrecompute(unittest.TestCase):
    def test_precompute_lines_speed(self):
        game_fast = ConnectFour(use_precomputed_lines=True)
        game_slow = ConnectFour(use_precomputed_lines=False)
        state = game_fast.getInitialState()
        # Fill a few moves so evaluateState doesn't immediately return terminal
        state = game_fast.applyAction(state, 0)
        state = game_fast.applyAction(state, 1)
        state = game_fast.applyAction(state, 0)
        state = game_fast.applyAction(state, 1)
        state = game_fast.applyAction(state, 0)

        def run(g):
            start = time.time()
            for _ in range(500):
                g.evaluateState('X', state)
            return time.time() - start

        slow_time = run(game_slow)
        fast_time = run(game_fast)
        self.assertLess(fast_time, slow_time)


if __name__ == "__main__":
    unittest.main()
