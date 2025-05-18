import sys
from types import SimpleNamespace
import unittest
import time

# Minimal pygame stub
sys.modules['pygame'] = SimpleNamespace(
    event=SimpleNamespace(pump=lambda: None),
    time=SimpleNamespace(delay=lambda x: None)
)

from mcts.Mcts import MCTS
from simple_games.tic_tac_toe import TicTacToe


class TestLegalCachePerformance(unittest.TestCase):
    def measure_search(self, cache_flag: bool) -> float:
        game = TicTacToe()
        state = game.getInitialState()
        mcts = MCTS(game=game, perspective_player="X", forced_check_depth=0,
                    num_iterations=500, max_depth=9, c_param=1.4,
                    cache_legal_actions=cache_flag)
        start = time.perf_counter()
        mcts.search(state)
        return time.perf_counter() - start

    def test_cache_flag_speed(self):
        # Average over a few runs to reduce noise
        off_times = [self.measure_search(False) for _ in range(3)]
        on_times = [self.measure_search(True) for _ in range(3)]
        avg_off = sum(off_times) / len(off_times)
        avg_on = sum(on_times) / len(on_times)
        # Assert that enabling caching does not make things slower
        self.assertLess(avg_on, avg_off)


if __name__ == '__main__':
    unittest.main()
