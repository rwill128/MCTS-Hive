import time
import unittest

from mcts.Mcts import MCTS, MCTSNode


class DummyGame:
    def evaluateState(self, *args, **kwargs):
        return 0.0

    def getOpponent(self, player):
        return "O" if player == "X" else "X"

    def getCurrentPlayer(self, state):
        return state["current_player"]


def time_weighted_choice(use_cache: bool) -> float:
    mcts = MCTS(game=DummyGame(), perspective_player="X", use_distance_cache=use_cache)
    state = {"board": {(0, 0): [("O", "Queen")]}, "current_player": "X"}
    actions = [("PLACE", None, (i % 5, i // 5)) for i in range(100)]
    start = time.perf_counter()
    for _ in range(1000):
        mcts.weightedActionChoice(state, actions)
    return time.perf_counter() - start


class TestWeightedChoicePerformance(unittest.TestCase):
    def test_distance_cache_is_faster(self):
        cached_time = time_weighted_choice(True)
        uncached_time = time_weighted_choice(False)
        # Cached version should be faster or equal
        self.assertLessEqual(cached_time, uncached_time)


if __name__ == "__main__":
    unittest.main()
