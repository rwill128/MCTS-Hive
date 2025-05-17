import sys
from types import SimpleNamespace
import unittest

# Stub pygame so tests can run headless
sys.modules['pygame'] = SimpleNamespace(
    event=SimpleNamespace(pump=lambda: None),
    time=SimpleNamespace(delay=lambda x: None)
)

from mcts.Mcts import MCTS
from simple_games.tic_tac_toe import TicTacToe
from simple_games.perfect_tic_tac_toe import PerfectTicTacToePlayer


class TestPerfectTicTacToe(unittest.TestCase):
    def play_game(self, perfect_first: bool) -> str:
        game = TicTacToe()
        state = game.getInitialState()
        perfect_role = "X" if perfect_first else "O"
        perfect_player = PerfectTicTacToePlayer(game, perfect_role)

        while not game.isTerminal(state):
            to_move = state["current_player"]
            if to_move == perfect_role:
                action = perfect_player.search(state)
            else:
                mcts = MCTS(
                    game=game,
                    perspective_player=to_move,
                    forced_check_depth=0,
                    num_iterations=50,
                    max_depth=9,
                    c_param=1.4,
                )
                action = mcts.search(state)
            state = game.applyAction(state, action)
        return game.getGameOutcome(state)

    def test_perfect_never_loses(self):
        # Run several games with the perfect player starting first and second
        for perfect_first in [True, False]:
            for _ in range(3):
                outcome = self.play_game(perfect_first)
                if perfect_first:
                    self.assertNotEqual(outcome, "O")
                else:
                    self.assertNotEqual(outcome, "X")


if __name__ == "__main__":
    unittest.main()
