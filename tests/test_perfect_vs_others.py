import sys
from types import SimpleNamespace
import random
import unittest

# Provide a minimal pygame stub so MCTS can import
sys.modules['pygame'] = SimpleNamespace(
    event=SimpleNamespace(pump=lambda: None),
    time=SimpleNamespace(delay=lambda x: None)
)

from mcts.Mcts import MCTS
from simple_games.tic_tac_toe import TicTacToe
from simple_games.perfect_tic_tac_toe import PerfectTicTacToePlayer


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def search(self, state):
        actions = self.game.getLegalActions(state)
        return random.choice(actions)


def play_game(game, player_x, player_o, seed=0):
    random.seed(seed)
    state = game.getInitialState()
    while not game.isTerminal(state):
        to_move = game.getCurrentPlayer(state)
        if to_move == "X":
            action = player_x.search(state)
        else:
            action = player_o.search(state)
        state = game.applyAction(state, action)
    return game.getGameOutcome(state)


class TestPerfectPlayer(unittest.TestCase):
    def setUp(self):
        self.game = TicTacToe()

    def assert_no_loss(self, outcome, loss_value, msg):
        self.assertNotEqual(outcome, loss_value, msg)

    def play_multiple(self, as_x, opponent_factory, games=20, seed_offset=0):
        """Helper to play several games and ensure no losses."""
        if as_x:
            perfect = PerfectTicTacToePlayer(self.game, perspective_player="X")
            loss = "O"
        else:
            perfect = PerfectTicTacToePlayer(self.game, perspective_player="O")
            loss = "X"
        for i in range(games):
            if as_x:
                opp = opponent_factory("O")
                outcome = play_game(self.game, perfect, opp, seed_offset + i)
            else:
                opp = opponent_factory("X")
                outcome = play_game(self.game, opp, perfect, seed_offset + i)
            self.assert_no_loss(outcome, loss, f"Perfect player lost game {i+1} as {'X' if as_x else 'O'}")

    def test_perfect_never_loses(self):
        def mcts_factory(role):
            return MCTS(
                game=self.game,
                perspective_player=role,
                forced_check_depth=0,
                num_iterations=100,
                max_depth=9,
                c_param=1.4,
            )

        def random_factory(role):
            return RandomPlayer(self.game)

        # Perfect as X against MCTS and random players
        self.play_multiple(True, mcts_factory, seed_offset=100)
        self.play_multiple(True, random_factory, seed_offset=200)

        # Perfect as O against MCTS and random players
        self.play_multiple(False, mcts_factory, seed_offset=300)
        self.play_multiple(False, random_factory, seed_offset=400)


if __name__ == "__main__":
    unittest.main()
