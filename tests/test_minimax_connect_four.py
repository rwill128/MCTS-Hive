import sys
from types import SimpleNamespace
import unittest

# Provide a minimal pygame stub so imports work
sys.modules['pygame'] = SimpleNamespace(
    event=SimpleNamespace(pump=lambda: None),
    time=SimpleNamespace(delay=lambda x: None)
)

from simple_games.connect_four import ConnectFour
from simple_games.minimax_connect_four import MinimaxConnectFourPlayer


class TestMinimaxConnectFour(unittest.TestCase):
    def test_finds_vertical_win(self):
        game = ConnectFour()
        state = game.getInitialState()
        board = state["board"]
        board[0][0] = "X"
        board[1][0] = "X"
        board[2][0] = "X"
        state["current_player"] = "X"
        player = MinimaxConnectFourPlayer(game, perspective_player="X", depth=4)
        best = player.search(state)
        self.assertEqual(best, 0)


if __name__ == "__main__":
    unittest.main()
