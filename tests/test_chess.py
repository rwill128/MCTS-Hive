import unittest

from simple_games.chess import Chess

class TestChess(unittest.TestCase):
    def test_initial_legal_move_count(self):
        game = Chess()
        state = game.getInitialState()
        moves = game.getLegalActions(state)
        self.assertEqual(len(moves), 20)

    def test_capture_king(self):
        game = Chess()
        state = {"board": [[None]*8 for _ in range(8)], "current_player": "w"}
        state["board"][0][0] = "bK"
        state["board"][7][4] = "wK"
        state["board"][7][0] = "wR"
        moves = game.getLegalActions(state)
        self.assertIn(((7,0),(0,0)), moves)
        new_state = game.applyAction(state, ((7,0),(0,0)))
        self.assertEqual(game.getGameOutcome(new_state), "w")

if __name__ == "__main__":
    unittest.main()
