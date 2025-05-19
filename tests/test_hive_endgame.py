import unittest
from HivePocket.HivePocket import HiveGame


class TestHiveEndgame(unittest.TestCase):
    """Endgame scenarios where a queen is completely surrounded."""

    def _surrounded_queen_state(self, winner: str):
        """Return (game, state) with the losing queen surrounded."""
        game = HiveGame()
        state = {
            "board": {},
            # The losing player is on move, but the game is already decided.
            "current_player": "Player2" if winner == "Player1" else "Player1",
            "pieces_in_hand": {
                "Player1": game.INITIAL_PIECES.copy(),
                "Player2": game.INITIAL_PIECES.copy(),
            },
            "move_number": 6,  # arbitrary
        }
        loser = "Player2" if winner == "Player1" else "Player1"
        # Place the losing queen at the origin
        state["board"][(0, 0)] = [(loser, "Queen")]
        # Surround the queen with six pieces from the winner
        for dq, dr in game.DIRECTIONS:
            state["board"][(dq, dr)] = [(winner, "Ant")]
        return game, state

    def test_player1_wins_by_surround(self):
        game, state = self._surrounded_queen_state("Player1")
        # Player2's queen is surrounded so Player1 should win
        self.assertEqual(game.getGameOutcome(state), "Player1")
        self.assertTrue(game.isTerminal(state))
        self.assertEqual(game.evaluateState("Player1", state), 10000)
        self.assertEqual(game.evaluateState("Player2", state), -10000)

    def test_player2_wins_by_surround(self):
        game, state = self._surrounded_queen_state("Player2")
        # Player1's queen is surrounded so Player2 should win
        self.assertEqual(game.getGameOutcome(state), "Player2")
        self.assertTrue(game.isTerminal(state))
        self.assertEqual(game.evaluateState("Player2", state), 10000)
        self.assertEqual(game.evaluateState("Player1", state), -10000)


if __name__ == "__main__":
    unittest.main()
