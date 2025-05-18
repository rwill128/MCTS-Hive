import os
import tempfile
import unittest
from examples.ttt_selfplay_rl import RLAgent

class TestRLAgentPersistence(unittest.TestCase):
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "vals.json")
            agent1 = RLAgent(lr=1.0, epsilon=0.0, storage_path=path)
            agent1.play_episode()
            self.assertTrue(agent1.values)
            agent2 = RLAgent(lr=1.0, epsilon=0.0, storage_path=path)
            self.assertEqual(agent1.values, agent2.values)

if __name__ == "__main__":
    unittest.main()
