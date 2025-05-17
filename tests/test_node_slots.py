import unittest
from mcts.Mcts import MCTSNode
from mcts.single_perspective import SingleTurnNode


class TestNodeSlots(unittest.TestCase):
    def test_mctsnode_slots(self):
        node = MCTSNode({'current_player': 'X'})
        with self.assertRaises(AttributeError):
            node.extra_attr = 1

    def test_singleturnnode_slots(self):
        node = SingleTurnNode({'current_player': 'X'})
        with self.assertRaises(AttributeError):
            node.extra_attr = 1


if __name__ == '__main__':
    unittest.main()
