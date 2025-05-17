import unittest
import math

# We'll assume you have these imports pointing to the exact same classes
# in your code base. Adjust the import paths as necessary.
from mcts.Mcts import MCTS, MCTSNode
from HivePocket.HivePocket import HiveGame


class MockHiveGame(HiveGame):
    """
    A minimal mock/override of HiveGame for specialized backprop tests.

    We only override what we need, so that we can forcibly control
    the 'getGameOutcome' or 'evaluateState' behavior if we want to test
    sign flipping or perspective changes. This is optional—you can also
    use your real HiveGame with carefully designed states.
    """
    def getGameOutcome(self, state):
        # For demonstration, let's assume that if "winner" is in state dict,
        # we return that, else None. This way we can easily simulate who wins.
        return state.get("winner", None)

    def evaluateState(self, state, weights=None):
        """
        Return a large positive number if the current_player is the winner,
        large negative if the current_player is the loser, else 0.
        This ensures we can test sign flipping from perspective of each node.
        """
        outcome = self.getGameOutcome(state)
        if outcome is None:
            return 0
        if outcome == "Draw":
            return 0
        if outcome == state["current_player"]:
            return +1000
        else:
            return -1000


class TestMCTSBackprop(unittest.TestCase):
    def setUp(self):
        """
        Create an MCTS instance and some MCTSNode objects to test backprop on.
        """
        self.game = MockHiveGame()
        # By default, we’ll use small iteration counts, etc. so we don't
        # accidentally overshadow the backprop test logic.
        self.mcts = MCTS(
            game=self.game,
            forced_check_depth=0,
            num_iterations=1,  # Not relevant for direct backprop tests
            max_depth=1,
            c_param=1.4,
            eval_func=None,    # We'll rely on our MockHiveGame's evaluateState
            weights=None
        )

    def test_simple_backprop_positive(self):
        """
        In this test, the leaf node's perspective is that the simulation value is +1000.
        We check that parent's and root's total_value accumulate the same +1000,
        visit_count increments by 1, etc.
        """

        # 1) Construct states for root, child, leaf with different current_players
        root_state = {
            "board": {},
            "current_player": "Player1",
            "winner": None,  # Not decided in root state
        }
        child_state = {
            "board": {},
            "current_player": "Player2",
            "winner": None,
        }
        leaf_state = {
            "board": {},
            "current_player": "Player1",
            # Let's declare "Player1" the eventual winner from leaf's perspective.
            "winner": "Player1"
        }

        # 2) Build the node chain: leaf -> child -> root
        root_node = MCTSNode(root_state, parent=None)
        child_node = MCTSNode(child_state, parent=root_node)
        leaf_node = MCTSNode(leaf_state, parent=child_node)

        # 3) We artificially produce a simulation_value by calling evaluateState or simulateRandomPlayout
        #    but you can also just fix a number for demonstration. We'll do the real call:
        sim_value = self.game.evaluateState(leaf_state)  # Should be +1000
        self.assertEqual(sim_value, 1000, "Leaf node should see +1000 for Player1's victory.")

        # 4) Call _backpropagate on the leaf_node, passing in sim_value
        self.mcts._backpropagate(leaf_node, simulation_value=sim_value, root_node=root_node)

        # 5) Now we assert that the leaf/child/root node got updated.
        #    Because the default MCTS code updates each ancestor with the *same* sim_value,
        #    (there’s no automatic sign flip in the default code),
        #    we expect each node's total_value == +1000, visit_count == 1.
        self.assertEqual(leaf_node.total_value, 1000)
        self.assertEqual(leaf_node.visit_count, 1)

        self.assertEqual(child_node.total_value, 1000)
        self.assertEqual(child_node.visit_count, 1)

        self.assertEqual(root_node.total_value, 1000)
        self.assertEqual(root_node.visit_count, 1)


    def test_simple_backprop_negative(self):
        """
        Similar to the above, but the leaf's perspective is that the simulation value is -1000.
        We'll confirm that *all* ancestors are updated with -1000 in total_value and visit_count +1.
        """

        root_state = {
            "board": {},
            "current_player": "Player2",  # i.e. from root's perspective, Player2 is to move
            "winner": None
        }
        child_state = {
            "board": {},
            "current_player": "Player1",
            "winner": None
        }
        leaf_state = {
            "board": {},
            "current_player": "Player2",
            # Suppose the outcome is "Player1" wins, so from leaf's perspective,
            # the evaluateState is negative (since leaf is Player2).
            "winner": "Player1"
        }

        root_node = MCTSNode(root_state, parent=None)
        child_node = MCTSNode(child_state, parent=root_node)
        leaf_node = MCTSNode(leaf_state, parent=child_node)

        sim_value = self.game.evaluateState(leaf_state)  # Should be -1000, from Player2's perspective
        self.assertEqual(sim_value, -1000,
                         "Leaf node with current_player=Player2 but winner=Player1 => negative score.")

        # Backprop:
        self.mcts._backpropagate(leaf_node, simulation_value=sim_value, root_node=root_node)

        # Each node in the chain accumulates -1000
        self.assertEqual(leaf_node.total_value, -1000)
        self.assertEqual(leaf_node.visit_count, 1)

        self.assertEqual(child_node.total_value, -1000)
        self.assertEqual(child_node.visit_count, 1)

        self.assertEqual(root_node.total_value, -1000)
        self.assertEqual(root_node.visit_count, 1)

    def test_backprop_player_switch_assertions(self):
        """
        Sometimes people want to verify that the node's 'state.current_player'
        is as expected: child is the *other* player, etc.

        This test doesn't change MCTS code, but shows how you'd check your assumptions.
        If you *modify* _backpropagate to, for example, flip the sign at each parent
        (reflecting that parents are opponents), you'd confirm that logic here.
        """

        root_state = {
            "board": {},
            "current_player": "Player1",
            "winner": None
        }
        child_state = {
            "board": {},
            "current_player": "Player2",
            "winner": None
        }
        leaf_state = {
            "board": {},
            "current_player": "Player1",
            "winner": "Player1"  # Leaf is winning from its own perspective
        }

        root_node = MCTSNode(root_state, parent=None)
        child_node = MCTSNode(child_state, parent=root_node)
        leaf_node = MCTSNode(leaf_state, parent=child_node)

        # We can enforce an expectation that root->child->leaf players are alternating.
        self.assertNotEqual(root_state["current_player"], child_state["current_player"])
        self.assertNotEqual(child_state["current_player"], leaf_state["current_player"])

        # Evaluate from leaf perspective => +1000
        sim_value = self.game.evaluateState(leaf_state)
        self.assertEqual(sim_value, 1000)

        # If you *modified* _backpropagate to do sign flipping for each parent's perspective,
        # you would do something like:
        #
        #   if node.parent is not None:
        #       #  Flip the sign if parent's current_player != node.state["current_player"]
        #
        # Then you’d expect: leaf => +1000, child => -1000, root => +1000 again, etc.
        #
        # For now, we call the standard code (no flips):
        self.mcts._backpropagate(leaf_node, sim_value, root_node)

        # So in the default scenario, each node sees +1000
        self.assertEqual(leaf_node.total_value, 1000)
        self.assertEqual(child_node.total_value, 1000)
        self.assertEqual(root_node.total_value, 1000)


if __name__ == "__main__":
    unittest.main()
