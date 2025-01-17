import math
import random

class MCTSNode:
    """
    A node in the game tree for MCTS.
    """
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        # children is a dict: action -> MCTSNode
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0

    def is_fully_expanded(self, game):
        """
        A node is fully expanded if we have
        one child for every possible action.
        """
        legal_actions = game.getLegalActions(self.state)
        return len(self.children) == len(legal_actions)

    def best_child(self, c_param=1.4):
        best_score = -float('inf')
        best_children = []

        for action, child in self.children.items():
            if child.visit_count == 0:
                # Give unvisited children a very large UCB to encourage exploration
                score = float('inf')
            else:
                exploit = child.total_value / child.visit_count
                explore = math.sqrt(2 * math.log(self.visit_count) / child.visit_count)
                score = exploit + c_param * explore

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return random.choice(best_children)

    def expand(self, game):
        """
        Expand one child node by applying one untried action.
        Returns the newly created child node.
        """
        legal_actions = game.getLegalActions(self.state)
        tried_actions = set(self.children.keys())
        untried_actions = [a for a in legal_actions if a not in tried_actions]

        if not untried_actions:
            return None  # No expansion possible

        action = random.choice(untried_actions)
        next_state = game.applyAction(self.state, action)

        child_node = MCTSNode(next_state, parent=self)
        self.children[action] = child_node
        return child_node

    def update(self, value):
        """
        Update this node’s statistics.
        `value` is the reward for the current player
        (or for whichever player we’re tracking).
        """
        self.visit_count += 1
        self.total_value += value

    def is_terminal(self, game):
        return game.isTerminal(self.state)

    def __str__(self):
        """
        String representation of the node showing
        its visit count, value, and state.
        """
        state_str = str(self.state)  # Convert state to string for display
        return (f"MCTSNode(state={state_str}, "
                f"visit_count={self.visit_count}, "
                f"total_value={self.total_value})")


class MCTS:
    """
    A generic MCTS implementation.
    """
    def __init__(self, game, num_iterations=1000, c_param=1.4):
        self.game = game
        self.num_iterations = num_iterations
        self.c_param = c_param

    def search(self, root_state):
        """
        Run MCTS search from the given root state and return the best action.
        """
        root_node = MCTSNode(root_state, parent=None)

        for _ in range(self.num_iterations):
            # 1. SELECT
            node = self._select(root_node)

            # 2. EXPAND
            if (not node.is_terminal(self.game)) and (not node.is_fully_expanded(self.game)):
                node = node.expand(self.game)

            # 3. SIMULATE
            final_state = self.game.simulateRandomPlayout(node.state)

            # 4. BACKPROP
            self._backpropagate(node, final_state, root_node)

        # Return the action leading to the best child of root
        best_action, best_child = self._best_action(root_node)
        return best_action

    def _select(self, node):
        """
        Selection phase: descend tree using UCB until
        reaching a node that is either terminal or not fully expanded.
        """
        while not node.is_terminal(self.game) and node.is_fully_expanded(self.game):
            node = node.best_child(c_param=self.c_param)
        return node

    def _backpropagate(self, node, final_state, root_node):
        """
        Backpropagation: update the path from node up to the root.
        We assume a two-player or multi-player approach,
        so we get the reward for the 'current' player at each ancestor.
        """
        # Figure out the winners or the reward for each player
        # In a simpler scenario, we might assume that the current player
        # at each node is the one whose reward we track. But let's do it
        # in a typical 2-player way, or a single vantage way.
        #
        # For multi-player, you might store stats per player, but we'll keep it simple:
        # We'll track the perspective of the node's "current player" from each step.

        # If you always want the reward from the root's perspective:
        root_player = self.game.getCurrentPlayer(root_node.state)

        while node is not None:
            # We check if the root player won, lost, or drew:
            reward_for_root = self.game.getReward(final_state, root_player)
            # This is +1 if root_player won, -1 if root_player lost, etc.
            node.update(reward_for_root)
            node = node.parent


    def _best_action(self, root_node):
        """
        Among children of the root, pick the action that has the highest visit_count
        (or highest average value, whichever you prefer).
        """
        best_visit_count = -float('inf')
        best_action = None
        best_child = None

        for action, child in root_node.children.items():
            if child.visit_count > best_visit_count:
                best_visit_count = child.visit_count
                best_action = action
                best_child = child

        return best_action, best_child
