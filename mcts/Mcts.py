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
        self._legal_actions_override = None  # new

    def prune_to_actions(self, game, allowed_actions):
        """
        Limit this node's expansions to 'allowed_actions' only.
        Also discard any children that aren't in the allowed set.
        """
        # Discard children for moves not in allowed_actions
        for action in list(self.children.keys()):
            if action not in allowed_actions:
                del self.children[action]

        # Store this for use in expand():
        self._legal_actions_override = set(allowed_actions)

    def is_fully_expanded(self, game):
        if self._legal_actions_override is not None:
            return len(self.children) == len(self._legal_actions_override)
        else:
            legal_actions = game.getLegalActions(self.state)
            return len(self.children) == len(legal_actions)

    def expand(self, game):
        # Retrieve the set of actions from our override or from the game
        if self._legal_actions_override is not None:
            legal_actions = self._legal_actions_override
        else:
            legal_actions = set(game.getLegalActions(self.state))

        tried_actions = set(self.children.keys())
        untried_actions = list(legal_actions - tried_actions)

        if not untried_actions:
            return None

        action = random.choice(untried_actions)
        next_state = game.applyAction(self.state, action)

        child_node = MCTSNode(next_state, parent=self)
        self.children[action] = child_node
        return child_node

    def best_child(self, c_param=1.4):
        best_score = -float('inf')
        best_children = []

        for action, child in self.children.items():
            if child.visit_count == 0:
                # Give unvisited children a large UCB to encourage exploration
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
    A generic MCTS implementation with optional shallow forced-move checks.
    """
    def __init__(self,
                 game,
                 win_reward=1.0,
                 lose_reward=-1.0,
                 draw_reward=0.0,
                 c_param=1.4,
                 num_iterations=1000,
                 do_forced_move_check=True):
        """
        :param game: an instance of some Game class with required interface
        :param win_reward: numeric reward if root player wins
        :param lose_reward: numeric reward if root player loses
        :param draw_reward: numeric reward if it's a draw
        :param c_param: exploration constant for UCB
        :param num_iterations: number of MCTS simulations
        :param do_forced_move_check: whether to perform the forced-move filter at the root
        """
        self.game = game
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.draw_reward = draw_reward
        self.c_param = c_param
        self.num_iterations = num_iterations
        self.do_forced_move_check = do_forced_move_check


    def search(self, root_state):
        root_node = MCTSNode(root_state, parent=None)

        if self.do_forced_move_check:
            forced_action, safe_actions = self._forced_move_check(root_state)

            if forced_action is not None:
                # Found a forced immediate win — just return it
                return forced_action

            if len(safe_actions) > 0:
                # We found some moves that do NOT lead to an immediate forced loss
                # So let's prune the root so that MCTS only explores safe_actions.
                # That way, forced-losing moves won't appear in the tree at all.
                root_node.prune_to_actions(self.game, safe_actions)
            else:
                # All moves appear to be losing (or we might have chosen not to treat draws as safe).
                # We can either:
                #   (a) Just do MCTS over all moves anyway, or
                #   (b) Return any move (since they are all losing).
                # For demonstration, let’s just run MCTS normally:
                pass

        # If we get here, run normal MCTS (or MCTS restricted to safe moves if we pruned).
        for _ in range(self.num_iterations):
            # 1. SELECT
            node = self._select(root_node)

            # 2. EXPAND
            if (not node.is_terminal(self.game)) and (not node.is_fully_expanded(self.game)):
                node = node.expand(self.game)

            # 3. SIMULATE
            final_state = self.game.simulateRandomPlayout(node.state)

            # 4. BACKPROP
            outcome = self.game.getGameOutcome(final_state)
            self._backpropagate(node, final_state, root_node, outcome)

        best_action, _ = self._best_action(root_node)
        return best_action

    def _select(self, node):
        """
        Selection phase: descend the tree using UCB until
        reaching a node that is either terminal or not fully expanded.
        """
        while not node.is_terminal(self.game) and node.is_fully_expanded(self.game):
            node = node.best_child(c_param=self.c_param)
        return node

    def _backpropagate(self, node, final_state, root_node, outcome):
        """
        Backpropagation: update the path from node up to the root.
        We consider the root player's perspective for the reward.
        """
        root_player = self.game.getCurrentPlayer(root_node.state)
        reward = self._getRewardFromOutcome(outcome, root_player)

        while node is not None:
            node.update(reward)
            node = node.parent

    def _getRewardFromOutcome(self, outcome, root_player):
        """
        Convert "Player1"|"Player2"|"Draw"|None into numeric reward
        from the perspective of `root_player`.
        """
        if outcome is None:
            # If the game wasn't terminal, no reward
            return 0.0

        if outcome == root_player:
            return self.win_reward
        elif outcome == "Draw":
            return self.draw_reward
        else:
            # The other player won => root_player lost
            return self.lose_reward

    def _best_action(self, root_node):
        """
        Among the children of the root, pick the action that
        has the highest visit_count (or you can pick best average value).
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

    # -------------------------------------------------------------------------
    # Shallow forced-move logic (1-ply check)
    # -------------------------------------------------------------------------
    def _forced_move_check(self, state):
        """
        Same shallow check:
          - Return an immediate winning action if found.
          - Otherwise return a tuple: (None, list_of_safe_actions).
        """
        current_player = self.game.getCurrentPlayer(state)
        legal_actions = self.game.getLegalActions(state)

        safe_actions = []
        for action in legal_actions:
            next_state = self.game.applyAction(state, action)
            outcome = self.game.getGameOutcome(next_state)

            # Case 1: Immediate win
            if outcome == current_player:
                # This action leads directly to a terminal win. Choose it outright.
                return action, []

            # If the outcome was the opponent or draw or None, we keep checking
            if outcome is not None:
                # If it's a terminal state but not a win for the current player,
                # it's either a loss or a draw. If it's a loss, we skip it;
                # if it's a draw, we can consider it safe or skip it — your choice.
                if outcome != "Draw":
                    # It's a forced immediate loss => skip
                    continue
                else:
                    # If it's a draw, call that "safe enough."
                    safe_actions.append(action)
            else:
                # Not terminal => check if the opponent can force an immediate win next turn
                opponent = self.game.getOtherPlayer(current_player)
                if self._opponent_forces_win(next_state, opponent):
                    # The opponent can force a win => skip
                    continue
                else:
                    safe_actions.append(action)

        # If we reach here, no immediate winning move was found.
        return None, safe_actions

    def _opponent_forces_win(self, state, opponent):
        """
        Check if the opponent has an immediate winning move in `state`.
        That is, we see if there's an action that leads to a terminal state
        with outcome == opponent.
        """
        possible_actions = self.game.getLegalActions(state)
        for opp_action in possible_actions:
            opp_next_state = self.game.applyAction(state, opp_action)
            outcome = self.game.getGameOutcome(opp_next_state)
            if outcome == opponent:
                return True
        return False
