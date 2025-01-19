import math
import random

from HivePocket.DrawGame import drawStatePygame


class MCTSNode:
    """
    A node in the game tree for MCTS.
    """
    def __init__(self, state, parent=None, forced_depth_left=0):
        self.state = state
        self.parent = parent
        self.children = {}  # action -> MCTSNode
        self.visit_count = 0
        self.total_value = 0.0

        # For optional pruning:
        self._legal_actions_override = None

        # This is how many plies of forced-check are left
        # to attempt at THIS node when we expand it.
        self.forced_depth_left = forced_depth_left

    def is_terminal(self, game):
        return game.isTerminal(self.state)

    def best_child(self, c_param=1.4):
        best_score = -float('inf')
        best_children = []

        for action, child in self.children.items():
            if child.visit_count == 0:
                score = float('inf')  # Encourage exploring unvisited
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
        self.visit_count += 1
        self.total_value += value

    def prune_to_actions(self, game, allowed_actions):
        # discard child nodes that aren't allowed
        for action in list(self.children.keys()):
            if action not in allowed_actions:
                del self.children[action]
        self._legal_actions_override = set(allowed_actions)

    def is_fully_expanded(self, game):
        if self._legal_actions_override is not None:
            return len(self.children) == len(self._legal_actions_override)
        else:
            legal_actions = game.getLegalActions(self.state)
            return len(self.children) == len(legal_actions)

    def expand(self, game, mcts):
        """
        Expand one child node by applying one untried action,
        but first do forced-check logic if forced_depth_left > 0.
        """
        # Possibly run forced checks at this node. This might prune some moves
        # or even short-circuit if a forced-win is found.
        # We'll do it once at the moment of expansion so we don't do forced checks repeatedly.
        if self.forced_depth_left > 0 and not self.is_terminal(game):
            forced_action, safe_actions = mcts._forced_move_check(self.state, self.forced_depth_left)
            if forced_action is not None:
                # forced_action leads directly to a forced win for the current player
                # so let's prune everything except that move
                safe_actions = [forced_action]
            if len(safe_actions) > 0:
                self.prune_to_actions(game, safe_actions)
            # If safe_actions is empty, we do nothing special
            # => we might expand "bad" moves anyway if we want.

            # We only want to do forced-check once, so we set forced_depth_left=0
            # after we've pruned (so that we don't keep re-running forced checks
            # each time we want to expand or select).
            self.forced_depth_left = 0

        # Now proceed with normal expansion:
        if self._legal_actions_override is not None:
            legal_actions = self._legal_actions_override
        else:
            legal_actions = set(game.getLegalActions(self.state))

        tried = set(self.children.keys())
        untried = list(legal_actions - tried)

        if not untried:
            return None  # fully expanded

        action = random.choice(untried)
        next_state = game.applyAction(self.state, action)

        # For the child, we might want forced_depth_left = (this node's forced_depth_left - 1),
        # because we've used one ply of forced checking. But that depends on your desired logic:
        child_forced_depth = max(0, self.forced_depth_left - 1)

        child_node = MCTSNode(next_state, parent=self, forced_depth_left=child_forced_depth)
        self.children[action] = child_node
        return child_node

    def __str__(self):
        return (f"MCTSNode("
                f"state={self.state}, "
                f"visit_count={self.visit_count}, "
                f"total_value={self.total_value}, "
                f"forced_depth_left={self.forced_depth_left})")

class MCTS:
    def __init__(self, game, forced_check_depth=1, num_iterations=1000, c_param=1.4,
                 win_reward=1.0, lose_reward=-1.0, draw_reward=0.0):
        self.game = game
        self.forced_check_depth = forced_check_depth
        self.num_iterations = num_iterations
        self.c_param = c_param
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.draw_reward = draw_reward

    def search(self, root_state):
        # Root node with forced_depth_left = self.forced_check_depth
        root_node = MCTSNode(root_state, None, forced_depth_left=self.forced_check_depth)

        # We do not forcibly run forced checks at the root.
        # Because the expand(...) method will do it anyway
        # the first time we try to expand the root.
        # Or, if you prefer to do it immediately, you can do so here.

        for _ in range(self.num_iterations):
            # 1. SELECT
            node = self._select(root_node)

            # 2. EXPAND
            if not node.is_terminal(self.game) and not node.is_fully_expanded(self.game):
                node = node.expand(self.game, self)

            # 3. SIMULATE
            final_state = self.game.simulateRandomPlayout(node.state)

            # print("Simulated a random playout")
            # print("Self.numiterations: " + str(self.num_iterations))
            # drawStatePygame(final_state)

            # 4. BACKPROP
            outcome = self.game.getGameOutcome(final_state)
            self._backpropagate(node, final_state, root_node, outcome)

        print("Actions info: " + str(root_node.children))

        best_action, best_child = self._best_action(root_node)
        return best_action

    def _select(self, node):
        while not node.is_terminal(self.game) and node.is_fully_expanded(self.game):
            node = node.best_child(c_param=self.c_param)
        return node

    def _best_action(self, root_node):
        best_action, best_child = None, None
        best_visits = -float('inf')
        for action, child in root_node.children.items():
            if child.visit_count > best_visits:
                best_action = action
                best_child = child
                best_visits = child.visit_count
        return best_action, best_child

    def _backpropagate(self, node, final_state, root_node, outcome):
        root_player = self.game.getCurrentPlayer(root_node.state)
        reward = self._getRewardFromOutcome(outcome, root_player)

        while node is not None:
            node.update(reward)
            node = node.parent

    def _getRewardFromOutcome(self, outcome, root_player):
        if outcome is None:
            return 0.0
        elif outcome == root_player:
            return self.win_reward
        elif outcome == "Draw":
            return self.draw_reward
        else:
            return self.lose_reward

    # =======================
    # forced-check subroutines
    # =======================
    def _forced_check_depth_limited(self, state, player, depth):
        """
        Minimax-like check: can `player` force a win in <= `depth` plies?
        """
        outcome = self.game.getGameOutcome(state)
        if outcome is not None:
            return (outcome == player)
        if depth == 0:
            return False

        to_move = self.game.getCurrentPlayer(state)
        actions = self.game.getLegalActions(state)

        if to_move == player:
            # If the current player can pick at least one move that
            # eventually leads to a forced win, we return True
            for a in actions:
                nxt = self.game.applyAction(state, a)
                if self._forced_check_depth_limited(nxt, player, depth - 1):
                    return True
            return False
        else:
            # Opponent's turn: if they have ANY move that avoids losing,
            # then we can't guarantee a forced win for `player`.
            for a in actions:
                nxt = self.game.applyAction(state, a)
                if not self._forced_check_depth_limited(nxt, player, depth - 1):
                    return False
            return True

    def _forced_move_check(self, state, depth):
        """
        Returns (forced_win_action, safe_actions) just like before, but we
        do a depth-limited check with `_forced_check_depth_limited(...)`.
        """
        current_player = self.game.getCurrentPlayer(state)
        legal_actions = self.game.getLegalActions(state)

        forced_win_action = None
        safe_actions = []

        for action in legal_actions:
            nxt_state = self.game.applyAction(state, action)
            outcome = self.game.getGameOutcome(nxt_state)

            # immediate check
            if outcome is not None:
                if outcome == current_player:
                    # forced immediate win
                    return action, []
                elif outcome == "Draw":
                    safe_actions.append(action)
                else:
                    # immediate loss => skip
                    continue
            else:
                # Not terminal, check deeper:
                if self._forced_check_depth_limited(nxt_state, current_player, depth - 1):
                    # leads to a forced win
                    return action, []
                else:
                    # see if the opponent can force a win from nxt_state
                    opp = self.game.getOtherPlayer(current_player)
                    if self._forced_check_depth_limited(nxt_state, opp, depth - 1):
                        # opponent can force a win => skip
                        continue
                    else:
                        safe_actions.append(action)

        return forced_win_action, safe_actions
