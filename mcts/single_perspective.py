import math
import random

class SingleTurnNode:
    def __init__(self, state, parent=None):
        """
        state: a game state in which it's perspective_player's turn
               OR a terminal state (from perspective_player's point of view).
        """
        self.state = state
        self.parent = parent
        self.children = {}       # Dict[action -> SingleTurnNode]
        self.visit_count = 0
        self.total_value = 0.0

    @property
    def avg_value(self):
        return self.total_value / self.visit_count if self.visit_count else 0

    def is_terminal(self, game):
        return game.isTerminal(self.state)

    def expand_one_child(self, actions, game, roll_forward_fn):
        """
        Expand by choosing one untried action from the perspective player's
        legal moves, applying it, then rolling forward the opponent's moves.
        """
        untried_actions = [a for a in actions if a not in self.children]
        if not untried_actions:
            return None  # No expansion possible

        action = random.choice(untried_actions)
        next_state = game.applyAction(self.state, action)

        # --- ROLL FORWARD OPPONENT TURNS ---
        next_state = roll_forward_fn(next_state, game)

        # Create child node
        child_node = SingleTurnNode(next_state, parent=self)
        self.children[action] = child_node
        return child_node

    def update(self, value):
        """Standard backprop: add `value` to total_value; increment visit_count."""
        self.visit_count += 1
        self.total_value += value

    def best_child(self, c_param):
        """
        Standard UCB for SingleTurnNodes.
        Only perspective player's actions exist as children, so no sign flips.
        """
        best_score = float('-inf')
        best_nodes = []
        for action, child in self.children.items():
            if child.visit_count == 0:
                # Guarantee we pick unvisited children first:
                return child
            exploit = child.avg_value
            explore = math.sqrt(2.0 * math.log(self.visit_count) / child.visit_count)
            score = exploit + c_param * explore
            if score > best_score:
                best_score = score
                best_nodes = [child]
            elif math.isclose(score, best_score):
                best_nodes.append(child)
        return random.choice(best_nodes)

class SinglePerspectiveMCTS:
    def __init__(self,
                 game,
                 perspective_player,
                 num_iterations=1000,
                 max_depth=20,
                 c_param=1.4,
                 eval_func=None,
                 weights=None):
        self.game = game
        self.perspective_player = perspective_player
        self.num_iterations = num_iterations
        self.max_depth = max_depth
        self.c_param = c_param
        # If no eval_func is provided, we fallback to the game's built-in
        self.eval_func = eval_func or game.evaluateState
        self.weights = weights

    def search(self, root_state, draw_callback=None):
        """
        Main entry point: run MCTS from root_state *assuming it's perspective_player's turn*.
        If it's not perspective_player's turn, we roll forward until it is, or until terminal.
        """
        # 1) Make sure we roll forward if it's not perspective_player's turn
        root_state = self._roll_forward(root_state, self.game)

        # If it's terminal after rolling forward, just return PASS or None
        if self.game.isTerminal(root_state):
            return None

        # 2) Create the root node
        root_node = SingleTurnNode(root_state)

        # 3) Run the iterations
        for i in range(self.num_iterations):
            leaf_node = self._select(root_node)
            # If leaf_node is not terminal, expand it:
            if not leaf_node.is_terminal(self.game):
                actions = self.game.getLegalActions(leaf_node.state)
                child = leaf_node.expand_one_child(actions, self.game, self._roll_forward)
                if child is not None:
                    leaf_node = child

            # 4) Simulation
            simulation_value = self._simulate(leaf_node.state)

            # 5) Backpropagation
            self._backpropagate(leaf_node, simulation_value)

            if draw_callback is not None and i % 50 == 0:
                draw_callback(root_node, i)

        # 6) Choose the best action from the root.
        return self._best_action(root_node)

    def _select(self, node):
        """
        Descend the tree along best_child (UCB) until
        we reach a node that isn't fully expanded or is terminal.
        """
        while not node.is_terminal(self.game):
            legal = self.game.getLegalActions(node.state)
            # If fully expanded => pick best child. Else, return node for expansion.
            if len(node.children) < len(legal):
                return node
            else:
                node = node.best_child(self.c_param)
        return node

    def _simulate(self, state):
        """
        Playout from `state` to a maximum depth or terminal,
        using random moves (or weighted random, up to you).
        Return a float evaluation from perspective_player's viewpoint.
        """
        return self.game.simulateRandomPlayout(
            state,
            perspectivePlayer=self.perspective_player,
            max_depth=self.max_depth,
            eval_func=self.eval_func,
            weights=self.weights
        )

    def _backpropagate(self, node, value):
        """
        Standard backprop: each ancestor gets `value` added.
        No sign flips because every node is from the same perspective.
        """
        while node is not None:
            node.update(value)
            node = node.parent

    def _best_action(self, root_node):
        """
        Pick the action with the highest visit count (or highest average, your call).
        """
        best_action = None
        best_visit = -float('inf')
        for action, child in root_node.children.items():
            if child.visit_count > best_visit:
                best_visit = child.visit_count
                best_action = action
        return best_action

    def _roll_forward(self, state, game):
        """
        If it's not perspective_player's turn, we apply the opponent's move(s)
        until either the game is terminal or it becomes perspective_player's turn again.
        For a 2-player game, that means we do exactly one opponent move
        (if it's not terminal), then return.

        If you had more than 2 players, you'd keep going until we circle back
        to perspective_player.  The logic is similar.
        """
        # While not perspective_player and not terminal => apply a random move (or policy)
        while (not game.isTerminal(state)
               and game.getCurrentPlayer(state) != self.perspective_player):
            legal = game.getLegalActions(state)
            if not legal:
                break
            # For real play, you might want an actual opponent policy or
            # another MCTS for the opponent. For now, random:
            opponent_action = random.choice(legal)
            state = game.applyAction(state, opponent_action)
        return state
