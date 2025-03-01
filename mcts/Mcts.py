import math
import random

import pygame

from HivePocket.HivePocket import hex_distance, find_queen_position


class MCTSNode:
    def __init__(self, state, parent=None, forced_depth_left=0):
        self.state = state
        self.parent = parent
        self.children = {}  # action -> MCTSNode
        self.visit_count = 0
        self.total_value = 0.0
        self._legal_actions_override = None
        self.forced_depth_left = forced_depth_left

    def is_terminal(self, game):
        return game.isTerminal(self.state)

    def best_child(self, c_param=1.4):
        best_score = -float('inf')
        best_children = []
        for action, child in self.children.items():
            if child.visit_count == 0:
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
        self.visit_count += 1
        self.total_value += value

    def prune_to_actions(self, game, allowed_actions):
        for action in list(self.children.keys()):
            if action not in allowed_actions:
                del self.children[action]
        self._legal_actions_override = set(allowed_actions)

    def average_value(self):
        if self.visit_count == 0:
            return 0  # Or some default value
        return self.total_value / self.visit_count

    def is_fully_expanded(self, game):
        if self._legal_actions_override is not None:
            return len(self.children) == len(self._legal_actions_override)
        else:
            legal_actions = game.getLegalActions(self.state)
            return len(self.children) == len(legal_actions)

    def expand(self, game, mcts):
        if self.forced_depth_left > 0 and not self.is_terminal(game):
            forced_action, safe_actions = mcts._forced_move_check(self.state, self.forced_depth_left)
            if forced_action is not None:
                safe_actions = [forced_action]
            if len(safe_actions) > 0:
                self.prune_to_actions(game, safe_actions)
            self.forced_depth_left = 0

        if self._legal_actions_override is not None:
            legal_actions = self._legal_actions_override
        else:
            legal_actions = set(game.getLegalActions(self.state))
        tried = set(self.children.keys())
        untried = list(legal_actions - tried)
        if not untried:
            return None
        action = random.choice(untried)
        next_state = game.applyAction(self.state, action)
        child_forced_depth = max(0, self.forced_depth_left - 1)
        child_node = MCTSNode(next_state, parent=self, forced_depth_left=child_forced_depth)
        self.children[action] = child_node
        return child_node

    def __str__(self):
        return (f"MCTSNode(state={self.state}, visit_count={self.visit_count}, "
                f"total_value={self.total_value}, forced_depth_left={self.forced_depth_left})")


class MCTS:
    def __init__(self, game, forced_check_depth=1, num_iterations=1000,
                 max_depth=20, c_param=1.4, eval_func=None, weights=None, perspective_player=None):
        self.game = game
        self.forced_check_depth = forced_check_depth
        self.num_iterations = num_iterations
        self.max_depth = max_depth
        self.c_param = c_param

        # Store the custom evaluation function; if none given, fall back to game.evaluateState
        self.eval_func = eval_func if eval_func is not None else self.game.evaluateState

        # New: which player is this MCTS “for”?
        if perspective_player is None:
            raise ValueError("Must specify perspective_player (e.g. 'Player1' or 'Player2').")
        self.perspective_player = perspective_player

        self.weights = weights

    def search(self, root_state, draw_callback=None):
        root_node = MCTSNode(root_state, None, forced_depth_left=self.forced_check_depth)

        assert root_state["current_player"] == self.perspective_player

        for i in range(self.num_iterations):

            pygame.event.pump()

            # (1) SELECTION
            node = self._select(root_node)

            # (2) EXPANSION
            if not node.is_terminal(self.game) and not node.is_fully_expanded(self.game):
                node = node.expand(self.game, self)

            # (3) SIMULATION
            # Pass eval_func into simulateRandomPlayout
            simulation_value = self.game.simulateRandomPlayout(
                node.state,
                self.perspective_player,
                max_depth=self.max_depth,
                eval_func=self.eval_func,
                weights=self.weights
            )

            # (4) BACKPROPAGATION
            self._backpropagate(node, simulation_value, root_node)

            if draw_callback is not None and i % 1 == 0:
                draw_callback(root_node, i)

            pygame.time.delay(1)

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

    def _backpropagate(self, node, simulation_value, root_node):
        """
        Standard MCTS backprop where each parent's perspective is toggled.
        """
        # good_for_perspective_player = False
        # if simulation_value > 0:
        #     good_for_perspective_player = True
        # if winner_at_end is not None and winner_at_end == self.perspective_player and node.state["current_player"] == self.perspective_player:
        #     print("Current player: " + node.state["current_player"])
        #     print("Winner at end: " + winner_at_end)
        #     print("Simulation value: " + str(simulation_value))
        #     assert simulation_value > 1
        # elif winner_at_end is not None and winner_at_end != self.perspective_player and node.state["current_player"] != self.perspective_player:
        #     print("Perspective player: " + self.perspective_player)
        #     print("Current player: " + node.state["current_player"])
        #     print("Winner at end: " + winner_at_end)
        #     print("Simulation value: " + str(simulation_value))
        #     assert simulation_value < 1

        while node is not None:
            # If the node's current_player does NOT match the node we're back-propagating from,
            # flip the sign.
            #     print("We're back-propping a node where the current player is not the perspective player, flipping the sign.")

            # if node.state["current_player"] == self.perspective_player and good_for_perspective_player:
            #     assert simulation_value > 0
            # if winner_at_end is not None:
            #     print("We have a winner")
            #     if node.state["current_player"] == winner_at_end:
            #         print("Current player: " + node.state["current_player"])
            #         print("Winner at end: " + winner_at_end)
            #         print("Simulation value: " + str(simulation_value))
            #         assert simulation_value > 1
            #     else:
            #         print("Current player: " + node.state["current_player"])
            #         print("Winner at end: " + winner_at_end)
            #         print("Simulation value: " + str(simulation_value))
            #         assert simulation_value < 1

            node.update(simulation_value)
            node = node.parent
            simulation_value = -simulation_value

    def _forced_check_depth_limited(self, state, player, depth):
        outcome = self.game.getGameOutcome(state)
        if outcome is not None:
            return (outcome == player)
        if depth == 0:
            return False
        to_move = self.game.getCurrentPlayer(state)
        actions = self.game.getLegalActions(state)
        if to_move == player:
            for a in actions:
                nxt = self.game.applyAction(state, a)
                if self._forced_check_depth_limited(nxt, player, depth - 1):
                    return True
            return False
        else:
            for a in actions:
                nxt = self.game.applyAction(state, a)
                if not self._forced_check_depth_limited(nxt, player, depth - 1):
                    return False
            return True

    def _forced_move_check(self, state, depth):
        current_player = self.game.getCurrentPlayer(state)
        legal_actions = self.game.getLegalActions(state)
        forced_win_action = None
        safe_actions = []
        for action in legal_actions:
            nxt_state = self.game.applyAction(state, action)
            outcome = self.game.getGameOutcome(nxt_state)
            if outcome is not None:
                if outcome == current_player:
                    return action, []
                elif outcome == "Draw":
                    safe_actions.append(action)
                else:
                    continue
            else:
                if self._forced_check_depth_limited(nxt_state, current_player, depth - 1):
                    return action, []
                else:
                    opp = self.game.getOpponent(current_player)
                    if self._forced_check_depth_limited(nxt_state, opp, depth - 1):
                        continue
                    else:
                        safe_actions.append(action)
        return forced_win_action, safe_actions

    def update_heatmap(self, root_node, screen):
        """
        Draws the board (using the game's state) and overlays the moves from the root node
        with a color shading proportional to the visit count.
        """
        # Draw the base board first.
        # (Assuming you have a draw_hive_board(state, screen) function that draws the board.)
        self.game.drawBoardWithHeatmapOverlay(root_node.state, root_node, screen)

    def weightedActionChoice(self, state, actions):
        board = state["board"]
        current_player = state["current_player"]
        enemy_player = self.game.getOpponent(current_player)
        enemy_queen_pos = find_queen_position(board, enemy_player)
        if enemy_queen_pos is None:
            return random.choice(actions)
        (eq_q, eq_r) = enemy_queen_pos
        weighted_moves = []
        for action in actions:
            if action[0] == "PLACE":
                _, insectType, (q, r) = action
                final_q, final_r = q, r
            elif action[0] == "MOVE":
                _, (fq, fr), (tq, tr) = action
                final_q, final_r = tq, tr
            else:
                final_q, final_r = 0, 0
            dist = hex_distance(final_q, final_r, eq_q, eq_r)
            steepness = 2.0
            weight = 1.0 / (1.0 + steepness * dist**2)
            weighted_moves.append((action, weight))
        total_weight = sum(w for (_, w) in weighted_moves)
        if total_weight == 0.0:
            return random.choice(actions)
        rnd = random.random() * total_weight
        cumulative = 0.0
        for (act, w) in weighted_moves:
            cumulative += w
            if rnd < cumulative:
                return act
        return actions[-1]