import math
import random
import pygame
from typing import Optional, Tuple

from HivePocket.HivePocket import hex_distance, find_queen_position


class MCTSNode:
    """A single node of the Monte‑Carlo Tree.

    Each node stores exactly one *game state* (after the move that led to it)
    and the edge statistics that originate **from its parent**.  Children are
    stored in a dict keyed by the *action* that leads to them so that we can
    recover the edge data easily.
    """

    def __init__(self, state: dict, parent: Optional["MCTSNode"] = None,
                 forced_depth_left: int = 0):
        self.state           = state
        self.parent          = parent
        self.children        = {}          # action -> MCTSNode
        self.visit_count     = 0
        self.total_value     = 0.0
        self._legal_override = None        # Optional[set] of forced actions
        self.forced_depth_left = forced_depth_left

    # ---------------------------------------------------------------------
    # Cheap helpers -------------------------------------------------------
    # ---------------------------------------------------------------------
    def is_terminal(self, game) -> bool:
        return game.isTerminal(self.state)

    def average_value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.total_value / self.visit_count

    # -----------------------------------------------------------------
    # Tree policy ------------------------------------------------------
    # -----------------------------------------------------------------
    def best_child(self, c_param: float) -> "MCTSNode":
        """UCB1 selection among this node's *expanded* children."""
        assert self.children, "best_child called on a leaf"
        best_score   = -float("inf")
        best_children = []
        ln_parent = math.log(self.visit_count + 1)  # +1 guards log(0)
        for action, child in self.children.items():
            if child.visit_count == 0:
                score = float("inf")
            else:
                exploit = child.total_value / child.visit_count
                explore = math.sqrt(2 * ln_parent / child.visit_count)
                score   = exploit + c_param * explore
            if score > best_score + 1e-12:
                best_score   = score
                best_children = [child]
            elif abs(score - best_score) <= 1e-12:
                best_children.append(child)
        return random.choice(best_children)

    def is_fully_expanded(self, game) -> bool:
        legal = self._legal_override if self._legal_override is not None else set(game.getLegalActions(self.state))
        return len(self.children) == len(legal)

    def prune_to_actions(self, allowed_actions):
        """Drop children not in *allowed_actions* and lock future expansions."""
        self.children = {a: n for a, n in self.children.items() if a in allowed_actions}
        self._legal_override = set(allowed_actions)

    def expand(self, game, mcts) -> "MCTSNode":
        """Expand on an untried legal action and return the created child."""
        if self.forced_depth_left > 0 and not self.is_terminal(game):
            forced, safe = mcts._forced_move_check(self.state, self.forced_depth_left)
            if forced is not None:
                safe = [forced]
            if safe:
                self.prune_to_actions(safe)
            self.forced_depth_left = 0  # do it only once

        legal = self._legal_override if self._legal_override is not None else set(game.getLegalActions(self.state))
        untried = list(legal - self.children.keys())
        assert untried, "expand called on fully expanded node"

        action = random.choice(untried)
        next_state = game.applyAction(self.state, action)
        child_forced = max(0, self.forced_depth_left - 1)
        child = MCTSNode(next_state, parent=self, forced_depth_left=child_forced)
        self.children[action] = child
        return child

    # -----------------------------------------------------------------
    # Back‑prop --------------------------------------------------------
    # -----------------------------------------------------------------
    def update(self, value: float):
        assert -1.0 - 1e-6 <= value <= 1.0 + 1e-6, "value outside [-1,1]"
        self.visit_count += 1
        self.total_value += value

    def __str__(self):
        return (f"MCTSNode(visits={self.visit_count}, avg={self.average_value():.3f}, "
                f"forced={self.forced_depth_left})")


# =============================================================================
# Main MCTS driver
# =============================================================================
class MCTS:
    """A minimal (but now more assertive) MCTS implementation for Hive."""

    def __init__(self, game, *, perspective_player: str,
                 forced_check_depth: int = 1, num_iterations: int = 1000,
                 max_depth: int = 20, c_param: float = 1.4,
                 eval_func=None, weights=None):

        self.game               = game
        self.forced_check_depth = forced_check_depth
        self.num_iterations     = num_iterations
        self.max_depth          = max_depth
        self.c_param            = c_param
        self.eval_func          = eval_func or self.game.evaluateState
        self.weights            = weights
        self.perspective_player = perspective_player

    # -----------------------------------------------------------------
    # Public API -------------------------------------------------------
    # -----------------------------------------------------------------
    def search(self, root_state: dict, draw_callback=None):
        assert root_state["current_player"] == self.perspective_player, (
            "Perspective player mismatch: the root state's current player must "
            "match the MCTS agent's perspective.")

        root = MCTSNode(root_state, None, self.forced_check_depth)

        for i in range(self.num_iterations):
            # --------------- HOUSEKEEPING ---------------
            pygame.event.pump()   # keep the window responsive

            # --------------- SELECTION ------------------
            node = self._select(root)

            # --------------- EXPANSION ------------------
            if not node.is_terminal(self.game) and not node.is_fully_expanded(self.game):
                node = node.expand(self.game, self)

            # --------------- SIMULATION -----------------
            sim_val = self.game.simulateRandomPlayout(
                node.state,
                self.perspective_player,
                max_depth=self.max_depth,
                eval_func=self.eval_func,
                weights=self.weights)
            assert -1.0 - 1e-6 <= sim_val <= 1.0 + 1e-6, "simulateRandomPlayout must return in [-1,1]"

            # --------------- BACKPROP -------------------
            self._backpropagate(node, sim_val)

            # --------------- UI CALLBACK ----------------
            if draw_callback is not None:
                draw_callback(root, i)
            pygame.time.delay(1)

        # Final sanity: did we update exactly *num_iterations* paths?
        assert root.visit_count == self.num_iterations, (
            f"root.visit_count={root.visit_count} but expected {self.num_iterations}")

        best_action = self._best_action(root)
        return best_action

    # -------------------------------------------------------------
    # Internal helpers --------------------------------------------
    # -------------------------------------------------------------
    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.is_terminal(self.game) and node.is_fully_expanded(self.game):
            node = node.best_child(self.c_param)
        return node

    def _best_action(self, root: MCTSNode):
        assert root.children, "No actions available from root"
        best_action, best_visits = None, -1
        for action, child in root.children.items():
            if child.visit_count > best_visits:
                best_action, best_visits = action, child.visit_count
        return best_action

    def _backpropagate(self, node: MCTSNode, value: float):
        while node is not None:
            node.update(value)
            node = node.parent
            value = -value  # flip perspective at each level

    # -------------------------------------------------------------
    # Forced‑move pruning (unchanged) ------------------------------
    # -------------------------------------------------------------
    def _forced_check_depth_limited(self, state, player, depth):
        outcome = self.game.getGameOutcome(state)
        if outcome is not None:
            return outcome == player
        if depth == 0:
            return False
        to_move = self.game.getCurrentPlayer(state)
        actions = self.game.getLegalActions(state)
        if to_move == player:
            return any(self._forced_check_depth_limited(self.game.applyAction(state, a), player, depth - 1)
                       for a in actions)
        else:
            return all(self._forced_check_depth_limited(self.game.applyAction(state, a), player, depth - 1)
                       for a in actions)

    def _forced_move_check(self, state, depth):
        current = self.game.getCurrentPlayer(state)
        legal   = self.game.getLegalActions(state)
        forced_win = None
        safe        = []
        for a in legal:
            nxt = self.game.applyAction(state, a)
            out = self.game.getGameOutcome(nxt)
            if out is not None:
                if out == current:
                    return a, []
                elif out == "Draw":
                    safe.append(a)
                continue
            if self._forced_check_depth_limited(nxt, current, depth - 1):
                return a, []
            opp = self.game.getOpponent(current)
            if not self._forced_check_depth_limited(nxt, opp, depth - 1):
                safe.append(a)
        return forced_win, safe

    # -------------------------------------------------------------
    # Misc convenience --------------------------------------------
    # -------------------------------------------------------------
    def weightedActionChoice(self, state, actions):
        board = state["board"]
        enemy  = self.game.getOpponent(state["current_player"])
        pos    = find_queen_position(board, enemy)
        if pos is None:
            return random.choice(actions)
        eq_q, eq_r = pos
        weights = []
        for act in actions:
            if act[0] == "PLACE":
                _, _, (q, r) = act
            else:
                _, _, (q, r) = act  # MOVE, we only look at destination
            d = hex_distance(q, r, eq_q, eq_r)
            w = 1.0 / (1.0 + 2.0 * d * d)
            weights.append(w)
        total = sum(weights)
        r = random.random() * total
        acc = 0.0
        for act, w in zip(actions, weights):
            acc += w
            if r < acc:
                return act
        return actions[-1]
