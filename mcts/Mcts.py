from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple

try:
    import pygame
except ImportError:  # pragma: no cover - pygame optional for headless use
    pygame = None

from HivePocket.HivePocket import hex_distance, find_queen_position
from .eval_cache import EvalCache


@dataclass(slots=True)
class MCTSNode:
    """A single node of the Monte‑Carlo Tree.

    Each node stores exactly one *game state* (after the move that led to it)
    and the edge statistics that originate **from its parent**.  Children are
    stored in a dict keyed by the *action* that leads to them so that we can
    recover the edge data easily.
    """

    state: dict
    parent: Optional["MCTSNode"] = None
    forced_depth_left: int = 0
    children: dict = field(default_factory=dict)
    visit_count: int = 0
    total_value: float = 0.0
    _legal_override: Optional[set] = None
    _legal_cache: Optional[set] = None

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

    def _get_legal(self, game, cache: bool) -> set:
        if self._legal_override is not None:
            return self._legal_override
        if cache:
            if self._legal_cache is None:
                self._legal_cache = set(game.getLegalActions(self.state))
            return self._legal_cache
        return set(game.getLegalActions(self.state))

    def is_fully_expanded(self, game, cache_legal_actions: bool = False) -> bool:
        legal = self._get_legal(game, cache_legal_actions)
        return len(self.children) == len(legal)

    def prune_to_actions(self, allowed_actions):
        """Drop children not in *allowed_actions* and lock future expansions."""
        self.children = {a: n for a, n in self.children.items() if a in allowed_actions}
        allowed = set(allowed_actions)
        self._legal_override = allowed
        self._legal_cache = allowed

    def expand(self, game, mcts) -> "MCTSNode":
        """Expand on an untried legal action and return the created child."""
        if self.forced_depth_left > 0 and not self.is_terminal(game):
            forced, safe = mcts._forced_move_check(self.state, self.forced_depth_left)
            if forced is not None:
                safe = [forced]
            if safe:
                self.prune_to_actions(safe)
            self.forced_depth_left = 0  # do it only once

        legal = self._get_legal(game, mcts.cache_legal_actions)
        untried = list(legal - self.children.keys())
        assert untried, "expand called on fully expanded node"

        action = random.choice(untried)
        next_state = game.applyAction(self.state, action)
        child_forced = max(0, self.forced_depth_left - 1)
        child = MCTSNode(next_state, parent=self, forced_depth_left=child_forced)
        if mcts.cache:
            key = repr(mcts.game.canonical_state_key(next_state))
            cached = mcts.cache.get(key)
            if cached:
                child.visit_count, child.total_value = cached
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
                 minimax_depth: int = 0,
                 eval_func=None, weights=None, cache: Optional[EvalCache] = None,
                 cache_legal_actions: bool = False):

        self.game               = game
        self.forced_check_depth = forced_check_depth
        self.num_iterations     = num_iterations
        self.max_depth          = max_depth
        self.c_param            = c_param
        self.eval_func          = eval_func or self.game.evaluateState
        self.weights            = weights
        self.perspective_player = perspective_player
        self.cache              = cache
        self.minimax_depth      = minimax_depth
        self.cache_legal_actions = cache_legal_actions

    # -----------------------------------------------------------------
    # Public API -------------------------------------------------------
    # -----------------------------------------------------------------
    def search(self, root_state: dict, draw_callback=None):
        assert root_state["current_player"] == self.perspective_player, (
            "Perspective player mismatch: the root state's current player must "
            "match the MCTS agent's perspective.")

        root = MCTSNode(root_state, None, self.forced_check_depth)
        if self.cache:
            key = repr(self.game.canonical_state_key(root_state))
            cached = self.cache.get(key)
            if cached:
                root.visit_count, root.total_value = cached
        start_visits = root.visit_count

        for i in range(self.num_iterations):
            # --------------- HOUSEKEEPING ---------------
            if pygame is not None and hasattr(pygame, "get_init") and pygame.get_init():
                pygame.event.pump()   # keep the window responsive

            # --------------- SELECTION ------------------
            node = self._select(root)

            # --------------- EXPANSION ------------------
            if not node.is_terminal(self.game) and not node.is_fully_expanded(self.game, self.cache_legal_actions):
                node = node.expand(self.game, self)

            # --------------- SIMULATION -----------------
            sim_val = self._simulate(node.state)
            assert -1.0 - 1e-6 <= sim_val <= 1.0 + 1e-6, "simulateRandomPlayout must return in [-1,1]"

            # --------------- BACKPROP -------------------
            self._backpropagate(node, sim_val)

            # --------------- UI CALLBACK ----------------
            if draw_callback is not None:
                draw_callback(root, i)
            if pygame is not None:
                pygame.time.delay(1)

        # Final sanity: did we update exactly *num_iterations* new paths?
        assert root.visit_count == start_visits + self.num_iterations, (
            f"root.visit_count={root.visit_count} but expected {start_visits + self.num_iterations}")

        if self.cache:
            self.cache.save()

        best_action = self._best_action(root)
        if self.minimax_depth > 0:
            mm_action, _ = self._minimax_search(root_state, self.minimax_depth)
            return mm_action
        return best_action

    # -------------------------------------------------------------
    # Internal helpers --------------------------------------------
    # -------------------------------------------------------------
    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.is_terminal(self.game) and node.is_fully_expanded(self.game, self.cache_legal_actions):
            node = node.best_child(self.c_param)
        return node

    def _best_action(self, root: MCTSNode):
        assert root.children, "No actions available from root"
        best_action, best_visits = None, -1
        for action, child in root.children.items():
            if child.visit_count > best_visits:
                best_action, best_visits = action, child.visit_count
        return best_action

    def _simulate(self, state: dict) -> float:
        """Run a default random playout from ``state``."""
        return self.game.simulateRandomPlayout(
            state,
            self.perspective_player,
            max_depth=self.max_depth,
            eval_func=self.eval_func,
            weights=self.weights,
        )

    def _backpropagate(self, node: MCTSNode, value: float):
        while node is not None:
            node.update(value)
            if self.cache:
                key = repr(self.game.canonical_state_key(node.state))
                self.cache.increment(key, value)
            node = node.parent
            value = -value  # flip perspective at each level

    # -------------------------------------------------------------
    # Forced‑move pruning (unchanged) ------------------------------
    # -------------------------------------------------------------
    def _forced_check_depth_limited(self, state, player, depth):
        """Return True if ``player`` can force a win within ``depth`` moves."""
        outcome = self.game.getGameOutcome(state)
        if outcome is not None:
            return outcome == player
        if depth == 0:
            return False
        to_move = self.game.getCurrentPlayer(state)
        actions = self.game.getLegalActions(state)
        if to_move == player:
            return any(
                self._forced_check_depth_limited(
                    self.game.applyAction(state, a), player, depth - 1)
                for a in actions)
        else:
            return all(
                self._forced_check_depth_limited(
                    self.game.applyAction(state, a), player, depth - 1)
                for a in actions)

    def _forced_move_check(self, state, depth):
        """Return a winning move or the subset of actions that are safe.

        Parameters
        ----------
        state : dict
            Current game state.
        depth : int
            How deep to search for forced wins.

        Returns
        -------
        Tuple[action | None, list[action]]
            ``(winning_action, safe_actions)`` where ``winning_action`` is a
            move that wins immediately or via a depth-limited forced line.  If
            no such move exists it is ``None`` and ``safe_actions`` contains the
            legal moves that do not allow the opponent a forced reply.
        """
        current = self.game.getCurrentPlayer(state)
        legal = self.game.getLegalActions(state)
        forced_win = None
        safe = []
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
    # Optional minimax search ------------------------------------
    # -------------------------------------------------------------
    def _minimax_value(self, state, depth) -> float:
        outcome = self.game.getGameOutcome(state)
        if outcome == self.perspective_player:
            return 1.0
        if outcome == "Draw":
            return 0.0
        if outcome is not None:
            return -1.0
        if depth == 0:
            return self.eval_func(self.perspective_player, state, self.weights)

        to_move = self.game.getCurrentPlayer(state)
        actions = self.game.getLegalActions(state)
        scores = []
        for action in actions:
            next_state = self.game.applyAction(state, action)
            score = self._minimax_value(next_state, depth - 1)
            scores.append(score)
        if to_move == self.perspective_player:
            return max(scores)
        else:
            return min(scores)

    def _minimax_search(self, state, depth) -> Tuple:
        actions = self.game.getLegalActions(state)
        best_score = -float("inf")
        best_actions = []
        for action in actions:
            next_state = self.game.applyAction(state, action)
            score = self._minimax_value(next_state, depth - 1)
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
        return random.choice(best_actions), best_score

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
