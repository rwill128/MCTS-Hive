from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import torch

# Forward-declare for type hints if GameInterface is in another module
# class GameInterface: ...


@dataclass(slots=True)
class AlphaZeroMCTSNode:
    """A node in the MCTS tree for AlphaZero-style search."""
    game_state: Any  # The state of the game this node represents
    parent: Optional[AlphaZeroMCTSNode] = None
    action_that_led_here: Optional[Any] = None  # The action taken from parent to reach this node
    
    children: Dict[Any, AlphaZeroMCTSNode] = field(default_factory=dict)
    visit_count: int = 0
    total_action_value: float = 0.0  # W(s,a) - sum of values from simulations through this child
    
    # Policy prior P(s,a) from the network for the action leading to this child
    # This is stored in the child node, but corresponds to P(parent_state, action_that_led_here)
    prior_probability: float = 0.0

    # Value estimate V(s) for this node's state, obtained from the network during expansion
    # This is used if the node itself is a leaf that gets expanded.
    _value_from_network: Optional[float] = None

    def __post_init__(self):
        # Ensure prior_probability is float, helps with PUCT calculation
        if not isinstance(self.prior_probability, float):
            self.prior_probability = float(self.prior_probability) if self.prior_probability is not None else 0.0


    @property
    def Q_value(self) -> float:
        """Mean action value Q(s,a)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_action_value / self.visit_count

    def is_leaf(self) -> bool:
        return not self.children

    def select_child_puct(self, c_puct: float) -> AlphaZeroMCTSNode:
        """Selects the child with the highest PUCT score."""
        if not self.children:
            raise ValueError("Cannot select child from a leaf node or unexpanded node.")

        best_score = -float('inf')
        best_child_node = None
        parent_visit_count = self.visit_count

        for action, child_node in self.children.items():
            exploitation_term = child_node.Q_value
            exploration_term = c_puct * child_node.prior_probability * \
                               (math.sqrt(parent_visit_count + 1e-8) / (1 + child_node.visit_count)) # Added 1e-8 for sqrt(0)
            
            score = exploitation_term + exploration_term

            if score > best_score:
                best_score = score
                best_child_node = child_node
        
        if best_child_node is None:
            if not self.children: # Should not happen due to earlier check but defensive
                 raise ValueError("Node has no children to select from in PUCT, this state should be terminal or unexpanded.")
            return random.choice(list(self.children.values())) # Fallback
            
        return best_child_node


class AlphaZeroMCTS:
    def __init__(self, game_interface: Any, model_fn: Callable, device: torch.device,
                 c_puct: float = 1.41,
                 dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25):
        """
        AlphaZero-style Monte Carlo Tree Search.

        Args:
            game_interface: An object that implements the game's logic, including:
                - get_current_player(state)
                - get_legal_actions(state) -> List[action]
                - apply_action(state, action) -> new_state
                - is_terminal(state) -> bool
                - get_game_outcome(state) -> value (e.g., 1 for P1 win, -1 for P2 win, 0 for draw)
                           Value should be from the perspective of the current player in 'state'.
                - encode_state_for_net(state, player_perspective) -> network_input
            model_fn: A callable that takes a batch of encoded game states and returns
                      a tuple (policy_logits_batch, value_batch).
                      - policy_logits_batch: (batch_size, num_actions)
                      - value_batch: (batch_size, 1)
            c_puct: Exploration constant for PUCT.
            dirichlet_alpha: Alpha parameter for Dirichlet noise.
            dirichlet_epsilon: Epsilon for Dirichlet noise mixing.
        """
        self.game = game_interface
        self.model_fn = model_fn
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self._apply_dirichlet_noise_to_root = False # Initialized here

    def _get_device(self) -> torch.device:
        return self.device

    def run_simulations(self, root_state: Any, num_simulations: int, 
                        current_player_perspective: Any, debug_mcts: bool = False) -> Tuple[Dict[Any, float], float]:
        if debug_mcts: print("\n--- MCTS run_simulations START ---")
        if debug_mcts: print(f"[MCTS] Root state player: {self.game.getCurrentPlayer(root_state)}, Perspective: {current_player_perspective}, Sims: {num_simulations}")

        root_node = AlphaZeroMCTSNode(game_state=self.game.copyState(root_state))
        self._apply_dirichlet_noise_to_root = True 

        for sim_idx in range(num_simulations):
            if debug_mcts and sim_idx < 3: print(f"  [MCTS] Simulation {sim_idx + 1}/{num_simulations}")
            current_node = root_node
            path = [current_node]

            # 1. Selection
            while not current_node.is_leaf() and not self.game.isTerminal(current_node.game_state):
                selected_child = current_node.select_child_puct(self.c_puct)
                if debug_mcts and sim_idx < 1 and len(path) < 3: # Log first few selections in first sim
                    print(f"    [MCTS Select] Parent V={current_node.visit_count}, Q={current_node.Q_value:.3f}")
                    for act, child in current_node.children.items():
                        print(f"      Child action {act}: V={child.visit_count}, Q={child.Q_value:.3f}, P={child.prior_probability:.3f}")
                    print(f"    [MCTS Select] Chose action {selected_child.action_that_led_here} (Child V={selected_child.visit_count}, Q={selected_child.Q_value:.3f})")
                current_node = selected_child
                path.append(current_node)
            
            leaf_node = current_node
            leaf_state = leaf_node.game_state
            value_from_net_for_leaf_player = 0.0
            value_estimate_for_root_player = 0.0
            
            # 2. Expansion & Evaluation
            if self.game.isTerminal(leaf_state):
                game_outcome = self.game.getGameOutcome(leaf_state)
                if debug_mcts and sim_idx < 3 : print(f"    [MCTS Eval] Leaf is terminal. Outcome: {game_outcome}")
                if game_outcome == "Draw": value_estimate_for_root_player = 0.0
                elif game_outcome == current_player_perspective: value_estimate_for_root_player = 1.0
                else: value_estimate_for_root_player = -1.0
            else:
                if debug_mcts and sim_idx < 3 : print(f"    [MCTS Eval] Leaf not terminal. Expanding state (player {self.game.getCurrentPlayer(leaf_state)})...")
                encoded_state = self.game.encode_state(leaf_state, self.game.getCurrentPlayer(leaf_state))
                encoded_state_tensor = encoded_state.unsqueeze(0).to(self._get_device()) if isinstance(encoded_state, torch.Tensor) else torch.from_numpy(encoded_state).unsqueeze(0).to(self._get_device())

                with torch.no_grad():
                    policy_logits, net_value_output = self.model_fn(encoded_state_tensor)
                
                policy_logits = policy_logits.squeeze(0)
                value_from_net_for_leaf_player = net_value_output.item()
                leaf_node._value_from_network = value_from_net_for_leaf_player
                if debug_mcts and sim_idx < 3 : print(f"    [MCTS Eval] Network raw policy logits (example): {policy_logits[:3]}... Value for leaf player: {value_from_net_for_leaf_player:.3f}")

                if self.game.getCurrentPlayer(leaf_state) != current_player_perspective:
                    value_estimate_for_root_player = -value_from_net_for_leaf_player
                else:
                    value_estimate_for_root_player = value_from_net_for_leaf_player
                if debug_mcts and sim_idx < 3 : print(f"    [MCTS Eval] Value estimate for root player perspective: {value_estimate_for_root_player:.3f}")

                legal_actions = self.game.getLegalActions(leaf_state)
                if legal_actions:
                    policy_priors_raw = torch.softmax(policy_logits, dim=0).cpu().numpy()
                    if debug_mcts and sim_idx < 3 : print(f"    [MCTS Expand] Softmaxed priors (example): {policy_priors_raw[:3]}...")
                    masked_policy_priors = {action: policy_priors_raw[action] for action in legal_actions}
                    
                    if leaf_node == root_node and self._apply_dirichlet_noise_to_root and legal_actions:
                        if debug_mcts and sim_idx < 3 : print(f"    [MCTS Expand] Applying Dirichlet noise (alpha={self.dirichlet_alpha}, eps={self.dirichlet_epsilon})")
                        num_legal = len(legal_actions)
                        dirichlet_noise = np.random.dirichlet([self.dirichlet_alpha] * num_legal)
                        for i, action in enumerate(legal_actions):
                            masked_policy_priors[action] = (1 - self.dirichlet_epsilon) * masked_policy_priors[action] + \
                                                             self.dirichlet_epsilon * dirichlet_noise[i]
                        if debug_mcts and sim_idx < 3 : print(f"    [MCTS Expand] Priors after noise (first few legal): {[(k, v) for k,v in list(masked_policy_priors.items())[:3]]}")
                        self._apply_dirichlet_noise_to_root = False

                    for action_int in legal_actions:
                        prior_p = masked_policy_priors.get(action_int, 0.0)
                        child_game_state = self.game.applyAction(self.game.copyState(leaf_state), action_int)
                        new_child_node = AlphaZeroMCTSNode(
                            game_state=child_game_state, parent=leaf_node,
                            action_that_led_here=action_int, prior_probability=float(prior_p) )
                        leaf_node.children[action_int] = new_child_node
                        if debug_mcts and sim_idx < 1 and len(path)==1: # Only for root expansion in first sim
                            print(f"      [MCTS Expand] Created child for action {action_int} with prior {prior_p:.3f}")
            
            # 3. Backpropagation
            if debug_mcts and sim_idx < 3 : print(f"    [MCTS Backprop] Path len: {len(path)}, Value for root: {value_estimate_for_root_player:.3f}")
            for node_in_path in reversed(path):
                node_in_path.visit_count += 1
                # Correct perspective for total_action_value update
                is_current_player_at_node_same_as_root_perspective = (self.game.getCurrentPlayer(node_in_path.game_state) == current_player_perspective)
                if node_in_path.parent is not None: # For children, this Q is from parent's perspective
                    # Value for root player must be negated if parent's turn was different from root player
                    if self.game.getCurrentPlayer(node_in_path.parent.game_state) == current_player_perspective:
                        node_in_path.total_action_value += value_estimate_for_root_player
                    else:
                        node_in_path.total_action_value -= value_estimate_for_root_player
                elif node_in_path == root_node: # For root node, directly use value_estimate_for_root_player
                     node_in_path.total_action_value += value_estimate_for_root_player
                
                if debug_mcts and sim_idx < 1 and len(path) - path.index(node_in_path) <=2 : # Log last few backprop steps
                    print(f"      [MCTS Backprop] Node (player {self.game.getCurrentPlayer(node_in_path.game_state)}), V_count={node_in_path.visit_count}, TotalQ={node_in_path.total_action_value:.3f}, AvgQ={node_in_path.Q_value:.3f}")
        
        action_probs: Dict[Any, float] = {}
        legal_root_actions = self.game.getLegalActions(root_state)
        if not root_node.children: 
            # Root was not expanded (e.g. terminal or 0 simulations that didn't even run once)
            if not self.game.isTerminal(root_state) and legal_root_actions:
                uniform_prob = 1.0 / len(legal_root_actions)
                for action_key in legal_root_actions: action_probs[action_key] = uniform_prob
            if debug_mcts: print(f"[MCTS] Root has no children. Legal: {legal_root_actions}. Returning uniform/empty: {action_probs}")
        else:
            # Policy is visit counts of children, normalized
            # total_visits_for_policy should be the sum of visits to all actions taken from the root.
            total_visits_for_policy = sum(child.visit_count for child in root_node.children.values())
            
            if total_visits_for_policy > 0 :
                 for action_int, child_node in root_node.children.items():
                    if action_int in legal_root_actions: # Ensure action is legal for current root
                        action_probs[action_int] = child_node.visit_count / total_visits_for_policy
                    # If a child exists for an action no longer legal (e.g. due to game rules not caught by MCTS state alone, rare for TTT)
                    # it won't be included in action_probs here if not in legal_root_actions.
            else: # No child was visited (can happen if num_simulations is very low, e.g., 0 or 1 before full expansion)
                if legal_root_actions:
                    uniform_prob = 1.0 / len(legal_root_actions)
                    for action_key in legal_root_actions: 
                        action_probs[action_key] = uniform_prob
                if debug_mcts: print(f"[MCTS] No child visits from root (total_visits_for_policy=0). Legal: {legal_root_actions}. Returning uniform: {action_probs}")
            
            if debug_mcts: print(f"[MCTS] Final action_probs from root (sum={sum(action_probs.values()):.2f}): {action_probs}")

        mcts_value_of_root = root_node.Q_value 
        if debug_mcts: print(f"[MCTS] MCTS estimated value of root: {mcts_value_of_root:.3f}")
        if debug_mcts: print("--- MCTS run_simulations END ---")
        return action_probs, mcts_value_of_root

    def get_action_policy(self, root_state: Any, num_simulations: int, temperature: float = 1.0, debug_mcts: bool = False) -> Tuple[Any, Dict[Any, float]]:
        current_player = self.game.getCurrentPlayer(root_state)
        mcts_policy_dict, _ = self.run_simulations(root_state, num_simulations, current_player, debug_mcts)
        
        actions = list(mcts_policy_dict.keys())
        if not actions: # No legal actions from MCTS policy (e.g. terminal state or error)
            # Try to get legal actions directly from game as fallback if mcts_policy is empty but state not terminal
            if not self.game.isTerminal(root_state):
                actions = self.game.getLegalActions(root_state)
                if not actions:
                    raise ValueError("MCTS policy and game both give no legal actions from non-terminal root state.")
                # If game gives actions, create a uniform policy for selection
                mcts_policy_dict = {action: 1.0/len(actions) for action in actions}
                policy_target_probs = np.array([mcts_policy_dict[action] for action in actions])
                print(f"Warning: MCTS returned empty policy for non-terminal state. Using uniform over {actions}")
            else: # Terminal state, no actions to choose
                 raise ValueError("No legal actions from root state in MCTS get_action_policy, state is likely terminal.")
        else:
            policy_target_probs = np.array([mcts_policy_dict[action] for action in actions])

        if not actions: # Should be caught above, but as a safeguard
            raise ValueError("No legal actions available for selection.")

        if np.isclose(temperature, 0.0) or len(actions) == 1: 
            chosen_action_idx_in_list = np.argmax(policy_target_probs) 
            chosen_action = actions[chosen_action_idx_in_list]
        else:
            # Temperature sampling based on policy probabilities (derived from visit counts)
            # AlphaZero uses pi(a|s) = N(s,a)^(1/T) / sum_b N(s,b)^(1/T)
            # Our mcts_policy_dict contains N(s,a)/sum_N. To use T, we should ideally work from N(s,a).
            # For simplicity with current return, if T=1, sample from probs. If T != 1, this is an approximation.
            # A more direct way would be to get N(s,a) counts from root_node.children after simulation.
            
            # Using probabilities directly with temperature (approximation if T!=1)
            powered_probs = np.power(policy_target_probs, 1.0 / temperature)
            final_probs_for_sampling = powered_probs / np.sum(powered_probs)
            # Ensure it sums to 1 due to potential float issues
            final_probs_for_sampling = final_probs_for_sampling / np.sum(final_probs_for_sampling)
            
            chosen_action_idx_in_list = np.random.choice(len(actions), p=final_probs_for_sampling)
            chosen_action = actions[chosen_action_idx_in_list]

        return chosen_action, mcts_policy_dict

# Example Game Interface (conceptual)
# Needs to be adapted for ConnectFour
class GameInterface:
    def getCurrentPlayer(self, state) -> any: raise NotImplementedError
    def getLegalActions(self, state) -> List[any]: raise NotImplementedError
    def applyAction(self, state, action) -> any: raise NotImplementedError
    def isTerminal(self, state) -> bool: raise NotImplementedError
    def getGameOutcome(self, state) -> any: raise NotImplementedError # e.g., 1 for P1 win, -1 P2, 0 Draw
    def encode_state(self, state, player_perspective) -> any: raise NotImplementedError # Returns tensor/numpy for NN
    def copyState(self, state) -> any: raise NotImplementedError
    # For ConnectFour, number of actions is fixed (BOARD_W)
    # def get_action_size(self) -> int: raise NotImplementedError 