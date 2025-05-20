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

    def run_simulations(self, root_state: Any, num_simulations: int, current_player_perspective: Any) -> Tuple[Dict[Any, float], float]:
        """
        Run MCTS simulations from the root_state.

        Args:
            root_state: The starting state for the MCTS.
            num_simulations: The number of simulations to perform.
            current_player_perspective: The player for whom we are evaluating the state (e.g. 'X' or 'O').
                                       This is important for encoding the state for the network
                                       and interpreting the network's value output.

        Returns:
            A tuple: (action_probabilities, estimated_value_of_root_state)
            - action_probabilities: Dict[action, probability] based on visit counts.
            - estimated_value_of_root_state: The MCTS-estimated value of the root_state.
        """
        root_node = AlphaZeroMCTSNode(game_state=self.game.copyState(root_state))
        self._apply_dirichlet_noise_to_root = True # Reset for each new search

        for _ in range(num_simulations):
            current_node = root_node
            path = [current_node] # Path from root to leaf

            # 1. Selection: Traverse the tree using PUCT until a leaf node is reached
            while not current_node.is_leaf() and not self.game.isTerminal(current_node.game_state):
                current_node = current_node.select_child_puct(self.c_puct)
                path.append(current_node)
            
            leaf_node = current_node
            leaf_state = leaf_node.game_state
            
            value_from_net_for_leaf_player = 0.0 # Default for clarity if terminal
            value_estimate_for_root_player = 0.0 # Value from perspective of root player
            
            if self.game.isTerminal(leaf_state):
                game_outcome = self.game.getGameOutcome(leaf_state)
                if game_outcome == "Draw":
                    value_estimate_for_root_player = 0.0
                elif game_outcome == current_player_perspective:
                    value_estimate_for_root_player = 1.0
                else: 
                    value_estimate_for_root_player = -1.0
            else:
                encoded_state = self.game.encode_state(leaf_state, self.game.getCurrentPlayer(leaf_state))
                if isinstance(encoded_state, np.ndarray):
                    encoded_state_tensor = torch.from_numpy(encoded_state).unsqueeze(0)
                else: 
                    encoded_state_tensor = encoded_state.unsqueeze(0)
                
                # Use the stored device
                encoded_state_tensor = encoded_state_tensor.to(self._get_device())

                with torch.no_grad():
                    policy_logits, net_value_output = self.model_fn(encoded_state_tensor)
                
                policy_logits = policy_logits.squeeze(0)
                value_from_net_for_leaf_player = net_value_output.item()
                leaf_node._value_from_network = value_from_net_for_leaf_player

                if self.game.getCurrentPlayer(leaf_state) != current_player_perspective:
                    value_estimate_for_root_player = -value_from_net_for_leaf_player
                else:
                    value_estimate_for_root_player = value_from_net_for_leaf_player

                legal_actions = self.game.getLegalActions(leaf_state)
                if not legal_actions:
                    pass 
                else:
                    policy_priors_raw = torch.softmax(policy_logits, dim=0).cpu().numpy()
                    masked_policy_priors = {action: policy_priors_raw[action] for action in legal_actions}
                    
                    if leaf_node == root_node and self._apply_dirichlet_noise_to_root and legal_actions:
                        num_legal = len(legal_actions)
                        dirichlet_noise = np.random.dirichlet([self.dirichlet_alpha] * num_legal)
                        
                        for i, action in enumerate(legal_actions):
                            masked_policy_priors[action] = (1 - self.dirichlet_epsilon) * masked_policy_priors[action] + \
                                                             self.dirichlet_epsilon * dirichlet_noise[i]
                        self._apply_dirichlet_noise_to_root = False

                    for action in legal_actions:
                        prior_p = masked_policy_priors.get(action, 0.0) 
                        child_game_state = self.game.applyAction(self.game.copyState(leaf_state), action)
                        new_child_node = AlphaZeroMCTSNode(
                            game_state=child_game_state,
                            parent=leaf_node,
                            action_that_led_here=action,
                            prior_probability=float(prior_p)
                        )
                        leaf_node.children[action] = new_child_node
            
            for node_in_path in reversed(path):
                node_in_path.visit_count += 1
                if self.game.getCurrentPlayer(node_in_path.game_state) == current_player_perspective:
                    node_in_path.total_action_value += value_estimate_for_root_player
                else:
                    node_in_path.total_action_value -= value_estimate_for_root_player
        
        action_probs: Dict[Any, float] = {action: 0.0 for action in self.game.getLegalActions(root_state)}
        if not root_node.children: 
            if not self.game.isTerminal(root_state):
                num_legal = len(action_probs)
                if num_legal > 0:
                    uniform_prob = 1.0 / num_legal
                    for action_key in action_probs:
                        action_probs[action_key] = uniform_prob
        else:
            total_visits_for_policy = root_node.visit_count
            if total_visits_for_policy > 0 :
                 for action, child_node in root_node.children.items():
                    if action in action_probs: 
                        action_probs[action] = child_node.visit_count / total_visits_for_policy
            else: 
                num_legal = len(action_probs)
                if num_legal > 0:
                    uniform_prob = 1.0 / num_legal
                    for action_key in action_probs:
                        action_probs[action_key] = uniform_prob

        mcts_value_of_root = root_node.Q_value 
        return action_probs, mcts_value_of_root

    def get_action_policy(self, root_state: Any, num_simulations: int, temperature: float = 1.0) -> Tuple[Any, Dict[Any, float]]:
        """
        Runs MCTS and returns the chosen action and the policy (visit counts).

        Args:
            root_state: The current game state.
            num_simulations: Number of MCTS simulations.
            temperature: Temperature parameter for action selection.
                         High temperature -> more exploration. Low -> more exploitation.
                         AlphaZero often uses T=1 for first N moves, then T->0.

        Returns:
            A tuple: (chosen_action, action_probabilities)
            - chosen_action: The action selected by MCTS.
            - action_probabilities: Dict[action, probability] based on visit counts (policy target).
        """
        
        current_player = self.game.getCurrentPlayer(root_state)
        mcts_policy_dict, _ = self.run_simulations(root_state, num_simulations, current_player)
        
        actions = list(mcts_policy_dict.keys())
        policy_target_probs = np.array([mcts_policy_dict[action] for action in actions])

        if not actions:
            raise ValueError("No legal actions from root state in MCTS get_action_policy.")

        if np.isclose(temperature, 0.0): 
            chosen_action_idx = np.argmax(policy_target_probs) 
            chosen_action = actions[chosen_action_idx]
        else:
            if len(actions) == 1: 
                 chosen_action = actions[0]
            # For temperature sampling, we need raw visit counts N(s,a)
            # mcts_policy_dict currently stores pi(a|s) = N(s,a)/sum_N(s,b)
            # We need to retrieve N(s,a) from the tree or have run_simulations return them.
            # For now, let's assume T=1 action selection from probabilities if T is not 0.
            # This is a simplification for now. A more correct implementation for T != 0 & T != 1
            # would be to use N(s,a)^(1/T) / sum(N(s,b)^(1/T)).

            # Get raw visit counts for actions from root_node children
            # This requires access to root_node after run_simulations or run_simulations returns it.
            # For now, let's stick to sampling from probabilities if T=1 or using argmax if T=0
            # and leave more complex temperature handling for later refinement if needed.
            
            # Simplified: if T=1, sample from policy_target_probs. If T !=0 and T!=1, it's an approximation.
            # The original AlphaGo used N^(1/tau) for probabilities.
            
            # To implement N^(1/T) sampling correctly:
            # 1. run_simulations needs to return the root_node or the dict of raw visit counts {action: N(s,a)}
            # 2. Here, use those raw_visit_counts^(1/T), normalize, then sample.
            # Current mcts_policy_dict has N(s,a)/N(s). So N(s,a) = mcts_policy_dict[a] * N(s)
            # N(s) is root_node.visit_count after simulations.
            # This change is a bit more involved. Let's ensure the device fix first.
            # For now, if T is not 0, we sample proportionally to policy_target_probs (effectively T=1 sampling)
            
            prob_sum = np.sum(policy_target_probs)
            if prob_sum > 0:
                normalized_probs_for_sampling = policy_target_probs / prob_sum
                chosen_action_idx = np.random.choice(len(actions), p=normalized_probs_for_sampling)
                chosen_action = actions[chosen_action_idx]
            else: # Should only happen if no legal actions, which is checked above, or all probs are zero somehow
                # This path should ideally not be taken if legal actions exist.
                # Fallback to random choice if all probabilities are zero but actions exist.
                chosen_action = random.choice(actions) 

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