from __future__ import annotations
import random
import numpy as np

class SumTree:
    """
    A SumTree data structure for prioritized experience replay.
    Each leaf node stores a priority value (p^alpha), and each parent node stores the sum
    of these priorities of its children. This allows for efficient O(log N) sampling
    and updating of priorities.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity  # Number of leaf nodes (experiences)
        self.tree_size = 2 * capacity - 1
        self.tree = np.zeros(self.tree_size, dtype=np.float64)  # Tree array, use float64 for precision
        self.data_pointer = 0  # Points to the  data index (0 to capacity-1) for the next write

    def _propagate(self, tree_leaf_idx: int, change: float):
        """Propagate a change in priority up the tree starting from a tree_leaf_idx."""
        parent = (tree_leaf_idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, data_idx: int, priority_p_alpha: float):
        """
        Update the priority (p^alpha) of an experience at a given data_idx (0 to capacity-1).
        """
        if not (0 <= data_idx < self.capacity):
            raise IndexError(f"data_idx {data_idx} is out of bounds for capacity {self.capacity}.")
        
        tree_leaf_idx = data_idx + self.capacity - 1 # Convert data_idx to tree's leaf index
        
        change = priority_p_alpha - self.tree[tree_leaf_idx]
        self.tree[tree_leaf_idx] = priority_p_alpha
        if tree_leaf_idx != 0: # Avoid propagating from root if capacity is 1 (tree_leaf_idx would be 0)
             self._propagate(tree_leaf_idx, change)

    def add(self, priority_p_alpha: float):
        """
        Add a new priority (p^alpha) to the tree. Overwrites oldest if full.
        Uses self.data_pointer to determine which data_idx to update.
        """
        self.update(self.data_pointer, priority_p_alpha)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def get_leaf_priority(self, data_idx: int) -> float:
        """Get the stored priority (p^alpha) of a leaf node given its data_idx."""
        if not (0 <= data_idx < self.capacity):
            raise IndexError(f"data_idx {data_idx} is out of bounds for capacity {self.capacity}.")
        tree_leaf_idx = data_idx + self.capacity - 1
        return self.tree[tree_leaf_idx]

    def sample_leaf_data_idx_and_priority(self, value_sum_sample: float) -> tuple[int, float]:
        """
        Find the data_idx (0 to capacity-1) and its stored priority (p^alpha) for a given sum sample.
        """
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= self.tree_size:  # Reached a leaf in the tree array
                tree_leaf_idx = parent_idx
                break
            else:
                if value_sum_sample <= self.tree[left_child_idx] + 1e-8: # Add epsilon for float comparisons
                    parent_idx = left_child_idx
                else:
                    value_sum_sample -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        
        data_idx = tree_leaf_idx - (self.capacity - 1) # Convert tree leaf index to data_idx
        return data_idx, self.tree[tree_leaf_idx]

    @property
    def total_priority_sum(self) -> float:
        """Total sum of all p^alpha priorities in the tree (root node value)."""
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float, beta_start: float, beta_epochs: int, epsilon: float = 1e-5):
        self.sum_tree = SumTree(capacity)
        self.data_buffer = [None] * capacity  # Stores (experience_tuple)
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent (0: uniform, 1: full priority)
        self.beta_start = beta_start
        self.beta_epochs = beta_epochs 
        self.current_beta = beta_start
        self.epsilon = epsilon  # Small constant added to TD errors before raising to alpha
        self.max_td_error_ever = 1.0 # Tracks max |TD_error| + epsilon for new samples, ensures p > 0

        self._current_epoch_for_beta_anneal = 0 # For beta annealing
        self._current_size = 0 # Number of items currently in buffer
        self._next_data_idx_to_write = 0 # Pointer for data_buffer, same as sum_tree.data_pointer

    def add(self, experience: tuple):
        """Add a new experience. New experiences get max priority to ensure they are sampled."""
        # Priority p = |TD_error| + epsilon. We store (p^alpha) in SumTree.
        # For new experiences, TD_error is unknown. Give them max priority.
        priority_val = self.max_td_error_ever # This is |TD_error_max| + eps
        priority_p_alpha = priority_val ** self.alpha
        
        # self.sum_tree.data_pointer is the data_idx for the next write.
        self.data_buffer[self.sum_tree.data_pointer] = experience
        self.sum_tree.add(priority_p_alpha) # This also updates sum_tree.data_pointer
        
        if self._current_size < self.capacity:
            self._current_size += 1
        # _next_data_idx_to_write is implicitly handled by sum_tree.data_pointer

    def sample(self, batch_size: int) -> tuple[list[tuple], np.ndarray, np.ndarray] | None:
        """
        Sample a batch of experiences, their IS weights, and their data indices.
        Returns None if not enough samples in buffer.
        """
        if self._current_size < batch_size:
            return None

        experiences = [None] * batch_size
        is_weights = np.empty(batch_size, dtype=np.float32)
        sampled_data_indices = np.empty(batch_size, dtype=np.int32) 

        total_p_alpha_sum = self.sum_tree.total_priority_sum
        if total_p_alpha_sum < 1e-8: # Avoid division by zero if sum is negligible
            # This can happen if all priorities are zero (e.g. initial state, or all epsilons are tiny)
            # Fallback to uniform sampling if absolutely necessary, though PER should ensure non-zero priorities.
            # print("Warning: Total priority sum in PER is near zero. Check epsilon and priority updates.")
            # For now, if sum is zero, we can't sample by priority.
            # A robust way is to ensure new samples always have some minimal priority > 0.
            # self.max_td_error_ever starts at 1.0, so this path should be rare.
            return None # Indicate failure to sample meaningfully

        # Anneal beta for IS weights
        if self.beta_epochs > 0:
             self.current_beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * 
                                 (self._current_epoch_for_beta_anneal / self.beta_epochs) )
        else:
            self.current_beta = self.beta_start # Constant beta if beta_epochs is 0

        segment_size = total_p_alpha_sum / batch_size

        for i in range(batch_size):
            a = segment_size * i
            b = segment_size * (i + 1)
            # Ensure b doesn't exceed total sum due to floating point issues
            b = min(b, total_p_alpha_sum)
            value_sum_sample = random.uniform(a, b)
            
            # data_idx is 0 to capacity-1, p_alpha is priority_val^alpha from tree
            data_idx, p_alpha = self.sum_tree.sample_leaf_data_idx_and_priority(value_sum_sample)
            
            experiences[i] = self.data_buffer[data_idx]
            sampled_data_indices[i] = data_idx
            
            # P(i) = p_i^alpha / sum_k(p_k^alpha)
            sampling_probability = p_alpha / total_p_alpha_sum
            if sampling_probability < 1e-10: # Guard against zero probability
                # This can happen if a priority was exactly zero (which epsilon should prevent for TD error based priorities)
                # Or if p_alpha is extremely small compared to total_p_alpha_sum
                # print(f"Warning: Sampled experience with near-zero probability: {sampling_probability}")
                sampling_probability = 1e-10 # Assign a tiny probability to avoid div by zero in IS weight

            # IS weight: (N * P(i))^-beta. N is current_size of buffer.
            is_weights[i] = np.power(self._current_size * sampling_probability, -self.current_beta)

        # Normalize IS weights by dividing by max_is_weight for stability
        max_is_weight = np.max(is_weights)
        if max_is_weight > 1e-8: # Avoid division by zero if all weights are zero
            is_weights /= max_is_weight
        else: # This case should be rare if sampling_probabilities are handled
            is_weights.fill(1.0)
        
        return experiences, is_weights, sampled_data_indices

    def update_priorities(self, data_indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities of experiences based on their new TD errors."""
        if len(data_indices) != len(td_errors):
            raise ValueError("data_indices and td_errors must have the same length.")
            
        for data_idx, td_error in zip(data_indices, td_errors):
            if not (0 <= data_idx < self.capacity):
                 print(f"Warning: Attempting to update priority for invalid data_idx {data_idx}. Capacity is {self.capacity}. Skipping.")
                 continue

            priority_val = np.abs(td_error) + self.epsilon # p = |td_error| + eps
            priority_p_alpha = priority_val ** self.alpha # Store p^alpha in SumTree
            
            self.sum_tree.update(data_idx, priority_p_alpha) 
            
            if priority_val > self.max_td_error_ever: # Track max of (|td_error|+eps), not (p^alpha)
                self.max_td_error_ever = priority_val
                
    def advance_epoch_for_beta_anneal(self):
        if self.beta_epochs > 0:
            self._current_epoch_for_beta_anneal +=1

    def __len__(self) -> int:
        return self._current_size 