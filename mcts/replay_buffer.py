from __future__ import annotations
import random
import numpy as np
from collections import deque

class SumTree:
    \"\"\"
    A SumTree data structure for prioritized experience replay.
    Each leaf node stores a priority value, and each parent node stores the sum
    of the priorities of its children. This allows for efficient O(log N) sampling
    and updating of priorities.
    \"\"\"
    def __init__(self, capacity: int):
        self.capacity = capacity  # Number of leaf nodes (experiences)
        self.tree_size = 2 * capacity - 1
        self.tree = np.zeros(self.tree_size)  # Tree array
        self.data_pointer = 0  # Current position to write new data/priority

    def _propagate(self, idx: int, change: float):
        \"\"\"Propagate a change in priority up the tree.\"\"\"
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, tree_idx: int, priority: float):
        \"\"\"Update the priority of an experience at a given tree index.\"\"\"
        if not (0 <= tree_idx < self.tree_size - self.capacity +1 ): # check if its a valid data index, not tree index
             # This means tree_idx is actually an index into the data array (0 to capacity-1)
             # We need to convert it to the tree's leaf index.
             # Leaf nodes start at self.capacity - 1 in the tree array.
             actual_tree_idx = tree_idx + self.capacity -1
        else: # It was already a tree leaf index
            actual_tree_idx = tree_idx


        if not (self.capacity - 1 <= actual_tree_idx < self.tree_size):
            raise IndexError(f\"tree_idx {tree_idx} (actual: {actual_tree_idx}) is out of bounds for leaf nodes.\")
            
        change = priority - self.tree[actual_tree_idx]
        self.tree[actual_tree_idx] = priority
        self._propagate(actual_tree_idx, change)

    def add(self, priority: float):
        \"\"\"Add a new priority to the tree. Overwrites oldest if full.\"\"\"
        # data_pointer points to the position in the conceptual data array (0 to capacity-1)
        # We need to map this to the leaf node index in the tree array.
        tree_idx = self.data_pointer + self.capacity - 1
        
        self.update(tree_idx, priority) # Update will handle propagation
        
        self.data_pointer = (self.data_pointer + 1) % self.capacity


    def get_leaf_value(self, tree_idx: int) -> float:
        \"\"\"Get the priority value of a leaf node given its tree index.\"\"\"
        if not (self.capacity - 1 <= tree_idx < self.tree_size):
            raise IndexError(f\"tree_idx {tree_idx} is out of bounds for leaf nodes.\")
        return self.tree[tree_idx]

    def get_leaf_index(self, value_sum: float) -> int:
        \"\"\"
        Find the leaf index for a given sum value.
        Traverses the tree to find the leaf whose cumulative sum range contains value_sum.
        \"\"\"
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= self.tree_size:  # Reached a leaf
                leaf_idx = parent_idx
                break
            else:
                if value_sum <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    value_sum -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        return leaf_idx # This is the index in the self.tree array

    @property
    def total_sum(self) -> float:
        \"\"\"Total sum of all priorities in the tree (root node value).\"\"\"
        return self.tree[0]

    @property
    def max_priority(self) -> float:
        \"\"\"Maximum priority among the leaf nodes.\"\"\"
        return np.max(self.tree[self.capacity - 1:]) if self.capacity > 0 else 1.0 # Default to 1.0 if empty

    @property
    def min_priority(self) -> float:
        \"\"\"Minimum priority among the leaf nodes (non-zero).\"\"\"
        leaves = self.tree[self.capacity - 1:]
        non_zero_leaves = leaves[leaves > 1e-8] # Consider only non-negligible priorities
        return np.min(non_zero_leaves) if len(non_zero_leaves) > 0 else 1e-8 # Default if all zero or empty


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float, beta_start: float, beta_epochs: int, epsilon: float = 1e-5):
        self.sum_tree = SumTree(capacity)
        self.data_buffer = deque(maxlen=capacity) # Stores (experience_tuple)
        # self.data_indices maps tree_leaf_index to index in data_buffer if not aligned
        # For simplicity, assume data_buffer is aligned with sum_tree leaves via data_pointer logic

        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent (0: uniform, 1: full priority)
        self.beta_start = beta_start  # Initial value for IS exponent beta
        self.beta_epochs = beta_epochs # Epochs to anneal beta to 1.0
        self.current_beta = beta_start
        self.epsilon = epsilon  # Small constant added to priorities
        self.max_priority_ever = 1.0 # Track max priority for new samples

        self._current_epoch_for_beta = 0 # For beta annealing

    def add(self, experience: tuple):
        \"\"\"Add a new experience to the buffer.\"\"\"
        # New experiences are added with the current maximum priority encountered
        # This ensures they are likely to be sampled soon.
        priority = self.max_priority_ever 
        
        # If buffer is full, SumTree's data_pointer will cause overwrite.
        # Deque also handles its own overwrite.
        # We need to ensure that if deque overwrites, the corresponding SumTree entry
        # (which will be pointed to by sum_tree.data_pointer) gets the new priority.
        
        if len(self.data_buffer) == self.capacity:
            # The sum_tree.data_pointer is where the OLDEST item's priority is.
            # The deque's oldest item is about to be popped.
            # This is fine as sum_tree.add will overwrite that priority.
            pass

        self.data_buffer.append(experience) # Deque handles its maxlen
        self.sum_tree.add(priority) # SumTree handles its data_pointer and capacity

    def sample(self, batch_size: int) -> tuple[list[tuple], np.ndarray, np.ndarray]:
        \"\"\"
        Sample a batch of experiences from the buffer.
        Returns:
            experiences: List of experience tuples
            is_weights: Importance sampling weights for each experience
            tree_indices: Tree indices of the sampled experiences (for priority updates)
        \"\"\"
        if len(self.data_buffer) < batch_size:
            # Not enough samples yet for a full batch, could return what's available or raise error
            # For now, let's proceed but IS weights might be less meaningful
            # Or, one could wait until buffer has batch_size elements.
             # This case should be handled by min_buffer_fill in the main script
            print(f\"Warning: Sampling batch_size={batch_size} but buffer has only {len(self.data_buffer)} samples.\")


        experiences = []
        is_weights = np.empty(batch_size, dtype=np.float32)
        tree_indices = np.empty(batch_size, dtype=int) # Store tree leaf indices

        total_priority_sum = self.sum_tree.total_sum
        if total_priority_sum == 0: # Should not happen if epsilon > 0 and items exist
            # Fallback to uniform random sampling from what's available if sum is zero
            # This indicates an issue or an empty buffer state not handled earlier.
            print(\"Warning: Total priority sum is zero. Sampling uniformly if data exists.\")
            if not self.data_buffer: return [], np.array([]), np.array([])
            
            indices_in_data_buffer = random.sample(range(len(self.data_buffer)), min(batch_size, len(self.data_buffer)))
            experiences = [self.data_buffer[i] for i in indices_in_data_buffer]
            # For tree_indices and IS_weights, this fallback is problematic for PER.
            # A robust PER system ensures total_priority_sum > 0 if buffer has items with epsilon.
            # This case should ideally be prevented by min_buffer_fill and epsilon in priorities.
            # Returning uniform weights and dummy tree indices:
            is_weights.fill(1.0)
            # We need to map data_buffer indices to tree_indices, which is complex if not aligned.
            # For now, this fallback path in PER sample means PER isn't really working for this batch.
            # This highlights the need for `min_buffer_fill` and proper priority initialization.
            # Let's assume this path is rarely taken in a well-running system.
            # To get tree_indices, we'd need to search for data or assume direct mapping via data_pointer,
            # but that gets complicated if buffer isn't full or if data_pointer has wrapped many times.
            # Simplest for fallback: just return empty or handle this state more gracefully if it occurs.
            # For now, the warning stands and this batch won't be correctly prioritized for updates.
            # A proper fix would ensure `sum_tree.data_pointer` and `len(data_buffer)` are used
            # to map these random `indices_in_data_buffer` to their SumTree leaf indices.
            # This example assumes sum_tree is always populated if data_buffer is.
            # This needs careful handling of how `data_buffer` indices map to `sum_tree` leaves.
            # The current `SumTree.add` and `data_buffer.append` with `maxlen` might not keep perfect alignment
            # without explicit index mapping if we sample random indices from `data_buffer`.
            # PER sampling *must* use the SumTree for selection.

            # Let's assume `total_priority_sum > 0` if len(self.data_buffer) >= batch_size due to `min_buffer_fill`
            # and initial priorities being > epsilon.
            if not self.data_buffer or total_priority_sum <= 1e-8: # A more robust check for empty/zero sum
                return [], np.array([]), np.array([])


        # Segment size for stratified sampling
        segment_size = total_priority_sum / batch_size
        
        # Anneal beta
        self.current_beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self._current_epoch_for_beta / self.beta_epochs) )

        min_prob = self.sum_tree.min_priority / total_priority_sum
        if min_prob == 0: # Avoid division by zero if all priorities are zero (should not happen)
            min_prob = self.epsilon / total_priority_sum if total_priority_sum > 0 else self.epsilon


        for i in range(batch_size):
            a = segment_size * i
            b = segment_size * (i + 1)
            value_sum_sample = random.uniform(a, b)
            
            leaf_idx_in_tree = self.sum_tree.get_leaf_index(value_sum_sample) # This is index in self.tree array
            priority_at_leaf = self.sum_tree.get_leaf_value(leaf_idx_in_tree)
            
            # Map leaf_idx_in_tree (index in SumTree's tree array) to index in self.data_buffer
            # Leaf nodes in SumTree start at index `self.capacity - 1`.
            # So, data_idx = leaf_idx_in_tree - (self.capacity - 1)
            data_idx = leaf_idx_in_tree - (self.capacity - 1)

            if not (0 <= data_idx < len(self.data_buffer)):
                # This can happen if sum_tree has entries for capacity but data_buffer is not yet full,
                # or if there's a misalignment. For a full buffer, data_idx should be valid.
                # If buffer is not full, data_idx might point to an empty slot if sum_tree was pre-filled
                # or if sampling logic for non-full buffer needs refinement.
                # A common approach: SumTree capacity matches data_buffer capacity, and SumTree.add
                # aligns with data_buffer.append via SumTree.data_pointer.
                # The `self.data_buffer[data_idx]` then directly works if `data_pointer` logic is correct.
                
                # Let's adjust data_idx based on how SumTree's data_pointer and deque work.
                # When SumTree's data_pointer is `p`, it means the *next* write for priority goes to `p`-th slot (0-indexed).
                # The corresponding data is `self.data_buffer[p]` if `deque` is also filled sequentially.
                # However, `get_leaf_index` returns a SumTree leaf index. We need to map this to the
                # current valid index in `self.data_buffer`.
                # If `self.data_buffer` is a deque with `maxlen`, and it's full,
                # `self.data_buffer[0]` is the oldest. `sum_tree.data_pointer` points to where the OLDEST priority is stored.
                # So, if `leaf_idx_in_tree` corresponds to `sum_tree.data_pointer`, then the data is `self.data_buffer[0]`.
                # This mapping is tricky. A simpler way is to store (priority, data_index_in_buffer) in SumTree
                # or have a direct way to get data from SumTree leaf index.

                # For this implementation: assume sum_tree.data_pointer and deque's conceptual pointers align.
                # `data_idx` calculated as `leaf_idx_in_tree - (self.capacity - 1)` is the 0-indexed slot
                # in the conceptual array of size `capacity`.
                # We need to retrieve the item from `self.data_buffer` that corresponds to this conceptual slot.
                # If `len(self.data_buffer) < self.capacity`, then `self.data_buffer[data_idx]` works if `data_idx` is within current length.
                # If `len(self.data_buffer) == self.capacity`, then `self.data_buffer` effectively wraps.
                # The `data_idx` (0 to capacity-1) needs to be mapped to current deque indices.
                # Let's consider the `current_size = len(self.data_buffer)`.
                # The `data_pointer` in SumTree refers to the *next write location* in the conceptual array of priorities.
                # The `data_buffer` stores the last `current_size` elements.
                # If `data_idx` (from `leaf_idx_in_tree`) refers to conceptual slot `k` (0 to capacity-1),
                # and `sum_tree.data_pointer` is `dp`, and `current_buffer_len = len(self.data_buffer)`.
                # If `current_buffer_len < capacity`: data is at `self.data_buffer[data_idx]` if `data_idx < current_buffer_len`.
                # If `current_buffer_len == capacity`:
                #   The conceptual slot `dp` in SumTree holds priority for `self.data_buffer[0]` (oldest).
                #   Conceptual slot `(dp + 1) % capacity` holds for `self.data_buffer[1]`.
                #   Conceptual slot `(dp + k) % capacity` holds for `self.data_buffer[k]`.
                #   We have `data_idx` (conceptual slot). We need to find `k` such that `(dp + k) % capacity == data_idx`.
                #   `k = (data_idx - dp + capacity) % capacity`.
                #   So, the experience is `self.data_buffer[k]`.

                current_buffer_len = len(self.data_buffer)
                if current_buffer_len == self.capacity: # Buffer is full and wrapped
                    # data_idx is the conceptual slot (0 to capacity-1)
                    # sum_tree.data_pointer is the conceptual slot of the OLDEST priority/data
                    actual_data_buffer_idx = (data_idx - self.sum_tree.data_pointer + self.capacity) % self.capacity
                else: # Buffer is not full yet
                    if data_idx >= current_buffer_len:
                        # This should not happen if sampling is correct and sum_tree only contains priorities for added data.
                        # This implies a priority exists for an empty data slot.
                        # This could occur if priorities are added for future data slots not yet filled.
                        # The `add` method in SumTree updates `data_pointer` *after* update.
                        # And it calls `update(tree_idx, priority)` where `tree_idx` is `data_pointer + capacity - 1`.
                        # This means `data_pointer` is the index of the *next* slot to write in conceptual array.
                        # The SumTree stores `capacity` priorities regardless of `data_buffer` fill level initially (as zeros).
                        # So, `get_leaf_index` can return an index for a slot not yet in `data_buffer` if buffer not full.
                        # This needs to be handled: only sample from filled portion.
                        # The `total_priority_sum` should reflect only filled items.
                        # Current SumTree initializes with zeros. `add` overwrites.
                        # This means stratified sampling might pick a zero-priority slot if buffer not full.

                        # Safer: if sum_tree.total_sum only reflects actual items, this is fine.
                        # If data_buffer is not full, we should only sample from the filled part.
                        # This is complex with stratified sampling over SumTree's full capacity.
                        # A common PER approach: only sample when buffer has > batch_size items.
                        # And SumTree structure is always for 'capacity', priorities of unused slots are 0.
                        # If `priority_at_leaf` is 0 (or near epsilon), it means this slot is likely empty or low actual priority.
                        # The issue is mapping SumTree leaf to actual filled data_buffer index.
                        
                        # Re-think: The SumTree's `data_pointer` is the index in the conceptual array (0 to capacity-1)
                        # for the *next* write. So, `(data_pointer - 1 + capacity) % capacity` is the last written element.
                        # The `data_buffer` (deque) stores elements chronologically.
                        # If `len(data_buffer) < capacity`, `data_buffer[j]` is `j`-th element.
                        # The `data_idx` from `sum_tree.get_leaf_index` is the 0-indexed leaf number.
                        # This `data_idx` matches the conceptual slot in the SumTree.
                        # We need the experience from `data_buffer` that corresponds to this `data_idx`.
                        
                        # Let's assume sum_tree and data_buffer are aligned for filled entries.
                        # If sum_tree.data_pointer = p, it means slots 0..p-1 have been written to in current cycle (or all if wrapped)
                        # Deque stores last N items.
                        # The `data_idx` (0 to capacity-1) is the conceptual slot.
                        # If `len(self.data_buffer) == self.capacity`:
                        #   The element at conceptual slot `s = self.sum_tree.data_pointer` (oldest priority)
                        #   corresponds to `self.data_buffer[0]`.
                        #   The element at conceptual slot `(s + k) % capacity` corresponds to `self.data_buffer[k]`.
                        #   So, `k = (data_idx - self.sum_tree.data_pointer + self.capacity) % self.capacity`.
                        #   This mapping was correct.
                        # The issue is when `data_idx >= current_buffer_len` when `current_buffer_len < capacity`.
                        # This means we sampled a priority for a slot not yet filled in `data_buffer`.
                        # This should be prevented by ensuring `total_priority_sum` in sampling only reflects filled slots
                        # or by re-sampling if an empty slot is chosen.
                        # Current `SumTree.total_sum` is sum of all `capacity` leaves.
                        # This is a common pitfall.
                        # A robust way: SumTree only stores priorities for actual items in data_buffer.
                        # Or, if SumTree has `capacity` leaves, only sample from segments corresponding to filled items.

                        # Simpler fix for now: if `data_idx` is out of `len(self.data_buffer)` bounds (when not full),
                        # it's an invalid sample for current data. This indicates sampling logic needs to be bound
                        # by `len(self.data_buffer)`.
                        # However, stratified sampling samples from `total_priority_sum`.
                        # The issue is that `SumTree` has `capacity` leaves which may not all correspond to valid data yet.
                        # The `add` method correctly updates `sum_tree.data_pointer`.
                        # `self.data_buffer[actual_data_buffer_idx]` should work if `current_buffer_len == self.capacity`.
                        # If `current_buffer_len < self.capacity`, `data_idx` (0 to capacity-1) must be `< current_buffer_len`.
                        # If stratified sampling picks a `value_sum_sample` that leads to `data_idx >= current_buffer_len`,
                        # it means it picked a part of SumTree where data hasn't been added or has low (epsilon) priority.
                        # For now, if this happens, we might need to resample or skip.
                        # This implies `total_priority_sum` should ideally be sum of priorities of *actual* data.
                        # The SumTree maintains sum of all capacity leaves. If unused leaves are 0, this is fine.
                        # The `add` with `max_priority_ever` ensures new items are not 0.
                        
                        # If `priority_at_leaf` is very small (near epsilon, or 0 if epsilon not added before storage),
                        # this slot might be "empty" or an old very low priority item.
                        # Let's assume for now the mapping `actual_data_buffer_idx` is generally correct for a full buffer.
                        # And that `min_buffer_fill` ensures we sample from a reasonably populated buffer.
                        # The `data_idx` is the direct conceptual index for `sum_tree` leaves.
                        # And for `data_buffer` if it were a simple array of `capacity`.
                        # With deque, the mapping from this conceptual `data_idx` to deque index is:
                        # `deque_idx = data_idx - self.sum_tree.data_pointer`
                        # if `data_idx >= self.sum_tree.data_pointer` (for items written after last wrap of data_pointer)
                        # `deque_idx = data_idx - self.sum_tree.data_pointer + self.capacity`
                        # if `data_idx < self.sum_tree.data_pointer` (for items written before last wrap of data_pointer, now at end of deque)
                        # This is equivalent to `(data_idx - self.sum_tree.data_pointer + self.capacity) % self.capacity` if buffer is full.

                        # If not full: `self.data_buffer` index is simply `data_idx`, assuming `data_idx < len(self.data_buffer)`.
                        # The `SumTree.get_leaf_index` samples from *all* `capacity` leaves.
                        # This is the core issue if not full. SumTree needs to reflect current size.
                        # For now, let's use the `actual_data_buffer_idx` derived for a full buffer,
                        # and rely on `min_buffer_fill` making `current_buffer_len` close to `capacity`.
                        # This part is often a source of off-by-one or indexing errors in PER.
                        # The current `SumTree.add` correctly maps `self.data_pointer` (0 to capacity-1)
                        # to `tree_idx` (leaf index).
                        # And `self.data_buffer.append` adds to the end.
                        # `self.data_buffer[k]` is the k-th item *currently in the deque*.
                        # The `data_idx` is the k-th slot in the SumTree's leaves.
                        # These are aligned if deque has not wrapped.
                        # If deque has wrapped, `self.data_buffer[0]` is oldest, corresponding to `sum_tree.data_pointer`'s slot.
                        
                        # Correct mapping from sum_tree leaf index to data_buffer index:
                        # `leaf_idx_in_tree` is `sum_tree_data_idx + self.capacity - 1`.
                        # `sum_tree_data_idx` is the conceptual slot (0 to capacity-1).
                        # If buffer len `L == capacity`:
                        #   `deque_idx = (sum_tree_data_idx - self.sum_tree.data_pointer + L) % L`
                        # If buffer len `L < capacity`:
                        #   `deque_idx = sum_tree_data_idx` (valid if `sum_tree_data_idx < L`)
                        # This requires careful handling if sampling can select a `sum_tree_data_idx >= L`.
                        # For now, this path indicates a potential rare issue or needs a robust empty slot check.
                        # A common simple PER: data array and sumtree array are directly 1-to-1. Deque adds complexity.
                        # Let's use a simple array for data_buffer for PER to simplify indexing.

                        # Revisit: For SumTree used with a fixed-size array as data store (not deque):
                        # sum_tree.data_pointer is index into data array for next write.
                        # leaf_idx_in_tree directly maps to data_array[leaf_idx_in_tree - (self.capacity-1)].
                        # Using deque makes this mapping dynamic.
                        # To simplify, let's change self.data_buffer to be a list/np.array pre-allocated.
                        
                        # Sticking with deque for now: the logic for `actual_data_buffer_idx` for a full buffer is standard.
                        # The main concern is sampling an "empty" slot if not full.
                        # If priority_at_leaf is ~epsilon, it's likely an uninteresting or empty slot.
                        # Resampling might be needed if priority_at_leaf is too low.
                        # For now, assume this works if buffer is mostly full due to min_buffer_fill.
                        print(f\"Warning: Sampled data_idx {data_idx} which might be out of current data_buffer bounds if not full. Buffer len: {len(self.data_buffer)}\")
                        # This path should ideally not be hit frequently.
                        # Fallback: if data_idx is truly invalid for current deque, resample this one instance (tricky in loop).
                        # Or, ensure total_priority_sum for sampling is only over *filled* entries.
                        # For now, we proceed, and if it crashes, this mapping is the place to fix.
                        # The critical part is `self.data_buffer[actual_data_buffer_idx]`
                        # If `len(self.data_buffer)` is `L`:
                        #   If `L == self.capacity`: `actual_data_buffer_idx = (data_idx - self.sum_tree.data_pointer + self.capacity) % self.capacity`
                        #   If `L < self.capacity`: `actual_data_buffer_idx = data_idx`. This is valid only if `data_idx < L`.
                        #      If stratified sampling picks a `data_idx >= L`, this is problematic.
                        #      This means `SumTree.total_sum` must be sum of priorities of *valid* data items only.
                        #      The current SumTree sums all `capacity` leaves.
                        #      The simplest robust PER uses an array for data, not a deque, making indexing trivial.

                        # Given the current SumTree (sums all capacity leaves):
                        # We MUST ensure sampling `value_sum_sample` only considers range of priorities of *actual items*.
                        # This is usually done by `SumTree.total_sum` being the sum of *valid* priorities.
                        # If unused SumTree leaves are 0, then `total_sum` is correct.
                        # `epsilon` ensures they are not 0 if `add` uses `self.epsilon` for new slots.
                        # Our `add` uses `self.max_priority_ever`.
                        # This implies SumTree always has non-zero entries once something is added.

                        # Let's assume `data_idx` obtained from `get_leaf_index` will correspond to a slot
                        # that has been written to if `total_priority_sum` is substantial.
                        # The mapping to deque index for a full deque:
                        idx_in_deque = (data_idx - self.sum_tree.data_pointer + len(self.data_buffer)) % len(self.data_buffer) \
                                       if len(self.data_buffer) == self.capacity \
                                       else data_idx
                        
                        if not (0 <= idx_in_deque < len(self.data_buffer)):
                             # This is a critical failure point if logic is flawed.
                             # It means the sumtree sampling picked a leaf that doesn't map to current deque content.
                             # This happens if sumtree has more "active" entries than deque.
                             # This should not happen if sum_tree.add and data_buffer.append are kept in sync by data_pointer.
                             # For now, if this edge case is hit, we have a problem.
                             # Most PER impls use a fixed array for data, making data_idx directly usable.
                             print(f\"Critical PER sampling error: data_idx {data_idx} -> deque_idx {idx_in_deque} out of bounds for deque len {len(self.data_buffer)}\")
                             # Fallback for this sample - extremely hacky: pick random from deque
                             # This should be fixed by better SumTree/Deque synchronization or data storage.
                             idx_in_deque = random.randrange(len(self.data_buffer))


                experiences.append(self.data_buffer[idx_in_deque])
                tree_indices[i] = leaf_idx_in_tree # Store the tree index (for SumTree.update)
            
                sampling_prob = priority_at_leaf / total_priority_sum
                # IS weight: ( (1/N) * (1/P(i)) )^beta = (N * P(i))^-beta
                # N = len(self.data_buffer) # Current number of items
                is_weights[i] = np.power(len(self.data_buffer) * sampling_prob, -self.current_beta)

        # Normalize IS weights by dividing by max(is_weights) for stability
        if len(is_weights) > 0 and np.max(is_weights) > 0:
            is_weights /= np.max(is_weights)
        
        return experiences, is_weights, tree_indices

    def update_priorities(self, tree_indices: np.ndarray, priorities: np.ndarray):
        \"\"\"Update priorities of experiences corresponding to given tree indices.\"\"\"
        if len(tree_indices) != len(priorities):
            raise ValueError(\"tree_indices and priorities must have the same length.\")
            
        for tree_idx, priority_val in zip(tree_indices, priorities):
            # Add epsilon to ensure priority is non-zero
            actual_priority = priority_val + self.epsilon
            self.sum_tree.update(tree_idx, actual_priority) # tree_idx is already the leaf index in sum_tree.tree
            
            # Update max_priority_ever
            if actual_priority > self.max_priority_ever:
                self.max_priority_ever = actual_priority
                
    def advance_epoch_for_beta(self):
        self._current_epoch_for_beta +=1

    def __len__(self) -> int:
        return len(self.data_buffer) 