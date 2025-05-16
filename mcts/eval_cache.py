import os
import pickle
from typing import Dict, Tuple


class EvalCache:
    """Simple persistent cache for MCTS evaluations."""

    def __init__(self, path: str = "mcts_eval_cache.pkl"):
        self.path = path
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self.data: Dict[str, Tuple[int, float]] = pickle.load(f)
        else:
            self.data: Dict[str, Tuple[int, float]] = {}

    def get(self, key: str):
        return self.data.get(key)

    def increment(self, key: str, value: float):
        visits, total = self.data.get(key, (0, 0.0))
        self.data[key] = (visits + 1, total + value)

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.data, f)
