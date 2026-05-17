import numpy as np


class EpsilonGreedy:
    """
    Epsilon-Greedy bandit.
    Explores a random arm with probability epsilon,
    otherwise exploits the arm with the highest estimated CTR.
    """

    name = "Epsilon-Greedy (ε=0.1)"
    color = "#E94560"

    def __init__(self, n_ads: int, epsilon: float = 0.1):
        self.n_ads = n_ads
        self.epsilon = epsilon
        self.counts = np.zeros(n_ads)
        self.values = np.zeros(n_ads)   # estimated CTR per ad

    def select_ad(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_ads)   # explore
        return int(np.argmax(self.values))          # exploit

    def update(self, ad_index: int, reward: int):
        self.counts[ad_index] += 1
        n = self.counts[ad_index]
        # Incremental mean update
        self.values[ad_index] += (reward - self.values[ad_index]) / n

    def estimated_ctrs(self) -> np.ndarray:
        return self.values.copy()
