import numpy as np


class UCB:
    """
    Upper Confidence Bound (UCB) bandit.
    Explores arms whose reward estimates are uncertain by adding a
    confidence bonus that shrinks as an arm is pulled more often.
    """

    name = "UCB"
    color = "#0F3460"

    def __init__(self, n_ads: int, c: float = 2.0):
        self.n_ads = n_ads
        self.c = c                      # exploration coefficient
        self.counts = np.zeros(n_ads)
        self.values = np.zeros(n_ads)
        self.t = 0                      # total rounds elapsed

    def select_ad(self) -> int:
        self.t += 1
        # Pull each arm at least once before applying UCB formula
        unvisited = np.where(self.counts == 0)[0]
        if len(unvisited) > 0:
            return int(unvisited[0])
        confidence = self.c * np.sqrt(np.log(self.t) / self.counts)
        return int(np.argmax(self.values + confidence))

    def update(self, ad_index: int, reward: int):
        self.counts[ad_index] += 1
        n = self.counts[ad_index]
        self.values[ad_index] += (reward - self.values[ad_index]) / n

    def estimated_ctrs(self) -> np.ndarray:
        return self.values.copy()
