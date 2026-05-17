import numpy as np


class ThompsonSampling:
    """
    Thompson Sampling bandit.
    Maintains a Beta(alpha, beta) distribution per arm.
    At each step, samples from every distribution and picks the highest —
    naturally balancing exploration and exploitation in a Bayesian way.
    """

    name = "Thompson Sampling"
    color = "#16213E"

    def __init__(self, n_ads: int):
        self.n_ads = n_ads
        self.alpha = np.ones(n_ads)   # successes (clicks) + 1
        self.beta  = np.ones(n_ads)   # failures  (no-click) + 1

    def select_ad(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, ad_index: int, reward: int):
        self.alpha[ad_index] += reward
        self.beta[ad_index]  += (1 - reward)

    def estimated_ctrs(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)
