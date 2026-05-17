import numpy as np
 
 
class AdEnvironment:
    """
    Simulates an ad server with N ads, each having a hidden
    true click-through rate (CTR). Calling show_ad() returns
    a stochastic click (1) or no-click (0).
    """
 
    def __init__(self, n_ads: int, seed: int = 42):
        np.random.seed(seed)
        self.n_ads = n_ads
        self.true_ctrs = np.random.uniform(0.02, 0.35, n_ads)
        self.best_ad = int(np.argmax(self.true_ctrs))
        self.best_ctr = self.true_ctrs[self.best_ad]
 
    def show_ad(self, ad_index: int) -> int:
        """Returns 1 (click) or 0 (no click) based on the ad's true CTR."""
        return int(np.random.random() < self.true_ctrs[ad_index])
 
    def summary(self):
        """Prints a table of all ads and their true CTRs."""
        print("\n      Ad Environment")
        print("─" * 40)
        for i, ctr in enumerate(self.true_ctrs):
            marker = "  ← BEST" if i == self.best_ad else ""
            print(f"  Ad {i:02d}  CTR = {ctr:.3f}{marker}")
        print()
