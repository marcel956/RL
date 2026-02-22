import numpy as np
import scipy.stats as stats

class Bandits:
    def __init__(self, num_arms, dist_mode, means="random", gap=None):
        self.num_arms = num_arms
        self.dist_mode = dist_mode
        
        if dist_mode not in ['bernoulli', 'gaussian']:
            raise ValueError('dist_mode must be bernoulli or gaussian')

        # 1. Initial Mean Generation
        if isinstance(means, str) and means == "random":
            if dist_mode == "gaussian":
                self.means = np.random.normal(0, 1, num_arms)
            else: # bernoulli
                self.means = np.random.uniform(0, 1, num_arms)
        else:
            self.means = np.array(means)

        # 2. Implement Reward Gap Logic
        if gap is not None:
            # Find the best mean (mu*)
            max_mean = np.max(self.means)
            # Sort means descending to identify ranks, then replace
            # mu_k = mu* - k * delta (where k=0 is the best arm)
            new_means = []
            for k in range(num_arms):
                val = max_mean - (k * gap)
                # For Bernoulli, floor the value at 0
                if dist_mode == 'bernoulli':
                    val = max(0, val)
                new_means.append(val)
            self.means = np.array(new_means)
            # Optional: Shuffle if you don't want the best arm to always be index 0
            np.random.shuffle(self.means)

        # 3. Setup Distribution Arms
        self.arms = {}
        for i in range(num_arms):
            if dist_mode == 'gaussian':
                self.arms[i] = stats.norm(loc=self.means[i], scale=1)
            else:
                self.arms[i] = stats.bernoulli(p=self.means[i])

    def pull_arm(self, arm_i):
        """Action: Drawing of a certain arm"""
        return self.arms[arm_i].rvs()

# Example Usage with Reward Gap
bandit_with_gap = Bandits(num_arms=5, dist_mode='gaussian', means="random", gap=0.5)
print(f"Means with gap 0.5: {bandit_with_gap.means}")
print(f"Sample reward from arm 0: {bandit_with_gap.pull_arm(0)}")


