import numpy as np

class ETC:
    def __init__(self, bandit, m):
        self.bandit = bandit
        self.m = m
        self.total_rounds = 0
        
        # Track counts and q_values just like the UCB module
        self.counts = np.zeros(self.bandit.num_arms)
        self.q_values = np.zeros(self.bandit.num_arms)
        self.best_arm = None

    def play(self):
        # 1. Determine which arm to pull
        if self.total_rounds < self.m * self.bandit.num_arms:
            # Exploration phase: pull each arm round-robin
            chosen_arm = self.total_rounds % self.bandit.num_arms
        else:
            # Exploitation phase: commit to the best arm found
            if self.best_arm is None:
                # Find the arm with the highest estimated mean (Q-value)
                self.best_arm = np.argmax(self.q_values)
            chosen_arm = self.best_arm
            
        # 2. Pull the arm
        reward = self.bandit.pull(chosen_arm)
        self.total_rounds += 1
        
        # 3. Update counts and Q-values incrementally
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        self.q_values[chosen_arm] += (1.0 / n) * (reward - self.q_values[chosen_arm])
        
        return chosen_arm, reward