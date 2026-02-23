import numpy as np

class ETC:
    def __init__(self, bandit, m):
        self.bandit = bandit
        self.m = m
        self.total_rounds = 0
        self.rewards_per_arm = np.zeros(self.bandit.num_arms)
        self.best_arm = None

    def play(self):
        if self.total_rounds < self.m * self.bandit.num_arms:
            chosen_arm = self.total_rounds % self.bandit.num_arms
            reward = self.bandit.pull(chosen_arm)
            self.rewards_per_arm[chosen_arm] += reward
        else:
            if self.best_arm is None:
                self.best_arm = np.argmax(self.rewards_per_arm)
            chosen_arm = self.best_arm
            reward = self.bandit.pull(chosen_arm)
        self.total_rounds += 1
        return chosen_arm, reward
