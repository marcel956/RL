import numpy as np

class Greedy:
    def __init__(self, bandit, epsilon = None, initial_q = None):
        self.bandit = bandit
        self.epsilon = epsilon
        self.total_rounds = 0

        # T_a: counts of how many times each arm was pulled
        self.counts = np.zeros(self.bandit.num_arms)

        self.best_arm = None
        self.initial_q = initial_q

        # \hat{Q}_a: empirical mean reward of each arm
        if initial_q is not None:
            self.q_values = np.array(initial_q, dtype=float)
        else:
            self.q_values = np.zeros(self.bandit.num_arms)



    def play(self):

        self.total_rounds += 1

        #Get Current Epsilon

        if callable(self.epsilon):
            current_eps = self.epsilon(self.total_rounds)
        else:
            current_eps = self.epsilon


        #Select Action
        if np.random.rand() < current_eps:
            #Explore
            chosen_arm = np.random.randint(self.bandit.num_arms)
        else:
            #Exploit
            max_q = np.max(self.q_values)
            self.best_arms = np.where(self.q_values == max_q)[0]
            chosen_arm = np.random.choice(self.best_arms)

        #Pull Arm
        reward = self.bandit.pull(chosen_arm)

        #Update Counts and Q-values
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]

        self.q_values[chosen_arm] += 1/n * (reward - self.q_values[chosen_arm])

        return chosen_arm, reward

