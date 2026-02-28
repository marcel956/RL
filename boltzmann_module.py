import numpy as np

class Boltzmann:
    def __init__(self, bandit, theta= None, C=None, noise_gen= None, initial_q = None):
        self.bandit = bandit
        self.theta = theta
        self.C = C
        self.noise_gen = noise_gen
        self.total_rounds = 0

        # T_a: counts of how many times each arm was pulled
        self.counts = np.zeros(self.bandit.num_arms)

        self.initial_q = initial_q

        # \hat{Q}_a: empirical mean reward of each arm
        if initial_q is not None:
            self.q_values = np.array(initial_q, dtype=float)
        else:
            self.q_values = np.zeros(self.bandit.num_arms)


    def play(self):

        self.total_rounds += 1

        #1 Formula C Mode:
        if self.C is not None:
            #Unexplored arms first
            unexplored_arms = np.where(self.counts == 0)[0]
        
            if len(unexplored_arms) > 0:
                chosen_arm = np.random.choice(unexplored_arms)
                return self.pull_and_update(chosen_arm)

            #Apply formula from assignment
            Z_a = np.random.gumbel(loc=0, scale=1, size = self.bandit.num_arms)
            exploration_term = np.sqrt(self.C / self.counts) * Z_a
            values = self.q_values + exploration_term


        #2 Arbitrary Distribution Mode
        elif self.noise_gen is not None:
            # If passed a custom distribution, use it here.
            # Assume theta=1.0 if they didn't explicitly pass a theta alongside it.
            current_theta = self.theta if self.theta is not None else 1.0
            noise = self.noise_gen(size=self.bandit.num_arms)
            values = (current_theta * self.q_values) + noise

        #3 Standard Gumbel from script
        elif self.theta is not None:
            g_a = np.random.gumbel(loc=0, scale=1, size=self.bandit.num_arms)
            values = (self.theta * self.q_values) +g_a

        else:
            raise ValueError("Please provide either 'theta, 'C' or a 'noise_generator'")


        #Select Action
        max_val = np.max(values)
        best_arms = np.where(values == max_val)[0]
        chosen_arm = np.random.choice(best_arms)


        return self.pull_and_update(chosen_arm)



    def pull_and_update(self, chosen_arm):
        #Helper method to execute the pull and update values
        reward = self.bandit.pull(chosen_arm)
        
        # Update the counts and Q-values
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        self.q_values[chosen_arm] += (1.0 / n) * (reward - self.q_values[chosen_arm])
        
        return chosen_arm, reward
