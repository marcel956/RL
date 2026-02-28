import numpy as np

class UCB:
    def __init__(self, bandit, delta = None, sigma=1, initial_q = None):
        self.bandit = bandit
        self.delta = delta
        self.sigma = sigma
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

        #Identify unexplored arms
        # np.where returns a tuple, so we take the first element [0] to get the array of indices
        unexplored_arms = np.where(self.counts == 0)[0]


        if len(unexplored_arms) > 0:
            # If any arm has a count of 0, its UCB is infinity. 
            # We pick the first unexplored arm we find.
            chosen_arm = unexplored_arms[0]
        else:
            #Calculate UCB values
            exploration_bonus = np.sqrt(( 2 * (self.sigma**2) * np.log(1 / self.delta)) / self.counts)
            ucb_values = self.q_values + exploration_bonus

            #Select Action
            max_ucb = np.max(ucb_values)
            best_arms = np.where(ucb_values == max_ucb)[0]
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

