import numpy as np

class Policy_Gradient:
    def __init__(self, bandit, alpha = 0.1, use_baseline=True):
        self.bandit = bandit
        self.alpha = alpha
        self.use_baseline = use_baseline
        self.total_rounds = 0

        #Theta are the preferences for each arm, no Qvalues here
        self.theta = np.zeros(self.bandit.num_arms)

        # Track the average reward for the baseline trick
        self.average_reward = 0.0

    def play(self):

        self.total_rounds += 1

        #1 Calculate probabilities with Softmax
        #Subtract maximum theta for stability
        # This prevents np.exp() from overflowing into infinity if theta gets too large,
        # but mathematically doesn't change the resulting probabilities.

        shifted_theta = self.theta - np.max(self.theta)
        exp_theta = np.exp(shifted_theta)
        probabilities = exp_theta / np.sum(exp_theta)

        #2 Select Action based on the calculated probabilities
        chosen_arm = np.random.choice(self.bandit.num_arms, p=probabilities)

        #3 Pull Arm
        reward = self.bandit.pull(chosen_arm)

        #4 Determine the baseline
        if self.use_baseline:
            #Incremental update of the average reward
            self.average_reward += (1.0 / self.total_rounds) * (reward - self.average_reward)
            baseline = self.average_reward
        else:
            baseline = 0.0

        #5 Update preferences
        # Need an array that is 1.0 for the chosen arm, and 0.0 everywhere else
        indicator = np.zeros(self.bandit.num_arms)
        indicator[chosen_arm] = 1

        # This single vectorized line applies the standard Policy Gradient update rule:
        # For chosen arm: theta += alpha * (reward - baseline) * (1 - prob)
        # For other arms: theta -= alpha * (reward - baseline) * prob
        self.theta += self.alpha * (reward - baseline) * (indicator - probabilities)

        return chosen_arm, reward





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
