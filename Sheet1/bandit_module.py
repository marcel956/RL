import numpy as np 

class MultiArmedBandit:
    def __init__(self, num_arms, dist_type='gaussian', means=None, delta=None):
        self.num_arms = num_arms
        self.dist_type = dist_type

        # 1. Setting the means
        if means is not None:
            self.means = np.array(means)
        else:
            #generate random means
            if self.dist_type == 'gaussian':
                #Standard normal: 0mean, variance 1
                self.means = np.random.randn(num_arms)
            else:
                #Uniform between 0 and 1
                self.means = np.random.rand(num_arms)

        # 2. Apply Reward gap delta (if specified)
        if delta is not None:
            #sort means desc.
            self.means = np.sort(self.means)[::-1]
            mu_star = self.means[0]

            for k in range(1, self.num_arms):
                new_mean = mu_star - k * delta
                #Bernoulli means must be between 0 and 1
                if self.dist_type == 'bernoulli':
                    new_mean = max(0, new_mean)
                self.means[k] = new_mean

    def pull(self, arm_index):
        mu = self.means[arm_index]

        if self.dist_type == 'gaussian':
            return np.random.normal(mu, 1)

        elif self.dist_type == 'bernoulli':
            return np.random.binomial(1, mu)













