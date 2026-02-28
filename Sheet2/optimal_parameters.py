import sys
from pathlib import Path

# 1. Path(__file__).parent gets you the 'Sheet2' folder.
# 2. .parent goes up one level to the 'RL' folder.
# 3. / "Sheet1" goes down into the 'Sheet1' folder.
sheet1_path = Path(__file__).parent.parent / "Sheet1"

# Add this folder to Python's search path
sys.path.append(str(sheet1_path))

# Now you can import it normally, without the dots!
from bandit_module import MultiArmedBandit
from etc_module import ETC
from greedy_module import Greedy
from ucb_module import UCB
from boltzmann_module import Boltzmann
from policy_gradient_module import Policy_Gradient


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats




num_arms = 5
N = 100  # Tip: Start with N=10 or N=100 to test if the code works before running N=1000!
n = 10000

# We use a dictionary to define our agents and their parameters easily
agent_configs = {
    "Greedy (eps=0.1)": lambda b: Greedy(b, epsilon=0.01),
    "UCB (delta=1/n)": lambda b: UCB(b, delta=1/n), # Common optimal param
    "Boltzmann (Gumbel)": lambda b: Boltzmann(b, theta=10),
    "Policy Gradient": lambda b: Policy_Gradient(b, alpha=0.1, use_baseline=True)
}

# Data structures to hold results for all algorithms
# Shape: (number of algorithms, N iterations, n rounds)
num_algos = len(agent_configs)
all_regrets_over_time = {name: np.zeros((N, n)) for name in agent_configs}

# Dictionaries for the boxplot data at time n
final_regrets = {name: np.zeros(N) for name in agent_configs}
final_estimates_error = {name: np.zeros(N) for name in agent_configs}
prob_optimal_arm = {name: np.zeros(N) for name in agent_configs}

print("Starting simulation...")

bandit = MultiArmedBandit(num_arms=num_arms, dist_type='bernoulli')
true_means = bandit.means
best_arm_index = np.argmax(true_means)
mu_star = np.max(true_means)

print(f"Fixed Bandit True Means: {true_means}")
print(f"Optimal Arm: {best_arm_index} (Mean: {mu_star:.3f})")


# Add this right below prob_optimal_arm = ...
final_tracked_values = {name: np.zeros((N, num_arms)) for name in agent_configs}


for i in range(N):
    if (i + 1) % 10 == 0:
        print(f"Running iteration {i + 1}/{N}...")

    for algo_name, agent_init in agent_configs.items():
        # Initialize the specific agent for this round
        agent = agent_init(bandit)
        
        # We track how many times the optimal arm was chosen to calculate probabilities
        optimal_pulls = 0

        for t in range(n):
            arm, reward = agent.play()
            
            # 1. Calculate Instantaneous Regret
            regret = mu_star - true_means[arm]
            all_regrets_over_time[algo_name][i, t] = regret
            
            if arm == best_arm_index:
                optimal_pulls += 1
                
        # --- CONSOLIDATED DATA COLLECTION AT THE END OF HORIZON n ---
        # Cumulative regret for this specific iteration
        final_regrets[algo_name][i] = np.sum(all_regrets_over_time[algo_name][i, :])
        
        # Probability of playing the optimal arm
        prob_optimal_arm[algo_name][i] = optimal_pulls / n
        
        if algo_name == "Policy Gradient":
            # PG doesn't have Q-values, so calculate and store its final probabilities
            shifted_theta = agent.theta - np.max(agent.theta)
            exp_theta = np.exp(shifted_theta)
            final_probs = exp_theta / np.sum(exp_theta)
            
            final_tracked_values[algo_name][i] = final_probs
            final_estimates_error[algo_name][i] = np.nan # PG has no direct mean estimates
        else:
            # For all others, store the final Q-value estimates
            final_tracked_values[algo_name][i] = agent.q_values.copy()
            
            # Calculate Mean Absolute Error (MAE)
            mae = np.mean(np.abs(true_means - agent.q_values))
            final_estimates_error[algo_name][i] = mae

# --- Print Final Outputs ---
print("\n" + "="*60)
print("FINAL ESTIMATES & PROBABILITIES (Averaged over N iterations)")
print("="*60)
print(f"True Bandit Means: \n{np.round(true_means, 3)}\n")

for algo_name in agent_configs.keys():
    # Calculate the average across all N iterations
    avg_final_vals = np.mean(final_tracked_values[algo_name], axis=0)
    
    if algo_name == "Policy Gradient":
        print(f"{algo_name} (Final Action Probabilities):")
        print(f"{np.round(avg_final_vals, 3)}\n")
    else:
        print(f"{algo_name} (Final Estimated Q-Values):")
        print(f"{np.round(avg_final_vals, 3)}\n")






#Plot a) Regrets with confidence intervals
plt.figure(figsize=(12, 6))

for algo_name in agent_configs.keys():
    # 1. Calculate cumulative regret for each iteration
    cumulative_regrets = np.cumsum(all_regrets_over_time[algo_name], axis=1)
    
    # 2. Calculate the mean across all N iterations
    mean_cum_regret = np.mean(cumulative_regrets, axis=0)
    
    # 3. Calculate 95% Confidence Interval
    # Standard Error = Standard Deviation / sqrt(N)
    std_dev = np.std(cumulative_regrets, axis=0)
    std_error = std_dev / np.sqrt(N)
    ci_95 = 1.96 * std_error
    
    # 4. Plot line and shaded area
    rounds = np.arange(n)
    line, = plt.plot(rounds, mean_cum_regret, label=algo_name)
    plt.fill_between(rounds, 
                     mean_cum_regret - ci_95, 
                     mean_cum_regret + ci_95, 
                     color=line.get_color(), alpha=0.2)

plt.title("Cumulative Regret over Time with 95% Confidence Intervals")
plt.xlabel("Rounds (n)")
plt.ylabel("Cumulative Regret")
plt.legend()
plt.grid(True)
plt.show()


#Plot b Boxplots 

# Helper function to extract data for boxplots cleanly
def extract_boxplot_data(data_dict):
    labels = list(data_dict.keys())
    # Drop NaNs (like the Policy Gradient in the estimates plot)
    data = [data_dict[label][~np.isnan(data_dict[label])] for label in labels]
    return data, labels

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# --- Plot (b)(i): Estimation Error ---
data_est, labels_est = extract_boxplot_data(final_estimates_error)
axes[0].boxplot(data_est, labels=labels_est)
axes[0].set_title("i) Mean Abs Error: Estimates vs Real Means")
axes[0].set_ylabel("Error")
axes[0].tick_params(axis='x', rotation=45)

# --- Plot (b)(ii): Probabilities for playing arms ---
# We plot the probability of playing the *optimal* arm, which is the most meaningful metric
data_prob, labels_prob = extract_boxplot_data(prob_optimal_arm)
axes[1].boxplot(data_prob, labels=labels_prob)
axes[1].set_title("ii) Probability of Playing the Optimal Arm")
axes[1].set_ylabel("Probability")
axes[1].set_ylim(0, 1.05)
axes[1].tick_params(axis='x', rotation=45)

# --- Plot (b)(iii): Final Regrets ---
data_regret, labels_regret = extract_boxplot_data(final_regrets)
axes[2].boxplot(data_regret, labels=labels_regret)
axes[2].set_title("iii) Cumulative Regret at Horizon n")
axes[2].set_ylabel("Cumulative Regret")
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()