import sys
from pathlib import Path

# Construct paths to import modules from sibling directories (Sheet1, Sheet2)
sheet1_path = Path(__file__).parent.parent / "Sheet1"
sheet2_path = Path(__file__).parent.parent / "Sheet2"

# Add folder to Python's search path
sys.path.append(str(sheet1_path))
sys.path.append(str(sheet2_path))


# Import modules
from bandit_module import MultiArmedBandit
from etc_module import ETC
from greedy_module import Greedy
from ucb_module import UCB
from boltzmann_module import Boltzmann
from policy_gradient_module import Policy_Gradient


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats



# --- Simulation Parameters ---
num_arms = 5
N = 100   # Number of independent simulation runs (for statistical significance)
n = 10000 # Number of rounds (time horizon) for each simulation run

# Use a dictionary to define our agents and their parameters 
agent_configs = {
    "ETC (m=100)": lambda b: ETC(b, m=100), #optimal: m=1287
    "Greedy (eps=0.01)": lambda b: Greedy(b, epsilon=0.01),
    "UCB (delta=1/n)": lambda b: UCB(b, delta=1/n), # Common optimal param
    "Boltzmann (Gumbel)": lambda b: Boltzmann(b, theta=10),
    "Policy Gradient": lambda b: Policy_Gradient(b, alpha=0.1, use_baseline=True)
}

# --- Data Structures for Storing Results ---
# Dictionaries to track metrics over time for each algorithm.
# Each value is a NumPy array of shape (N, n) to store the metric's value
# at each time step 't' for each independent run 'i'.
all_regrets_over_time = {name: np.zeros((N, n)) for name in agent_configs}
optimal_action_over_time = {name: np.zeros((N, n)) for name in agent_configs}
mae_over_time = {name: np.zeros((N, n)) for name in agent_configs}


# Dictionaries to store a single, final value at the end of each run (for boxplots).
# Each value is a NumPy array of shape (N).
final_regrets = {name: np.zeros(N) for name in agent_configs}
final_estimates_error = {name: np.zeros(N) for name in agent_configs}
prob_optimal_arm = {name: np.zeros(N) for name in agent_configs}
final_tracked_values = {name: np.zeros((N, num_arms)) for name in agent_configs}

# --- Bandit Initialization ---
print("Starting simulation...")
bandit = MultiArmedBandit(num_arms=num_arms, dist_type='bernoulli', means=[0.9, 0.8, 0.5, 0.3, 0.1])
true_means = bandit.means
best_arm_index = np.argmax(true_means)
mu_star = np.max(true_means)

print(f"Fixed Bandit True Means: {true_means}")
print(f"Optimal Arm: {best_arm_index} (Mean: {mu_star:.3f})")


# --- Main Simulation Loop ---
for i in range(N): # Loop over N independent runs for statistical validity
    if (i + 1) % 10 == 0:
        print(f"Running iteration {i + 1}/{N}...")

    for algo_name, agent_init in agent_configs.items():
        # Initialize the specific agent for this round
        agent = agent_init(bandit)

        for t in range(n): # Loop over the time horizon n (a single agent run)
            arm, reward = agent.play()
            
            # Calculate and store instantaneous regret for this step
            regret = mu_star - true_means[arm]
            all_regrets_over_time[algo_name][i, t] = regret

            
            # Track if the optimal action was taken at step t
            if arm == best_arm_index:
                 optimal_action_over_time[algo_name][i, t] = 1
                

            # Track MAE over time (skip Policy Gradient since it doesn't estimate means)
            if algo_name != "Policy Gradient":
                mae = np.mean(np.abs(true_means - agent.q_values))
                mae_over_time[algo_name][i, t] = mae
            else:
                mae_over_time[algo_name][i, t] = np.nan
                

        # --- CONSOLIDATED DATA COLLECTION AT THE END OF HORIZON n ---
        # Cumulative regret for this specific iteration
        final_regrets[algo_name][i] = np.sum(all_regrets_over_time[algo_name][i, :])
        
        # Probability of playing the optimal arm
        prob_optimal_arm[algo_name][i] = np.mean(optimal_action_over_time[algo_name][i, :])  

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



## Explanation for a):
# typical cumulative regret rates:

# ETC: If tuned perfectly and commits to the optimal arm after exploration the regret should be constant (flat) after the exploration phase, 
# but if the wrong arm is chosen it has linear regret O(t)

# epsilon-greedy: Linear regret O(t), because it explores uniformly with probability epsilon

# UCB: Logarithmic regret O(log t), it bounds exploration, resulting in a curve that continuously flattens out over time but never stops exploring

# Boltzmann: Linear regret O(t), similar to epsilon-greedy. with fixed temperature theta it never stops pulling the bad arms

# policy gradient: depending on the learning rate, it can achieve constant regret if it converges confidently on the optimal arm, or linear regret
# if it gets stuck on a suboptimal arm


# Which performs best?
# Theoretically the best choice is UCB since it is guaranteed to achieve the optimal regret bound O(log t) without needing to know anything in advance.
# 
# empirically in the experiment: policy gradient performed the best in this specific simulation, because the optimal arm was distinct enough. The softmax
# preference was isolated quickly, so the probability of pulling a suboptimal arm went to near zero, resulting in flat regret. But the success relies 
# heavily on the well tuned learning rate.
# 
# 
# The rates arise via the reward gap delta, which is the difference in mean between the optimal arm and the mean of a pulled suboptimal arm. For linear regret
# the algorithms with constant exploration pull a suboptimal arm a fixed percentage of the time. Each time they pull a suboptimal arm it gets a penalty of delta_i.

# Therefore the constant in front of the linear t term is proportional to the exploration rate and the sum of the gaps:
# C_linear = sum_(i!=optimal) epsilon * delta_i     (epsilon-greedy example)
# it pays a fixed tax on every round forever

# UCB is different it only explores an arm until it is confident that the arm is worse than the current best. Taking longer for that if the reward gap is small.
# So the constant in front of the log t term iss inversely proportional to the gap.
# C_log = sum_(i!=optimal) = 1/delta_i
# The harder it is to tell 2 arms apart, the larger the constant becomes because the algorithm needs more time to explore that.






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


## Explanation for b):

# Exploration-Exploitation tradeoff: The algorithm must choose between gathering more information about the arms (exploration) to make better future decisions,
# or choosing the best-known arm right now to maximize immediate rewards (exploitation)

# Committal Behavior: This describes how decisively an algorithm "locks in" on a specific arm. Strong committal is great if it locks onto the true optimal arm,
# but bad if it gets stuck on a suboptimal ones


#ETC:
# Behavior: ETC forces a strict seperation of the trade, it purely explores first for m x K rounds and the n commits 100%
# Graph: In Graph (ii), the probability of playing the optimal arm is a very tight line near the top, showing complete committal. 
# Because m=100 was a sufficient exploration phase for these specific means, there are no outliers pulling the wrong arm. This results 
# in a very predictable cumulative regret in Graph (iii) and low estimation error in Graph (i) since every arm was sampled 100 times.

#Epsilon-greedy:
# This algorithm never fully commits. it is forced to explore randomly 10% of the time
# Graph: (ii) shows the median probability of playing the optimal arm around 0.9, which aligns with 90% of the time. But it has a big trail 
# pf outliers dropping all the way to 0. So it has poor committal behavior. Sometimes it failed to find the optimal arm, leading to the 
# massive regret outliers in Graph (iii)

#UCB:
# Behavior: Optimism in the face of uncertainty. Explores aggressively early on but smoothly transition to exploitation.
# Graph: (ii) shows a very tight box around 0.9 probability. It commits strongly, but not 100%, because its confidence bounds 
# force it to occasionally check the other arms. That makes it avoid the outliers seen in ϵ-greedy,
# resulting in a very low, consistent cumulative regret in Graph (iii). It is the safest value-based algorithm.

#Boltzmann:
# Behavior: It commits softly based on the relative differences in estimated values.
# Graph: In this specific setup, Boltzmann struggles heavily. Graph (ii) and (iii) show massive variance (huge boxes). 
# This indicates that the temperature parameter (theta=10) is not well-tuned for the reward gaps of the specific bandit. It is 
# failing to confidently commit to the optimal arm, bouncing between exploration and exploitation inefficiently.

#Policy gradient:
# Behavior: It uses an exponential, soft-max committal. Once it finds the optimal arm, its preference for it skyrockets, 
# driving the probability of exploring other arms to near zero.
# Graph: (ii) shows it has the highest and tightest probability of playing the optimal arm. It commits hard and fast. 
# Because it found the right arm quickly, Graph (iii) shows it has the absolute lowest cumulative regret of all the algorithms tested here. 
# (Graph (i) is blank because PG does not estimate arm means!).


#Plots for c=

# --- Plot c) Metrics Over Time ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot c)(i): Probability of Optimal Action Over Time
for algo_name in agent_configs.keys():
    # Calculate the mean probability across all N iterations at each time step
    prob_over_time = np.mean(optimal_action_over_time[algo_name], axis=0)
    axes[0].plot(np.arange(n), prob_over_time, label=algo_name)

axes[0].set_title("Probability of Selecting Optimal Arm Over Time")
axes[0].set_xlabel("Rounds (n)")
axes[0].set_ylabel("% Optimal Action")
axes[0].legend()
axes[0].grid(True)

# Plot c)(ii): Mean Absolute Error Over Time
for algo_name in agent_configs.keys():
    if algo_name == "Policy Gradient":
        continue # Skip PG
    mae_curve = np.mean(mae_over_time[algo_name], axis=0)
    axes[1].plot(np.arange(n), mae_curve, label=algo_name)

axes[1].set_title("Mean Absolute Error of Q-values Over Time")
axes[1].set_xlabel("Rounds (n)")
axes[1].set_ylabel("MAE")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()


## Explanation for c):
# There are 2 other goals one could imagine. The first from the Reinforcement learning context is just best arm identification.
# Since the agent is being trained in a safe simulation, one does not care about regret only about finding the best arm in a 
# short amount of time. The metric used to explain that is the probability of choosing the right arm over time (left plot).
#  
# Graph: The policy gradient rises smoothly and rapidly to 1.0, which means that it isolates the best arm and  that quickly
# For ETC one can see a massive block at the beginning which is the random exploration phase of the algorithm, after that it snaps close to 1.
# UCB climbs high but never gets really solid because its always forced to still explore a bit.
# For this objective policy gradient performs the best.
#
#
# The 2nd goal could be System Identification. Here one wants the agent to accurately map out the entire environment. 
# The metric are the estimates of arm means / Mean Absolute Error over time (right plot)
# 
# Graph: UCB drops very fast but then plateaus. Its designed to minimize regret, so if it notices that one arm is bad, it will stop pulling it
# so the estimate of that bad arm will never get more accurate.
# Epsilon-greedy drops much slower, but continuously drops downwards. Because it always explores a little and still gathers data on the bad arms. 
# ETC drops very fast during exploration, but then stays flat. It never explores anymore so it cant gather new data on the other arms.
# The best algorithm here is one that never stops explore like epsilon greedy (given enough time). UCB the "best" algorithm for regret is bad at 
# this objective. Tuning the parameters to focus on this goal would help all algorithms.
# 
#
#Metric: Number of Correctly Chosen Actions
# This metric is the direct inverse of cumulative regret. Every time an algorithm chooses the optimal arm, it incurs a regret of 0. 
# Therefore, the algorithm with the highest number of correctly chosen actions will inherently be the one with the lowest cumulative regret.
# At the end of the time horizon (n=10000), Policy Gradient achieved the highest total number of correctly chosen actions because its exponential 
# committal behavior allowed it to exclusively pull the optimal arm for the vast majority of the simulation.




## Explanation for d):
#Optimal Parameters
# The calculated optimal parameter for ETC was m=1287 in this specific test. Its the huge exploration phase to guarantee finding the best arm. But
# running this optimal m leads to a way higher cumulative regret than with the guessed m=100.
#
#Parameter Estimation:
# One can use a hyperparameter grid search to find the numerically best parameters for each algorithm. It simple but expensive, it means just testing 
# out a lot of different values to see which one leads to the lowest regret.
#
#Cheating:
# ETC, epsilon-greedy and Boltzmann all require cheating. To calculate their  optimal parameters one must know the rewards gaps in advance. So you need
# information about the model that you typically don't have.
# UCB does not require cheating because it doesn't need prior knowledge of the model to calculate its optimal delta.
#
#Conclusion:
# Based on these findings UCB is the best algorithm. While policy gradient and ETC performed better in the test in a)-c), this was created by cheating
# and selecting manually tuned parameters that performed well. If the environments would change the performance of those algorithms could change drastically.
# Only UCB balances the exploration-exploitation tradeoff dynamically and robustly. Achieving optimal bounds without prior knowledge.
