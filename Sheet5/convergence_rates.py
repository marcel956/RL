import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sheet4_path = Path(__file__).parent.parent / "Sheet4"

# Add this folder to Python's search path
sys.path.append(str(sheet4_path))

from gridworld import gridworld
from hard_policy_evaluation import policy_evaluation, value_iteration, monte_carlo_optimal_policy, worst_value_iteration
from game_dynamic_algorithms import policy_iteration, value_iteration, policy_evaluation
from dynamic_programming import policy_evaluation_finiteMDP, optimal_control






# 1. Define the rewards
rewards = {
    (3, 3): {"type": "goal", "reward_type": "deterministic", "value": 10, "is_terminal": True},
}

noise_dirs = {"up": 0.25, "down": 0.25, "left": 0.25, "right": 0.25}

# 2. Create the environments
env = gridworld(
    m=4, n=4, 
    reward_structure=rewards, 
    default_reward=0, 
    wall_behavior="reflect", 
    start_state=(0, 0), 
    wind_direction="right", 
    wind_prob=0, 
    slip_prob=0, 
    noise_prob=0.3, 
    noise_directions=noise_dirs
)


gamma = 0.9
V_star, pi_policy, pi_errors = policy_iteration(env, gamma, max_steps=None, use_Q=False)

# Assuming you collected these lists from your modified functions
pi_V, pi_policy, pi_errors = value_iteration(env, gamma, V_star)
vi_V, vi_policy, vi_errors = policy_iteration(env, gamma, V_star)

# 1. Set up the plot
plt.figure(figsize=(8, 6))

# 2. Plot the error lists
# We use a marker like 'o' to see exactly where each iteration lands
plt.plot(vi_errors, label='Value Iteration', marker='o', color='blue')
plt.plot(pi_errors, label='Policy Iteration', marker='s', color='red')

# 3. Set the Y-axis to a Logarithmic Scale! 
plt.yscale('log')

# 4. Make it look nice
plt.title('Empirical Convergence Rates: VI vs PI')
plt.xlabel('Number of Iterations')
plt.ylabel('Max Error (Log Scale): $||V_n - V^*||_{\infty}$')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()

# 5. Show the graph
plt.show()