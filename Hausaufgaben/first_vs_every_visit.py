import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sheet4_path = Path(__file__).parent.parent / "Sheet4"
sheet5_path = Path(__file__).parent.parent / "Sheet5"
sheet6_path = Path(__file__).parent.parent / "Sheet6"


# Add this folder to Python's search path
sys.path.append(str(sheet4_path))
sys.path.append(str(sheet5_path))
sys.path.append(str(sheet6_path))

from gridworld import gridworld
from hard_policy_evaluation import policy_evaluation, value_iteration, monte_carlo_optimal_policy, worst_value_iteration
from game_dynamic_algorithms import policy_iteration, value_iteration, policy_evaluation
from dynamic_programming import policy_evaluation_finiteMDP, optimal_control
from sample_based_algorithms import monte_carlo_V





# --- 1. Environment Setup (From Previous Step) ---
rewards = {
    (4, 4): {"type": "goal", "reward_type": "deterministic", "value": 20, "is_terminal": True},
    (4, 0): {"type": "bomb", "reward_type": "deterministic", "value": -10, "is_terminal": True},
    (2, 2): {"type": "trap", "reward_type": "deterministic", "value": -5, "is_terminal": False},
    (1, 3): {"type": "bonus", "reward_type": "deterministic", "value": 5, "is_terminal": False},
}

noise_dirs = {"up": 0.25, "down": 0.25, "left": 0.25, "right": 0.25}

# Note: Assuming your custom `gridworld` class is imported or defined above this
env = gridworld(
    m=5, n=5, 
    reward_structure=rewards, 
    default_reward=-1, 
    wall_behavior="reflect", 
    start_state=(0, 0), 
    wind_direction="right", 
    wind_prob=0.0, 
    slip_prob=0.0, 
    noise_prob=0.3, 
    noise_directions=noise_dirs
)

# --- 2. Define the Policy ---
# We will use a Uniform Random Policy. 
# Since your function supports lists for random choice, we assign all 4 directions to every state.
actions = ["up", "down", "left", "right"]
policy = {state: actions for state in env.allowed_actions.keys()}

# --- 3. Run the Monte Carlo Tests ---
num_episodes = 5000
gamma = 0.9  # Discount factor is highly recommended for environments with heavy looping

print(f"Running tests for {num_episodes} episodes...\n")

# Run First-Visit MC
V_first, N_first = monte_carlo_V(env, policy, num_episodes, gamma=gamma, first_visit=True)

# Run Every-Visit MC
V_every, N_every = monte_carlo_V(env, policy, num_episodes, gamma=gamma, first_visit=False)

# --- 4. Helper Function to Visualize the Grids ---
def print_grid(data_dict, m, n, title, is_int=False):
    print(f"--- {title} ---")
    for y in range(n):
        row_str = ""
        for x in range(m):
            val = data_dict.get((x, y), 0.0)
            if is_int:
                row_str += f"{int(val):>6} | "
            else:
                row_str += f"{val:>6.2f} | "
        print(row_str[:-2]) # drop the last pipe
    print("\n")

# --- 5. Output the Results ---
print("### VALUE FUNCTIONS V(s) ###")
print_grid(V_first, 5, 5, "First-Visit MC: V(s)")
print_grid(V_every, 5, 5, "Every-Visit MC: V(s)")

print("### VISIT COUNTS N(s) ###")
print_grid(N_first, 5, 5, "First-Visit MC: N(s)", is_int=True)
print_grid(N_every, 5, 5, "Every-Visit MC: N(s)", is_int=True)