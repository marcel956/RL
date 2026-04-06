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
#from backpropagation import visualize_grid




# 1. Define the rewards
rewards = {
    (3, 3): {"type": "goal", "reward_type": "deterministic", "value": 10, "is_terminal": True},
}

noise_dirs = {"up": 0.25, "down": 0.25, "left": 0.25, "right": 0.25}

# 2. Create the environments
env_zero = gridworld(
    m=4, n=4, 
    reward_structure=rewards, 
    default_reward=0, 
    wall_behavior="reflect", 
    start_state=(0, 0), 
    wind_direction="right", 
    wind_prob=0, 
    slip_prob=0, 
    noise_prob=0, 
    noise_directions=noise_dirs
)

env_plus = gridworld(
    m=4, n=4, 
    reward_structure=rewards, 
    default_reward=1, 
    wall_behavior="reflect", 
    start_state=(0, 0), 
    wind_direction="right", 
    wind_prob=0, 
    slip_prob=0, 
    noise_prob=0, 
    noise_directions=noise_dirs
)

env_minus = gridworld(
    m=4, n=4, 
    reward_structure=rewards, 
    default_reward=-1, 
    wall_behavior="reflect", 
    start_state=(0, 0), 
    wind_direction="right", 
    wind_prob=0, 
    slip_prob=0, 
    noise_prob=0, 
    noise_directions=noise_dirs
)


# 3. Loop through the environments
environment_list = [
    ("Zero Environment", env_zero),
    ("Plus Environment", env_plus),
    ("Minus Environment", env_minus)
]

print("=" * 60)
print(f"{'GRIDWORLD POLICY EVALUATION':^60}")
print("=" * 60)



for env_name, env in environment_list:
    print(f"\n\n--- {env_name} ---")

    differences = []

    V_pi = value_iteration(env, gamma=0.9, max_steps=None, use_Q=False)[0]

    for t in range(1,21):
        V_all_oc = optimal_control(env, gamma=0.9, T=t)[0]

        V_oc = V_all_oc[0]

        diff = np.max([np.abs(V_pi[s] - V_oc[s]) for s in env.allowed_actions])
        differences.append(diff)

    plt.plot(range(1, 21), differences, label=env_name)

# After the loop finishes:
plt.yscale('log') # Just like Task 7!
plt.xlabel('Time Horizon T')
plt.ylabel('Max Difference from Infinite V')
plt.title('Convergence of Finite to Infinite Horizon')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()


print("\n" + "=" * 60)





