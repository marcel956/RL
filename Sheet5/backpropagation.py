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





def visualize_grid(V, policy=None, grid_size=(4, 4)):
    # 1. Create an empty 2D array for the heatmap
    v_array = np.zeros(grid_size)
    
    # 2. Fill the array with values from the V dictionary
    for state, value in V.items():
        if type(state) == tuple and len(state) == 2:
            row, col = state
            v_array[row, col] = value

    # 3. Set up the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 4. Draw the heatmap using matplotlib's imshow
    # 'YlGnBu' is a nice colormap that goes from Yellow (low) to Blue (high)
    cax = ax.imshow(v_array, cmap="YlGnBu")

    # Draw gridlines to separate the squares cleanly
    ax.set_xticks(np.arange(-0.5, grid_size[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size[0], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0) # Hide the actual tick marks
    ax.set_xticks([]) # Hide coordinate numbers
    ax.set_yticks([])

    # 5. Overlay the numbers and policy arrows
    arrow_map = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
    
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            # Print the V-value
            val = v_array[row, col]
            ax.text(col, row - 0.15, f"{val:.2f}", ha='center', va='center', color='black')
            
            # Print the policy arrow
            if policy is not None:
                state = (row, col)
                if state in policy:
                    action = policy[state]
                    if action in arrow_map:
                        ax.text(col, row + 0.2, arrow_map[action], 
                                ha='center', va='center', fontsize=20, color='red', weight='bold')

    plt.title("Value Function and Policy")
    plt.show()








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
    noise_prob=0, 
    noise_directions=noise_dirs
)

for max_steps in range(1,7):

    V, policy, x = policy_iteration(env, gamma=0.9, V_star=None, max_steps=max_steps, use_Q=False)

    visualize_grid(V, policy)

