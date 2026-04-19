import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

sheet4_path = Path(__file__).parent.parent / "Sheet4"
sheet5_path = Path(__file__).parent.parent / "Sheet5"


# Add this folder to Python's search path
sys.path.append(str(sheet4_path))
sys.path.append(str(sheet5_path))


from gridworld import gridworld
from hard_policy_evaluation import policy_evaluation, value_iteration, monte_carlo_optimal_policy, worst_value_iteration
from game_dynamic_algorithms import policy_iteration, value_iteration, policy_evaluation
from dynamic_programming import policy_evaluation_finiteMDP, optimal_control
from sample_based_algorithms import monte_carlo_Q, monte_carlo_V, totally_async_policy_evaluation, Q_learning, RMSE_evaluation, Q_into_policy, Q_into_V, evaluate_pit_stop






# 1. Define the rewards
rewards = {
    (3, 3): {"type": "goal", "reward_type": "deterministic", "value": 10, "is_terminal": True},
    (1, 1): {"type": "bomb", "reward_type": "deterministic", "value": -10, "is_terminal": True},
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


# Only evaluated one type of Q learning but can easily add more

# Calculate optimal V
V_opt, V_policy, _ = value_iteration(env, gamma=0.9, max_steps=None, use_Q=False)



# Q learning evaluation by chunk
total_episodes = 100
chunk_size = 5

# Completely random policy
random_policy = {s: env.allowed_actions[s] for s in env.allowed_actions.keys()}


# Drunk expert policy
drunk_expert_policy = {}

for state, actions in env.allowed_actions.items():
    if state in V_policy:
        best_action = V_policy[state]
        # Duplicate the best action 10 times in the list, then add the rest!
        # Example output: ['up', 'up', 'up', 'up', 'up', 'up', 'up', 'up', 'up', 'up', 'up', 'down', 'left', 'right']
        drunk_expert_policy[state] = [best_action] * 10 + actions
    else:
        drunk_expert_policy[state] = actions


# Policy biased towards down right
biased_policy = {}

for state, actions in env.allowed_actions.items():
    weighted_list = []
    for action in actions:
        # If the action is 'down' or 'right', add it to the list 5 times!
        if action in ['down', 'right']:
            weighted_list.extend([action] * 5)
        else:
            weighted_list.append(action)
            
    biased_policy[state] = weighted_list











# --- Put the policies in a dictionary to loop over ---
policies_to_test = {
    "Random Policy": random_policy,
    "Drunk Expert": drunk_expert_policy,
    "Biased Policy": biased_policy
}

all_results = {}


for policy_name, policy in policies_to_test.items():
    
    print(f"Training with {policy_name}...")

    # Lists to hold plotting data
    episodes_x = []
    rmse_y = []
    avg_score_y = []
    correct_rate_y = []
    start_q_value_y = []


    # Empty tables to start
    current_Q = None
    current_N = None

    start_time = time.time()



    for current_episode in range(0, total_episodes, chunk_size):
        
        # 1. Train for a chunk of episodes, passing in the current memory
        current_Q, current_N = Q_learning(env, random_policy, chunk_size, gamma=0.9, schedule_type="constant", Q=current_Q, N=current_N)
        
        # 2. Extract V_curr from Q (Finding the max action value for each state)
        V_curr = Q_into_V(env, current_Q)
            
        # 3. Calculate the error
        error = RMSE_evaluation(env, V_opt, V_curr)
        
        # 4. Save the data for the plot
        episodes_x.append(current_episode + chunk_size)
        rmse_y.append(error)

        avg_score, correct_rate, start_q_value = evaluate_pit_stop(env, current_Q, V_policy, (0, 0), chunk_size)

        avg_score_y.append(avg_score)
        correct_rate_y.append(correct_rate)
        start_q_value_y.append(start_q_value)

    end_time = time.time()
    print(f"Q-Learning Total Runtime: {end_time - start_time} seconds")


    # Save all the data into the master dictionary
    all_results[policy_name] = {
        "episodes": episodes_x,
        "rmse": rmse_y,
        "score": avg_score_y,
        "correct": correct_rate_y,
        "start_q": start_q_value_y
    }






# --- Plot the results ---
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Q-Learning Exploration Methods Comparison")

# Pick colors so they stay consistent across the 4 plots
colors = {"Random Policy": "blue", "Drunk Expert": "orange", "Biased Policy": "green"}

# Loop through our saved results and draw the lines!
for policy_name, data in all_results.items():
    axs[0, 0].plot(data["episodes"], data["rmse"], label=policy_name, color=colors[policy_name])
    axs[0, 1].plot(data["episodes"], data["correct"], label=policy_name, color=colors[policy_name])
    axs[1, 0].plot(data["episodes"], data["score"], label=policy_name, color=colors[policy_name])
    axs[1, 1].plot(data["episodes"], data["start_q"], label=policy_name, color=colors[policy_name])

# Format Plot 1: RMSE
axs[0, 0].set_title("RMSE (Lower is better)")
axs[0, 0].set_ylabel("Error")
axs[0, 0].legend() # Only need one legend!

# Format Plot 2: Correct Action Rate
axs[0, 1].set_title("Correct Action Rate (Higher is better)")
axs[0, 1].set_ylabel("% Correct")
axs[0, 1].set_ylim([0, 1.1]) 

# Format Plot 3: Average Score
axs[1, 0].set_title(f"Avg Score over {chunk_size} episodes")
axs[1, 0].set_ylabel("Score")
axs[1, 0].set_xlabel("Training Episodes")

# Format Plot 4: Start State Q-Value
axs[1, 1].set_title("Start State Q-Value (Backprop Effect)")
axs[1, 1].set_ylabel("Max Q-Value at (0,0)")
axs[1, 1].set_xlabel("Training Episodes")

plt.tight_layout()
plt.show()










