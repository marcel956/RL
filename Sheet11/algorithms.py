import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time

sheet10_path = Path(__file__).parent.parent / "Sheet10"

sys.path.append(str(sheet10_path))

import gymnasium as gym
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor 
from sb3_contrib import ARS, TQC, TRPO
from stable_baselines3.common.evaluation import evaluate_policy
from REINFORCE import REINFORCE

start_time_total = time.time()

# --- Define Environments ---
envs_discrete = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
envs_continuous = ["Pendulum-v1", "MountainCarContinuous-v0"]
all_envs = envs_discrete + envs_continuous

# The minimum score required to consider an episode "solved" or "successful"
env_success_thresholds = {
    "CartPole-v1": 500.0,             # Max possible score
    "Acrobot-v1": -499.0,             # Reaching the top before the 500-step timeout
    "MountainCar-v0": -199.0,         # Reaching the flag before the 200-step timeout
    "Pendulum-v1": -200.0,            # Pendulum doesn't have a strict win, but >-200 is great
    "MountainCarContinuous-v0": 90.0  # Environment standard for success
}


# --- Define Algorithms & Training Parameters ---
all_algs = [A2C, DDPG, PPO, SAC, TD3, ARS, TQC, TRPO, REINFORCE]
continuous_only_algs = ["DDPG", "SAC", "TD3", "TQC"]

# Set Total Training steps and frequency of evaluation
total_timesteps = 100000
eval_freq = 5000

# Seed list
seed_list = [7, 42, 1337, 12345, 666]

# List to save all results
evaluation_results = []


# --- Main Evaluation Loop ---

# Loop through all environments
for env_name in all_envs:
    print(f"\n{'='*40}")
    print(f"Evaluating Environment: {env_name}")
    print(f"{'='*40}")
    
    # Get the success threshold for evaluation later
    threshold = env_success_thresholds[env_name]

    #Loop through all algorithms
    for alg in all_algs:

        alg_name = alg.__name__

        # Check if algorithm can be used with the environment
        if alg_name in continuous_only_algs and env_name not in envs_continuous:
            continue


        # Loop through all seeds in list
        for seed in seed_list:

            print(f"--- Testing {alg_name} on {env_name} (Seed: {seed}) ---")

            # Create the environment
            env = Monitor(gym.make(env_name))

            # Initialize model
            model = alg("MlpPolicy", env, seed=seed, verbose=0)

            # Track the cumulative training time
            cumulative_training_time = 0.0

            # Train the model while looping through evaluation frequency chunks
            for current_step in range(eval_freq, total_timesteps + 1, eval_freq):


                # Train the model with timer
                start_time = time.time()
                model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)
                end_time = time.time()

                # Calculate Training time
                cumulative_training_time += end_time - start_time

                # Evaluate the model ---
                episode_rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True, return_episode_rewards=True)
                
                # Calculate mean and std
                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                mean_length = np.mean(episode_lengths)


                # Calculate success rate
                successful_episodes = sum(1 for reward in episode_rewards if reward >= threshold)
                success_rate = successful_episodes / len(episode_rewards) * 100
                
                # Store the result
                evaluation_results.append({
                    "Environment": env_name,
                    "Algorithm": alg_name,
                    "Seed": seed,
                    "Current Step": current_step,
                    "Mean Reward": mean_reward,
                    "Standard Deviation Reward": std_reward,
                    "Success Rate": success_rate,
                    "Training Time": cumulative_training_time,
                    "Mean Episode Length": mean_length
                })
            
            # Close the environment to free up memory
            env.close()
            print(f"Success")

            # Save Checkpoint  
            # Save a backup after every single seed finishes
            backup_df = pd.DataFrame(evaluation_results)
            backup_df.to_csv("rl_evaluation_results_backup.csv", index=False)


# Turn result data into a Pandas DataFrame
df = pd.DataFrame(evaluation_results)
print(df.head()) # Preview 

# Save data to CSV file
df.to_csv("rl_evaluation_results.csv", index=False)
print("Data successfully saved to rl_evaluation_results.csv!")

end_time_total = time.time()
total_run_time = end_time_total - start_time_total
print(f"Total RunTime: {total_run_time} seconds")





