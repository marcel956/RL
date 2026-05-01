import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

sheet4_path = Path(__file__).parent.parent / "Sheet4"
sheet5_path = Path(__file__).parent.parent / "Sheet5"
sheet6_path = Path(__file__).parent.parent / "Sheet6"

sys.path.append(str(sheet4_path))
sys.path.append(str(sheet5_path))
sys.path.append(str(sheet6_path))

from gridworld import gridworld
from sample_based_algorithms import Q_into_policy
from Q_learning import Q_learning, double_Q_learning
from hard_policy_evaluation import policy_evaluation








# Define the environment:
rewards = {
    (3, 1): {"type": "goal", "reward_type": "deterministic", "value": 1, "is_terminal": True},
    (0, 0): {"type": "fake goal", "reward_type": "deterministic", "value": 0.65, "is_terminal": True},

    # Choice type for stochastic region
    (2, 2): {"type": "stochastic region", "reward_type": "choice", "values": [-2.1, 2], "is_terminal": False},
    (2, 3): {"type": "stochastic region", "reward_type": "choice", "values": [-2.1, 2], "is_terminal": False},
    (3, 2): {"type": "stochastic region", "reward_type": "choice", "values": [-2.1, 2], "is_terminal": False},
    (3, 3): {"type": "stochastic region", "reward_type": "choice", "values": [-2.1, 2], "is_terminal": False}
}

noise_dirs = {"up": 0.25, "down": 0.25, "left": 0.25, "right": 0.25}

env = gridworld(
    m=4, n=4, 
    reward_structure=rewards, 
    default_reward=[-0.05, 0.05], 
    wall_behavior="reflect", 
    start_state=(0, 3), 
    wind_direction="right", 
    wind_prob=0, 
    slip_prob=0, 
    noise_prob=0.2, 
    noise_directions=noise_dirs
)



# Parameters to test
epsilon = 0.25
num_episodes=25000
schedule= "constant"

# Run Standard Q-Learning

print("--- Standard Q-Learning ---")
start = time.time()
Q_standard = Q_learning(
    env, 
    epsilon=epsilon, 
    num_episodes=num_episodes, 
    gamma=0.9, 
    schedule_type=schedule 
)
print(f"\nFinished in {time.time() - start:.2f} seconds")

# Convert Q-table to a deterministic policy to visualize it
policy_standard = Q_into_policy(env, Q_standard)

# Evaluate the Standard Q-Learning Policy
V_standard = policy_evaluation(env, policy_standard, gamma=0.9)
print(f"\nPolicy Value: {V_standard[(0, 3)]:.4f}")

# Print policy visual
print("\nStandard Q-Learning Policy:")
env.visualize_policy(policy_standard)

# Run Double Q-Learning
print("\n--- Double Q-Learning ---")
start = time.time()
Q_double = double_Q_learning(
    env, 
    epsilon=epsilon, 
    num_episodes=num_episodes, 
    gamma=0.9, 
    schedule_type=schedule
)
print(f"\nFinished in {time.time() - start:.2f} seconds")

policy_double = Q_into_policy(env, Q_double)

# Evaluate the Double Q-Learning Policy
V_double = policy_evaluation(env, policy_double, gamma=0.9)
print(f"\nPolicy Value: {V_double[(0, 3)]:.4f}")

# Print policy visual
print("\nDouble Q-Learning Policy:")
env.visualize_policy(policy_double)



#====================
# Parameter Sweep
#====================

# print("\nParameter Sweep ")

# # Parameters
# test_epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,  0.4]
# test_schedules = ["constant", "1/n", "1/sqrt(n)"] 
# test_episodes = 25000
# gamma = 0.9

# results = []

# for eps in test_epsilons:
#     for sched in test_schedules:
#         print(f"Training with Epsilon: {eps:<4} | Schedule: {sched:<10} ", end="")
        
#         # 1. Train and Evaluate Standard Q-Learning
#         Q_std = Q_learning(
#             env, epsilon=eps, num_episodes=test_episodes, gamma=gamma, schedule_type=sched
#         )
#         policy_std = Q_into_policy(env, Q_std)
#         V_std = policy_evaluation(env, policy_std, gamma=gamma)
#         score_std = V_std[(0, 3)]
        
#         # 2. Train and Evaluate Double Q-Learning
#         Q_dbl = double_Q_learning(
#             env, epsilon=eps, num_episodes=test_episodes, gamma=gamma, schedule_type=sched
#         )
#         policy_dbl = Q_into_policy(env, Q_dbl)
#         V_dbl = policy_evaluation(env, policy_dbl, gamma=gamma)
#         score_dbl = V_dbl[(0, 3)]
        
#         # 3. Store the results
#         results.append({
#             "epsilon": eps, 
#             "schedule": sched, 
#             "score_std": score_std, 
#             "score_dbl": score_dbl
#         })
#         print(f"-> Std Q: {score_std:>7.4f} | Double Q: {score_dbl:>7.4f}")


# # Final report
# print("\n\n" + "="*60)
# print(" Parameter Sweep Results ")
# print("="*60)
# print(f"{'Epsilon':<10} | {'Schedule':<12} | {'Standard Q Value':<18} | {'Double Q Value':<18}")
# print("-" * 65)

# for r in results:
#     # Format the scores to highlight the optimal 0.6561 vs trap values
#     print(f"{r['epsilon']:<10.2f} | {r['schedule']:<12} | {r['score_std']:<18.4f} | {r['score_dbl']:<18.4f}")
    
# print("="*60)