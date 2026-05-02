import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

sheet4_path = Path(__file__).parent.parent / "Sheet4"
sheet5_path = Path(__file__).parent.parent / "Sheet5"
sheet6_path = Path(__file__).parent.parent / "Sheet6"
sheet8_path = Path(__file__).parent.parent / "Sheet8"



sys.path.append(str(sheet4_path))
sys.path.append(str(sheet5_path))
sys.path.append(str(sheet6_path))
sys.path.append(str(sheet8_path))


from gridworld import gridworld
from sample_based_algorithms import Q_into_policy
from Q_learning import Q_learning
from hard_policy_evaluation import policy_evaluation
from SARSA import SARSA










# Define the environment:
rewards = {
    (0, 11): {"type": "goal", "reward_type": "deterministic", "value": 100, "is_terminal": True},

    # Cliff
    (0, 1): {"type": "bomb", "reward_type": "deterministic", "value": -100, "is_terminal": True},
    (0, 2): {"type": "bomb", "reward_type": "deterministic", "value": -100, "is_terminal": True},
    (0, 3): {"type": "bomb", "reward_type": "deterministic", "value": -100, "is_terminal": True},
    (0, 4): {"type": "bomb", "reward_type": "deterministic", "value": -100, "is_terminal": True},
    (0, 5): {"type": "bomb", "reward_type": "deterministic", "value": -100, "is_terminal": True},
    (0, 6): {"type": "bomb", "reward_type": "deterministic", "value": -100, "is_terminal": True},
    (0, 7): {"type": "bomb", "reward_type": "deterministic", "value": -100, "is_terminal": True},
    (0, 8): {"type": "bomb", "reward_type": "deterministic", "value": -100, "is_terminal": True},
    (0, 9): {"type": "bomb", "reward_type": "deterministic", "value": -100, "is_terminal": True},
    (0, 10): {"type": "bomb", "reward_type": "deterministic", "value": -100, "is_terminal": True},


}

noise_dirs = {"up": 0.25, "down": 0.25, "left": 0.25, "right": 0.25}

env = gridworld(
    m=4, n=12, 
    reward_structure=rewards, 
    default_reward=-1, 
    wall_behavior="reflect", 
    start_state=(0, 0), 
    wind_direction="up", 
    wind_prob=0, 
    slip_prob=0, 
    noise_prob=0, 
    noise_directions=noise_dirs
)




print("--- Q-Learning ---")
start = time.time()
QL_Q = Q_learning(env, epsilon=0.1, num_episodes=10000, gamma=0.9, schedule_type="constant")

print(f"\nFinished in {time.time() - start:.2f} seconds")

# Convert Q-table to a deterministic policy to visualize it
QL_policy = Q_into_policy(env, QL_Q)

# Evaluate the Standard Q-Learning Policy
V_standard = policy_evaluation(env, QL_policy, gamma=0.9)
print(f"\nPolicy Value: {V_standard[(0, 0)]:.4f}")

# Print policy visual
print("\nQ-Learning Policy:")
env.visualize_policy(QL_policy)



print("--- SARSA ---")
start = time.time()
SARSA_Q = SARSA(env, epsilon=0.1, num_episodes=10000, gamma=0.9, alpha_schedule="constant", epsilon_schedule="constant")

print(f"\nFinished in {time.time() - start:.2f} seconds")

# Convert Q-table to a deterministic policy to visualize it
SARSA_policy = Q_into_policy(env, SARSA_Q)

# Evaluate the Standard Q-Learning Policy
V_standard = policy_evaluation(env, SARSA_policy, gamma=0.9)
print(f"\nPolicy Value: {V_standard[(0, 0)]:.4f}")

# Print policy visual
print("\nSARSA Policy:")
env.visualize_policy(SARSA_policy)


