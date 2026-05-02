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
from sample_based_algorithms import Q_into_policy, calculate_bias
from Q_learning import Q_learning
from hard_policy_evaluation import policy_evaluation
from SARSA import SARSA








# ==========================================
# THE OVERESTIMATION BIAS EXPERIMENT
# ==========================================

# 1. Build the Casino Environment (1 row, 3 columns)
rewards = {
    (0, 0): {"type": "goal", "reward_type": "deterministic", "value": 1.0, "is_terminal": True},

    (0, 2): {"type": "casino", "reward_type": "choice", "values": [-15.0, 15.0], "is_terminal": True},
    (2, 2): {"type": "casino", "reward_type": "choice", "values": [-15.0, 15.0], "is_terminal": True},
    (1, 3): {"type": "casino", "reward_type": "choice", "values": [-15.0, 15.0], "is_terminal": True}
}

noise_dirs = {"up": 0.25, "down": 0.25, "left": 0.25, "right": 0.25}

env = gridworld(
    m=3, n=4, 
    reward_structure=rewards, 
    default_reward=0,  
    wall_behavior="reflect", 
    start_state=(1, 1), # Start in the middle
    wind_direction="right", wind_prob=0, slip_prob=0, noise_prob=0, noise_directions=noise_dirs
)

gamma = 0.9
epsilon = 0.1
episodes = 20000 

print("\n" + "="*50)
print("--- Q-Learning ---")
print("="*50)
QL_Q = Q_learning(env, epsilon=epsilon, num_episodes=episodes, gamma=gamma, schedule_type="1/n")

# Evaluate and get metrics
QL_policy = Q_into_policy(env, QL_Q)
V_true_QL = policy_evaluation(env, QL_policy, gamma=gamma)

print("\nQ-Learning Bias Metrics:")
calculate_bias(env, QL_Q, V_true_QL, gamma=gamma)
print("\nQ-Learning Final Policy:")
env.visualize_policy(QL_policy)


print("\n" + "="*50)
print("--- SARSA ---")
print("="*50)
SARSA_Q = SARSA(env, epsilon=epsilon, num_episodes=episodes, gamma=gamma, alpha_schedule="1/n", epsilon_schedule="constant")

# Evaluate and get metrics
SARSA_policy = Q_into_policy(env, SARSA_Q)
V_true_SARSA = policy_evaluation(env, SARSA_policy, gamma=gamma)

print("\nSARSA Bias Metrics:")
calculate_bias(env, SARSA_Q, V_true_SARSA, gamma=gamma)
print("\nSARSA Final Policy:")
env.visualize_policy(SARSA_policy)