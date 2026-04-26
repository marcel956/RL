import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

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
from sample_based_algorithms import monte_carlo_Q, monte_carlo_V, totally_async_policy_evaluation, Q_learning, RMSE_evaluation, Q_into_policy, Q_into_V, evaluate_pit_stop
from actor_critic import epsilon_greedy_actor, SARSA_critic, general_actor_critic





import time

# --- 1. Environment Setup ---
rewards = {
    (3, 3): {"type": "goal", "reward_type": "deterministic", "value": 10, "is_terminal": True},
    (1, 1): {"type": "bomb", "reward_type": "deterministic", "value": -10, "is_terminal": True},
}

noise_dirs = {"up": 0.25, "down": 0.25, "left": 0.25, "right": 0.25}

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

# --- 2. Grid Visualizer Helper ---
def print_policy(env, policy, title):
    print(f"\n=== {title} ===")
    for r in range(env.m):
        row_str = ""
        for c in range(env.n):
            state = (r, c)
            if state in env.terminal_states:
                if state == (3, 3): row_str += " [GOAL] "
                elif state == (1, 1): row_str += " [BOMB] "
                else: row_str += " [TERM] "
            else:
                # Find the most common action in the policy's list for this state
                actions = policy[state]
                best_action = max(set(actions), key=actions.count)
                
                # Map strings to arrow symbols
                symbols = {"up": "  ↑   ", "down": "  ↓   ", "left": "  ←   ", "right": "  →   "}
                row_str += symbols.get(best_action, "  ?   ")
        print(row_str)
    print("=" * (env.n * 8))


# --- 3. Run the Actor-Critic Comparisons! ---
print("Starting Actor-Critic Experiments...")
iterations = 15
episodes_per_eval = 50
gamma_val = 0.9

# TEST 1: SARSA Critic
print("\nTraining with SARSA Critic...")
start = time.time()
policy_sarsa = general_actor_critic(
    env, iterations, gamma_val, 
    critic_eval_fn=SARSA_critic, 
    actor_improve_fn=epsilon_greedy_actor, 
    num_episodes=episodes_per_eval
)
print(f"Finished in {time.time() - start:.2f} seconds")
print_policy(env, policy_sarsa, "SARSA Critic Policy")


# TEST 2: Totally Async Critic
print("\nTraining with Async Critic...")
start = time.time()
policy_async = general_actor_critic(
    env, iterations, gamma_val, 
    critic_eval_fn=totally_async_policy_evaluation, 
    actor_improve_fn=epsilon_greedy_actor, 
    num_episodes=episodes_per_eval,
    output="Q" # Passing our kwargs!
)
print(f"Finished in {time.time() - start:.2f} seconds")
print_policy(env, policy_async, "Async Critic Policy")


# TEST 3: Monte Carlo Critic
print("\nTraining with Monte Carlo Critic...")
start = time.time()
policy_mc = general_actor_critic(
    env, iterations, gamma_val, 
    critic_eval_fn=monte_carlo_Q, 
    actor_improve_fn=epsilon_greedy_actor, 
    num_episodes=episodes_per_eval,
    first_visit=True # Passing our kwargs!
)
print(f"Finished in {time.time() - start:.2f} seconds")
print_policy(env, policy_mc, "Monte Carlo Critic Policy")

