import numpy as np

from gridworld import gridworld
from hard_policy_evaluation import policy_evaluation, value_iteration, monte_carlo_optimal_policy, worst_value_iteration


# The best, the worst, uniform, and random policies

def evaluate_policy(env, policy, N, T, gamma):

    # Initialize score
    score = np.zeros(N)

    # Loop through episodes
    for i in range(N):

        # Reset environment
        state = env.reset()

        # Loop through time steps
        for t in range(T):

            # Take step
            policy_action = policy[state]

            # If policy is a list, use uniformly random
            if type(policy_action) == list:
                action = np.random.choice(policy_action)

            # If it's deterministic, use it
            else:
                action = policy_action


            next_state, reward, is_terminal = env.step(state, action)

            # Update score and state
            score[i] += (gamma ** t) * reward
            state = next_state

            # Check if terminal
            if is_terminal:
                break

    # Return mean score
    return np.mean(score)







# 1. Define the rewards
rewards = {
    (3, 3): {"type": "goal", "reward_type": "deterministic", "value": 10, "is_terminal": True},
    (1, 1): {"type": "bomb", "reward_type": "deterministic", "value": -10, "is_terminal": True},
}

noise_dirs = {"up": 0.25, "down": 0.25, "left": 0.25, "right": 0.25}

# 2. Create the environments
wind_env = gridworld(
    m=4, n=4, 
    reward_structure=rewards, 
    default_reward=-1, 
    wall_behavior="reflect", 
    start_state=(0, 0), 
    wind_direction="right", 
    wind_prob=0.3, 
    slip_prob=0, 
    noise_prob=0, 
    noise_directions=noise_dirs
)

slip_env = gridworld(
    m=4, n=4, 
    reward_structure=rewards, 
    default_reward=-1, 
    wall_behavior="reflect", 
    start_state=(0, 0), 
    wind_direction="right", 
    wind_prob=0, 
    slip_prob=0.3, 
    noise_prob=0, 
    noise_directions=noise_dirs
)

noise_env = gridworld(
    m=4, n=4, 
    reward_structure=rewards, 
    default_reward=-1, 
    wall_behavior="reflect", 
    start_state=(0, 0), 
    wind_direction="right", 
    wind_prob=0, 
    slip_prob=0, 
    noise_prob=0.3, 
    noise_directions=noise_dirs
)


# 3. Loop through the environments
environment_list = [
    ("Wind Environment", wind_env),
    ("Slip Environment", slip_env),
    ("Noise Environment", noise_env)
]

print("=" * 60)
print(f"{'GRIDWORLD POLICY EVALUATION':^60}")
print("=" * 60)

for env_name, env in environment_list:
    print(f"\n\n--- {env_name} ---")

    # The best policy via value iteration:
    best_policy = value_iteration(env, gamma=0.9)[1]

    # The worst policy via worst value iteration:
    worst_policy = worst_value_iteration(env, gamma=0.9)[1]

    # The uniform policy:
    uniform_policy = env.allowed_actions

    # The fixed random policy:
    fixed_random_policy = {}

    for state in env.allowed_actions:
        if state in env.terminal_states:
            continue
        fixed_random_policy[state] = np.random.choice(env.allowed_actions[state])

    # Single line alternative:
    # fixed_random_policy = {state: np.random.choice(actions) for state, actions in env.allowed_actions.items()}


    # Loop through the policies and evaluate:
    policy_list = [
        ("Best Policy", best_policy),
        ("Worst Policy", worst_policy),
        ("Uniform Random Policy", uniform_policy),
        ("Fixed Random Policy", fixed_random_policy)
    ]

    for policy_name, policy in policy_list:
        score = evaluate_policy(env, policy, N=1000, T=16, gamma=0.9)

        print(f"\n{policy_name:<22} | Mean Score: {score:>7.3f}")

        # Optional visual print:
        #env.visualize_policy(policy, prefix=" Policy visual:  ")

print("\n" + "=" * 60)
