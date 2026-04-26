import sys
from pathlib import Path
import numpy as np

sheet4_path = Path(__file__).parent.parent / "Sheet4"
sheet6_path = Path(__file__).parent.parent / "Sheet6"

sys.path.append(str(sheet4_path))
sys.path.append(str(sheet6_path))

from gridworld import gridworld
from sample_based_algorithms import step_size_scheduler




def get_epsilon_greedy(env, Q, state, epsilon):

    # Roll dice
    if np.random.random() < epsilon:
        # Explore
        return np.random.choice(env.allowed_actions[state])
    
    else :
        # Exploit
        best_action = None
        best_value = float("-inf")

        for action in env.allowed_actions[state]:
            if Q[(state, action)] > best_value:
                best_value = Q[(state, action)]
                best_action = action
        return best_action






def SARSA(env, num_episodes, gamma=1.0, alpha_schedule="constant", epsilon=0.1, epsilon_schedule="constant", Q=None, N=None):

    # Initialize variables for Q values and N
    if Q is None and N is None:
        Q = {(state, action): 0.0 for state in env.allowed_actions.keys() for action in env.allowed_actions[state]}

        N = {state: 0.0 for state in env.allowed_actions.keys()}


    for _ in range(num_episodes):

        state = env.reset()

        epsilon = step_size_scheduler(epsilon_schedule, N[state])

        action = get_epsilon_greedy(env, Q, state, epsilon)

        while True:

            # Update N & play the game
            N[state] += 1

            next_state, reward, is_terminal = env.step(state, action)

            if is_terminal:
                future_value = 0.0
                next_action = None
            else:

                # Calculate next action & future value
                epsilon = step_size_scheduler(epsilon_schedule, N[next_state])

                next_action = get_epsilon_greedy(env, Q, next_state, epsilon)

                future_value = Q[(next_state, next_action)]

            # SARSA update rule:
            current_alpha = step_size_scheduler(alpha_schedule, N[state])

            Q[state, action] = Q[state, action] + current_alpha * (reward + gamma * future_value - Q[state, action])

            # set up for next iteration
            state = next_state
            action = next_action
    
            if is_terminal:
                break

    return Q, N