import sys
from pathlib import Path
import numpy as np


sheet6_path = Path(__file__).parent.parent / "Sheet4"
sys.path.append(str(sheet6_path))

from sample_based_algorithms import step_size_scheduler







def epsilon_greedy_actor(env, Q, epsilon=0.5):

    policy = {}

    for state in env.allowed_actions.keys():

        if state in env.terminal_states:
            continue

        actions = env.allowed_actions[state]

        # Find best action:
        best_action = None
        best_value = float("-inf")

        for action in actions:      
            if Q[(state, action)] > best_value:
                best_value = Q[(state, action)]
                best_action = action


        # Build a weighted list
        # If epsilon is 0.1, the best action should be picked ~90% of the time.
        # Simulate this by adding the best action to the list many times.

        # Calculate how many dublicates are needed to achieve epsilon ratio
        num_duplicates = int((1 - epsilon) / (epsilon / len(actions)))

        policy[state] = [best_action] * num_duplicates + actions

    return policy



def SARSA_critic(env, policy, num_episodes, gamma=1.0, alpha_schedule="constant", Q=None, N=None):


    # Initialize variables for Q values and N
    if Q is None and N is None:
        Q = {(state, action): 0.0 for state in env.allowed_actions.keys() for action in env.allowed_actions[state]}

        N = {state: 0.0 for state in env.allowed_actions.keys()}


    for _ in range(num_episodes):

        state = env.reset()

        # Choose action randomly from policy
        action = np.random.choice(policy[state])

        while True:

            # Update N & play the game
            N[state] += 1

            next_state, reward, is_terminal = env.step(state, action)

            if is_terminal:
                future_value = 0.0
                next_action = None
            else:

                # Random next action & future value
                next_action = np.random.choice(policy[next_state])

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
















def general_actor_critic(env, num_iterations, gamma, critic_eval_fn, actor_improve_fn, num_episodes=50, **critic_kwargs):


    # Start with a random policy
    policy = {}

    for state in env.allowed_actions.keys():
  
        # Skip terminal states
        if state in env.terminal_states:
            continue

        policy[state] = env.allowed_actions[state].copy()




    for i in range(num_iterations):

        # Critic (Policy Evaluation)
        # Run evaluation algorithm
        Q, _ = critic_eval_fn(env, policy, num_episodes, gamma=gamma, **critic_kwargs)

        # Actor (Policy Improvement)
        # Run improvement algorithm
        policy = actor_improve_fn(env, Q)



    return policy