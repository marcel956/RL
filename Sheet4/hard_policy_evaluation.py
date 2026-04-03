import numpy as np

from gridworld import gridworld
from multi_step_bandit import MultiStepBandit

def policy_evaluation(env, policy, gamma=1.0, epsilon=1e-6):

    # Initialize value function
    V = {state: 0 for state in env.allowed_actions.keys()}

    # Make sure terminal states are 0
    for state in env.terminal_states:
        
        V[state] = 0

    delta = 1

    while delta > epsilon:

        delta = 0

        # Initialize V_new for the updated values
        V_new = V.copy()

        for state in env.allowed_actions:
            # Skip terminal states
            if state in env.terminal_states:
                continue

            V_old = V[state]

            # 1. Look up what the policy says to do
            policy_action = policy[state]

            # Get transition probabilities and exp reward
            trans_prob = env.transition_probabilities[(state, policy_action)]

            # Calculate expected return
            expected_return = 0

            for next_state, prob in trans_prob.items():

                expected_reward = env.get_expected_rewards(state, policy_action, next_state)

                expected_return += prob * (expected_reward + gamma * V[next_state])

            # Update V_new
            V_new[state] = expected_return

            # Calculate delta
            delta = max(delta, abs(V_old - V_new[state]))

        V = V_new

    return V



def value_iteration(env, gamma=1.0, epsilon=1e-6):

    # Initialize value function and optimal policy
    V = {state: 0 for state in env.allowed_actions.keys()}
    optimal_policy = {}


    # Make sure terminal states are 0
    for state in env.terminal_states:
        
        V[state] = 0

    delta = 1

    while delta > epsilon:

        delta = 0

        # Initialize V_new for the updated values
        V_new = V.copy()

        for state in env.allowed_actions:
            # Skip terminal states
            if state in env.terminal_states:
                continue

            V_old = V[state]

            max_return = -np.inf
            best_action = None

            # 1. Loop through every allowed action
            for action in env.allowed_actions[state]:

                # Get transition probabilities and exp reward
                trans_prob = env.transition_probabilities[(state, action)]

                # Calculate expected return
                expected_return = 0

                for next_state, prob in trans_prob.items():

                    expected_reward = env.get_expected_rewards(state, action, next_state)

                    expected_return += prob * (expected_reward + gamma * V[next_state])

                # Calculate max return and save best action
                if max_return < expected_return:
                    max_return = expected_return
                    best_action = action


            # Update V_new and optimal policy
            V_new[state] = max_return
            optimal_policy[state] = best_action

            # Calculate delta
            delta = max(delta, abs(V_old - V_new[state]))


        V = V_new

    return V, optimal_policy

        

def monte_carlo_optimal_policy(env, num_episodes=5000, gamma=1.0):
    # 1. Run MC with a completely random policy to explore everything
    random_policy = env.allowed_actions
    Q_values = env.monte_carlo(random_policy, num_episodes, gamma)
    
    optimal_policy = {}
    
    # 2. Look at the results and pick the best action for each state
    for state in env.allowed_actions.keys():
        if state in env.terminal_states:
            continue
            
        best_action = None
        max_q = -np.inf
        
        for action in env.allowed_actions[state]:
            # Get the Q-value our MC algorithm estimated (default to 0 if unexplored)
            q_val = Q_values.get((state, action), 0)
            
            if q_val > max_q:
                max_q = q_val
                best_action = action
                
        optimal_policy[state] = best_action
        
    return optimal_policy, Q_values



def worst_value_iteration(env, gamma=1.0, epsilon=1e-6):

    # Initialize value function and optimal policy
    V = {state: 0 for state in env.allowed_actions.keys()}
    optimal_policy = {}


    # Make sure terminal states are 0
    for state in env.terminal_states:
        
        V[state] = 0

    delta = 1

    while delta > epsilon:

        delta = 0

        # Initialize V_new for the updated values
        V_new = V.copy()

        for state in env.allowed_actions:
            # Skip terminal states
            if state in env.terminal_states:
                continue

            V_old = V[state]

            min_return = np.inf
            best_action = None

            # 1. Loop through every allowed action
            for action in env.allowed_actions[state]:

                # Get transition probabilities and exp reward
                trans_prob = env.transition_probabilities[(state, action)]

                # Calculate expected return
                expected_return = 0

                for next_state, prob in trans_prob.items():

                    expected_reward = env.get_expected_rewards(state, action, next_state)

                    expected_return += prob * (expected_reward + gamma * V[next_state])

                # Calculate max return and save best action
                if min_return > expected_return:
                    min_return = expected_return
                    best_action = action


            # Update V_new and optimal policy
            V_new[state] = min_return
            optimal_policy[state] = best_action

            # Calculate delta
            delta = max(delta, abs(V_old - V_new[state]))


        V = V_new

    return V, optimal_policy







