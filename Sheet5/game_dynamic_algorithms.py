import numpy as np


def policy_evaluation(env, policy, gamma=1.0, epsilon=1e-6, max_steps=None, async_update=False, use_Q=False):

    # Initialize value function or Q function
    if use_Q:
        # Initialize Q function to 0 for all state-action pairs
        V = {(state, action): 0 for state, actions in env.allowed_actions.items() for action in actions}

        # Make sure terminal states are 0
        for state in env.terminal_states:
            for action in env.allowed_actions[state]:
                V[(state, action)] = 0

    else:
        # Initialize value function to 0 for all states
        V = {state: 0 for state in env.allowed_actions.keys()}

        # Make sure terminal states are 0
        for state in env.terminal_states:
            V[state] = 0

    delta = 1
    step_counter = 0

    # Loop until convergence or max_steps reached
    while delta > epsilon:

        delta = 0

        # Initialize V_new for the updated values
        V_new = V.copy()

        # Iterate over all states
        for state in env.allowed_actions:

            # Skip terminal states
            if state in env.terminal_states:
                continue

            # Calculate Q matrix
            if use_Q:
                # Loop through every allowed action
                for action in env.allowed_actions[state]:

                    # Get transition probabilities and exp reward
                    V_old = V[state, action]

                    trans_prob = env.transition_probabilities[(state, action)]

                    expected_return = 0

                    for next_state, prob in trans_prob.items():

                        expected_reward = env.get_expected_rewards(state, action, next_state)   

                        # Edge case where next state is terminal
                        if next_state in env.terminal_states:

                            expected_return += prob * expected_reward

                        else:

                            expected_return += prob * (expected_reward + gamma * V[(next_state, policy[next_state])])


                    if async_update:
                        # Update V asynchronously and calculate delta
                        V[state, action] = expected_return
                        delta = max(delta, abs(V_old - V[state, action]))

                    else:
                        # Update V_new and calculate delta
                        V_new[state, action] = expected_return
                        delta = max(delta, abs(V_old - V_new[state, action]))

            # Calculate V vector
            else:

                V_old = V[state]

                # Look up what the policy says to do
                policy_action = policy[state]

                # Get transition probabilities and exp reward
                trans_prob = env.transition_probabilities[(state, policy_action)]

                # Calculate expected return
                expected_return = 0

                for next_state, prob in trans_prob.items():

                    expected_reward = env.get_expected_rewards(state, policy_action, next_state)

                    expected_return += prob * (expected_reward + gamma * V[next_state])


                if async_update:
                    # Update V asynchronously and calculate delta
                    V[state] = expected_return
                    delta = max(delta, abs(V_old - V[state]))

                else:
                    # Update V_new and calculate delta
                    V_new[state] = expected_return
                    delta = max(delta, abs(V_old - V_new[state]))


        if not async_update:
            V = V_new


        # Increment step counter
        step_counter += 1

        # Check if max_steps reached
        if max_steps is not None and step_counter >= max_steps:
            break

    return V









# added error tracking with V-star. can not simultaneously be used with useQ
def value_iteration(env, gamma=1.0, V_star=None, epsilon=1e-6, max_steps=None, async_update=False, use_Q=False):

    # Initialize value function and optimal policy
    if use_Q:
        # Initialize Q function to 0 for all state-action pairs
        V = {(state, action): 0 for state, actions in env.allowed_actions.items() for action in actions}

        # Make sure terminal states are 0
        for state in env.terminal_states:
            for action in env.allowed_actions[state]:
                V[(state, action)] = 0

    else:
        # Initialize value function to 0 for all states
        V = {state: 0 for state in env.allowed_actions.keys()}

        # Make sure terminal states are 0
        for state in env.terminal_states:
            V[state] = 0

    optimal_policy = {}
    error_history = []


    delta = 1
    step_counter = 0

    # Loop until convergence or max_steps reached
    while delta > epsilon:

        delta = 0

        # Initialize V_new for the updated values
        V_new = V.copy()

        # Iterate over all states
        for state in env.allowed_actions:
            # Skip terminal states
            if state in env.terminal_states:
                continue

            max_return = -np.inf
            best_action = None

            if use_Q:
                # Update Q function
                for action in env.allowed_actions[state]:

                    V_old = V[state, action]

                    # Get transition probabilities
                    trans_prob = env.transition_probabilities[(state, action)]

                    expected_return = 0

                    # Calculate expected return
                    for next_state, prob in trans_prob.items(): 

                        # Get expected reward
                        expected_reward = env.get_expected_rewards(state, action, next_state)

                        if next_state in env.terminal_states:
                            # Terminal state has 0 future value
                            expected_return += prob * expected_reward
                        else:
                            # Get all possible actions for the next state
                            future_actions = env.allowed_actions[next_state]

                            # Find max Q-value for the next state
                            future_q_values = [V[(next_state, next_action)] for next_action in future_actions]

                            max_future_q = max(future_q_values)

                            # Add expected future value based on max Q
                            expected_return += prob * (expected_reward + gamma * max_future_q)

                    # Calculate max return and save best action
                    if max_return < expected_return:
                        max_return = expected_return
                        best_action = action

                    # Update V_new and optimal policy
                    if async_update:
                        # Update V asynchronously in-plac and calculate delta
                        V[state, action] = expected_return
                        delta = max(delta, abs(V_old - V[state, action]))

                    else:
                        # Update V_new synchronously and calculate delta
                        V_new[state, action] = expected_return
                        delta = max(delta, abs(V_old - V_new[state, action]))

            else:
                # Update Value function
                V_old = V[state]

                # 1. Loop through every allowed action
                for action in env.allowed_actions[state]:

                    # Get transition probabilities
                    trans_prob = env.transition_probabilities[(state, action)]

                    # Calculate expected return
                    expected_return = 0

                    for next_state, prob in trans_prob.items():

                        # Get expected reward
                        expected_reward = env.get_expected_rewards(state, action, next_state)

                        # Add expected future value
                        expected_return += prob * (expected_reward + gamma * V[next_state])

                    # Calculate max return and save best action
                    if max_return < expected_return:
                        max_return = expected_return
                        best_action = action

                # Update V_new and optimal policy
                if async_update:
                    # Update V asynchronously in-place and calculate delta
                    V[state] = max_return
                    delta = max(delta, abs(V_old - V[state]))

                else:
                    # Update V_new synchronously and calculate delta
                    V_new[state] = max_return
                    delta = max(delta, abs(V_old - V_new[state]))

            # Store best action for current state
            optimal_policy[state] = best_action

        if not async_update:
            # Synchronous update of the value function
            V = V_new

        if V_star is not None:
            # Calculate the biggest difference between the current V and the perfect V_star
            current_error = max([abs(V[state] - V_star[state]) for state in env.allowed_actions])

            # Append it to the error history
            error_history.append(current_error)


        step_counter += 1

        # Check max_steps condition
        if max_steps is not None and step_counter >= max_steps:
            break

    return V, optimal_policy, error_history















def policy_iteration(env, gamma, V_star=None, epsilon=1e-6, max_steps=None, use_Q=False):

    # Initialize policy
    policy = {}
    error_history = []


    # Set random initial policy
    for state in env.allowed_actions:

        # Skip terminal states
        if state in env.terminal_states:
            continue

        policy[state] = np.random.choice(env.allowed_actions[state])

    policy_stable = False

    # Loop until policy does not change
    while not policy_stable:

        # Policy evaluation step
        V = policy_evaluation(env, policy, gamma=gamma, epsilon=epsilon, max_steps=max_steps, use_Q=use_Q)

        # Calculate Error, if V_star is specified
        if V_star is not None:
            # Calculate the biggest difference between the current V and the perfect V_star
            current_error = max([abs(V[state] - V_star[state]) for state in env.allowed_actions])

            # Append it to a tracking list!
            error_history.append(current_error)



        # Policy improvement step
        policy_stable = True
    
        # Iterate over all states
        for state in env.allowed_actions:

            # Skip terminal states
            if state in env.terminal_states:
                continue
            
            # Store old policy action
            old_action = policy[state]

            max_return = -np.inf
            best_action = None

            if use_Q:
                # Evaluate Q function directly
                for action in env.allowed_actions[state]:

                    # Look up expected return from Q function
                    expected_return = V[(state, action)]

                    # Calculate max return and save best action
                    if max_return < expected_return:
                        max_return = expected_return
                        best_action = action

            else:
                # Evaluate Value function
                # Loop through every allowed action
                for action in env.allowed_actions[state]:

                    # Get transition probabilities
                    trans_prob = env.transition_probabilities[(state, action)]

                    # Calculate expected return
                    expected_return = 0

                    for next_state, prob in trans_prob.items():

                        # Get expected reward
                        expected_reward = env.get_expected_rewards(state, action, next_state)

                        # Add expected future value
                        expected_return += prob * (expected_reward + gamma * V[next_state])

                    # Calculate max return and save best action
                    if max_return < expected_return:
                        max_return = expected_return
                        best_action = action

            # Update policy with best action
            policy[state] = best_action


            # Check if policy changed
            if old_action != best_action:
                policy_stable = False


    return V, policy, error_history
