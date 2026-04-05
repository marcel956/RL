import numpy as np



def policy_evaluation_finiteMDP(env, T, policy, gamma=1.0):


    # Set up master lists to hold dictionaries for all time steps
    V_all = [None] * (T + 1)

    # Initialize every state worth 0 at time T
    V_T = {state: 0 for state in env.allowed_actions.keys()}

    # Create an empty set of predecessors for every state (including terminals)
    predecessors = {state: set() for state in env.allowed_actions.keys()}

    # Make sure terminal states are 0
    for state in env.terminal_states:
         V_T[state] = 0
         predecessors[state] = set()

    # Loop through all forward transitions to build the backward map
    for (state, action), trans_prob in env.transition_probabilities.items():
        for next_state in trans_prob.keys():
            # next_state, can be reached from 'state'
            predecessors[next_state].add(state) 

    V_all[T] = V_T


    # Set active states
    active_states = set(env.terminal_states)


    for t in reversed(range(T)):

        # Set V and policy for this time step
        V_t = {}



        # Iterate over all states
        for state in env.allowed_actions:

            if state not in active_states:
                V_t[state] = 0
                continue


            # Skip terminal states and save them as zero
            if state in env.terminal_states:
                V_t[state] = 0
                continue


            action = policy[t][state]

            # Get transition probabilities
            trans_prob = env.transition_probabilities[(state, action)]

            # Calculate expected return
            expected_return = 0

            for next_state, prob in trans_prob.items():

                # Get expected reward
                expected_reward = env.get_expected_rewards(state, action, next_state)

                # Add expected future value
                expected_return += prob * (expected_reward + gamma * V_all[t+1][next_state])


            # Update V
            V_t[state] = expected_return


        # Update active states
        new_active_states = set(active_states)
        for state in active_states:
            if state in predecessors:
                new_active_states.update(predecessors[state])

        active_states = new_active_states


        # Save V and policy into master list
        V_all[t] = V_t

    return V_all





































# Optimal control Algorithm, optimized for gridworld with active steps. All steps that aren't adjacent to ones that have reward will be skipped
def optimal_control(env, gamma=1.0, T=0):

    # Set up master lists to hold dictionaries for all time steps
    V_all = [None] * (T + 1)
    policy_all = [None] * T

    # Initialize every state worth 0 at time T
    V_T = {state: 0 for state in env.allowed_actions.keys()}

    # Create an empty set of predecessors for every state (including terminals)
    predecessors = {state: set() for state in env.allowed_actions.keys()}

    # Make sure terminal states are 0
    for state in env.terminal_states:
         V_T[state] = 0
         predecessors[state] = set()

    # Loop through all forward transitions to build the backward map
    for (state, action), trans_prob in env.transition_probabilities.items():
        for next_state in trans_prob.keys():
            # next_state, can be reached from 'state'
            predecessors[next_state].add(state) 

    V_all[T] = V_T


    # Set active states
    active_states = set(env.terminal_states)


    for t in reversed(range(T)):

        # Set V and policy for this time step
        V_t = {}
        policy_t = {}



        # Iterate over all states
        for state in env.allowed_actions:

            if state not in active_states:
                V_t[state] = 0
                continue


            # Skip terminal states and save them as zero
            if state in env.terminal_states:
                V_t[state] = 0
                continue


            max_return = -np.inf
            best_action = None

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
                    expected_return += prob * (expected_reward + gamma * V_all[t+1][next_state])

                # Calculate max return and save best action
                if max_return < expected_return:
                    max_return = expected_return
                    best_action = action

            
            # Update V and policy
            V_t[state] = max_return
            policy_t[state] = best_action


        # Update active states
        new_active_states = set(active_states)
        for state in active_states:
            if state in predecessors:
                new_active_states.update(predecessors[state])

        active_states = new_active_states


        # Save V and policy into master list
        V_all[t] = V_t
        policy_all[t] = policy_t

    return V_all, policy_all

                            