import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sheet4_path = Path(__file__).parent.parent / "Sheet4"

# Add this folder to Python's search path
sys.path.append(str(sheet4_path))

from gridworld import gridworld
from hard_policy_evaluation import policy_evaluation, value_iteration, monte_carlo_optimal_policy, worst_value_iteration
from game_dynamic_algorithms import policy_iteration, value_iteration, policy_evaluation
from dynamic_programming import policy_evaluation_finiteMDP, optimal_control







def step_size_scheduler(schedule_type, visit_count, initial_alpha):
# Ensure visit_count is at least 1 so we never divide by zero!
    n = max(1, visit_count) 
    
    if schedule_type == "constant":
        return 0.1  # Always 10%
    elif schedule_type == "1/n":
        return initial_alpha / n
    elif schedule_type == "1/sqrt(n)":
        return initial_alpha / np.sqrt(n)


       





def monte_carlo_Q(env, policy, num_episodes, gamma=1.0, first_visit=False, Q=None, N=None):

        # Initialize variables for Q values and returns
        if Q is None and N is None:
            Q = {}

            N = {}

        # Loop through number of episodes
        for i in range(num_episodes):

            # Reset state and episode
            state = env.reset()

            episode = []

            while True:

                # 1. Look up what the policy says to do
                policy_action = policy[state]
                
                # 2. If the policy gave is a list, pick randomly
                if type(policy_action) == list:
                    action = np.random.choice(policy_action)
                # 3. If the policy gave is a single integer, use it
                else:
                    action = policy_action

                # 4. Take the step
                next_state, reward, is_terminal = env.step(state, action)

                episode.append((state, action, reward))

                state = next_state

                # 5. Check if terminal
                if is_terminal:
                    break

            G = 0

            # Loop through episode backwards
            for t in reversed(range(len(episode))):
                 
                step_state, step_action, step_reward = episode[t]

                # Calculate return
                G = gamma * G + step_reward

                state_action = (step_state, step_action)

                # First visit check
                if first_visit: 
                    earlier_state_actions = [(step[0], step[1]) for step in episode[:t]]
                    if state_action in earlier_state_actions:
                        continue


                if state_action not in Q:
                    Q[state_action] = 0.0
                    N[state_action] = 0

                # Calculate Q value
                N[state_action] += 1
                Q[state_action] = Q[state_action] + 1/N[state_action] * (G - Q[state_action])

        

        return Q, N



















def monte_carlo_V(env, policy, num_episodes, gamma=1.0, first_visit=False, V=None, N=None):

        # Initialize variables for V values and returns
        if V is None and N is None:
            V = {state: 0.0 for state in env.allowed_actions.keys()}

            N = {state: 0 for state in env.allowed_actions.keys()}

        # Loop through number of episodes
        for i in range(num_episodes):

            # Reset state and episode
            state = env.reset()

            episode = []

            while True:

                # 1. Look up what the policy says to do
                policy_action = policy[state]
                
                # 2. If the policy gave is a list, pick randomly
                if type(policy_action) == list:
                    action = np.random.choice(policy_action)
                # 3. If the policy gave is a single integer, use it
                else:
                    action = policy_action

                # 4. Take the step
                next_state, reward, is_terminal = env.step(state, action)

                episode.append((state, action, reward))

                state = next_state

                # 5. Check if terminal
                if is_terminal:
                    break

            G = 0

            # Loop through episode backwards
            for t in reversed(range(len(episode))):
                 
                step_state, step_action, step_reward = episode[t]

                # Calculate return
                G = gamma * G + step_reward

                # First visit check
                if first_visit: 
                    earlier_steps = [step[0] for step in episode[:t]]
                    if step_state in earlier_steps:
                        continue


                if step_state not in V:
                    V[step_state] = 0.0
                    N[step_state] = 0

                # Calculate V value
                N[step_state] += 1
                V[step_state] = V[step_state] + 1/N[step_state] * (G - V[step_state])

        

        return V, N

















def totally_async_policy_evaluation(env, policy, num_episodes, schedule_type="constant", gamma=1.0, V=None, N=None):

        # Initialize variables for V values and returns
        if V is None and N is None:
            V = {state: 0.0 for state in env.allowed_actions.keys()}
            
            N = {state: 0 for state in env.allowed_actions.keys()}

        # Loop through number of episodes
        for i in range(num_episodes):

            # Reset state and episode
            state = env.reset()


            while True:

                # 1. Look up what the policy says to do
                policy_action = policy[state]
                
                # 2. If the policy gave is a list, pick randomly
                if type(policy_action) == list:
                    action = np.random.choice(policy_action)
                # 3. If the policy gave is a single integer, use it
                else:
                    action = policy_action

                # 4. Take the step
                next_state, reward, is_terminal = env.step(state, action)



                if state not in V:
                    V[state] = 0.0
                    N[state] = 0
                if next_state not in V:
                    V[next_state] = 0.0

                # Calculate V value
                N[state] += 1


                alpha = step_size_scheduler(schedule_type, N[state], 1)

                future_value = 0.0 if is_terminal else V[next_state]

                G = reward + gamma * future_value

                V[state] = V[state] + alpha * (G - V[state])
            

                state = next_state

                # 5. Check if terminal
                if is_terminal:
                    break


        return V, N






def Q_learning(env, policy, num_episodes, gamma=1.0, schedule_type="constant", Q=None, N=None):


    # Initialize variables for Q values and N
    if Q is None and N is None:
        Q = {(state, action): 0.0 for state in env.allowed_actions.keys() for action in env.allowed_actions[state]}

        N = {(state, action): 0.0 for state in env.allowed_actions.keys() for action in env.allowed_actions[state]}

    # Loop through number of episodes
    for i in range(num_episodes):

        # Reset state and episode
        state = env.reset()


        while True:

            # 1. Look up what the policy says to do
            policy_action = policy[state]
                
            # 2. If the policy gave is a list, pick randomly
            if type(policy_action) == list:
                    action = np.random.choice(policy_action)
            # 3. If the policy gave is a single integer, use it
            else:
                action = policy_action

            # 4. Take the step
            next_state, reward, is_terminal = env.step(state, action)


            state_action = (state, action)


            N[state_action] += 1

                # Calculate Q value
            if is_terminal:
                future_value = 0.0
            else:
                future_value = max([Q[next_state, future_action] for future_action in env.allowed_actions[next_state]])


            G = reward + gamma * future_value

            alpha = step_size_scheduler(schedule_type, N[state_action], 1)

            Q[state_action] = Q[state_action] + alpha * (G - Q[state_action])
            

            state = next_state

            # 5. Check if terminal
            if is_terminal:
                break


    return Q, N




# Evaluation helper functions:



def Q_into_policy(env, Q):

    policy = {}

    for state in env.allowed_actions.keys():

        best_action = None
        best_value = float("-inf")

        for action in env.allowed_actions[state]:

            if Q[(state, action)] > best_value:
                best_value = Q[(state, action)]
                best_action = action


        policy[state] = best_action

    return policy


def Q_into_V(env, Q):

    V = {}

    for state in env.allowed_actions.keys():

        V[state] = max(Q[state, action] for action in env.allowed_actions[state])

    return V



def RMSE_evaluation(env, V_opt, V_curr):

    diff = []

    for state in env.allowed_actions.keys():
        
        squared_err = (V_opt[state] - V_curr[state]) **2

        diff.append(squared_err)

    mean = np.mean(diff)
    RMSE = np.sqrt(mean)

    return RMSE



def evaluate_pit_stop(env, Q_curr, V_opt_policy, start_state, k_timesteps):
    
    # --- Extract the Greedy Policy from Q ---
    current_greedy_policy = Q_into_policy(env, Q_curr)

    # --- Correct Action Rate ---
    # Compare our greedy policy to the perfect DP policy
    correct_actions = 0
    total_states = len(env.allowed_actions.keys()) - len(env.terminal_states)
    
    for state in env.allowed_actions.keys():

        if state in env.terminal_states:
            continue

        if current_greedy_policy[state] == V_opt_policy[state]:
            correct_actions += 1
            
    correct_rate = correct_actions / total_states

    # --- Average Score over k episodes ---
    total_reward = 0
    max_eval_steps = 100

    for _ in range(k_timesteps):
        state = env.reset()
        steps_taken = 0

        while steps_taken < max_eval_steps:
            # The agent strictly follows what it thinks is best
            action = current_greedy_policy[state] 
            next_state, reward, is_terminal = env.step(state, action)
            total_reward += reward
            state = next_state

            steps_taken += 1

            if is_terminal:
                break
                
    avg_score = total_reward / k_timesteps

    # --- Start State Q-Value ---
    # Find the max Q-value for the starting square
    start_q_value = max([Q_curr[(start_state, a)] for a in env.allowed_actions[start_state]])

    return avg_score, correct_rate, start_q_value