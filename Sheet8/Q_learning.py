import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sheet4_path = Path(__file__).parent.parent / "Sheet4"
sheet5_path = Path(__file__).parent.parent / "Sheet5"
sheet6_path = Path(__file__).parent.parent / "Sheet6"



# Add this folder to Python's search path
sys.path.append(str(sheet4_path))
sys.path.append(str(sheet5_path))
sys.path.append(str(sheet6_path))



from gridworld import gridworld
from sample_based_algorithms import step_size_scheduler













def Q_learning(env, epsilon, num_episodes, gamma=1.0, schedule_type="constant", max_steps=1000):


    # Initialize variables for Q values and N
    Q = {(state, action): 0.0 for state in env.allowed_actions.keys() for action in env.allowed_actions[state]}

    N = {(state, action): 0.0 for state in env.allowed_actions.keys() for action in env.allowed_actions[state]}

    # Loop through number of episodes
    for i in range(num_episodes):

        # Reset state and episode
        state = env.reset()

        step_count = 0


        while True:

            step_count += 1

            # Explore with epsilon-greedy
            if np.random.rand() < epsilon:
                action = np.random.choice(env.allowed_actions[state])
            # Choose best action from Q
            else:
                best_action = None
                best_value = -float('inf')
                for a in env.allowed_actions[state]:
                    if Q[(state, a)] > best_value:
                        best_value = Q[(state, a)]
                        best_action = a
                action = best_action



            # Take the step
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
            if is_terminal or step_count >= max_steps:
                break

    return Q









def double_Q_learning(env, epsilon, num_episodes, gamma=1.0, schedule_type="constant", max_steps=1000):


    # Initialize variables for Q values and N

    QA = {(state, action): 0.0 for state in env.allowed_actions.keys() for action in env.allowed_actions[state]}
    QB = {(state, action): 0.0 for state in env.allowed_actions.keys() for action in env.allowed_actions[state]}

    N = {(state, action): 0.0 for state in env.allowed_actions.keys() for action in env.allowed_actions[state]}

    # Loop through number of episodes
    for i in range(num_episodes):

        # Reset state and episode
        state = env.reset()

        step_count = 0


        while True:

            step_count += 1

            # Explore with epsilon-greedy
            if np.random.rand() < epsilon:
                action = np.random.choice(env.allowed_actions[state])
            # Choose best action from Q
            else:
                best_action = None
                best_value = float("-inf")
                for action in env.allowed_actions[state]:
                    combined_value = QA[state, action] + QB[state, action]
                    if combined_value > best_value:
                        best_value = combined_value
                        best_action = action
                action = best_action


            # Take the step
            next_state, reward, is_terminal = env.step(state, action)


            state_action = (state, action)


            N[state_action] += 1

            alpha = step_size_scheduler(schedule_type, N[state_action], 1)

            # Calculate Q value for A and B if its terminal
            if is_terminal:
                if np.random.rand() < 0.5:
                    QA[state_action] += alpha * (reward - QA[state_action])
                    break
                else:
                    QB[state_action] += alpha * (reward - QB[state_action])
                    break




            # Coinflip A and B and then calculate the Q value
            if np.random.rand() < 0.5:

                best_next_action = max(env.allowed_actions[next_state], key=lambda a: QA[next_state, a])

                future_value = QB[next_state, best_next_action]

                G = reward + gamma * future_value

                QA[state_action] = QA[state_action] + alpha * (G - QA[state_action])

            else:

                best_next_action = max(env.allowed_actions[next_state], key=lambda a: QB[next_state, a])

                future_value = QA[next_state, best_next_action]

                G = reward + gamma * future_value

                QB[state_action] = QB[state_action] + alpha * (G - QB[state_action])
            

            state = next_state

            # 5. Check if terminal
            if is_terminal or step_count >= max_steps:
                break

    # Calculate average of QA and QB
    final_Q = {k: (QA[k] + QB[k]) / 2 for k in QA.keys()}

    return final_Q






