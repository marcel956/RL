import numpy as np

class gridworld:
    def __init__(self, m, n, reward_structure, default_reward, wall_behavior, start_state, wind_direction, wind_prob=0, slip_prob=0, noise_prob=0, noise_directions={}):

        self.m = m
        self.n = n
        self.reward_structure = reward_structure
        self.default_reward = default_reward
        self.start_state = start_state
        self.wall_behavior = wall_behavior
        self.wind_prob = wind_prob
        self.wind_direction = wind_direction
        self.slip_prob = slip_prob
        self.noise_prob = noise_prob
        self.noise_directions = noise_directions

        self.normal_prob = 1.0 - self.wind_prob - self.slip_prob - self.noise_prob


        # Build expected rewards matrix and terminal states list
        expected_rewards = np.ones((m, n)) *  default_reward
        terminal_states = []

        for i in reward_structure:

            if reward_structure[i]["reward_type"] == "deterministic":

                expected_rewards[i[0], i[1]] = reward_structure[i]["value"]
            
            elif reward_structure[i]["reward_type"] == "normal":

                expected_rewards[i[0], i[1]] = reward_structure[i]["mean"]

            elif reward_structure[i]["reward_type"] == "binomial":

                expected_rewards[i[0], i[1]] = reward_structure[i]["n"] * reward_structure[i]["p"]

            if reward_structure[i]["is_terminal"]:
                terminal_states.append((i[0], i[1]))

        self.expected_rewards = expected_rewards
        self.terminal_states = terminal_states

        # Build allowed actions
        allowed_actions = {}

        for row in range(m):
            for col in range(n):
                
                possible_actions = ["up", "down", "left", "right"]

                if row == 0:
                    possible_actions.remove("up")
                elif row == m - 1:
                    possible_actions.remove("down")
                if col == 0:
                    possible_actions.remove("left")
                elif col == n - 1:
                    possible_actions.remove("right")

                allowed_actions[(row, col)] = possible_actions

        self.allowed_actions = allowed_actions


        # Build transition probabilities
        transition_probabilities = {}

        # Calculate adjacent directions for slip
        adjacent_directions = {
            "up": ["left", "right"],
            "down": ["left", "right"],
            "left": ["up", "down"],
            "right": ["up", "down"]
        }

        # Loop through every state and direction
        for row in range(m):
            for col in range(n):

                state = (row, col)

                #skip direction calc if state terminal
                if state in terminal_states:
                    continue

                for direction in ["up", "down", "left", "right"]:

                    # Calculate probability for every direction, considering events
                    dir_probs = {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0}

                    dir_probs[direction] += self.normal_prob

                    dir_probs[self.wind_direction] += self.wind_prob

                    for slip_dire in adjacent_directions[direction]:
                        dir_probs[slip_dire] += self.slip_prob / 2 

                    for noise_dire in ["up", "down", "left", "right"]:
                        dir_probs[noise_dire] += self.noise_prob * self.noise_directions[noise_dire]

                    # Calculate next state (outcome) of those actions
                    outcomes = {}

                    for dir, prob in dir_probs.items():

                        # Skip impossible directions
                        if prob == 0:
                            continue 

                        # calculate next state
                        if dir == "up":
                            next_state = (state[0] - 1, state[1])
                        elif dir == "down":
                            next_state = (state[0] + 1, state[1])
                        elif dir == "left": 
                            next_state = (state[0], state[1] - 1)       
                        elif dir == "right":
                            next_state = (state[0], state[1] + 1)

                        # Apply wall behavior:
                        if dir not in self.allowed_actions[state]:
                            next_state = state

                        # Add probabilities together if they lead to the same outcome
                        if next_state in outcomes:
                            outcomes[next_state] += prob 
                        else:
                            outcomes[next_state] = prob

                    transition_probabilities[(state, direction)] = outcomes

        self.transition_probabilities = transition_probabilities





    def step(self, state, action): # Game dynamics function


        final_action = action

        #
        events = ["normal", "wind", "slip", "noise"]
        event_probs = [self.normal_prob, self.wind_prob, self.slip_prob, self.noise_prob]

        triggered_event = np.random.choice(events, p=event_probs)


        # Apply triggered Event:
        # Apply wind:
        if triggered_event == "wind":
            # Move in the wind direction after already moving
            final_action = self.wind_direction

        #Apply slip:
        elif triggered_event == "slip":
            # Build slip action list and remove chosen action
            slip_actions = ["up", "down", "left", "right"]
            slip_actions.remove(action) 

            # Remove action opposite to the chosen action
            if action == "up":
                slip_actions.remove("down")
            elif action == "down":
                slip_actions.remove("up")
            elif action == "left":
                slip_actions.remove("right")
            elif action == "right":
                slip_actions.remove("left")

            # Choose random slip
            slip_direction = np.random.choice(slip_actions)
            final_action = slip_direction

        # Apply noise:
        elif triggered_event == "noise":

            # Extract choices and probs from the dictionary
            choices = list(self.noise_directions.keys())
            probs = list(self.noise_directions.values())

            noise_direction = np.random.choice(choices, p=probs)

            final_action = noise_direction


        # Calculate next state
        if final_action == "up":
            next_state = (state[0] - 1, state[1])
        elif final_action == "down":
            next_state = (state[0] + 1, state[1])
        elif final_action == "left":
            next_state = (state[0], state[1] - 1)
        elif final_action == "right":
            next_state = (state[0], state[1] + 1)

    
        # Apply wall behavior:
        if final_action not in self.allowed_actions[state]:
            if self.wall_behavior == "reflect":
                next_state = state
            elif self.wall_behavior == "prohibited":    
                raise ValueError("Action prohibited by wall behavior")




        # Calculate reward
        if next_state in self.reward_structure:
            if self.reward_structure[next_state]["reward_type"] == "deterministic":
                reward = self.reward_structure[next_state]["value"]
            elif self.reward_structure[next_state]["reward_type"] == "normal":
                reward = np.random.normal(self.reward_structure[next_state]["mean"], self.reward_structure[next_state]["std"])
            elif self.reward_structure[next_state]["reward_type"] == "binomial":
                reward = np.random.binomial(self.reward_structure[next_state]["n"], self.reward_structure[next_state]["p"])
        else: 
            reward = self.default_reward


        # Check if next state is terminal
        if next_state in self.terminal_states:
            is_terminal = True
        else:
            is_terminal = False


        return next_state, reward, is_terminal


    def reset(self):
        return self.start_state
                

    def monte_carlo(self, policy, num_episodes, gamma=1.0):

        # Initialize variables for Q values and returns
        Q = {}

        returns = {}

        # Loop through number of episodes
        for i in range(num_episodes):

            # Reset state and episode
            state = self.reset()

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
                next_state, reward, is_terminal = self.step(state, action)

                episode.append((state, action, reward))

                state = next_state

                # 5. Check if terminal
                if is_terminal:
                    break

            G = 0

            # Loop through episode backwards
            for step_state, step_action, step_reward in reversed(episode):
                
                # Calculate return
                G = gamma * G + step_reward

                state_action = (step_state, step_action)

                if state_action not in returns:
                    returns[state_action] = []

                returns[state_action].append(G)

                # Calculate Q value
                Q[state_action] = np.mean(returns[state_action])

        return Q




    def visualize_policy(self, policy):
        # Map string actions to unicode arrows
        action_symbols = {
            "up": "↑", 
            "down": "↓", 
            "left": "←", 
            "right": "→"
        }
        
        print("-" * (self.n * 5)) # Print a top border
        
        for row in range(self.m):
            row_string = "|"
            
            for col in range(self.n):
                state = (row, col)
                
                # Check if it's a terminal state first
                if state in self.terminal_states:
                    # If it's a positive reward, print G for Goal. Otherwise B for Bomb.
                    if self.reward_structure[state].get("value", 0) > 0:
                        row_string += " G  |"
                    else:
                        row_string += " B  |"
                
                # Otherwise, print the arrow for the policy's action
                else:
                    # .get() prevents crashes if a state is somehow missing from the policy
                    action = policy.get(state, "up") 
                    symbol = action_symbols[action]
                    row_string += f" {symbol}  |"
                    
            print(row_string)
            print("-" * (self.n * 5)) # Print a row border




    

#Tests:

# 1. Define your rules
rewards = {
    (0, 3): {"type": "goal", "reward_type": "deterministic", "value": 10, "is_terminal": True},
    (1, 1): {"type": "bomb", "reward_type": "deterministic", "value": -10, "is_terminal": True},
    (2, 2): {"type": "bonus", "reward_type": "normal", "mean": 5, "std": 1.0, "is_terminal": False}
}

noise_dirs = {"up": 0.25, "down": 0.25, "left": 0.25, "right": 0.25}

# 2. Create the environment
env = gridworld(
    m=3, n=4, 
    reward_structure=rewards, 
    default_reward=-1, 
    wall_behavior="reflect", 
    start_state=(2, 0), 
    wind_direction="right", 
    wind_prob=0.1, 
    slip_prob=0.1, 
    noise_prob=0.05, 
    noise_directions=noise_dirs
)

# 3. Make a dummy policy that just tells the agent to always go right or up
dummy_policy = {}
for r in range(env.m):
    for c in range(env.n):
        dummy_policy[(r, c)] = "right" if r > 0 else "up"

# 4. Test your visualizer
print("--- My Policy ---")
env.visualize_policy(dummy_policy)

# 5. Test your Monte Carlo
print("\n--- Running Monte Carlo ---")
q_values = env.monte_carlo(dummy_policy, num_episodes=1000)
print(f"Estimated Q-Value for starting at (2,0) and going right: {q_values.get(((2,0), 'right'), 0):.2f}")





# ==========================================
# GRIDWORLD RANDOM WALK TEST SCRIPT
# ==========================================

# 1. Define your rules
rewards = {
    (0, 3): {"type": "goal", "reward_type": "deterministic", "value": 10, "is_terminal": True},
    (1, 1): {"type": "bomb", "reward_type": "deterministic", "value": -10, "is_terminal": True},
    (2, 2): {"type": "bonus", "reward_type": "normal", "mean": 5, "std": 1.0, "is_terminal": False}
}

noise_dirs = {"up": 0.25, "down": 0.25, "left": 0.25, "right": 0.25}

# 2. Create the environment
env = gridworld(
    m=3, n=4, 
    reward_structure=rewards, 
    default_reward=-1, 
    wall_behavior="reflect", 
    start_state=(2, 0), 
    wind_direction="right", 
    wind_prob=0.1, 
    slip_prob=0.1, 
    noise_prob=0.05, 
    noise_directions=noise_dirs
)

# 3. Use the allowed_actions dictionary as a Random Policy
random_grid_policy = env.allowed_actions

# 4. VERY IMPORTANT BUG FIX FOR MONTE CARLO
# (You must update the type-checking inside your monte_carlo function so it doesn't break on strings!)
# Find this inside your monte_carlo function and replace it:
"""
                # 2. If the policy gave is a list, pick randomly
                if type(policy_action) == list:
                    # REMOVED int() because actions are strings here!
                    action = np.random.choice(policy_action) 
                # 3. If the policy gave is a single string/integer, use it
                else:
                    action = policy_action
"""

# 5. Run Monte Carlo!
# Using 2000 episodes and gamma=0.9 to prevent infinite random wandering loops
print("\n--- Running Monte Carlo Random Walk ---")
# 
grid_q_values = env.monte_carlo(random_grid_policy, num_episodes=2000, gamma=0.9)

print("\nEstimated Q-Values for our Random Walk (First 15 printed):")
count = 0
for state_action, value in sorted(grid_q_values.items()):
    print(f"State: {state_action[0]} | Action: {state_action[1]:<5} -> Expected Return: {value:.2f}")
    count += 1
    if count >= 15:
        break