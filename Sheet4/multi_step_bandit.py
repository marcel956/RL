import numpy as np 


# Start state is (0,0) and looks a bit off, maybe change
class MultiStepBandit:
    def __init__(self, branch_config, reward_structure, default_reward, start_state=(0,0)):

        self.start_state = start_state
        self.branch_config = branch_config
        self.reward_structure = reward_structure
        self.default_reward = default_reward


        # Build allowed actions and terminal states
        allowed_actions = {}
        transition_probabilities = {}
        terminal_states = ["terminal"]

        # Allowed starting action
        
        allowed_actions[(0, 0)] = list(range(len(self.branch_config)))

        # Build transition probabilities for start step

        for branch in range(len(self.branch_config)):

            transition_probabilities[((0,0), branch)] = {(branch, 1): 1.0}


        for branch in branch_config:

            # m_i = num_steps
            # List of all terminal states
            m_i = self.branch_config[branch]["m_i"]

            #terminal_states.append((branch_id, step_id))

            # Allowed actions for every state
            for step in range(1, m_i + 1):

                state = (branch, step)

                num_actions = self.branch_config[branch]["actions_per_step"][step]

                allowed_actions[state] = list(range(num_actions))

                # Build deterministic transition probabilities

                for action in range(num_actions):
                    if step + 1 <= m_i:   
                        next_state = (branch, step + 1)

                    else:
                        next_state = "terminal"
                        
                    transition_probabilities[(state, action)] = {next_state: 1.0}       
                

        self.allowed_actions = allowed_actions
        self.terminal_states = terminal_states

        # Build expected rewards
        expected_rewards = {}

        for state in allowed_actions:

            for action in allowed_actions[state]:

                # Use get to avoid crashes
                reward_info = reward_structure.get((state, action))

                # Check if reward_info exists
                if reward_info:
                    if reward_info["reward_type"] == "deterministic":

                        expected_rewards[(state, action)] = reward_info["value"]
                    
                    elif reward_info["reward_type"] == "normal":

                        expected_rewards[(state, action)] = reward_info["mean"]

                    elif reward_info["reward_type"] == "binomial":

                        expected_rewards[(state, action)] = reward_info["n"] * reward_info["p"]

                else:
                    expected_rewards[(state, action)] = default_reward

        self.expected_rewards = expected_rewards






    def step(self, state, action):

        # Set parameters
        is_terminal = False

        branch_id = state[0]
        step_id = state[1]


        # Handle Start State
        if step_id == 0:
            if action < 0 or action >= len(self.branch_config):
                raise ValueError
            
            next_state = (action, 1)

        # Normal Next State calculation
        else:
            num_action = self.branch_config[branch_id]["actions_per_step"][step_id]

            # Check if state is legal
            if action < 0 or action >= num_action:
                raise ValueError

            # Calculate next State
            else:  
                if step_id + 1 <= self.branch_config[branch_id]["m_i"]:
                    next_state = (branch_id, step_id + 1)

                else:
                    next_state = "terminal"
                    is_terminal = True

        # Calculate reward
        # Use get to avoid crashes
        reward_info = self.reward_structure.get((state, action))

        # Check if reward_info exists
        if reward_info:
            if reward_info["reward_type"] == "deterministic":
                reward = reward_info["value"]

            elif reward_info["reward_type"] == "normal":
                reward = np.random.normal(reward_info["mean"], reward_info["std"])

            elif reward_info["reward_type"] == "binomial":
                reward = np.random.binomial(reward_info["n"], reward_info["p"])

        else:
            reward = self.default_reward
        
    
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
                    action = int(np.random.choice(policy_action))
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







# ==========================================
# TEST SCRIPT
# ==========================================

# 1. Define the Tree Structure
# Branch 0 has 2 steps (3 arms, then 2 arms)
# Branch 1 has 1 step (5 arms)
config = {
    0: {"m_i": 2, "actions_per_step": {1: 3, 2: 2}}, 
    1: {"m_i": 1, "actions_per_step": {1: 5}} 
}

# 2. Define Custom Rewards
# Let's make Branch 1, Step 1, Action 4 the "Jackpot"
rewards = {
    ((1, 1), 4): {"reward_type": "deterministic", "value": 100}
}

# 3. Initialize the Environment
bandit_env = MultiStepBandit(
    branch_config=config, 
    reward_structure=rewards, 
    default_reward=-1 # Every step costs -1 unless specified
)

# 4. Create a dummy policy
# Let's tell the agent to always pick action 0 (which avoids the jackpot!)
dummy_policy = {}
for state, actions in bandit_env.allowed_actions.items():
    dummy_policy[state] = 0 # Just pick the first available arm

# 5. Run Monte Carlo!
print("Running Monte Carlo evaluation...")
q_values = bandit_env.monte_carlo(dummy_policy, num_episodes=500)

print("\nEstimated Q-Values for our Dummy Policy:")
for state_action, value in sorted(q_values.items()):
    print(f"State: {state_action[0]} | Action: {state_action[1]} -> Expected Return: {value:.2f}")


# ==========================================
# TEST SCRIPT (RANDOM POLICY)
# ==========================================

# 1. Define the Tree Structure
config = {
    0: {"m_i": 2, "actions_per_step": {1: 3, 2: 2}}, 
    1: {"m_i": 1, "actions_per_step": {1: 5}} 
}

# 2. Define Custom Rewards (Action 4 on Branch 1 is the Jackpot)
rewards = {
    ((1, 1), 4): {"reward_type": "deterministic", "value": 100}
}

# 3. Initialize the Environment
bandit_env = MultiStepBandit(
    branch_config=config, 
    reward_structure=rewards, 
    default_reward=-1 
)

# 4. Use allowed_actions as our Random Policy!
random_policy = bandit_env.allowed_actions

# 5. Run Monte Carlo!
print("Running Monte Carlo evaluation with a Random Policy...")
q_values = bandit_env.monte_carlo(random_policy, num_episodes=2000)

print("\nEstimated Q-Values for our Random Policy:")
for state_action, value in sorted(q_values.items()):
    print(f"State: {state_action[0]} | Action: {state_action[1]} -> Expected Return: {value:.2f}")