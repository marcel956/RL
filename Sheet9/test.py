
import gymnasium as gym

from stable_baselines3 import PPO, DQN



test = "pendulum"


if test == "car":
    env = gym.make("MountainCar-v0")

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=4e-3,           # Optimized step size
        batch_size=128,               # Process bigger chunks of memory
        buffer_size=10000,            # Total replay memory capacity
        learning_starts=1000,         # Collect 1,000 steps of pure driving data before training begins
        gamma=0.98,                   # Slightly lower than default 0.99 (makes it care more about immediate velocity changes)
        target_update_interval=600,   # How often to sync target network weights
        train_freq=16,                # Train only once every 16 steps (stops it from over-correcting)
        gradient_steps=8,             # Do 8 optimization steps per training phase
        exploration_fraction=0.2,     # Fades random actions out faster (20% of training), letting it use momentum
        exploration_final_eps=0.07,   # Keeps a 7% chance of random actions to fine-tune
        policy_kwargs=dict(net_arch=[256, 256]), # Uses a wider neural network to calculate physics states
        verbose=1                     # Prints progress tables
    )

    print("Training the agent...")
    # Let it run for 120,000 timesteps to hit full convergence
    model.learn(total_timesteps=120000)

    # Open a new environment with rendering turned on
    eval_env = gym.make("MountainCar-v0", render_mode="human")
    obs, info = eval_env.reset()

    # Let the AI play for up to 1000 frames
    for i in range(1000):
        # Predict the best action. deterministic=True means it takes the BEST action it knows, no random guessing.
        action, _states = model.predict(obs, deterministic=True)
        
        # Pass action to the environment
        obs, reward, terminated, truncated, info = eval_env.step(action)
        
        # If it reaches the flag (terminated) or runs out of time (truncated), reset
        if terminated or truncated:
            print("Episode finished! Resetting...")
            obs, info = eval_env.reset()

    eval_env.close()

elif test == "pendulum":

    print("Setting up the Pendulum environment...")
    env = gym.make("Pendulum-v1")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,           # Good step size for continuous adjustments
        n_steps=1024,                 # Reduce batch collection window (default is 2048)
        batch_size=64,                # Optimize in small, precise batches of memory
        n_epochs=10,                  # Run through the data 10 times per update to learn deeply
        gae_lambda=0.95,              # Smoothes out the variance in physics calculations
        gamma=0.9,                    # Lower discount factor: makes it care heavily about immediate gravity impacts
        policy_kwargs=dict(net_arch=[64, 64]), # Leaner neural network works better for basic physics
        verbose=1
    )

    print("Training the agent...")
    # Let's bump practice time up slightly to 60,000 steps to guarantee stability
    model.learn(total_timesteps=60000)
    print("Training complete! Let's watch it balance.")

    # ---------------------------------------------------------
    # PHASE 2: Watching the AI play
    # ---------------------------------------------------------
    eval_env = gym.make("Pendulum-v1", render_mode="human")
    obs, info = eval_env.reset()

    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        
        if terminated or truncated:
            obs, info = eval_env.reset()

    eval_env.close()







