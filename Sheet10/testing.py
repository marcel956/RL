import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from REINFORCE import REINFORCE
from mini_batch_Reinforce import mini_batch_REINFORCE


def run_comparison():
    # 1. Create the environments
    # CartPole-v1 has a maximum reward/step limit of 500.
    env_name = "CartPole-v1"
    train_env = gym.make(env_name)
    eval_env = gym.make(env_name)

    total_timesteps = 50000
    print(f"Starting training for {total_timesteps} timesteps on {env_name}...\n")




    # =========================================================================
    # 2. Train Custom Standard REINFORCE
    # =========================================================================
    print("--- Training Standard REINFORCE ---")
    # CRITICAL: n_steps must be equal to or greater than the max episode length (500)
    # so that the rollout buffer captures complete trajectories without mid-way updates.
    reinforce_model = REINFORCE(
        "MlpPolicy",
        train_env,
        n_steps=500,  # Full episode length for CartPole
        gae_lambda=1.0,  # Pure Monte Carlo returns
        learning_rate=1e-3,
        verbose=0,
    )
    reinforce_model.learn(total_timesteps=total_timesteps)

    # Evaluate REINFORCE
    mean_reward_rf, std_reward_rf = evaluate_policy(
        reinforce_model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"Standard REINFORCE Mean Reward: {mean_reward_rf} +/- {std_reward_rf}\n")

    # =========================================================================
    # 3. Train Standard A2C
    # =========================================================================
    print("--- Training A2C ---")
    # Using n_steps=500 here ensures A2C gets a stable batch size equivalent to REINFORCE
    a2c_model = A2C(
        "MlpPolicy",
        train_env,
        n_steps=500,  
        learning_rate=1e-3,
        verbose=0,
    )
    a2c_model.learn(total_timesteps=total_timesteps)

    # Evaluate A2C
    mean_reward_a2c, std_reward_a2c = evaluate_policy(
        a2c_model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"A2C Mean Reward: {mean_reward_a2c} +/- {std_reward_a2c}\n")


    # =========================================================================
    # 5. Final Comparison Summary
    # =========================================================================
    print("==========================================")
    print("FINAL RESULTS (Max possible score is 500):")
    print("==========================================")
    print(f"Standard REINFORCE:             {mean_reward_rf:.2f}")
    print(f"Standard A2C:                   {mean_reward_a2c:.2f}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    run_comparison()