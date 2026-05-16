from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance

SelfREINFORCE = TypeVar("SelfREINFORCE", bound="REINFORCE")


class mini_batch_REINFORCE(OnPolicyAlgorithm):
    """
    Advantage Actor Critic (A2C)

    Paper: https://arxiv.org/abs/1602.01783
    Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (https://github.com/hill-a/stable-baselines)

    Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
    :param use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`a2c_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 7e-4,
        batch_size_K: int = 32,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        normalize_advantage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=1,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.normalize_advantage = normalize_advantage

        self.batch_size_K = batch_size_K

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        if _init_setup_model:
            self._setup_model()




    def _sample_geometric(self, p: float) -> int:

        return np.random.geometric(p)
    

    def collect_custom_batch(self):

        batch_data =[]

        # Grab the raw environment from the SB3 wrapper
        raw_env = self.env.envs[0].unwrapped

        for i in range(self.batch_size_K):
            # Sample the first horizon T_i
            p1 = 1.0 - self.gamma
            T_i = self._sample_geometric(p1)
            
            # Run the environment up to step T_i to find state_Ti and action_Ti

            # Initialize variables to hold our targets
            state_Ti = None
            action_Ti = None
            
            obs, info = raw_env.reset()

            for t in range(T_i):
                # Get action from poplicy
                action, _states = self.policy.predict(obs, deterministic=True)

                # Tracker for early termination:
                terminated_early = False

                # If this is our target step, freeze and save them!
                if t == T_i - 1:
                    state_Ti = obs
                    action_Ti = action
                
                # Step in the environment
                obs, reward, terminated, truncated, info = raw_env.step(action)

                # If the episode ends before we hit T_i, break early
                if terminated or truncated:
                    terminated_early = True
                    # Fallback: if we haven't saved state_Ti yet, grab the last valid one
                    if state_Ti is None:
                        state_Ti = obs
                        action_Ti = action
                    break


            rewards_pred = []

            if not terminated_early:
                # Sample the second horizon T_i
                p2 = 1.0 - np.sqrt(self.gamma)
                T2_i = self._sample_geometric(p2)

                raw_env.state = state_Ti

   

                for t in range(T2_i):

                    # Get action from poplicy or previous rollout
                    if t == 0:
                        action = action_Ti
                    else:
                        action, _states = self.policy.predict(obs, deterministic=True)

                    # Step in the environment
                    obs, reward, terminated, truncated, info = raw_env.step(action)

                    # Save the reward
                    rewards_pred.append(reward)

                    # If the episode ends, break early
                    if terminated or truncated:
                        break

            # Calculate the exponentially discounted return weight
            discounted_return = 0.0
            gamma_half = np.sqrt(self.gamma)
            
            for t_prime, r in enumerate(rewards_pred):
                discounted_return += (gamma_half ** t_prime) * r

            # Package up the data point for this iteration
            batch_data.append({
                "state": state_Ti,
                "action": action_Ti,
                "return_weight": discounted_return
            })

        return batch_data









    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch policy to training mode and update learning rate
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        # Collect your custom infinite-horizon geometric batch
        batch = self.collect_custom_batch()

        # Convert everything into PyTorch tensors
        states = th.tensor(np.array([item["state"] for item in batch]), dtype=th.float32, device=self.device)
        actions = th.tensor(np.array([item["action"] for item in batch]), device=self.device)
        return_weights = th.tensor(np.array([item["return_weight"] for item in batch]), dtype=th.float32, device=self.device)

        # If action space is discrete, flatten actions to long integers for PyTorch compatibility
        if isinstance(self.action_space, spaces.Discrete):
            actions = actions.long().flatten()

        # Evaluate the actions to get their log probabilities
        _, log_prob, _ = self.policy.evaluate_actions(states, actions)

        # Calculate the exact Algorithm 33 loss
        loss = 1 / (1-self.gamma)  * (return_weights * log_prob).mean()


        # Optimization step
        self.policy.optimizer.zero_grad()
        loss.backward()

        # Clip grad norm
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

def learn(
        self: "mini_batch_REINFORCE",
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "REINFORCE_Infinite",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "mini_batch_REINFORCE":
        # FIX: Call SB3's internal setup to initialize self._logger and setup callbacks
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        self.start_time = np.atleast_1d(0.0)
        timesteps_counter = 0
        
        while timesteps_counter < total_timesteps:
            self.train()
            timesteps_counter += self.batch_size_K
            self.num_timesteps = timesteps_counter
            
        return self
