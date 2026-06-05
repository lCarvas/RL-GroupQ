from typing import TYPE_CHECKING, Protocol

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    import gymnasium
    from stable_baselines3.common.base_class import BaseAlgorithm


class EnvFactoryContinuous(Protocol):
    """
    Using Protocol to allow for better type hinting as Callable does not
    support specifying default arguments.
    """  # noqa: D205

    def __call__(
        self,
        sofa_bias: float = ...,
        lam: float = ...,
        malfunction_prob: float = ...,
        noise_std: float = ...,
        missing_prob: float = ...,
        n_missing: int = ...,
        event_prob: float = ...,
    ) -> gymnasium.Env:
        """Signature of the environment factory function."""
        ...


class EpisodeReturnsCallback(BaseCallback):
    """Callback to save the return of each episode during training."""

    def __init__(self) -> None:
        super().__init__()
        self.episode_returns = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_returns.append(info["episode"]["r"])
        return True


class LearningCurveCallback(BaseCallback):
    """Save training returns, evaluate checkpoints and keep the best policy.

    Parameters
    ----------
    eval_env : gymnasium.Env
        The environment to use for evaluation
    eval_freq : int, optional
        The frequency of evaluations, in timesteps, by default 2000
    n_eval_episodes : int, optional
        The number of episodes to evaluate, by default 5
    validation_first_seed : int, optional
        The first seed to use for evaluation, by default 20000
    """

    def __init__(
        self,
        eval_env_factory: EnvFactoryContinuous,
        eval_freq: int = 2000,
        n_eval_episodes: int = 20,
        validation_first_seed: int = 20000,
    ) -> None:
        # Save the settings used for checkpoint evaluation.
        super().__init__()
        self.eval_env_factory = eval_env_factory
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.validation_first_seed = validation_first_seed

        # Store information that will be plotted after training.
        self.episode_returns = []
        self.eval_steps = []
        self.eval_means = []
        self.eval_stds = []

        # Start without a selected checkpoint.
        self.best_mean_return = -np.inf
        self.best_step = None
        self.best_parameters = None

    def _on_step(self) -> bool:
        # Save the return whenever a training episode is complete.
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_returns.append(info["episode"]["r"])

        # Evaluate only at the chosen checkpoint frequency.
        if self.num_timesteps % self.eval_freq == 0:
            returns = evaluate_sb3_fixed_seeds(
                self.model,
                self.eval_env_factory,
                n_episodes=self.n_eval_episodes,
                first_seed=self.validation_first_seed,
            )
            mean_return = float(returns.mean())

            # Save checkpoint results for the evaluation plot.
            self.eval_steps.append(self.num_timesteps)
            self.eval_means.append(mean_return)
            self.eval_stds.append(float(returns.std()))

            # Save a copy of the model when it is the best checkpoint so far.
            if mean_return > self.best_mean_return:
                self.best_mean_return = mean_return
                self.best_step = self.num_timesteps
                self.best_parameters = self.model.get_parameters()

        return True


def evaluate_sb3_fixed_seeds(
    model: BaseAlgorithm,
    env_factory: EnvFactoryContinuous,
    n_episodes: int = 20,
    first_seed: int = 10000,
) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
    """Evaluate one Stable Baselines policy using fixed episode seeds.

    Parameters
    ----------
    model : DQN | PPO
        The Stable Baselines model to evaluate.
    env_factory : EnvFactory
        A factory function that creates a new environment instance when called.
    n_episodes : int, optional
        The number of episodes to evaluate the policy on, by default 20
    first_seed : int, optional
        The first seed to use for evaluation, by default 10000

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.float32]],
        The evaluation returns.
    """
    returns = []

    env = env_factory()
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=first_seed + episode)
        total_return = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(
                int(np.asarray(action).item())
            )
            total_return += float(reward)
            done = terminated or truncated

        env.close()
        returns.append(total_return)

    return np.asarray(returns, dtype=np.float32)
