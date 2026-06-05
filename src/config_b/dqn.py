from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from utils import (
    LearningCurveCallback,
    evaluate_sb3_fixed_seeds,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from utils import EnvFactoryContinuous


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def dqn_objective(
    trial: optuna.Trial,
    env_factory: EnvFactoryContinuous,
    gamma: float,
    trial_training_returns: MutableMapping[int, Any],
    timesteps: int = 50000,
    eval_freq: int = 10000,
    n_eval_episodes: int = 5,
    *,
    seed: int = 42,
    verbose: int = 0,
) -> float:
    """Objective function for tuning a DQN with Optuna.

    Parameters
    ----------
    trial : optuna.Trial
        The Optuna trial object
    env_factory : EnvFactoryContinuous
        A factory function to create the environment
    gamma : float
        The discount factor
    trial_training_returns : MutableMapping[int, Any]
        A mapping to store the training returns for each episode during the trial
    timesteps : int, optional
        The total number of timesteps to train, by default 50000
    eval_freq : int, optional
        The frequency of evaluations, in timesteps, by default 10000
    n_eval_episodes : int, optional
        The number of episodes to evaluate, by default 5
    seed : int, optional
        The random seed, by default 42
    verbose : int, optional
        The level of verbosity, by default 0

    Returns
    -------
    float
        The mean reward obtained during evaluation

    Raises
    ------
    optuna.exceptions.TrialPruned
        If the trial is pruned by Optuna
    """
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    buffer_size = trial.suggest_int("buffer_size", 1000, 500000)
    learning_starts = trial.suggest_int("learning_starts", 1000, 10000)
    batch_size = trial.suggest_int("batch_size", 32, 256, step=32)
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16])
    tau = trial.suggest_float("tau", 1e-3, 1.0, log=True)
    target_update_interval = trial.suggest_int(
        "target_update_interval", 1, 1000
    )
    exploration_fraction = trial.suggest_float(
        "exploration_fraction", 0.01, 0.5
    )
    exploration_final_eps = trial.suggest_float(
        "exploration_final_eps", 0.01, 0.1
    )

    train_env = Monitor(env_factory())

    eval_callback = LearningCurveCallback(
        eval_env_factory=env_factory,
        n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq,
    )

    dqn_model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        train_freq=train_freq,
        tau=tau,
        gamma=gamma,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1,
        exploration_final_eps=exploration_final_eps,
        verbose=verbose,
        seed=seed,
        device=device,
    )

    try:
        dqn_model.learn(total_timesteps=timesteps, callback=eval_callback)

        if eval_callback.best_parameters is not None:
            dqn_model.set_parameters(
                eval_callback.best_parameters,  # pyright: ignore[reportArgumentType]
                exact_match=True,
            )

        validation_returns = evaluate_sb3_fixed_seeds(
            dqn_model, env_factory, n_episodes=n_eval_episodes
        )
        trial_training_returns[trial.number] = (
            eval_callback.episode_returns.copy()
        )

        trial.set_user_attr(
            "episode_returns", eval_callback.episode_returns.copy()
        )
        trial.set_user_attr("selected_step", eval_callback.best_step)
        trial.set_user_attr("mean_return", float(validation_returns.mean()))
        trial.set_user_attr("std_return", float(validation_returns.std()))
    finally:
        train_env.close()

    return np.mean(validation_returns, dtype=np.float64)


def tune_dqn(
    env_factory: EnvFactoryContinuous,
    *,
    gamma: float,
    n_trials: int = 50,
    eval_freq: int = 10000,
    n_eval_episodes: int = 5,
    timesteps: int = 50000,
    seed: int = 42,
    verbose: int = 0,
) -> optuna.Study:
    """Tune DQN hyperparameters using Optuna.

    Parameters
    ----------
    env_factory : EnvFactoryContinuous
        A factory function to create the environment
    gamma : float
        The discount factor
    n_trials : int, optional
        The number of trials to run, by default 50
    eval_freq : int, optional
        The frequency of evaluations, in timesteps, by default 10000
    n_eval_episodes : int, optional
        The number of episodes to evaluate, by default 5
    timesteps : int, optional
        The total number of timesteps to train, by default 50000
    seed : int, optional
        The random seed, by default 42
    verbose : int, optional
        The level of verbosity, by default 0

    Returns
    -------
    optuna.Study
        The optimised Optuna study object
    """
    trial_training_returns: dict[int, Any] = {}

    sampler = optuna.samplers.RandomSampler(seed=seed)
    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner
    )

    study.optimize(
        lambda trial: dqn_objective(
            trial,
            env_factory,
            gamma,
            trial_training_returns,
            timesteps=timesteps,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            seed=seed,
            verbose=verbose,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,
    )

    return study
