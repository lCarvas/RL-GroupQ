from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from utils import (
    LearningCurveCallback,
    evaluate_sb3_fixed_seeds,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from utils import EnvFactoryContinuous

device = torch.device("cpu")


def ppo_objective(
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
    """Objective function for tuning PPO with Optuna.

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
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64])
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    gae_lambda = trial.suggest_float("gae_lambda", 0.85, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 4.0, log=True)

    train_env = Monitor(env_factory())

    eval_callback = LearningCurveCallback(
        eval_env_factory=env_factory,
        n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq,
    )

    ppo_model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=verbose,
        seed=seed,
        device=device,
    )

    try:
        ppo_model.learn(total_timesteps=timesteps, callback=eval_callback)

        if eval_callback.best_parameters is not None:
            ppo_model.set_parameters(
                eval_callback.best_parameters,  # pyright: ignore[reportArgumentType]
                exact_match=True,
            )

        validation_returns = evaluate_sb3_fixed_seeds(
            ppo_model, env_factory, n_episodes=n_eval_episodes
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


def tune_ppo(
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
    """Tune PPO hyperparameters using Optuna.

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
        lambda trial: ppo_objective(
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
