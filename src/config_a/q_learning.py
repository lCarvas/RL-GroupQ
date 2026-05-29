from typing import TYPE_CHECKING, cast

import numpy as np
import optuna
from tqdm import trange

from config_a.utils import (
    EnvFactory,
    evaluate_deterministic_policy,
    greedy_policy_from_q,
)

if TYPE_CHECKING:
    import gymnasium


def train_q_learning(
    env: gymnasium.Env,
    n_states: int,
    n_actions: int,
    gamma: float,
    episodes: int = 10_000,
    alpha: float = 0.1,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.9995,
    seed: int = 42,
    progress_desc: str = "Q-learning training",
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    """Tabular Q-learning.

    Parameters
    ----------
    env : gymnasium.Env
        The environment to train on
    n_states : int
        Number of states in the environment
    n_actions : int
        Number of actions in the environment
    gamma : float
        The discount factor
    episodes : int, optional
        Number of episodes to train for, by default 10_000
    alpha : float, optional
        Learning rate, by default 0.1
    epsilon_start : float, optional
        Initial exploration rate, by default 1.0
    epsilon_end : float, optional
        Minimum exploration rate, by default 0.05
    epsilon_decay : float, optional
        Exploration decay rate, by default 0.9995
    seed : int, optional
        Random seed, by default 42
    progress_desc : str, optional
        Description for tqdm, by default "Q-learning training"

    Returns
    -------
    tuple[
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.ndarray[tuple[int], np.dtype[np.float64]],
    ]
        The trained Q-table and episode returns
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    q_table: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.zeros(
        (n_states, n_actions), dtype=np.float64
    )
    returns: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(
        episodes, dtype=np.float64
    )
    epsilon: float = epsilon_start

    for episode in trange(episodes, desc=progress_desc):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000_000)))
        done = False
        total_reward = 0.0

        while not done:
            state = int(obs)
            if rng.random() < epsilon:
                action = int(rng.integers(n_actions))
            else:
                action = int(np.argmax(q_table[state]))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done: bool = terminated or truncated
            next_state = int(next_obs)

            target = (
                reward if done else reward + gamma * np.max(q_table[next_state])
            )
            q_table[state, action] += alpha * (target - q_table[state, action])

            total_reward += cast("float", reward)
            obs = next_obs

        returns[episode] = total_reward
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return q_table, returns


def objective_q_learning(
    trial: optuna.Trial,
    env_factory: EnvFactory,
    n_states: int,
    n_actions: int,
    gamma: float,
    train_episodes: int = 20_000,
    validation_episodes: int = 10_000,
    seed: int = 42,
) -> np.float64:
    """Objective function for tuning Q-learning with Optuna.

    Parameters
    ----------
    trial : optuna.Trial
        The Optuna trial object
    env_factory : EnvFactory
        A factory function to create the environment
    n_states : int
        The number of states in the environment
    n_actions : int
        The number of actions in the environment
    gamma : float
        The discount factor
    train_episodes : int, optional
        The number of episodes to train for, by default 20_000
    validation_episodes : int, optional
        The number of episodes to validate for, by default 10_000
    seed : int, optional
        The random seed, by default 42

    Returns
    -------
    float
        The mean validation return
    """
    alpha = trial.suggest_float("alpha", 0.005, 0.5, log=True)
    epsilon_end = trial.suggest_float("epsilon_end", 0.01, 0.2, log=True)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.7, 0.99995)

    train_env = env_factory()
    try:
        q_table, _ = train_q_learning(
            train_env,
            n_states=n_states,
            n_actions=n_actions,
            gamma=gamma,
            episodes=train_episodes,
            alpha=alpha,
            epsilon_start=1.0,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            seed=seed,
            progress_desc=f"Q-learning trial {trial.number + 1}",
        )
    finally:
        train_env.close()

    greedy_policy = greedy_policy_from_q(q_table)
    validation_env = env_factory()
    try:
        validation_returns, _ = evaluate_deterministic_policy(
            validation_env,
            greedy_policy,
            n_episodes=validation_episodes,
            seed=seed,
        )
    finally:
        validation_env.close()

    return np.mean(validation_returns, dtype=np.float64)


def tune_q_learning(
    env_factory: EnvFactory,
    n_states: int,
    n_actions: int,
    gamma: float,
    n_trials: int = 10,
    train_episodes: int = 20_000,
    validation_episodes: int = 10_000,
    seed: int = 42,
) -> optuna.Study:
    """Tune Q-learning hyperparameters using Optuna.

    Parameters
    ----------
    env_factory : EnvFactory
        A factory function to create the environment
    n_states : int
        The number of states in the environment
    n_actions : int
        The number of actions in the environment
    gamma : float
        The discount factor.
    n_trials : int, optional
        The number of trials to run, by default 8
    train_episodes : int, optional
        The number of episodes to train for, by default 2_000
    validation_episodes : int, optional
        The number of episodes to validate for, by default 200
    seed : int, optional
        The random seed, by default 42

    Returns
    -------
    optuna.Study
        The optimised study object
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(
        lambda trial: objective_q_learning(
            trial,
            env_factory,
            n_states=n_states,
            n_actions=n_actions,
            gamma=gamma,
            train_episodes=train_episodes,
            validation_episodes=validation_episodes,
            seed=seed,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,
    )

    return study
