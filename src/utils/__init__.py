from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from tqdm import trange

if TYPE_CHECKING:
    import gymnasium


class EnvFactory(Protocol):
    """
    Using Protocol to allow for better type hinting as Callable does not
    support specifying default arguments.
    """  # noqa: D205

    def __call__(
        self, sofa_bias: float = ..., lam: float = ...
    ) -> gymnasium.Env:
        """Signature of the environment factory function."""
        ...


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


def greedy_policy_from_q(
    q_table: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Convert a Q-table into a deterministic greedy policy."""
    policy: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.zeros_like(
        q_table
    )
    policy[np.arange(q_table.shape[0]), np.argmax(q_table, axis=1)] = 1.0
    return policy


def evaluate_deterministic_policy(
    env: gymnasium.Env,
    policy: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    n_episodes: int = 500,
    seed: int = 42,
    progress_desc: str = "Policy evaluation",
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float32]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
]:
    """Simulate a deterministic policy in the environment."""
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    lengths: list[int] = []

    for _ in trange(n_episodes, desc=progress_desc):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000_000)))
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = int(np.argmax(policy[int(obs)]))
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += cast("float", reward)
            steps += 1
            done = terminated or truncated

        returns.append(total_reward)
        lengths.append(steps)

    return (
        np.asarray(returns, dtype=np.float32),
        np.asarray(lengths, dtype=np.int32),
    )
