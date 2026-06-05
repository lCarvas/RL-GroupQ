import numpy as np
from tqdm import trange

from utils import greedy_policy_from_q


def policy_evaluation(
    policy: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    transition_matrix: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    reward_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    gamma: float = 1.0,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Evaluate a policy.

    Parameters
    ----------
    policy : np.ndarray[tuple[int, int], np.dtype[np.float64]]
        Policy matrix of shape (n_states, n_actions) where each row is a
        probability distribution over actions.
    transition_matrix : np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
        Transition matrix of shape (n_states, n_actions, n_states).
    reward_matrix : np.ndarray[tuple[int, int], np.dtype[np.float64]]
        Reward matrix of shape (n_states, n_actions).
    terminal_states : list[int]
        List of terminal state indices.
    gamma : float, optional
        Discount factor, by default 1.0


    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.float64]]
        State values under the given policy.
    """
    n_states = policy.shape[0]
    # i discovered the power of einsum and im in love
    policy_reward: np.ndarray[tuple[int], np.dtype[np.float64]] = np.einsum(
        "sa,sa->s", policy, reward_matrix, dtype=np.float64
    )
    policy_transition: np.ndarray[tuple[int, int], np.dtype[np.float64]] = (
        np.einsum("sa,san->sn", policy, transition_matrix, dtype=np.float64)
    )

    # with gamma = 1, the equation matrix A is singular, that is, it is not of
    # full rank and thus doesn't not have an inverse
    equation_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]] = (
        np.eye(n_states) - gamma * policy_transition
    )

    # since A is singular, have to use least squares instead of solve.
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
    return np.linalg.lstsq(equation_matrix, policy_reward, rcond=None)[
        0
    ].astype(np.float64)


def policy_iteration(
    transition_matrix: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    reward_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    gamma: float = 1.0,
    max_policy_iter: int = 1000,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    """Policy iteration algorithm.

    Parameters
    ----------
    transition_matrix : np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
        Transition matrix of shape (n_states, n_actions, n_states).
    reward_matrix : np.ndarray[tuple[int, int], np.dtype[np.float64]]
        Reward matrix of shape (n_states, n_actions).
    terminal_states : list[int]
        List of terminal state indices.
    gamma : float, optional
        Discount factor, by default 1.0
    max_policy_iter : int, optional
        Maximum number of policy iteration iterations, by default 1000

    Returns
    -------
    tuple[
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.ndarray[tuple[int], np.dtype[np.float64]]
    ]
        Tuple of (policy, values) where policy is a matrix of shape
        (n_states, n_actions), and values is a vector of shape (n_states,).
    """
    n_states, n_actions = transition_matrix.shape[:2]

    policy: np.ndarray[tuple[int, int], np.dtype[np.float64]] = (
        np.ones((n_states, n_actions), dtype=np.float64) / n_actions
    )

    values: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(
        n_states, dtype=np.float64
    )

    for _ in trange(max_policy_iter, desc="Policy Iteration"):
        values = policy_evaluation(
            policy,
            transition_matrix,
            reward_matrix,
            gamma=gamma,
        )

        new_policy = greedy_policy_from_q(
            reward_matrix
            + gamma
            * np.einsum(
                "san,n->sa", transition_matrix, values, dtype=np.float64
            )
        )

        n_changed = int(
            np.sum(np.argmax(new_policy, axis=1) != np.argmax(policy, axis=1))
        )

        if n_changed == 0:
            print(f"Converged after {_ + 1} iterations.")
            break

        policy = new_policy
    else:
        print(
            f"Warning: Policy iteration did not converge after {max_policy_iter} iterations."
        )

    return policy, values
