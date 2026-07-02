from typing import Dict, List, Tuple

import numpy as np

State = int
Action = str


def value_iteration(
    states: List[State],
    actions: List[Action],
    transitions: Dict[Tuple[State, Action], List[Tuple[State, float]]],
    reward_func,
    gamma: float,
    tol: float = 1e-6,
    max_iters: int = 1000,
):
    V: Dict[State, float] = {s: 0.0 for s in states}

    for _ in range(max_iters):
        V_new: Dict[State, float] = {}

        for s in states:
            q_best = -np.inf

            for a in actions:
                exp_next = sum(
                    prob * V[s_next]
                    for s_next, prob in transitions[(s, a)]
                )
                q_sa = reward_func(s, a) + gamma * exp_next

                if q_sa > q_best:
                    q_best = q_sa

            V_new[s] = q_best

        delta = max(abs(V_new[s] - V[s]) for s in states)
        V = V_new

        if delta < tol:
            break

    policy: Dict[State, Action] = {}

    for s in states:
        best_a = max(
            actions,
            key=lambda a: reward_func(s, a)
            + gamma
            * sum(
                prob * V[s_next]
                for s_next, prob in transitions[(s, a)]
            ),
        )
        policy[s] = best_a

    return V, policy


def policy_evaluation(
    policy: Dict[State, Action],
    states: List[State],
    transitions: Dict[Tuple[State, Action], List[Tuple[State, float]]],
    reward_func,
    gamma: float,
    tol: float = 1e-6,
    max_iters: int = 1000,
) -> Dict[State, float]:
    V: Dict[State, float] = {s: 0.0 for s in states}

    for _ in range(max_iters):
        V_new: Dict[State, float] = {}

        for s in states:
            a = policy[s]
            exp_next = sum(
                prob * V[s_next]
                for s_next, prob in transitions[(s, a)]
            )
            V_new[s] = reward_func(s, a) + gamma * exp_next

        delta = max(abs(V_new[s] - V[s]) for s in states)
        V = V_new

        if delta < tol:
            break

    return V


def policy_improvement(
    V: Dict[State, float],
    policy_old: Dict[State, Action],
    states: List[State],
    actions: List[Action],
    transitions: Dict[Tuple[State, Action], List[Tuple[State, float]]],
    reward_func,
    gamma: float,
):
    policy_new: Dict[State, Action] = {}
    stable = True

    for s in states:
        best_a = max(
            actions,
            key=lambda a: reward_func(s, a)
            + gamma
            * sum(
                prob * V[s_next]
                for s_next, prob in transitions[(s, a)]
            ),
        )
        policy_new[s] = best_a

        if best_a != policy_old[s]:
            stable = False

    return policy_new, stable


def policy_iteration(
    states: List[State],
    actions: List[Action],
    transitions: Dict[Tuple[State, Action], List[Tuple[State, float]]],
    reward_func,
    gamma: float,
    tol: float = 1e-6,
    max_iters: int = 1000,
):
    policy: Dict[State, Action] = {s: "off" for s in states}

    for _ in range(max_iters):
        V = policy_evaluation(
            policy=policy,
            states=states,
            transitions=transitions,
            reward_func=reward_func,
            gamma=gamma,
            tol=tol,
            max_iters=max_iters,
        )
        policy_new, stable = policy_improvement(
            V=V,
            policy_old=policy,
            states=states,
            actions=actions,
            transitions=transitions,
            reward_func=reward_func,
            gamma=gamma,
        )

        policy = policy_new

        if stable:
            break

    return V, policy
