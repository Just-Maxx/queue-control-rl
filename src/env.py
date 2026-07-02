from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np

State = int
Action = str

N = 50
gamma = 0.7

lambda_arrival = 0.8
mu = 0.7

p_fast_0 = 0.1
p_fast_1 = 0.3
p_fast_2 = 0.6

alpha = 1.0
c_off = 0.0
c_norm = 0.5
c_fast = 1.5

tol = 1e-6
max_iters = 1000

S: List[State] = list(range(N + 1))
A: List[Action] = ["off", "norm", "fast"]

arrival_probs = {
    0: 1 - lambda_arrival,
    1: lambda_arrival,
}

service_probs = {
    "off": {0: 1.0},
    "norm": {0: 1 - mu, 1: mu},
    "fast": {0: p_fast_0, 1: p_fast_1, 2: p_fast_2},
}


def reward(s: State, a: Action) -> float:
    cost = {
        "off": c_off,
        "norm": c_norm,
        "fast": c_fast,
    }
    return -alpha * s - cost[a]


R = reward


def build_transition_table(
    n: int = N,
    states: List[State] = S,
    actions: List[Action] = A,
    arrival_distribution: Dict[int, float] = arrival_probs,
    service_distribution: Dict[Action, Dict[int, float]] = service_probs,
) -> Dict[Tuple[State, Action], List[Tuple[State, float]]]:
    transitions: Dict[Tuple[State, Action], List[Tuple[State, float]]] = {}

    for s in states:
        for a in actions:
            next_state_probs = defaultdict(float)

            for y, py in service_distribution[a].items():
                for x, px in arrival_distribution.items():
                    served = min(s, y)
                    s_next = min(n, s - served + x)
                    next_state_probs[s_next] += py * px

            transitions[(s, a)] = list(next_state_probs.items())

    for key, rows in transitions.items():
        total = sum(prob for _, prob in rows)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Probabilities for {key} sum to {total}, not 1."
            )

    return transitions


P = build_transition_table()


class QueueControlEnv:
    def __init__(
        self,
        n: int,
        states: List[State],
        actions: List[Action],
        arrival_probs: Dict[int, float],
        service_probs: Dict[Action, Dict[int, float]],
        reward_func: Callable[[State, Action], float],
        max_steps: int = 100,
        start_state: State = 0,
        random_seed: int | None = None,
    ):
        self.n = n
        self.states = states
        self.actions = actions
        self.arrival_probs = arrival_probs
        self.service_probs = service_probs
        self.reward_func = reward_func
        self.max_steps = max_steps
        self.start_state = start_state

        self.rng = np.random.default_rng(random_seed)

        self.state = start_state
        self.step_count = 0

    def reset(
        self,
        start_state: State | None = None,
        random_start: bool = False,
    ) -> State:
        if random_start:
            self.state = int(self.rng.choice(self.states))
        elif start_state is None:
            self.state = self.start_state
        else:
            if start_state not in self.states:
                raise ValueError(
                    f"Недопустимое начальное состояние: {start_state}"
                )
            self.state = start_state

        self.step_count = 0
        return self.state

    def sample_from_distribution(self, probs: Dict[int, float]) -> int:
        values = list(probs.keys())
        probabilities = list(probs.values())

        if not np.isclose(sum(probabilities), 1.0):
            raise ValueError("Сумма вероятностей должна быть равна 1")

        return int(self.rng.choice(values, p=probabilities))

    def step(self, action: Action) -> Tuple[State, float, bool]:
        if action not in self.actions:
            raise ValueError(f"Недопустимое действие: {action}")

        current_state = self.state

        service = self.sample_from_distribution(self.service_probs[action])
        arrival = self.sample_from_distribution(self.arrival_probs)

        served = min(current_state, service)
        next_state = min(self.n, current_state - served + arrival)

        reward_value = self.reward_func(current_state, action)

        self.state = next_state
        self.step_count += 1

        done = self.step_count >= self.max_steps

        return next_state, reward_value, done


def create_env(random_seed: int = 42, max_steps: int = 100) -> QueueControlEnv:
    return QueueControlEnv(
        n=N,
        states=S,
        actions=A,
        arrival_probs=arrival_probs,
        service_probs=service_probs,
        reward_func=R,
        max_steps=max_steps,
        start_state=0,
        random_seed=random_seed,
    )
