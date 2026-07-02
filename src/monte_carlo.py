from collections import defaultdict

import numpy as np

from metrics import discounted_episode_return


def generate_episode(
    env,
    policy,
    start_state=None,
    random_start=False,
):
    episode = []

    state = env.reset(
        start_state=start_state,
        random_start=random_start,
    )
    done = False

    while not done:
        action = policy(state)
        next_state, reward, done = env.step(action)

        episode.append((state, action, reward, next_state))

        state = next_state

    return episode


def choose_random_action(actions, rng):
    return str(rng.choice(actions))


def choose_greedy_action(Q, state, actions, rng):
    q_values = [Q[(state, action)] for action in actions]
    max_q = max(q_values)

    best_actions = [
        action
        for action in actions
        if Q[(state, action)] == max_q
    ]

    return str(rng.choice(best_actions))


def choose_greedy_action_deterministic(Q, state, actions):
    q_values = [Q[(state, action)] for action in actions]
    max_q = max(q_values)

    for action in actions:
        if Q[(state, action)] == max_q:
            return action


def choose_epsilon_greedy_action(Q, state, actions, epsilon, rng):
    if rng.random() < epsilon:
        return choose_random_action(actions, rng)

    return choose_greedy_action(Q, state, actions, rng)


def epsilon_greedy_action_with_prob(Q, state, actions, epsilon, rng):
    greedy_action = choose_greedy_action_deterministic(
        Q,
        state,
        actions,
    )

    if rng.random() < epsilon:
        action = str(rng.choice(actions))
    else:
        action = greedy_action

    action_prob = epsilon / len(actions)

    if action == greedy_action:
        action_prob += 1.0 - epsilon

    return action, action_prob


def build_greedy_policy(Q, states, actions):
    policy = {}

    for state in states:
        policy[state] = choose_greedy_action_deterministic(
            Q,
            state,
            actions,
        )

    return policy


def monte_carlo_exploring_starts(
    env,
    states,
    actions,
    num_episodes,
    alpha_lr,
    gamma,
    random_seed=None,
):
    rng = np.random.default_rng(random_seed)

    Q = defaultdict(float)
    returns_history = []

    for _ in range(num_episodes):
        episode = []

        start_state = int(rng.choice(states))
        first_action = str(rng.choice(actions))

        state = env.reset(start_state=start_state)
        next_state, reward, done = env.step(first_action)

        episode.append((state, first_action, reward, next_state))

        state = next_state

        while not done:
            action = choose_greedy_action(Q, state, actions, rng)
            next_state, reward, done = env.step(action)

            episode.append((state, action, reward, next_state))

            state = next_state

        G = 0.0

        for state, action, reward, _ in reversed(episode):
            G = reward + gamma * G

            Q[(state, action)] += alpha_lr * (
                G - Q[(state, action)]
            )

        returns_history.append(discounted_episode_return(episode, gamma))

    policy = build_greedy_policy(Q, states, actions)

    return Q, policy, returns_history


def monte_carlo_epsilon_greedy(
    env,
    states,
    actions,
    num_episodes,
    alpha_lr,
    gamma,
    epsilon,
    random_seed=None,
    random_start=False,
):
    rng = np.random.default_rng(random_seed)
    Q = defaultdict(float)
    returns_history = []

    for _ in range(num_episodes):
        episode = []

        state = env.reset(random_start=random_start)
        done = False

        while not done:
            action = choose_epsilon_greedy_action(
                Q,
                state,
                actions,
                epsilon,
                rng,
            )

            next_state, reward, done = env.step(action)

            episode.append((state, action, reward, next_state))

            state = next_state

        G = 0.0

        for state, action, reward, _ in reversed(episode):
            G = reward + gamma * G

            Q[(state, action)] += alpha_lr * (
                G - Q[(state, action)]
            )

        returns_history.append(discounted_episode_return(episode, gamma))

    policy = build_greedy_policy(Q, states, actions)

    return Q, policy, returns_history


def off_policy_mc_weighted_importance_sampling(
    env,
    states,
    actions,
    num_episodes,
    gamma,
    epsilon,
    random_seed=None,
    random_start=True,
):
    rng = np.random.default_rng(random_seed)

    Q = defaultdict(float)
    C = defaultdict(float)
    returns_history = []

    for _ in range(num_episodes):
        episode = []

        state = env.reset(random_start=random_start)
        done = False

        while not done:
            action, action_prob = epsilon_greedy_action_with_prob(
                Q,
                state,
                actions,
                epsilon,
                rng,
            )

            next_state, reward, done = env.step(action)

            episode.append(
                (state, action, reward, next_state, action_prob)
            )

            state = next_state

        G = 0.0
        W = 1.0

        for state, action, reward, _, action_prob in reversed(episode):
            G = reward + gamma * G

            C[(state, action)] += W

            Q[(state, action)] += (
                W / C[(state, action)]
            ) * (
                G - Q[(state, action)]
            )

            greedy_action = choose_greedy_action_deterministic(
                Q,
                state,
                actions,
            )

            if action != greedy_action:
                break

            W /= action_prob

        returns_history.append(
            discounted_episode_return(
                [
                    (state, action, reward, next_state)
                    for state, action, reward, next_state, _ in episode
                ],
                gamma,
            )
        )

    policy = build_greedy_policy(Q, states, actions)

    return Q, policy, returns_history


def off_policy_mc_weighted_diagnostics(
    env,
    states,
    actions,
    num_episodes,
    gamma,
    epsilon,
    value_reference,
    policy_reference,
    policy_match_ratio,
    q_to_value,
    value_sse,
    random_seed=None,
    random_start=True,
):
    rng = np.random.default_rng(random_seed)

    Q = defaultdict(float)
    C = defaultdict(float)

    episode_lengths = []
    used_update_counts = []
    break_counts = 0

    for _ in range(num_episodes):
        episode = []

        state = env.reset(random_start=random_start)
        done = False

        while not done:
            action, action_prob = epsilon_greedy_action_with_prob(
                Q,
                state,
                actions,
                epsilon,
                rng,
            )

            next_state, reward, done = env.step(action)

            episode.append(
                (state, action, reward, next_state, action_prob)
            )

            state = next_state

        G = 0.0
        W = 1.0
        used_updates = 0
        was_break = False

        for state, action, reward, _, action_prob in reversed(episode):
            G = reward + gamma * G

            C[(state, action)] += W

            Q[(state, action)] += (
                W / C[(state, action)]
            ) * (
                G - Q[(state, action)]
            )

            used_updates += 1

            greedy_action = choose_greedy_action_deterministic(
                Q,
                state,
                actions,
            )

            if action != greedy_action:
                was_break = True
                break

            W /= action_prob

        episode_lengths.append(len(episode))
        used_update_counts.append(used_updates)

        if was_break:
            break_counts += 1

    policy = build_greedy_policy(Q, states, actions)
    V_mc_weighted = q_to_value(Q, states, actions)
    value_error_sse = value_sse(V_mc_weighted, value_reference, states)

    diagnostics = {
        "mean_episode_length": np.mean(episode_lengths),
        "mean_used_updates": np.mean(used_update_counts),
        "min_used_updates": np.min(used_update_counts),
        "max_used_updates": np.max(used_update_counts),
        "break_ratio": break_counts / num_episodes,
        "policy_match": policy_match_ratio(policy, policy_reference, states),
        "value_sse": value_error_sse,
    }

    return Q, policy, diagnostics
