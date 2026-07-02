import time

import pandas as pd
from joblib import Parallel, delayed

from env import A, N, R, S, arrival_probs, service_probs
from env import QueueControlEnv, create_env
from metrics import policy_match_ratio, q_to_value, value_sse
from monte_carlo import (
    monte_carlo_epsilon_greedy,
    monte_carlo_exploring_starts,
    off_policy_mc_weighted_importance_sampling,
)


def summarize_by(df_raw, group_columns):
    return (
        df_raw
        .groupby(group_columns)
        .agg(
            mean_value_sse=("value_sse", "mean"),
            min_value_sse=("value_sse", "min"),
            max_value_sse=("value_sse", "max"),
            std_value_sse=("value_sse", "std"),
            mean_policy_match=("policy_match", "mean"),
            min_policy_match=("policy_match", "min"),
            max_policy_match=("policy_match", "max"),
            std_policy_match=("policy_match", "std"),
            mean_training_time=("training_time", "mean"),
            std_training_time=("training_time", "std"),
        )
        .reset_index()
    )


def run_mc_es_once_light(
    num_episodes,
    alpha_lr,
    gamma,
    reference_policy,
    reference_values,
    max_steps=100,
    random_seed=42,
):
    env_train = create_env(
        random_seed=random_seed,
        max_steps=max_steps,
    )

    start_time = time.perf_counter()

    Q_mc_es, PI_mc_es, _ = monte_carlo_exploring_starts(
        env=env_train,
        states=S,
        actions=A,
        num_episodes=num_episodes,
        alpha_lr=alpha_lr,
        gamma=gamma,
        random_seed=random_seed,
    )

    training_time = time.perf_counter() - start_time
    policy_match = policy_match_ratio(PI_mc_es, reference_policy, S)
    V_mc_es = q_to_value(Q_mc_es, S, A)
    value_error_sse = value_sse(V_mc_es, reference_values, S)

    return {
        "num_episodes": num_episodes,
        "alpha_lr": alpha_lr,
        "max_steps": max_steps,
        "seed": random_seed,
        "policy_match": policy_match,
        "training_time": training_time,
        "value_sse": value_error_sse,
    }


def run_mc_eps_once_light(
    num_episodes,
    alpha_lr,
    gamma,
    epsilon,
    reference_policy,
    reference_values,
    max_steps=100,
    random_seed=42,
    random_start=True,
):
    env_train = QueueControlEnv(
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

    start_time = time.perf_counter()

    Q_mc_eps, PI_mc_eps, _ = monte_carlo_epsilon_greedy(
        env=env_train,
        states=S,
        actions=A,
        num_episodes=num_episodes,
        alpha_lr=alpha_lr,
        gamma=gamma,
        epsilon=epsilon,
        random_seed=random_seed,
        random_start=random_start,
    )

    training_time = time.perf_counter() - start_time
    policy_match = policy_match_ratio(PI_mc_eps, reference_policy, S)
    V_mc_eps = q_to_value(Q_mc_eps, S, A)
    value_error_sse = value_sse(V_mc_eps, reference_values, S)

    return {
        "num_episodes": num_episodes,
        "alpha_lr": alpha_lr,
        "epsilon": epsilon,
        "max_steps": max_steps,
        "seed": random_seed,
        "policy_match": policy_match,
        "training_time": training_time,
        "value_sse": value_error_sse,
    }


def run_mc_weighted_once_light(
    num_episodes,
    gamma,
    epsilon,
    reference_policy,
    reference_values,
    max_steps=100,
    random_seed=42,
    random_start=True,
):
    env_train = QueueControlEnv(
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

    start_time = time.perf_counter()

    Q_mc_weighted, PI_mc_weighted, _ = off_policy_mc_weighted_importance_sampling(
        env=env_train,
        states=S,
        actions=A,
        num_episodes=num_episodes,
        gamma=gamma,
        epsilon=epsilon,
        random_seed=random_seed,
        random_start=random_start,
    )

    training_time = time.perf_counter() - start_time
    policy_match = policy_match_ratio(PI_mc_weighted, reference_policy, S)
    V_mc_weighted = q_to_value(Q_mc_weighted, S, A)
    value_error_sse = value_sse(V_mc_weighted, reference_values, S)

    return {
        "num_episodes": num_episodes,
        "epsilon": epsilon,
        "max_steps": max_steps,
        "seed": random_seed,
        "policy_match": policy_match,
        "training_time": training_time,
        "value_sse": value_error_sse,
    }


def experiment_mc_es_num_episodes_parallel(
    episodes_list,
    seeds,
    alpha_lr,
    gamma,
    reference_policy,
    reference_values,
    max_steps=100,
    n_jobs=6,
):
    rows = Parallel(n_jobs=n_jobs)(
        delayed(run_mc_es_once_light)(
            num_episodes=num_episodes,
            alpha_lr=alpha_lr,
            gamma=gamma,
            reference_policy=reference_policy,
            reference_values=reference_values,
            max_steps=max_steps,
            random_seed=seed,
        )
        for num_episodes in episodes_list
        for seed in seeds
    )

    df_raw = pd.DataFrame(rows)
    df_summary = summarize_by(df_raw, "num_episodes")

    return df_raw, df_summary


def experiment_mc_es_alpha_parallel(
    alpha_list,
    seeds,
    num_episodes,
    gamma,
    reference_policy,
    reference_values,
    max_steps=100,
    n_jobs=6,
):
    rows = Parallel(n_jobs=n_jobs)(
        delayed(run_mc_es_once_light)(
            num_episodes=num_episodes,
            alpha_lr=alpha_lr,
            gamma=gamma,
            reference_policy=reference_policy,
            reference_values=reference_values,
            max_steps=max_steps,
            random_seed=seed,
        )
        for alpha_lr in alpha_list
        for seed in seeds
    )

    df_raw = pd.DataFrame(rows)
    df_summary = summarize_by(df_raw, "alpha_lr")

    return df_raw, df_summary


def experiment_mc_es_max_steps_parallel(
    max_steps_list,
    seeds,
    num_episodes,
    alpha_lr,
    gamma,
    reference_policy,
    reference_values,
    n_jobs=6,
):
    rows = Parallel(n_jobs=n_jobs)(
        delayed(run_mc_es_once_light)(
            num_episodes=num_episodes,
            alpha_lr=alpha_lr,
            gamma=gamma,
            reference_policy=reference_policy,
            reference_values=reference_values,
            max_steps=max_steps,
            random_seed=seed,
        )
        for max_steps in max_steps_list
        for seed in seeds
    )

    df_raw = pd.DataFrame(rows)
    df_summary = summarize_by(df_raw, "max_steps")

    return df_raw, df_summary


def experiment_mc_es_local_grid_parallel(
    num_episodes_list,
    alpha_list,
    max_steps_list,
    seeds,
    gamma,
    reference_policy,
    reference_values,
    n_jobs=6,
):
    rows = Parallel(n_jobs=n_jobs)(
        delayed(run_mc_es_once_light)(
            num_episodes=num_episodes,
            alpha_lr=alpha_lr,
            gamma=gamma,
            reference_policy=reference_policy,
            reference_values=reference_values,
            max_steps=max_steps,
            random_seed=seed,
        )
        for num_episodes in num_episodes_list
        for alpha_lr in alpha_list
        for max_steps in max_steps_list
        for seed in seeds
    )

    df_raw = pd.DataFrame(rows)
    df_summary = summarize_by(
        df_raw,
        ["num_episodes", "alpha_lr", "max_steps"],
    )

    return df_raw, df_summary


def experiment_mc_eps_epsilon_parallel(
    epsilon_list,
    seeds,
    num_episodes,
    alpha_lr,
    gamma,
    reference_policy,
    reference_values,
    max_steps=100,
    random_start=True,
    n_jobs=6,
):
    rows = Parallel(n_jobs=n_jobs)(
        delayed(run_mc_eps_once_light)(
            num_episodes=num_episodes,
            alpha_lr=alpha_lr,
            gamma=gamma,
            epsilon=epsilon,
            reference_policy=reference_policy,
            reference_values=reference_values,
            max_steps=max_steps,
            random_seed=seed,
            random_start=random_start,
        )
        for epsilon in epsilon_list
        for seed in seeds
    )

    df_raw = pd.DataFrame(rows)
    df_summary = summarize_by(df_raw, "epsilon")

    return df_raw, df_summary


def experiment_mc_eps_local_grid_parallel(
    num_episodes_list,
    alpha_list,
    epsilon_list,
    max_steps_list,
    seeds,
    gamma,
    reference_policy,
    reference_values,
    random_start=True,
    n_jobs=6,
):
    rows = Parallel(n_jobs=n_jobs)(
        delayed(run_mc_eps_once_light)(
            num_episodes=num_episodes,
            alpha_lr=alpha_lr,
            gamma=gamma,
            epsilon=epsilon,
            reference_policy=reference_policy,
            reference_values=reference_values,
            max_steps=max_steps,
            random_seed=seed,
            random_start=random_start,
        )
        for num_episodes in num_episodes_list
        for alpha_lr in alpha_list
        for epsilon in epsilon_list
        for max_steps in max_steps_list
        for seed in seeds
    )

    df_raw = pd.DataFrame(rows)
    df_summary = summarize_by(
        df_raw,
        ["num_episodes", "alpha_lr", "epsilon", "max_steps"],
    )

    return df_raw, df_summary


def experiment_mc_weighted_epsilon_parallel(
    epsilon_list,
    seeds,
    num_episodes,
    gamma,
    reference_policy,
    reference_values,
    max_steps=100,
    random_start=True,
    n_jobs=6,
):
    rows = Parallel(n_jobs=n_jobs)(
        delayed(run_mc_weighted_once_light)(
            num_episodes=num_episodes,
            gamma=gamma,
            epsilon=epsilon,
            reference_policy=reference_policy,
            reference_values=reference_values,
            max_steps=max_steps,
            random_seed=seed,
            random_start=random_start,
        )
        for epsilon in epsilon_list
        for seed in seeds
    )

    df_raw = pd.DataFrame(rows)
    df_summary = summarize_by(df_raw, "epsilon")

    return df_raw, df_summary


def experiment_mc_weighted_max_steps_parallel(
    max_steps_list,
    seeds,
    num_episodes,
    gamma,
    epsilon,
    reference_policy,
    reference_values,
    random_start=True,
    n_jobs=6,
):
    rows = Parallel(n_jobs=n_jobs)(
        delayed(run_mc_weighted_once_light)(
            num_episodes=num_episodes,
            gamma=gamma,
            epsilon=epsilon,
            reference_policy=reference_policy,
            reference_values=reference_values,
            max_steps=max_steps,
            random_seed=seed,
            random_start=random_start,
        )
        for max_steps in max_steps_list
        for seed in seeds
    )

    df_raw = pd.DataFrame(rows)
    df_summary = summarize_by(df_raw, "max_steps")

    return df_raw, df_summary


def experiment_mc_weighted_grid_parallel(
    num_episodes_list,
    epsilon_list,
    max_steps_list,
    seeds,
    gamma,
    reference_policy,
    reference_values,
    random_start=True,
    n_jobs=6,
):
    rows = Parallel(n_jobs=n_jobs)(
        delayed(run_mc_weighted_once_light)(
            num_episodes=num_episodes,
            gamma=gamma,
            epsilon=epsilon,
            reference_policy=reference_policy,
            reference_values=reference_values,
            max_steps=max_steps,
            random_seed=seed,
            random_start=random_start,
        )
        for num_episodes in num_episodes_list
        for epsilon in epsilon_list
        for max_steps in max_steps_list
        for seed in seeds
    )

    df_raw = pd.DataFrame(rows)
    df_summary = summarize_by(
        df_raw,
        ["num_episodes", "epsilon", "max_steps"],
    )

    return df_raw, df_summary
