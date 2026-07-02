"""Полная быстрая проверка файлов из src.

Запуск из корня репозитория:
    python scripts/full_check.py

Скрипт проверяет импорты всех модулей из src и делает короткие пробные
запуски основных функций. Это не повторяет тяжелые эксперименты из ноутбука.
"""

import matplotlib.pyplot as plt
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))


def assert_condition(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    from dynamic_programming import policy_iteration, value_iteration
    from env import A, P, R, S, gamma, max_iters, tol, create_env
    from experiments import (
        experiment_mc_eps_epsilon_parallel,
        experiment_mc_es_num_episodes_parallel,
        experiment_mc_weighted_epsilon_parallel,
        run_mc_eps_once_light,
        run_mc_es_once_light,
        run_mc_weighted_once_light,
    )
    from metrics import policy_match_ratio, q_to_value, value_sse
    from monte_carlo import (
        generate_episode,
        monte_carlo_epsilon_greedy,
        monte_carlo_exploring_starts,
        off_policy_mc_weighted_diagnostics,
        off_policy_mc_weighted_importance_sampling,
    )
    from plotting import plot_metric_by_parameter, plot_policy, plot_value_function

    print("Полная проверка src-кода проекта")

    print("\n1. Проверка MDP и динамического программирования")
    value_vi, policy_vi = value_iteration(
        states=S,
        actions=A,
        transitions=P,
        reward_func=R,
        gamma=gamma,
        tol=tol,
        max_iters=max_iters,
    )
    value_pi, policy_pi = policy_iteration(
        states=S,
        actions=A,
        transitions=P,
        reward_func=R,
        gamma=gamma,
        tol=tol,
        max_iters=max_iters,
    )

    match_vi_pi = policy_match_ratio(policy_vi, policy_pi, S)
    assert_condition(match_vi_pi > 0.99, "Политики VI и PI почти не совпали")
    print(f"VI/PI работают, совпадение политик: {match_vi_pi:.3f}")

    print("\n2. Проверка генерации эпизода")
    env = create_env(random_seed=42, max_steps=5)
    episode = generate_episode(env, policy=lambda state: "off")
    assert_condition(len(episode) == 5, "Эпизод должен иметь длину 5")
    print("Генерация эпизода работает")

    print("\n3. Проверка основных Monte Carlo методов")
    for name, func, kwargs in [
        (
            "MC Exploring Starts",
            monte_carlo_exploring_starts,
            {
                "num_episodes": 5,
                "alpha_lr": 0.01,
                "gamma": gamma,
                "random_seed": 42,
            },
        ),
        (
            "MC epsilon-greedy",
            monte_carlo_epsilon_greedy,
            {
                "num_episodes": 5,
                "alpha_lr": 0.01,
                "gamma": gamma,
                "epsilon": 0.1,
                "random_seed": 42,
                "random_start": True,
            },
        ),
        (
            "Off-policy weighted IS",
            off_policy_mc_weighted_importance_sampling,
            {
                "num_episodes": 5,
                "gamma": gamma,
                "epsilon": 0.1,
                "random_seed": 42,
                "random_start": True,
            },
        ),
    ]:
        env = create_env(random_seed=42, max_steps=10)
        q_values, policy, returns = func(
            env=env,
            states=S,
            actions=A,
            **kwargs,
        )
        assert_condition(len(returns) == 5, f"{name}: неверное число returns")
        assert_condition(
            set(policy.keys()) == set(S),
            f"{name}: политика не для всех состояний",
        )

        value_estimate = q_to_value(q_values, S, A)
        _ = value_sse(value_estimate, value_vi, S)
        print(f"{name}: работает")

    print("\n4. Проверка диагностической функции off-policy MC")
    env = create_env(random_seed=42, max_steps=10)
    _, _, diagnostics = off_policy_mc_weighted_diagnostics(
        env=env,
        states=S,
        actions=A,
        num_episodes=5,
        gamma=gamma,
        epsilon=0.1,
        value_reference=value_vi,
        policy_reference=policy_vi,
        policy_match_ratio=policy_match_ratio,
        q_to_value=q_to_value,
        value_sse=value_sse,
        random_seed=42,
        random_start=True,
    )
    assert_condition("value_sse" in diagnostics, "Нет value_sse в diagnostics")
    print("Диагностика работает")

    print("\n5. Проверка функций из experiments.py")
    _ = run_mc_es_once_light(
        num_episodes=5,
        alpha_lr=0.01,
        gamma=gamma,
        reference_policy=policy_vi,
        reference_values=value_vi,
        max_steps=10,
        random_seed=42,
    )
    _ = run_mc_eps_once_light(
        num_episodes=5,
        alpha_lr=0.01,
        gamma=gamma,
        epsilon=0.1,
        reference_policy=policy_vi,
        reference_values=value_vi,
        max_steps=10,
        random_seed=42,
    )
    _ = run_mc_weighted_once_light(
        num_episodes=5,
        gamma=gamma,
        epsilon=0.1,
        reference_policy=policy_vi,
        reference_values=value_vi,
        max_steps=10,
        random_seed=42,
    )
    print("Одиночные функции experiments работают")

    df_raw, df_summary = experiment_mc_es_num_episodes_parallel(
        episodes_list=[5],
        seeds=[42],
        alpha_lr=0.01,
        gamma=gamma,
        reference_policy=policy_vi,
        reference_values=value_vi,
        max_steps=10,
        n_jobs=1,
    )
    assert_condition(
        len(df_raw) == 1 and len(df_summary) == 1,
        "Ошибка MC ES experiment",
    )

    df_raw, df_summary = experiment_mc_eps_epsilon_parallel(
        epsilon_list=[0.1],
        seeds=[42],
        num_episodes=5,
        alpha_lr=0.01,
        gamma=gamma,
        reference_policy=policy_vi,
        reference_values=value_vi,
        max_steps=10,
        n_jobs=1,
    )
    assert_condition(
        len(df_raw) == 1 and len(df_summary) == 1,
        "Ошибка MC epsilon experiment",
    )

    df_raw, df_summary = experiment_mc_weighted_epsilon_parallel(
        epsilon_list=[0.1],
        seeds=[42],
        num_episodes=5,
        gamma=gamma,
        reference_policy=policy_vi,
        reference_values=value_vi,
        max_steps=10,
        n_jobs=1,
    )
    assert_condition(
        len(df_raw) == 1 and len(df_summary) == 1,
        "Ошибка weighted experiment",
    )
    print("Параллельные experiment-функции работают в режиме n_jobs=1")

    print("\n6. Проверка plotting.py")
    plot_value_function(
        S[:5],
        value_vi,
        title="Проверка V",
        show=False,
        close=True,
    )
    plot_policy(
        S[:5],
        policy_vi,
        title="Проверка политики",
        show=False,
        close=True,
    )
    plot_metric_by_parameter(
        df_summary,
        x_column="epsilon",
        y_column="mean_value_sse",
        title="Проверка графика",
        show=False,
        close=True,
    )
    print("Функции графиков вызываются без ошибок")

    print("\nВсе проверки src-кода пройдены.")


if __name__ == "__main__":
    main()
