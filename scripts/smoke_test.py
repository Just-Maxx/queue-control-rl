"""Быстрая проверка кода проекта.

Запуск из корня репозитория:
    python scripts/smoke_test.py

Скрипт не воспроизводит все эксперименты из ноутбука, а только проверяет,
что основные модули импортируются и базовые алгоритмы запускаются без ошибок.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))


def main() -> None:
    from dynamic_programming import policy_iteration, value_iteration
    from env import A, P, R, S, gamma, max_iters, tol, create_env
    from metrics import policy_match_ratio, q_to_value, value_sse
    from monte_carlo import monte_carlo_epsilon_greedy

    print("Проверка проекта queue-control-rl")
    print(f"Количество состояний: {len(S)}")
    print(f"Действия: {A}")

    print("\n1. Проверка Value Iteration...")
    value_vi, policy_vi = value_iteration(
        states=S,
        actions=A,
        transitions=P,
        reward_func=R,
        gamma=gamma,
        tol=tol,
        max_iters=max_iters,
    )
    print("Value Iteration успешно завершен")

    print("\n2. Проверка Policy Iteration...")
    value_pi, policy_pi = policy_iteration(
        states=S,
        actions=A,
        transitions=P,
        reward_func=R,
        gamma=gamma,
        tol=tol,
        max_iters=max_iters,
    )
    print("Policy Iteration успешно завершен")

    match = policy_match_ratio(policy_vi, policy_pi, S)
    print(f"Совпадение политик VI и PI: {match:.3f}")

    print("\n3. Короткий запуск Monte Carlo epsilon-greedy...")
    env = create_env(random_seed=42, max_steps=20)
    q_mc, policy_mc, returns = monte_carlo_epsilon_greedy(
        env=env,
        states=S,
        actions=A,
        num_episodes=10,
        alpha_lr=0.01,
        gamma=gamma,
        epsilon=0.1,
        random_seed=42,
        random_start=True,
    )
    value_mc = q_to_value(q_mc, S, A)
    error_sse = value_sse(value_mc, value_vi, S)

    print(f"Количество эпизодов MC: {len(returns)}")
    print(f"SSE относительно Value Iteration: {error_sse:.2f}")
    print("\nВсе базовые проверки пройдены.")


if __name__ == "__main__":
    main()
