from pathlib import Path

import matplotlib.pyplot as plt


def _save_figure(fig, save_path):
    if save_path is None:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_value_function(
    states,
    values,
    title="Функция ценности",
    xlabel="Состояние",
    ylabel="Значение функции ценности",
    show=True,
    save_path=None,
    close=False,
):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=110)

    ax.plot(states, [values[s] for s in states], marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, save_path)

    if show:
        plt.show()

    if close:
        plt.close(fig)

    return fig, ax


def plot_policy(
    states,
    policy,
    title="Политика управления",
    xlabel="Состояние",
    ylabel="Действие",
    show=True,
    save_path=None,
    close=False,
):
    action_to_num = {"off": 0, "norm": 1, "fast": 2}
    policy_numeric = [action_to_num[policy[s]] for s in states]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)

    ax.step(states, policy_numeric, where="mid")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["off", "norm", "fast"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, save_path)

    if show:
        plt.show()

    if close:
        plt.close(fig)

    return fig, ax


def plot_metric_by_parameter(
    df_summary,
    x_column,
    y_column,
    title,
    xlabel=None,
    ylabel=None,
    show=True,
    save_path=None,
    close=False,
):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=110)

    ax.plot(df_summary[x_column], df_summary[y_column], marker="o")
    ax.set_xlabel(xlabel or x_column)
    ax.set_ylabel(ylabel or y_column)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, save_path)

    if show:
        plt.show()

    if close:
        plt.close(fig)

    return fig, ax
