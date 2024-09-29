import random
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from project import Observation
from train.base_learner import build_network
from train.config import LEARNING_KINDS
from train.environment import encode_observation

from .metrics import WIN_RESULTS, summarize_hands
from .run_store import RunStore

plt.switch_backend("Agg")

KIND_COLORS = {
    "dqn": "#2a78d6",
    "mc": "#eb6834",
    "noob": "#1baf7a",
    "apprehensive": "#eda100",
    "aggressive": "#e87ba4",
    "random": "#008300",
}
FALLBACK_COLOR = "#4a3aa7"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
MUTED = "#898781"
INK = "#0b0b0b"
SECONDARY = "#52514e"


def color_for_kind(kind: str) -> str:
    return KIND_COLORS.get(kind, FALLBACK_COLOR)


def rolling_mean(values: List[float], window: int) -> List[float]:
    means = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
            means.append(total / window)
        else:
            means.append(total / (index + 1))
    return means


def group_by_kind(hand_records: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for record in hand_records:
        grouped.setdefault(record["kind"], []).append(record)
    return grouped


def style_axes(axes, title: str, x_label: str, y_label: str) -> None:
    axes.set_facecolor(SURFACE)
    axes.set_title(title, color=INK, fontsize=12)
    axes.set_xlabel(x_label, color=SECONDARY, fontsize=10)
    axes.set_ylabel(y_label, color=SECONDARY, fontsize=10)
    axes.tick_params(colors=MUTED, labelsize=9)
    axes.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_color(GRID)


def build_figure(title: str, x_label: str, y_label: str):
    figure, axes = plt.subplots(figsize=(8, 5))
    figure.patch.set_facecolor(SURFACE)
    style_axes(axes, title, x_label, y_label)
    return figure, axes


def save_figure(figure, path: str) -> None:
    figure.savefig(path, dpi=120, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)


def add_legend(axes) -> None:
    legend = axes.legend(frameon=False, fontsize=9, labelcolor=SECONDARY, loc="best")
    for line in legend.get_lines():
        line.set_linewidth(2)


def plot_loss_curves(loss_records: List[dict], path: str, window: int = 200) -> None:
    figure, axes = build_figure("Training loss", "train step", "loss (rolling mean)")
    for kind in sorted({record["kind"] for record in loss_records}):
        losses = [record["loss"] for record in loss_records if record["kind"] == kind]
        axes.plot(
            rolling_mean(losses, window),
            color=color_for_kind(kind),
            linewidth=2,
            label=kind,
        )
    add_legend(axes)
    save_figure(figure, path)


def plot_win_rates(hand_records: List[dict], path: str, window: int = 2000) -> None:
    figure, axes = build_figure(
        "Rolling win rate", "hands played", "win rate (rolling mean)"
    )
    for kind, records in sorted(group_by_kind(hand_records).items()):
        wins = [
            (
                1.0
                if record["result"] in WIN_RESULTS
                or record["result"] == "PLAYER_BLACKJACK"
                else 0.0
            )
            for record in records
        ]
        axes.plot(
            rolling_mean(wins, window),
            color=color_for_kind(kind),
            linewidth=2,
            label=kind,
        )
    add_legend(axes)
    save_figure(figure, path)


def plot_reward_comparison(summaries: Dict[str, dict], path: str) -> None:
    figure, axes = build_figure(
        "Average reward per hand", "avg reward (fraction of bet)", ""
    )
    kinds = sorted(summaries, key=lambda kind: summaries[kind]["avg_reward"])
    rewards = [summaries[kind]["avg_reward"] for kind in kinds]
    colors = [color_for_kind(kind) for kind in kinds]
    bars = axes.barh(kinds, rewards, color=colors, height=0.6)
    axes.axvline(0, color=MUTED, linewidth=1)
    for reward_bar, reward in zip(bars, rewards):
        axes.text(
            reward_bar.get_width() + (0.005 if reward >= 0 else -0.005),
            reward_bar.get_y() + reward_bar.get_height() / 2,
            f"{reward:.3f}",
            va="center",
            ha="left" if reward >= 0 else "right",
            color=SECONDARY,
            fontsize=9,
        )
    save_figure(figure, path)


def plot_result_breakdown(summaries: Dict[str, dict], path: str) -> None:
    figure, axes = build_figure("Hand results by agent", "", "share of hands")
    categories = ["win", "blackjack", "push", "dealer win", "bust"]
    kinds = sorted(summaries)
    bar_width = 0.8 / max(len(kinds), 1)
    for kind_index, kind in enumerate(kinds):
        summary = summaries[kind]
        hands = summary["hands"]
        shares = [
            summary["wins"] / hands,
            summary["blackjacks"] / hands,
            summary["pushes"] / hands,
            (summary["losses"] - summary["busts"]) / hands,
            summary["busts"] / hands,
        ]
        positions = [
            category_index + kind_index * bar_width
            for category_index in range(len(categories))
        ]
        axes.bar(
            positions,
            shares,
            width=bar_width * 0.9,
            color=color_for_kind(kind),
            label=kind,
        )
    axes.set_xticks(
        [
            category_index + 0.4 - bar_width / 2
            for category_index in range(len(categories))
        ]
    )
    axes.set_xticklabels(categories)
    add_legend(axes)
    save_figure(figure, path)


def plot_bankrolls(
    hand_records: List[dict], path: str, players_per_kind: int = 2
) -> None:
    figure, axes = build_figure("Bankroll trajectories", "hands played", "money")
    for kind, records in sorted(group_by_kind(hand_records).items()):
        player_ids = []
        for record in records:
            if record["player_id"] not in player_ids:
                player_ids.append(record["player_id"])
            if len(player_ids) >= players_per_kind:
                break
        for sample_index, player_id in enumerate(player_ids):
            money_series = [
                record["money_after"]
                for record in records
                if record["player_id"] == player_id
            ]
            axes.plot(
                money_series,
                color=color_for_kind(kind),
                linewidth=2,
                alpha=1.0 if sample_index == 0 else 0.45,
                label=kind if sample_index == 0 else None,
            )
    add_legend(axes)
    save_figure(figure, path)


def plot_epsilon_schedule(hand_records: List[dict], path: str) -> None:
    figure, axes = build_figure("Exploration schedule", "hands played", "epsilon")
    for kind, records in sorted(group_by_kind(hand_records).items()):
        epsilons = [record["epsilon"] for record in records]
        if not any(epsilons):
            continue
        axes.plot(epsilons, color=color_for_kind(kind), linewidth=2, label=kind)
    add_legend(axes)
    save_figure(figure, path)


ACTION_LABELS = ["hit", "stay", "double"]
ACTION_LETTERS = ["H", "S", "D"]
ACTION_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
LAYER_TITLES = ["input → hidden 1", "hidden 1 → hidden 2", "hidden 2 → q values"]
UPCARD_VALUES = list(range(2, 12))


def load_weights_by_kind(run_store: RunStore) -> Dict[str, list]:
    weights_by_kind = {}
    for kind in LEARNING_KINDS:
        stored = run_store.read_json(f"weights_{kind}.json")
        if stored is not None:
            weights_by_kind[kind] = stored["weights"]
    return weights_by_kind


def draw_weight_panel(figure, axes, matrix, scale, title) -> None:
    image = axes.imshow(matrix, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
    axes.set_title(title, color=INK, fontsize=10)
    axes.tick_params(colors=MUTED, labelsize=8)
    for spine in axes.spines.values():
        spine.set_color(GRID)
    colorbar = figure.colorbar(image, ax=axes, shrink=0.85)
    colorbar.ax.tick_params(colors=MUTED, labelsize=7)
    colorbar.outline.set_edgecolor(GRID)


def plot_weight_heatmaps(weights_by_kind: Dict[str, list], path: str) -> None:
    kinds = sorted(weights_by_kind)
    figure, axes_grid = plt.subplots(
        len(kinds), 3, figsize=(12, 3.6 * len(kinds)), squeeze=False
    )
    figure.patch.set_facecolor(SURFACE)
    for column, layer_index in enumerate([0, 2, 4]):
        scale = max(
            abs(value)
            for kind in kinds
            for row in weights_by_kind[kind][layer_index]
            for value in row
        )
        for row_index, kind in enumerate(kinds):
            matrix = weights_by_kind[kind][layer_index]
            draw_weight_panel(
                figure,
                axes_grid[row_index][column],
                matrix,
                scale,
                f"{kind}: {LAYER_TITLES[column]} ({len(matrix)}×{len(matrix[0])})",
            )
    figure.suptitle(
        "learned weights (blue negative, red positive)", color=INK, fontsize=12
    )
    figure.tight_layout()
    save_figure(figure, path)


def build_policy_grid(network, soft: bool) -> List[List[int]]:
    totals = range(12, 22) if soft else range(4, 22)
    grid = []
    for total in totals:
        row = []
        for upcard in UPCARD_VALUES:
            observation = Observation(
                player_total=total,
                is_soft=soft,
                dealer_upcard_value=upcard,
                can_double=True,
                money=1000,
            )
            q_values = network.forward([encode_observation(observation)])[0]
            row.append(q_values.index(max(q_values)))
        grid.append(row)
    return grid


def draw_policy_panel(axes, grid, totals, title) -> None:
    axes.imshow(
        grid,
        cmap=ListedColormap(ACTION_COLORS),
        vmin=0,
        vmax=2,
        aspect="auto",
    )
    for row_index, row in enumerate(grid):
        for column_index, action_index in enumerate(row):
            axes.text(
                column_index,
                row_index,
                ACTION_LETTERS[action_index],
                ha="center",
                va="center",
                color="#fcfcfb",
                fontsize=7,
                fontweight="bold",
            )
    axes.set_title(title, color=INK, fontsize=10)
    axes.set_xticks(range(len(UPCARD_VALUES)))
    axes.set_xticklabels(
        ["A" if upcard == 11 else str(upcard) for upcard in UPCARD_VALUES]
    )
    axes.set_yticks(range(len(totals)))
    axes.set_yticklabels([str(total) for total in totals])
    axes.tick_params(colors=MUTED, labelsize=8)
    axes.set_xlabel("dealer upcard", color=SECONDARY, fontsize=9)
    axes.set_ylabel("player total", color=SECONDARY, fontsize=9)
    for spine in axes.spines.values():
        spine.set_color(GRID)


def plot_policy_charts(weights_by_kind: Dict[str, list], path: str) -> None:
    kinds = sorted(weights_by_kind)
    figure, axes_grid = plt.subplots(
        len(kinds), 2, figsize=(11, 5.2 * len(kinds)), squeeze=False
    )
    figure.patch.set_facecolor(SURFACE)
    for row_index, kind in enumerate(kinds):
        network = build_network([32, 32], random.Random(0))
        network.set_training(False).set_weights(weights_by_kind[kind])
        draw_policy_panel(
            axes_grid[row_index][0],
            build_policy_grid(network, soft=False),
            list(range(4, 22)),
            f"{kind}: hard totals",
        )
        draw_policy_panel(
            axes_grid[row_index][1],
            build_policy_grid(network, soft=True),
            list(range(12, 22)),
            f"{kind}: soft totals",
        )
    legend_patches = [
        Patch(facecolor=color, label=label)
        for color, label in zip(ACTION_COLORS, ACTION_LABELS)
    ]
    figure.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        frameon=False,
        labelcolor=SECONDARY,
    )
    figure.suptitle("learned policy (double assumed available)", color=INK, fontsize=12)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(figure, path)


def render_network_plots(run_store: RunStore) -> List[str]:
    weights_by_kind = load_weights_by_kind(run_store)
    if not weights_by_kind:
        return []
    rendered = []
    heatmap_path = run_store.plot_path("network_weights.png")
    plot_weight_heatmaps(weights_by_kind, heatmap_path)
    rendered.append(heatmap_path)
    policy_path = run_store.plot_path("policy_charts.png")
    plot_policy_charts(weights_by_kind, policy_path)
    rendered.append(policy_path)
    return rendered


def render_all(run_store: RunStore) -> List[str]:
    hand_records = run_store.read_jsonl("metrics.jsonl")
    loss_records = run_store.read_jsonl("loss.jsonl")
    if not hand_records:
        return []
    summaries = summarize_hands(hand_records)
    rendered = []

    plots = [
        ("win_rates.png", lambda path: plot_win_rates(hand_records, path)),
        (
            "reward_comparison.png",
            lambda path: plot_reward_comparison(summaries, path),
        ),
        (
            "result_breakdown.png",
            lambda path: plot_result_breakdown(summaries, path),
        ),
        ("bankrolls.png", lambda path: plot_bankrolls(hand_records, path)),
        (
            "epsilon_schedule.png",
            lambda path: plot_epsilon_schedule(hand_records, path),
        ),
    ]
    if loss_records:
        plots.insert(0, ("loss.png", lambda path: plot_loss_curves(loss_records, path)))

    for filename, renderer in plots:
        path = run_store.plot_path(filename)
        renderer(path)
        rendered.append(path)
    rendered.extend(render_network_plots(run_store))
    return rendered
