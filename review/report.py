import argparse
import json
import os
from typing import List

from .metrics import format_summary, summarize_hands
from .plots import render_all
from .run_store import RunStore, latest_run_path


def open_run_store(run_path: str) -> RunStore:
    base_dir, run_name = os.path.split(os.path.normpath(run_path))
    return RunStore(base_dir=base_dir or ".", run_name=run_name)


def load_config(store: RunStore) -> dict:
    config_path = os.path.join(store.run_path, "config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def build_report(run_path: str) -> str:
    store = open_run_store(run_path)
    hand_records = store.read_jsonl("metrics.jsonl")
    if not hand_records:
        raise Exception(f"No metrics found in {run_path}")
    config = load_config(store)
    lines = [f"run: {store.run_path}"]
    if config:
        lines.append(
            f"agents: {config.get('agents', '?')}  tables: {config.get('tables', '?')}  "
            f"workers: {config.get('workers', '?')}  hands: {config.get('hands', '?')}  "
            f"seed: {config.get('seed', '?')}"
        )
    lines.append("")
    lines.append(format_summary(summarize_hands(hand_records)))
    return "\n".join(lines)


def build_comparison(run_paths: List[str]) -> str:
    rewards_by_kind = {}
    run_names = []
    for run_path in run_paths:
        store = open_run_store(run_path)
        run_name = os.path.basename(store.run_path)
        run_names.append(run_name)
        summaries = summarize_hands(store.read_jsonl("metrics.jsonl"))
        for kind, summary in summaries.items():
            rewards_by_kind.setdefault(kind, {})[run_name] = summary["avg_reward"]

    header = f"{'agent':<14}" + "".join(f"{name:>18}" for name in run_names)
    lines = ["avg reward per hand across runs", header, "-" * len(header)]
    for kind in sorted(rewards_by_kind):
        row = f"{kind:<14}"
        for run_name in run_names:
            reward = rewards_by_kind[kind].get(run_name)
            row += f"{reward:>18.4f}" if reward is not None else f"{'-':>18}"
        lines.append(row)
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Summarize blackjack training runs")
    parser.add_argument("run_paths", nargs="*")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args(argv)

    run_paths = args.run_paths if args.run_paths else [latest_run_path()]
    for run_path in run_paths:
        print(build_report(run_path))
        print()
        if args.plots:
            rendered = render_all(open_run_store(run_path))
            for path in rendered:
                print(f"plot: {path}")
            print()

    if len(run_paths) > 1:
        print(build_comparison(run_paths))


if __name__ == "__main__":
    main()
