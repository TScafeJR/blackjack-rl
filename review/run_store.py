import json
import os
from datetime import datetime
from typing import List, Optional


def latest_run_path(base_dir: str = "runs") -> str:
    if not os.path.isdir(base_dir):
        raise Exception(f"No runs directory at {base_dir}")
    run_names = sorted(
        name
        for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
    )
    if not run_names:
        raise Exception(f"No runs found in {base_dir}")
    return os.path.join(base_dir, run_names[-1])


class RunStore:
    def __init__(self, **kwargs):
        base_dir = kwargs.get("base_dir", "runs")
        run_name = kwargs.get("run_name")
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_path = os.path.join(base_dir, run_name)
        self.plots_path = os.path.join(self.run_path, "plots")
        os.makedirs(self.plots_path, exist_ok=True)

    def append_jsonl(self, filename: str, records: List[dict]) -> None:
        if not records:
            return
        path = os.path.join(self.run_path, filename)
        with open(path, "a", encoding="utf-8") as jsonl_file:
            for record in records:
                jsonl_file.write(json.dumps(record) + "\n")

    def append_hands(self, records: List[dict]) -> None:
        self.append_jsonl("metrics.jsonl", records)

    def append_losses(self, records: List[dict]) -> None:
        self.append_jsonl("loss.jsonl", records)

    def read_json(self, filename: str) -> Optional[dict]:
        path = os.path.join(self.run_path, filename)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as json_file:
            return json.load(json_file)

    def read_jsonl(self, filename: str) -> List[dict]:
        path = os.path.join(self.run_path, filename)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as jsonl_file:
            return [json.loads(line) for line in jsonl_file if line.strip()]

    def save_json(self, filename: str, payload: dict) -> None:
        path = os.path.join(self.run_path, filename)
        with open(path, "w", encoding="utf-8") as json_file:
            json.dump(payload, json_file, indent=4)

    def save_config(self, config_payload: dict) -> None:
        self.save_json("config.json", config_payload)

    def save_weights(self, kind: str, weights: list) -> None:
        self.save_json(f"weights_{kind}.json", {"weights": weights})

    def save_report(self, text: str) -> None:
        path = os.path.join(self.run_path, "report.txt")
        with open(path, "w", encoding="utf-8") as report_file:
            report_file.write(text + "\n")

    def plot_path(self, filename: str) -> str:
        return os.path.join(self.plots_path, filename)
