from typing import Dict, List

WIN_RESULTS = ["PLAYER_WIN", "DEALER_BUST"]
LOSS_RESULTS = ["DEALER_WIN", "BUST"]


class MetricsCollector:
    def __init__(self):
        self.hand_records: List[dict] = []
        self.loss_records: List[dict] = []

    def record_hands(self, records: List[dict]) -> None:
        self.hand_records.extend(records)

    def record_loss(self, record: dict) -> None:
        self.loss_records.append(record)


def summarize_hands(hand_records: List[dict]) -> Dict[str, dict]:
    summaries: Dict[str, dict] = {}
    for record in hand_records:
        kind = record["kind"]
        if kind not in summaries:
            summaries[kind] = {
                "hands": 0,
                "total_reward": 0.0,
                "net_profit": 0.0,
                "wins": 0,
                "blackjacks": 0,
                "pushes": 0,
                "busts": 0,
                "losses": 0,
                "rebuys": 0,
            }
        summary = summaries[kind]
        summary["hands"] += 1
        summary["total_reward"] += record["reward"]
        summary["net_profit"] += record["reward"] * record["bet"]
        result = record["result"]
        if result in WIN_RESULTS:
            summary["wins"] += 1
        elif result == "PLAYER_BLACKJACK":
            summary["blackjacks"] += 1
        elif result == "PUSH":
            summary["pushes"] += 1
        elif result == "BUST":
            summary["busts"] += 1
        if result in LOSS_RESULTS:
            summary["losses"] += 1
        summary["rebuys"] = max(summary["rebuys"], record.get("rebuys", 0))

    for summary in summaries.values():
        hands = summary["hands"]
        summary["avg_reward"] = summary["total_reward"] / hands
        summary["win_rate"] = (summary["wins"] + summary["blackjacks"]) / hands
        summary["push_rate"] = summary["pushes"] / hands
        summary["bust_rate"] = summary["busts"] / hands
    return summaries


def format_summary(summaries: Dict[str, dict]) -> str:
    header = (
        f"{'agent':<14}{'hands':>8}{'win %':>8}{'push %':>8}{'bust %':>8}"
        f"{'bj %':>7}{'avg reward':>12}{'net profit':>12}{'rebuys':>8}"
    )
    lines = [header, "-" * len(header)]
    for kind in sorted(summaries):
        summary = summaries[kind]
        hands = summary["hands"]
        lines.append(
            f"{kind:<14}{hands:>8}"
            f"{summary['win_rate'] * 100:>7.1f}%"
            f"{summary['push_rate'] * 100:>7.1f}%"
            f"{summary['bust_rate'] * 100:>7.1f}%"
            f"{summary['blackjacks'] / hands * 100:>6.1f}%"
            f"{summary['avg_reward']:>12.4f}"
            f"{summary['net_profit']:>12.0f}"
            f"{summary['rebuys']:>8}"
        )
    return "\n".join(lines)
