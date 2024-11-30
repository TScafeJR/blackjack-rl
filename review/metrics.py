from typing import Dict, List

WIN_RESULTS = ["PLAYER_WIN", "DEALER_BUST"]
LOSS_RESULTS = ["DEALER_WIN", "BUST"]
DEFAULT_BASE_BET = 10


def record_base_bet(record: dict) -> int:
    return record.get("base_bet") or DEFAULT_BASE_BET


def record_bet_units(record: dict) -> float:
    units = record.get("bet_units")
    if units:
        return float(units)
    return record["bet"] / record_base_bet(record)


def record_net_units(record: dict) -> float:
    return record["reward"] * record["bet"] / record_base_bet(record)


def records_by_kind(hand_records: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for record in hand_records:
        grouped.setdefault(record["kind"], []).append(record)
    return grouped


def tail_records(hand_records: List[dict], share: float) -> List[dict]:
    if share >= 1.0:
        return hand_records
    kept: List[dict] = []
    for records in records_by_kind(hand_records).values():
        keep_count = max(1, int(len(records) * share))
        kept.extend(records[-keep_count:])
    return kept


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
                "net_units": 0.0,
                "wagered_units": 0.0,
            }
        summary = summaries[kind]
        summary["hands"] += 1
        summary["total_reward"] += record["reward"]
        summary["net_profit"] += record["reward"] * record["bet"]
        summary["net_units"] += record_net_units(record)
        summary["wagered_units"] += record_bet_units(record)
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
        summary["avg_units"] = summary["net_units"] / hands
        summary["avg_bet_units"] = summary["wagered_units"] / hands
        summary["win_rate"] = (summary["wins"] + summary["blackjacks"]) / hands
        summary["push_rate"] = summary["pushes"] / hands
        summary["bust_rate"] = summary["busts"] / hands
    return summaries


COUNT_BUCKET_LIMIT = 5


def count_bucket(record: dict) -> int:
    bucket = round(record.get("true_count", 0.0))
    return int(max(-COUNT_BUCKET_LIMIT, min(COUNT_BUCKET_LIMIT, bucket)))


def summarize_by_count(hand_records: List[dict]) -> Dict[str, Dict[int, dict]]:
    by_kind: Dict[str, Dict[int, dict]] = {}
    for record in hand_records:
        if "true_count" not in record:
            continue
        buckets = by_kind.setdefault(record["kind"], {})
        bucket = buckets.setdefault(
            count_bucket(record), {"hands": 0, "net_units": 0.0, "wagered_units": 0.0}
        )
        bucket["hands"] += 1
        bucket["net_units"] += record_net_units(record)
        bucket["wagered_units"] += record_bet_units(record)
    for buckets in by_kind.values():
        for bucket in buckets.values():
            bucket["avg_units"] = bucket["net_units"] / bucket["hands"]
            bucket["avg_bet_units"] = bucket["wagered_units"] / bucket["hands"]
            bucket["ev_per_unit"] = bucket["net_units"] / bucket["wagered_units"]
    return by_kind


def format_count_table(hand_records: List[dict]) -> str:
    by_kind = summarize_by_count(hand_records)
    if not by_kind:
        return "no true count recorded in this run"
    buckets = list(range(-COUNT_BUCKET_LIMIT, COUNT_BUCKET_LIMIT + 1))
    header = f"{'agent':<14}{'metric':<12}" + "".join(
        f"{bucket:>+8}" for bucket in buckets
    )
    lines = ["bet size and profit by true count", header, "-" * len(header)]
    for kind in sorted(by_kind):
        for label, key in [("avg bet", "avg_bet_units"), ("units/hand", "avg_units")]:
            row = f"{kind:<14}{label:<12}"
            for bucket in buckets:
                stats = by_kind[kind].get(bucket)
                row += f"{stats[key]:>8.2f}" if stats else f"{'-':>8}"
            lines.append(row)
    return "\n".join(lines)


def format_summary(summaries: Dict[str, dict]) -> str:
    header = (
        f"{'agent':<14}{'hands':>8}{'win %':>8}{'push %':>8}{'bust %':>8}"
        f"{'bj %':>7}{'ev/unit':>10}{'avg bet':>9}{'units/hand':>12}"
        f"{'net profit':>12}{'rebuys':>8}"
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
            f"{summary['avg_reward']:>10.4f}"
            f"{summary['avg_bet_units']:>9.2f}"
            f"{summary['avg_units']:>12.4f}"
            f"{summary['net_profit']:>12.0f}"
            f"{summary['rebuys']:>8}"
        )
    return "\n".join(lines)
