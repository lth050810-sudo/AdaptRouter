"""Closed expert quality/dollar objective; no parameter-count or thinking bonuses.

Prices are the user's fixed experimental tariff, not an assertion of provider billing.
All operations here are offline; no model calls or changes to collection state.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable, Sequence

EXPERTS = (
    "gpt-6.1-terra",
    "deepseek-flash",
    "claude-sonnet-5-5",
    "gemini-4-argon",
    "kimi-k3.1-pro",
)
PRICES_PER_MILLION = {
    "gpt-6.1-terra": (1.00, 5.00),
    "deepseek-flash": (0.15, 0.60),
    "claude-sonnet-5-5": (2.00, 10.00),
    "gemini-4-argon": (2.00, 10.00),
    "kimi-k3.1-pro": (2.00, 10.00),
}
LAMBDA_GRID = (0.0, 0.0025, 0.005, 0.01, 0.02)
MAX_QUALITY_LOSS = 0.005  # 0.5 percentage points, selected on validation only.


def _number(x, name, lower=0.0, upper=math.inf):
    if type(x) not in (int, float) or not math.isfinite(x) or not lower <= x <= upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    return float(x)


def usage_cost_usd(
    model: str, input_tokens: int | None, output_tokens: int | None
) -> float | None:
    """One expert attempt's input/output cost. Output includes charged reasoning tokens.

    Callers must not add reasoning_tokens again when already in output_tokens.
    Missing either counter is unknown, including timeout failures without usage.
    """
    if model not in PRICES_PER_MILLION:
        raise ValueError(f"No user tariff for exact model {model!r}")
    if input_tokens is None or output_tokens is None:
        return None
    for value in (input_tokens, output_tokens):
        if type(value) is not int or value < 0:
            raise ValueError("Token counts must be nonnegative integers or null")
    pin, pout = PRICES_PER_MILLION[model]
    return (input_tokens * pin + output_tokens * pout) / 1_000_000.0


def fit_cost_scale(train_rows: Iterable[dict]) -> float:
    """One scalar: mean of task-specific mean known arm costs, TRAIN ONLY.

    It rescales the optimization coefficient; experimental cost stays raw USD.
    A task with more sampled questions cannot dominate this normalization.
    """
    grouped = defaultdict(list)
    seen = 0
    for row in train_rows:
        if row.get("split") != "train":
            raise ValueError("Cost-scale fitting accepts training split only")
        seen += 1
        for arm in row["arms"]:
            value = arm.get("cost_usd")
            if value is not None:
                grouped[row["dataset"]].append(_number(value, "cost_usd"))
    means = [sum(values) / len(values) for values in grouped.values() if values]
    if not seen or not means or sum(means) <= 0:
        raise ValueError("Positive observed training cost required")
    return sum(means) / len(means)


def reward_vector(
    qualities: Sequence[float],
    costs_usd: Sequence[float | None],
    lambda_cost: float,
    cost_scale: float,
) -> tuple[list[float], list[bool]]:
    """R_i = q_i - lambda * c_i / fixed_train_scale for fully cost-observed rows.

    With ANY missing arm cost the WHOLE question uses quality-only rewards.
    This prevents an unknown-cost expert from obtaining an artificial cheap reward.
    Individual quality labels remain usable; callers must log cost-applied counts.
    """
    if len(qualities) != len(EXPERTS) or len(costs_usd) != len(EXPERTS):
        raise ValueError("Exactly five aligned expert outcomes required")
    q = [_number(x, "quality", upper=1.0) for x in qualities]
    lam = _number(lambda_cost, "lambda_cost")
    scale = _number(cost_scale, "cost_scale", lower=1e-20)
    mask = [c is not None for c in costs_usd]
    costs = [None if c is None else _number(c, "cost_usd") for c in costs_usd]
    if not all(mask):
        return q, mask
    return [value - lam * c / scale for value, c in zip(q, costs)], mask


def quality_first_target(
    qualities: Sequence[float],
    costs_usd: Sequence[float | None],
    lambda_cost: float,
    cost_scale: float,
    temperature: float = 0.05,
) -> dict:
    rewards, mask = reward_vector(qualities, costs_usd, lambda_cost, cost_scale)
    temp = _number(temperature, "temperature", lower=1e-20)
    maximum = max(rewards)
    mass = [math.exp((r - maximum) / temp) for r in rewards]
    total = sum(mass)
    return {
        "target_probs": [x / total for x in mass],
        "rewards": rewards,
        "cost_applied": all(mask) and lambda_cost > 0,
        "cost_mask": mask,
        "quality": list(qualities),
        "cost_usd": list(costs_usd),
    }


def task_macro_weights(rows: Sequence[dict]) -> list[float]:
    """Weights sum to one; each dataset has equal total weight."""
    counts = defaultdict(int)
    for row in rows:
        counts[row["dataset"]] += 1
    if not rows:
        raise ValueError("No rows")
    return [1.0 / (len(counts) * counts[row["dataset"]]) for row in rows]


def select_validation_candidate(
    candidates: Sequence[dict], max_quality_loss=MAX_QUALITY_LOSS
) -> dict:
    """Pick cheapest complete-cost candidate within 0.5pp of best validation quality.

    Candidates must bind exactly the SAME validation item IDs and dataset coverage.
    If no cost-complete qualified candidate exists, pick best quality and explicitly
    state that no cost-based selection was possible. Never consume test metrics.
    """
    if not candidates:
        raise ValueError("No validation candidates")
    tolerance = _number(max_quality_loss, "max_quality_loss", upper=1.0)
    first_ids = candidates[0].get("validation_ids_sha256")
    first_sets = candidates[0].get("dataset_names")
    if not first_ids or not first_sets:
        raise ValueError("Validation identity and dataset coverage required")
    for c in candidates:
        if c.get("split") != "validation" or c.get("quality_complete") is not True:
            raise ValueError("Selection requires complete validation scores")
        if (
            c.get("validation_ids_sha256") != first_ids
            or c.get("dataset_names") != first_sets
        ):
            raise ValueError(
                "All candidates must use identical validation questions/datasets"
            )
        _number(c["macro_quality"], "macro_quality", upper=1.0)
    best_q = max(c["macro_quality"] for c in candidates)
    qualified = [
        c for c in candidates if c["macro_quality"] + 1e-12 >= best_q - tolerance
    ]
    known = [
        c
        for c in qualified
        if c.get("cost_complete") is True and c.get("macro_cost_usd") is not None
    ]
    if known:
        for c in known:
            _number(c["macro_cost_usd"], "macro_cost_usd")
        winner = min(
            known,
            key=lambda c: (c["macro_cost_usd"], -c["macro_quality"], str(c["id"])),
        )
        rule = "min_macro_usd_within_validation_quality_tolerance"
    else:
        winner = min(qualified, key=lambda c: (-c["macro_quality"], str(c["id"])))
        rule = "quality_only_cost_incomplete"
    return {
        "selected": dict(winner),
        "rule": rule,
        "best_macro_quality": best_q,
        "quality_floor": best_q - tolerance,
        "max_quality_loss": tolerance,
        "qualified_candidate_ids": [c["id"] for c in qualified],
        "test_used_for_selection": False,
    }
