"""Offline full-information feedback schema used by the Table 3 trainer."""

from __future__ import annotations
import json
import math
from pathlib import Path
from .reward import EXPERTS


def read_rows(path, require_labels=True):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    else:
        value = json.loads(path.read_text())
        rows = value if isinstance(value, list) else value["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("A nonempty row list is required")
    seen = set()
    group_splits = {}
    for row in rows:
        key = (row["dataset"], row["task_id"])
        if key in seen:
            raise ValueError("Duplicate task_id within a dataset")
        seen.add(key)
        if require_labels and tuple(row["expert_ids"]) != EXPERTS:
            raise ValueError("Five exact expert IDs/order differ")
        if require_labels and (
            len(row["qualities"]) != 5
            or any(
                type(q) not in (int, float) or not math.isfinite(q) or not 0 <= q <= 1
                for q in row["qualities"]
            )
        ):
            raise ValueError("Missing/invalid quality")
        if require_labels and (
            len(row["costs_usd"]) != 5
            or any(
                c is not None
                and (type(c) not in (int, float) or not math.isfinite(c) or c < 0)
                for c in row["costs_usd"]
            )
        ):
            raise ValueError("Invalid cost")
        if row["split"] not in ("train", "validation", "test"):
            raise ValueError("Explicit split required")
        group = row["group_id"]
        prior = group_splits.setdefault(group, row["split"])
        if prior != row["split"]:
            raise ValueError("Document/image family crosses split")
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError("Complete input is missing")
    return rows


def validate_schema(rows, phase, expected_datasets=10, expected_refit_rows=1328):
    names = {r["dataset"] for r in rows if r["split"] != "test"}
    if len(names) != expected_datasets:
        raise ValueError(
            f"Need {expected_datasets} training datasets, got {sorted(names)}"
        )
    if (
        phase == "refit"
        and len([r for r in rows if r["split"] != "test"]) != expected_refit_rows
    ):
        raise ValueError(
            f"Final refit requires {expected_refit_rows} collected questions"
        )
    for name in names:
        for split in ("train", "validation"):
            if not any(r["dataset"] == name and r["split"] == split for r in rows):
                raise ValueError("Every dataset needs train and validation groups")
