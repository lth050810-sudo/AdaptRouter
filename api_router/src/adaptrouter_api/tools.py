"""Explicit asset download and CPU-only feedback/recipe preparation."""

from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from .data import read_rows, validate_schema
from .reward import fit_cost_scale

BASE_ID = "startlux-models/StartLux-Decision-v1.1-0.8B-Instant"
BASE_REVISION = "6496ae641cc71b149ce9dad69bd97f2a44e7bf11"
ARCHIVED_FEEDBACK_SHA256 = (
    "8861d97f08a6eac980168a4a52e585ace987a85bd2b7eb8ae575ae8dd8740709"
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_base(out):
    """Network is used only when this explicit setup command is invoked."""
    from huggingface_hub import HfApi, snapshot_download

    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(BASE_ID, revision=BASE_REVISION, files_metadata=True)
    assert info.sha == BASE_REVISION
    names = [
        f.rfilename
        for f in info.siblings
        if "/" not in f.rfilename
        and f.rfilename != ".gitattributes"
        and (
            f.rfilename.endswith((".json", ".safetensors", ".txt", ".jinja", ".md"))
            or f.rfilename in ("LICENSE", "NOTICE")
        )
    ]
    snapshot_download(
        BASE_ID, revision=BASE_REVISION, allow_patterns=names, local_dir=root
    )
    rows = []
    for item in info.siblings:
        name = item.rfilename
        if name not in names:
            continue
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Invalid upstream path")
        digest = sha(path)
        if path.stat().st_size != item.size:
            raise ValueError("Downloaded size mismatch: " + name)
        if item.lfs and item.lfs.sha256 != digest:
            raise ValueError("Downloaded upstream digest mismatch: " + name)
        if not item.lfs:
            blob = hashlib.sha1(
                f"blob {path.stat().st_size}\0".encode() + path.read_bytes()
            ).hexdigest()
            if blob != item.blob_id:
                raise ValueError("Downloaded upstream Git blob mismatch: " + name)
        rows.append(
            {
                "file": name,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "upstream_sha256": item.lfs.sha256 if item.lfs else None,
            }
        )
    manifest = {
        "complete": True,
        "model_id": BASE_ID,
        "revision": BASE_REVISION,
        "files": rows,
    }
    (root / "DOWNLOAD_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    from .native.model_assets import verify_model_snapshot

    result = verify_model_snapshot(root, BASE_ID, BASE_REVISION)
    print(
        json.dumps(
            {
                "downloaded": True,
                "directory": str(root),
                "verified_files": result["verified_file_count"],
            }
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="adaptrouter-api")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-data")
    validate.add_argument("--data", required=True)
    validate.add_argument("--unlabeled", action="store_true")
    prepare = commands.add_parser("prepare-refit")
    prepare.add_argument("--data", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--lambda-cost", type=float, default=0.02)
    prepare.add_argument("--epochs", type=int, default=3)
    prepare.add_argument("--expected-datasets", type=int, default=10)
    prepare.add_argument("--expected-refit-rows", type=int, default=1328)
    download = commands.add_parser("download-base")
    download.add_argument("--out", default=".artifacts/startlux-base")
    args = parser.parse_args(argv)
    if args.command == "download-base":
        download_base(args.out)
        return
    rows = read_rows(args.data, require_labels=not getattr(args, "unlabeled", False))
    if args.command == "validate-data":
        counts = {
            split: sum(r["split"] == split for r in rows)
            for split in ("train", "validation", "test")
        }
        print(
            json.dumps(
                {
                    "schema_valid": True,
                    "rows": len(rows),
                    "split_counts": counts,
                    "datasets": sorted({r["dataset"] for r in rows}),
                    "matches_archived_feedback_sha256": sha(args.data)
                    == ARCHIVED_FEEDBACK_SHA256,
                }
            )
        )
        return
    validate_schema(rows, "refit", args.expected_datasets, args.expected_refit_rows)
    if not math.isfinite(args.lambda_cost) or args.lambda_cost <= 0 or args.epochs < 1:
        raise ValueError(
            "The Table 3 global recipe requires a finite positive cost coefficient and epoch count"
        )
    train = [r for r in rows if r["split"] == "train"]
    scale = fit_cost_scale(
        {
            "split": "train",
            "dataset": r["dataset"],
            "arms": [{"cost_usd": c} for c in r["costs_usd"]],
        }
        for r in train
    )
    value = {
        "family": "startlux",
        "epoch": args.epochs,
        "lambda_cost": args.lambda_cost,
        "data_sha256": sha(args.data),
        "cost_scale": scale,
        "router_input_policy": "question_length_head_tail",
        "selection_rule": "fixed_user_supplied_recipe; train-only cost scale; no test outcomes",
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {
                "prepared": True,
                "selection": str(target),
                "training_cost_scale": scale,
                "lambda_cost": args.lambda_cost,
                "epochs": args.epochs,
            }
        )
    )
