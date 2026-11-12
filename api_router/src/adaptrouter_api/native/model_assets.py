"""Read-only validation of both generations of pinned local model manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _digest(path: Path, algorithm: str) -> str:
    if algorithm not in {"sha256", "git_blob_sha1"}:
        raise ValueError(f"Unsupported manifest digest: {algorithm}")
    h = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
    if algorithm == "git_blob_sha1":
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_model_snapshot(
    path: str | Path, model_id: str, revision: str, verify_hashes: bool = True
) -> dict[str, Any]:
    """Check identity, every listed file and weight-index coverage; never download.

    ``verify_hashes=False`` is explicitly size-only evidence, unsuitable as a
    replacement for the execution preflight's full hash verification.
    """
    root = Path(path).resolve()
    modern, original = (
        root / "DOWNLOAD_MANIFEST.json",
        root / "JEV_DOWNLOAD_MANIFEST.json",
    )
    manifest_path = modern if modern.is_file() else original
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest_path == modern:
        if manifest.get("complete") is not True or manifest.get("model_id") != model_id:
            raise ValueError(
                "Downloaded manifest is incomplete or identifies another checkpoint"
            )
        rows = [
            dict(r, hash_type="sha256", hash=r["sha256"])
            for r in manifest.get("files", [])
        ]
    else:
        if manifest.get("repo_id") != model_id:
            raise ValueError("Original manifest identifies another checkpoint")
        rows = manifest.get("verified_files", [])
    if manifest.get("revision") != revision or not revision or not rows:
        raise ValueError("Pinned model revision or file manifest mismatch")
    names = [r["file"] for r in rows]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate manifest file entries")
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if model_id.startswith("startlux-models/"):
        required |= {"decision_config.json", "preprocessor_config.json"}
    index = root / "model.safetensors.index.json"
    if index.is_file():
        required |= {index.name} | set(
            json.loads(index.read_text())["weight_map"].values()
        )
    else:
        required.add("model.safetensors")
    if not required <= set(names):
        raise ValueError(
            "Manifest lacks required configuration or indexed model shards"
        )
    verified = []
    for row in rows:
        item = (root / row["file"]).resolve()
        if (
            not item.is_relative_to(root)
            or not item.is_file()
            or type(row["bytes"]) is not int
            or item.stat().st_size != row["bytes"]
        ):
            raise ValueError(f"Model file missing or size mismatch: {row['file']}")
        if row["hash_type"] not in {"sha256", "git_blob_sha1"} or not row.get("hash"):
            raise ValueError("File lacks a supported recorded digest")
        digest = _digest(item, row["hash_type"]) if verify_hashes else None
        if verify_hashes and (
            digest != row["hash"]
            or (row.get("upstream_sha256") and digest != row["upstream_sha256"])
        ):
            raise ValueError(f"Model file digest mismatch: {row['file']}")
        verified.append(
            {
                "file": row["file"],
                "bytes": row["bytes"],
                "hash_type": row["hash_type"],
                "expected_hash": row["hash"],
                "actual_hash": digest,
            }
        )
    return {
        "model_id": model_id,
        "revision": revision,
        "snapshot": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "verified_file_count": len(rows),
        "verified_bytes": sum(r["bytes"] for r in rows),
        "verification": "all_file_hashes_and_sizes" if verify_hashes else "sizes_only",
        "files": verified,
    }
