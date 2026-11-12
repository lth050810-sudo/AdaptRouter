"""Explicit local-only loading; training and routing never call expert APIs."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from .native.startlux import StartLuxTrainable
from .observation import configure_tokenizer

BASE_ID = "startlux-models/StartLux-Decision-v1.1-0.8B-Instant"
BASE_REVISION = "6496ae641cc71b149ce9dad69bd97f2a44e7bf11"
GLOBAL_CHECKPOINT_SHA256 = (
    "f79c6be7c863a1efc299be20e4ce2b8d15ba0203961f6bfd5324a134231324e5"
)


def load_training_policy(base_path, source_dir, device, max_input_tokens):
    configure_tokenizer(base_path)
    result = StartLuxTrainable.from_pretrained(
        base_path,
        source_dir=source_dir,
        base_id=BASE_ID,
        base_revision=BASE_REVISION,
        device=device,
        max_input_tokens=max_input_tokens,
        max_images=16,
    )
    if any(
        not p.requires_grad or p.dtype != torch.float32 for p in result.parameters()
    ):
        raise RuntimeError("Full FP32 parameter contract violated")
    return result


def load_release_policy(model_dir, source_dir, device):
    """Load the published lossless FP32 global checkpoint with its native readout."""
    from transformers import AutoConfig, AutoTokenizer, AutoProcessor
    import transformers

    root = Path(model_dir).resolve()
    provenance = json.loads((root / "release_provenance.json").read_text())
    if provenance["base_model"] != {"id": BASE_ID, "revision": BASE_REVISION}:
        raise ValueError("Release has another base model")
    if provenance["source_checkpoint"]["sha256"] != GLOBAL_CHECKPOINT_SHA256:
        raise ValueError("This loader expects the published Table 3 global checkpoint")
    files = json.loads((root / "release_files.json").read_text())["files"]
    weights = set(
        json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ].values()
    )
    required = weights | {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "decision_config.json",
        "model.safetensors.index.json",
        "chat_template.jinja",
        "merges.txt",
        "vocab.json",
        "video_preprocessor_config.json",
    }
    if not required <= set(files):
        raise ValueError("Release manifest is missing model assets")
    verified = {}
    for name in sorted(required):
        item = (root / name).resolve()
        if (
            not item.is_relative_to(root)
            or not item.is_file()
            or item.stat().st_size != files[name]["bytes"]
        ):
            raise ValueError("Missing release asset: " + name)
        h = hashlib.sha256()
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != files[name]["sha256"]:
            raise ValueError("Release digest mismatch: " + name)
        verified[name] = h.hexdigest()
    config = AutoConfig.from_pretrained(root, local_files_only=True)
    if config.model_type != "qwen3_5" or config.jev_prompt_attention != "bidirectional":
        raise ValueError("Native Instant configuration required")
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        root,
        local_files_only=True,
        dtype=torch.float32,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    processor = AutoProcessor.from_pretrained(root, local_files_only=True)
    configure_tokenizer(root)
    result = StartLuxTrainable(
        model,
        tokenizer=tokenizer,
        processor=processor,
        source_dir=source_dir,
        decision_config=json.loads((root / "decision_config.json").read_text()),
        base_id=BASE_ID,
        base_revision=BASE_REVISION,
        max_input_tokens=8192,
        max_images=16,
        evidence_kind="verified_hf_global_release",
    )
    result.download_provenance = {
        "release_source_checkpoint_sha256": GLOBAL_CHECKPOINT_SHA256,
        "verified_files": verified,
    }
    return result
