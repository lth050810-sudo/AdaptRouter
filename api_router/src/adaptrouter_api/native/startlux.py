"""Full-parameter, native-image StartLux Instant letter readout.

Loads explicitly supplied, clean pinned upstream source; never imports the audit
snapshot and never downloads weights. Training and decide() share one forward.
The original causal LM forward is not used in place of Instant attention.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types
from typing import Any

import torch
from torch import nn

from .decision import DecisionRequest, DecisionResult
from .image_io import image_data_uri

UPSTREAM_COMMIT = "cb1d11f1ac9e9b43b95ef3a6468fa29b607c40fd"
PROFILE_FIELDS = (
    "task_success_rate",
    "mean_input_div10000",
    "mean_output_div10000",
    "tasks_div10",
    "reserved_4",
    "reserved_5",
)
REGISTERED_BASES = {
    "startlux-models/StartLux-Decision-v1.1-0.8B-Instant": "6496ae641cc71b149ce9dad69bd97f2a44e7bf11",
}


def verify_download_manifest(
    model_path: str | Path, base_id: str, revision: str
) -> dict[str, Any]:
    """Verify identity, completeness, sizes and all downloaded file digests."""
    from .model_assets import verify_model_snapshot

    provenance = verify_model_snapshot(
        model_path, base_id, revision, verify_hashes=True
    )
    required = {"decision_config.json", "preprocessor_config.json"}
    if not required <= {row["file"] for row in provenance["files"]}:
        raise ValueError("StartLux manifest lacks decision and processor configuration")
    return provenance


def load_upstream(source_dir: str | Path) -> types.SimpleNamespace:
    source = Path(source_dir).resolve()
    files = (
        "startlux_decide/prompt.py",
        "startlux_decide/images.py",
        "startlux_decide/backends/bidirectional.py",
    )
    manifest_path = source / "UPSTREAM.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        revision = manifest.get("revision")
        if revision != UPSTREAM_COMMIT:
            raise ValueError(
                "Bundled upstream revision differs from the training revision"
            )
        hashes = {
            p: hashlib.sha256((source / p).read_bytes()).hexdigest() for p in files
        }
        if hashes != manifest.get("files"):
            raise ValueError("Bundled native prompt or forward implementation changed")
    else:
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != UPSTREAM_COMMIT:
            raise ValueError("StartLux source revision is not the registered commit")
        if subprocess.run(
            ["git", "-C", str(source), "diff", "--quiet", "HEAD", "--", *files],
            check=False,
        ).returncode:
            raise ValueError(
                "StartLux forward/prompt source has unregistered modifications"
            )
        hashes = {
            p: hashlib.sha256((source / p).read_bytes()).hexdigest() for p in files
        }
    # Namespace packages avoid executing upstream CLI/server initialization.
    package_name = (
        "_jev_startlux_" + hashlib.sha256(str(source).encode()).hexdigest()[:12]
    )
    for suffix, directory in (
        ("", source / "startlux_decide"),
        (".backends", source / "startlux_decide/backends"),
    ):
        key = package_name + suffix
        if key not in sys.modules:
            package = types.ModuleType(key)
            package.__path__ = [str(directory)]
            sys.modules[key] = package
    modules = {
        name: importlib.import_module(package_name + "." + module)
        for name, module in (
            ("prompt", "prompt"),
            ("images", "images"),
            ("bidirectional", "backends.bidirectional"),
        )
    }
    return types.SimpleNamespace(
        **modules,
        record={"commit": revision, "source_dir": str(source), "sha256": hashes},
    )


class StartLuxTrainable(nn.Module):
    provider = "local"
    max_output_tokens = 0
    max_retries = 0

    def __init__(
        self,
        model: nn.Module,
        *,
        tokenizer: Any,
        processor: Any,
        source_dir: str | Path,
        decision_config: dict[str, Any],
        base_id: str,
        base_revision: str,
        training_mode: str = "full",
        max_input_tokens: int = 2048,
        max_images: int = 2,
        min_pixels: int = 65536,
        max_pixels: int = 262144,
        profile_fields: tuple[str, ...] = PROFILE_FIELDS,
        evidence_kind: str = "unverified_components",
        input_protocol_id: str = "whole_question_v1",
    ):
        super().__init__()
        if training_mode != "full":
            raise ValueError("The current Router requires full-parameter training")
        if (
            type(max_input_tokens) is not int
            or max_input_tokens < 1
            or not 0 <= max_images <= 16
        ):
            raise ValueError("Finite context/image limits required")
        if decision_config.get("prompt_attention") != "bidirectional":
            raise ValueError(
                "This Router requires the Instant bidirectional checkpoint"
            )
        if not 0 < min_pixels <= max_pixels:
            raise ValueError("Invalid processor pixel bounds")
        if input_protocol_id != "whole_question_v1":
            raise ValueError(
                "Only the whole_question_v1 StartLux input protocol is active"
            )
        self.input_protocol_id = input_protocol_id
        self.upstream = load_upstream(source_dir)
        self.base_id, self.base_revision = base_id, base_revision
        self.model_id, self.revision = base_id, base_revision
        self.tokenizer, self.processor = tokenizer, processor
        self.decision_config = decision_config
        self.max_input_tokens, self.max_images = max_input_tokens, max_images
        self.min_pixels, self.max_pixels = min_pixels, max_pixels
        self.profile_fields, self.evidence_kind = tuple(profile_fields), evidence_kind
        if len(set(self.profile_fields)) != len(self.profile_fields) or any(
            not isinstance(name, str) or not name for name in self.profile_fields
        ):
            raise ValueError("Profile field names must be unique nonempty strings")
        self.policy_version = 0
        self.training_kind = "untrained"
        self.training_mode = training_mode
        self.temperature = float(decision_config["temperature_by_type"]["choice"])
        if not torch.isfinite(torch.tensor(self.temperature)) or self.temperature <= 0:
            raise ValueError("Choice temperature must be finite and positive")
        letters = tuple(decision_config["letter_token_ids"])
        if len(letters) != 26 or len(set(letters)) != 26:
            raise ValueError("Need 26 distinct registered letter tokens")
        if (
            tokenizer is not None
            and tuple(self.upstream.prompt.check_tokenizer(tokenizer)) != letters
        ):
            raise ValueError(
                "Tokenizer letter readout differs from registered decision_config"
            )
        self.register_buffer(
            "letter_ids", torch.tensor(letters, dtype=torch.long), persistent=False
        )
        # FP32 master parameters keep small policy-gradient updates representable.
        # CUDA forwards use BF16 autocast; text, vision and the letter readout
        # all remain in the optimizer, with no adapter or frozen Router body.
        model.float().requires_grad_(True)
        model.eval()
        # The upstream check runs real forwards. On CUDA it allows only fused attention kernels, which reject
        # FP32 grouped-query inputs, so it runs under the same BF16 autocast as every Router forward.
        device = model.model.language_model.embed_tokens.weight.device
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            self.bidirectional_check = self.upstream.bidirectional.enable(
                model.model.language_model, tokens=8
            )
        self.model = model
        self.assert_full_trainable()
        self.initialization_usage = {
            "category": "synthetic_upstream_semantics_check",
            "input_tokens": 144,
            "output_tokens": 0,
            "model_forward_calls": 9,
            "note": "upstream enable: 9 forwards of batch=2, tokens=8; not deployment or benchmark usage",
        }

    def train(self, mode: bool = True):
        super().train(mode)
        # eval() controls dropout, not autograd. Collection and differentiable
        # replay retain the same deterministic Instant readout.
        self.model.eval()
        return self

    @property
    def base_model(self):
        return self.model

    @property
    def body(self):
        return self.base_model.model.language_model

    @property
    def device(self):
        return self.body.embed_tokens.weight.device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        source_dir: str | Path,
        base_id: str,
        base_revision: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        if kwargs.get("input_protocol_id", "whole_question_v1") != "whole_question_v1":
            raise ValueError(
                "Only the whole_question_v1 StartLux input protocol is active"
            )
        if REGISTERED_BASES.get(base_id) != base_revision:
            raise ValueError(
                "Base model/revision differs from a registered native Instant model"
            )
        if kwargs.get("training_mode", "full") != "full" or dtype != torch.float32:
            raise ValueError("Full Router training requires FP32 master parameters")
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer
        import transformers

        model_path = Path(model_path)
        download_provenance = verify_download_manifest(
            model_path, base_id, base_revision
        )
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        if (
            config.model_type != "qwen3_5"
            or getattr(config, "jev_prompt_attention", None) != "bidirectional"
        ):
            raise ValueError("Expected the registered Qwen3.5 Instant config")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model_class = getattr(transformers, config.architectures[0])
        model = model_class.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=dtype,
            device_map={"": device},
            attn_implementation="sdpa",
        )
        decision_config = json.loads((model_path / "decision_config.json").read_text())
        kwargs.setdefault("evidence_kind", "real_base_checkpoint")
        result = cls(
            model,
            tokenizer=tokenizer,
            processor=processor,
            source_dir=source_dir,
            decision_config=decision_config,
            base_id=base_id,
            base_revision=base_revision,
            **kwargs,
        )
        result.download_provenance = download_provenance
        return result

    def forward_encoded(
        self, encoded: dict[str, torch.Tensor], option_count: int
    ) -> torch.Tensor:
        """Scaled logits [1,N]; no inference_mode, no detached hidden states."""
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            return self._forward_encoded(encoded, option_count)

    def _forward_encoded(
        self, encoded: dict[str, torch.Tensor], option_count: int
    ) -> torch.Tensor:
        if type(option_count) is not int or not 1 <= option_count <= 26:
            raise ValueError("One joint readout supports at most 26 options")
        ids = encoded["input_ids"].to(self.device)
        if (
            ids.ndim != 2
            or ids.shape[0] != 1
            or not 0 < ids.shape[1] <= self.max_input_tokens
        ):
            raise ValueError(
                "Expected one bounded unpadded prompt; truncation is not implicit"
            )
        if "pixel_values" in encoded:
            inner = self.base_model.model
            pixels = encoded["pixel_values"].to(self.device)
            grid = encoded["image_grid_thw"].to(self.device)
            embeds = inner.get_input_embeddings()(ids)
            features = inner.get_image_features(
                pixels, grid, return_dict=True
            ).pooler_output
            features = torch.cat(list(features), 0).to(embeds.device, embeds.dtype)
            mask, _ = inner.get_placeholder_mask(
                ids, inputs_embeds=embeds, image_features=features
            )
            embeds = embeds.masked_scatter(mask, features)
            token_types = encoded.get(
                "mm_token_type_ids",
                (ids == self.base_model.config.image_token_id).int(),
            ).to(self.device)
            positions, _ = inner.get_rope_index(
                ids,
                image_grid_thw=grid,
                attention_mask=None,
                mm_token_type_ids=token_types,
            )
            hidden = self.upstream.bidirectional.text_forward(
                self.body, inputs_embeds=embeds, position_ids=positions
            )
        else:
            hidden = self.upstream.bidirectional.text_forward(self.body, ids)
        rows = self.base_model.get_output_embeddings().weight.index_select(
            0, self.letter_ids[:option_count].to(self.device)
        )
        # Keep the original LM letter rows differentiable as part of full tuning.
        with torch.autocast(device_type=self.device.type, enabled=False):
            return (hidden[:, -1].float() @ rows.float().T) / self.temperature

    def prepare(
        self, request: DecisionRequest
    ) -> tuple[dict[str, torch.Tensor], list[int], dict[str, Any]]:
        if (
            request.history
            or request.public_memory is not None
            or request.attempt_index != 1
        ):
            raise ValueError(
                "Whole-question routing cannot receive previous expert calls or routing memory"
            )
        if request.budget is not None and (
            request.budget.remaining_calls != 1
            or request.budget.remaining_env_steps != 0
        ):
            raise ValueError(
                "Whole-question routing permits one expert call and no environment steps"
            )
        mask = request.feasible_mask()
        indices = [i for i, allowed in enumerate(mask) if allowed]
        if len(indices) > 26:
            raise ValueError(
                "More than 26 feasible candidates requires a separately registered protocol"
            )
        if len(indices) == 1:
            return (
                {},
                indices,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "visual_tokens": 0,
                    "visual_in_input": True,
                    "coverage": "complete",
                    "source": "deterministic_singleton_no_model_call",
                },
            )
        if self.tokenizer is None:
            raise RuntimeError(
                "A real checkpoint tokenizer is required to prepare requests"
            )
        references = list(request.images)
        media = self.upstream.images.load_media(
            [image_data_uri(ref) for ref in references],
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            max_images=self.max_images,
        )
        state = {
            "question": request.task,
            "routing_unit": "whole_question",
            "routing_context": request.current,
            "images": ["<image>"] * len(request.images),
            "budget": (
                {
                    key: value
                    for key, value in asdict(request.budget).items()
                    if key in {"remaining_calls", "remaining_tokens"}
                }
                if request.budget
                else None
            ),
        }
        state_text = self.upstream.prompt.state_text(state)
        if state_text.count("<image>") != len(references):
            raise ValueError("Literal image markers conflict with the supplied images")
        if references:
            state_text = self.upstream.images.place_images(state_text, len(references))
        options = []
        for i in indices:
            expert = request.experts[i]
            if expert.profile and len(expert.profile) != len(self.profile_fields):
                raise ValueError("Profile dimensions require named fields")
            description = {
                "description": expert.description,
                "modalities": sorted(expert.modalities),
                "profile": dict(
                    zip(
                        self.profile_fields,
                        expert.profile or (None,) * len(self.profile_fields),
                    )
                ),
                "missing_fields": expert.missing_fields,
                "revision": expert.revision,
                "context_window": expert.context_window,
                "actions": expert.action_capabilities,
                "evidence_sources": expert.evidence_sources,
                "price_metadata": expert.price_metadata,
            }
            options.append(
                {
                    "id": expert.expert_id,
                    "criterion": json.dumps(description, ensure_ascii=False),
                }
            )
        row = {
            "type": "choice",
            "state": state_text,
            "instructions": getattr(
                self,
                "routing_instruction",
                "Select one frozen expert to answer the entire original question. Missing capability evidence is unknown.",
            ),
            "options": options,
        }
        prompt = self.upstream.prompt.plain_prompt(
            self.tokenizer, self.upstream.prompt.messages(row)
        )
        if references:
            if self.processor is None:
                raise ValueError("Images require the native checkpoint processor")
            raw = self.processor(
                text=[prompt],
                images=media.images,
                size=media.processor_size,
                return_tensors="pt",
            )
            encoded = {
                key: value
                for key, value in raw.items()
                if key
                in {"input_ids", "pixel_values", "image_grid_thw", "mm_token_type_ids"}
            }
            visual_tokens = int(
                (encoded["input_ids"] == self.base_model.config.image_token_id).sum()
            )
            if visual_tokens != sum(media.tokens):
                raise ValueError(
                    "Native processor and upstream visual-token accounting disagree"
                )
        else:
            encoded = {
                "input_ids": torch.tensor(
                    [self.tokenizer.encode(prompt, add_special_tokens=False)]
                )
            }
            visual_tokens = 0
        tokens = encoded["input_ids"].numel()
        if tokens > self.max_input_tokens:
            raise ValueError(
                f"Prompt exceeds registered token cap: actual={tokens}, cap={self.max_input_tokens}; no implicit question or candidate truncation"
            )
        usage = {
            "input_tokens": tokens,
            "output_tokens": 0,
            "visual_tokens": visual_tokens,
            "visual_in_input": True,
            "coverage": "complete",
            "source": "native_processor_exact_one_pass",
            "model_id": self.base_id,
            "model_revision": self.base_revision,
            "tokenizer_id": self.base_id,
            "provider": self.provider,
            "raw_usage": {
                "processed_input_tokens": tokens,
                "image_tokens_subset": visual_tokens,
                "encoder_calls": 1,
            },
        }
        return encoded, indices, usage

    def count_input_tokens(self, request: DecisionRequest) -> int:
        return self.prepare(request)[2]["input_tokens"]

    def forward_request(self, request: DecisionRequest) -> dict[str, Any]:
        encoded, indices, usage = self.prepare(request)
        if len(indices) == 1:
            valid_logits = torch.zeros((1, 1), device=self.device)
        else:
            valid_logits = self.forward_encoded(encoded, len(indices))
        all_logits = valid_logits.new_full((1, len(request.experts)), float("-inf"))
        all_logits[:, indices] = valid_logits
        return {
            "logits": all_logits,
            "probabilities": all_logits.softmax(-1),
            "input_tokens": usage["input_tokens"],
            "usage": usage,
        }

    @torch.no_grad()
    def decide(self, request: DecisionRequest) -> DecisionResult:
        self.eval()
        output = self.forward_request(request)
        result = DecisionResult(
            request.expert_ids,
            tuple(output["probabilities"][0].cpu().tolist()),
            request.feasible_mask(),
            (
                "raw_selection_readout"
                if self.training_kind == "untrained"
                else "trained_policy"
            ),
            metadata={
                "usage": output["usage"],
                "policy_version": self.policy_version,
                "training_kind": self.training_kind,
                "evidence_kind": self.evidence_kind,
                "prompt_attention": "bidirectional",
                "upstream": self.upstream.record,
                "input_protocol_id": self.input_protocol_id,
                "selection_mode": "sample",
            },
        )
        result.validate_for(request)
        return result

    def trainable_named_parameters(self):
        return [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    def assert_full_trainable(self):
        named = list(self.named_parameters())
        if (
            self.training_mode != "full"
            or not named
            or any(
                not p.requires_grad or p.dtype != torch.float32 or "lora_" in n
                for n, p in named
            )
        ):
            raise RuntimeError(
                "Every Router parameter must be trainable in FP32; adapters are unsupported"
            )

    def router_state_dict(self):
        """Complete Router parameters and persistent buffers, including tied rows."""
        self.assert_full_trainable()
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.state_dict().items()
        }

    def validate_router_state_dict(self, state):
        self.assert_full_trainable()
        expected = self.state_dict()
        if not isinstance(state, dict) or state.keys() != expected.keys():
            raise ValueError(
                "Full Router tensor names differ from the complete model state"
            )
        for name, tensor in expected.items():
            value = state[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != tensor.shape
                or value.dtype != tensor.dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Invalid full Router tensor: {name}")
        aliases = {}
        for name, tensor in self.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(tensor), []).append(name)
        for names in aliases.values():
            if any(not torch.equal(state[names[0]], state[name]) for name in names[1:]):
                raise ValueError("Full Router state disagrees on tied parameters")

    def load_router_state_dict(self, state):
        self.validate_router_state_dict(state)
        self.load_state_dict(state, strict=True)

    def training_state(self):
        return {
            "policy_version": self.policy_version,
            "training_kind": self.training_kind,
            "training_mode": self.training_mode,
        }

    @staticmethod
    def validate_training_state(state):
        if not isinstance(state, dict) or state.get("training_mode") != "full":
            raise ValueError("Full Router training state is required")
        if type(state.get("policy_version")) is not int or state["policy_version"] < 0:
            raise ValueError("Invalid policy version")
        if (
            not isinstance(state.get("training_kind"), str)
            or not state["training_kind"]
        ):
            raise ValueError("Invalid training kind")

    def load_training_state(self, state):
        self.validate_training_state(state)
        self.policy_version, self.training_kind = (
            state["policy_version"],
            state["training_kind"],
        )

    def architecture_manifest(self) -> dict[str, Any]:
        """Static architecture and precision contract for exact full-state resume."""
        return {
            "base_id": self.base_id,
            "base_revision": self.base_revision,
            "upstream": {
                key: self.upstream.record[key] for key in ("commit", "sha256")
            },
            "training_mode": "full",
            "parameter_dtype": "float32",
            "cuda_compute_dtype": "bfloat16",
            "shared_readout": "trainable original LM letter rows; variable prefix of 26 option letters",
            "letter_token_ids": self.letter_ids.cpu().tolist(),
            "max_options": 26,
            "max_input_tokens": self.max_input_tokens,
            "max_images": self.max_images,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "temperature": self.temperature,
            "profile_fields": list(self.profile_fields),
            "input_protocol_id": self.input_protocol_id,
            "trainable_shapes": {
                n: list(p.shape) for n, p in self.trainable_named_parameters()
            },
            "trainable_dtypes": {
                n: str(p.dtype) for n, p in self.trainable_named_parameters()
            },
            "frozen_parameter_count": sum(
                p.numel() for p in self.parameters() if not p.requires_grad
            ),
            "trainable_parameter_count": sum(
                p.numel() for _, p in self.trainable_named_parameters()
            ),
        }

    def save_router(self, output_dir: str | Path) -> None:
        """Export complete native model weights, processor and routing metadata."""
        self.assert_full_trainable()
        output = Path(output_dir)
        model_dir = output / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(model_dir, safe_serialization=True)
        if self.processor is not None:
            self.processor.save_pretrained(model_dir)
        elif self.tokenizer is not None:
            self.tokenizer.save_pretrained(model_dir)
        (model_dir / "decision_config.json").write_text(
            json.dumps(self.decision_config, indent=2) + "\n"
        )
        manifest = {
            "format_version": 3,
            "training_mode": "full",
            "architecture": self.architecture_manifest(),
            "training_state": self.training_state(),
            "evidence_kind": self.evidence_kind,
            "download_provenance": getattr(self, "download_provenance", None),
            "bidirectional_check": self.bidirectional_check,
            "initialization_usage": self.initialization_usage,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def load_router(self, output_dir: str | Path) -> None:
        """Restore a complete matching Router; never interpret a LoRA export as full weights."""
        from safetensors.torch import load_file

        output = Path(output_dir)
        manifest = json.loads((output / "manifest.json").read_text())
        if (
            manifest.get("format_version") != 3
            or manifest.get("training_mode") != "full"
        ):
            raise ValueError(
                "Expected complete full-parameter Router export; LoRA exports cannot resume this run"
            )
        if manifest["architecture"] != self.architecture_manifest():
            raise ValueError("Saved full Router architecture mismatch")
        self.validate_training_state(manifest["training_state"])
        model_dir = output / "model"
        index = model_dir / "model.safetensors.index.json"
        files = (
            sorted(set(json.loads(index.read_text())["weight_map"].values()))
            if index.exists()
            else ["model.safetensors"]
        )
        if not files or any(
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
            for name in files
        ):
            raise ValueError("Invalid full Router weight shard index")
        state = {}
        for name in files:
            shard = load_file(model_dir / name)
            if state.keys() & shard.keys():
                raise ValueError("Duplicate full Router tensors across weight shards")
            state.update(shard)
        # Transformers saves shared parameters once. Restore every tied name
        # before strict validation, while rejecting conflicting duplicate values.
        aliases = {}
        for name, parameter in self.model.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(parameter), []).append(name)
        for names in aliases.values():
            present = [name for name in names if name in state]
            if present:
                if any(
                    not torch.equal(state[present[0]], state[name])
                    for name in present[1:]
                ):
                    raise ValueError("Full Router export disagrees on tied parameters")
                for name in names:
                    state.setdefault(name, state[present[0]])
        self.load_router_state_dict(
            {"model." + name: tensor for name, tensor in state.items()}
        )
        self.load_training_state(manifest["training_state"])
