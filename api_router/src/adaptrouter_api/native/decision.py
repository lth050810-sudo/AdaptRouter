"""Complete-question inputs and model-list selection probabilities.

This module does not load weights, fetch images, infer capabilities from model
names, or equate a selection softmax with probability of task success. It places
no restriction on the backbone size or context/capability representation width.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import math
from numbers import Real
from pathlib import Path, PureWindowsPath
from typing import Literal, Mapping, Protocol

Provenance = Literal["raw_selection_readout", "trained_policy"]


class NoFeasibleExpertError(ValueError):
    """The visible observation has no available, modality-compatible expert."""


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _finite(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class ImageReference:
    """Caller-authorized local image or inline data; validation performs no I/O.

    ``authorized`` records the caller's assertion that this image may be used.
    Actual filesystem access and decoded-image validation belong to the adapter.
    Remote URLs and network-share paths are deliberately unsupported here.
    """

    uri: str
    mime_type: str
    authorized: bool = False

    def __post_init__(self) -> None:
        _nonempty(self.uri, "image uri")
        if self.authorized is not True:
            raise ValueError("image reference must be explicitly authorized")
        if self.mime_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
            raise ValueError("unsupported image MIME type")
        if self.uri.startswith("data:"):
            prefix = f"data:{self.mime_type};base64,"
            if not self.uri.startswith(prefix):
                raise ValueError("data image MIME type must match mime_type")
            encoded = self.uri[len(prefix) :]
            if not encoded:
                raise ValueError("empty inline image")
            try:
                base64.b64decode(encoded, validate=True)
            except (ValueError, base64.binascii.Error) as exc:
                raise ValueError("invalid base64 image data") from exc
        else:
            if "://" in self.uri or self.uri.startswith(("//", "\\\\")):
                raise ValueError("only local paths or inline data images are accepted")
            if not (
                Path(self.uri).is_absolute() or PureWindowsPath(self.uri).is_absolute()
            ):
                raise ValueError("local image path must be absolute")


@dataclass(frozen=True)
class ExpertDescriptor:
    expert_id: str
    description: str
    modalities: frozenset[str]
    profile: tuple[float | None, ...] = ()
    available: bool = True
    revision: str | None = None
    profile_version: str | None = None
    evidence_sources: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    context_window: int | None = None
    action_capabilities: tuple[str, ...] = ()
    price_metadata: Mapping[str, object] | None = None
    service_status: str = "available"

    def __post_init__(self) -> None:
        _nonempty(self.expert_id, "expert_id")
        _nonempty(self.description, "expert description")
        object.__setattr__(self, "modalities", frozenset(self.modalities))
        object.__setattr__(self, "profile", tuple(self.profile))
        for name in ("evidence_sources", "missing_fields", "action_capabilities"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
            for value in getattr(self, name):
                _nonempty(value, name)
        for name in ("revision", "profile_version"):
            if getattr(self, name) is not None:
                _nonempty(getattr(self, name), name)
        if self.context_window is not None and (
            type(self.context_window) is not int or self.context_window <= 0
        ):
            raise ValueError("context_window must be a positive integer or None")
        _nonempty(self.service_status, "service_status")
        if self.price_metadata is not None:
            object.__setattr__(self, "price_metadata", dict(self.price_metadata))
        if not self.modalities or not self.modalities <= {"text", "image"}:
            raise ValueError("modalities must contain text and/or image")
        if type(self.available) is not bool:
            raise ValueError("expert availability must be boolean")
        for value in self.profile:
            if value is not None:
                _finite(value, "expert profile value")


@dataclass(frozen=True)
class DecisionEvent:
    """Historical record type; current requests reject prior execution events."""

    expert_id: str
    action: str | None
    feedback: str
    observation: str
    images: tuple[ImageReference, ...] = ()
    response: str | None = None
    logical_node_id: str | None = None
    attempt_index: int | None = None

    def __post_init__(self) -> None:
        _nonempty(self.expert_id, "history expert_id")
        if self.action is not None and not isinstance(self.action, str):
            raise ValueError("history action must be a string or None")
        if not isinstance(self.feedback, str) or not isinstance(self.observation, str):
            raise ValueError("history feedback and observation must be strings")
        if self.response is not None and not isinstance(self.response, str):
            raise ValueError("history response must be a string or None")
        if self.logical_node_id is not None:
            _nonempty(self.logical_node_id, "history logical_node_id")
        if self.attempt_index is not None and (
            type(self.attempt_index) is not int or self.attempt_index < 1
        ):
            raise ValueError("history attempt_index must be positive or None")
        object.__setattr__(self, "images", tuple(self.images))
        if any(not isinstance(image, ImageReference) for image in self.images):
            raise ValueError("history images must be validated ImageReference objects")


@dataclass(frozen=True)
class BudgetState:
    """Resources for one complete-question execution."""

    remaining_calls: int
    remaining_env_steps: int
    remaining_tokens: int
    remaining_runtime_seconds: float

    def __post_init__(self) -> None:
        for name in ("remaining_calls", "remaining_env_steps", "remaining_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if _finite(self.remaining_runtime_seconds, "remaining_runtime_seconds") < 0:
            raise ValueError("remaining_runtime_seconds must be nonnegative")


@dataclass(frozen=True)
class DecisionRequest:
    """One complete question and its candidate models, without prior attempts.

    Historical field names remain readable to reject old request records with
    an explicit protocol error. They cannot enable execution history.
    """

    task: str
    current: str
    experts: tuple[ExpertDescriptor, ...]
    text_observation_available: bool
    history: tuple[DecisionEvent, ...] = ()
    images: tuple[ImageReference, ...] = ()
    availability_mask: tuple[bool, ...] | None = None
    budget: BudgetState | None = None
    logical_node_id: str | None = None
    attempt_index: int = 1
    routing_call_id: str | None = None
    public_memory: dict | None = None

    def __post_init__(self) -> None:
        _nonempty(self.task, "task")
        if not isinstance(self.current, str):
            raise ValueError("current observation must be a string")
        if self.budget is not None and not isinstance(self.budget, BudgetState):
            raise ValueError("budget must be a validated BudgetState or None")
        if self.history or self.public_memory is not None:
            raise ValueError(
                "Whole-question routing cannot include intermediate execution history or public memory"
            )
        if self.budget is not None and (
            self.budget.remaining_calls != 1 or self.budget.remaining_env_steps != 0
        ):
            raise ValueError(
                "Whole-question routing permits one expert call and no environment steps"
            )
        for name in ("logical_node_id", "routing_call_id"):
            if getattr(self, name) is not None:
                _nonempty(getattr(self, name), name)
        if type(self.attempt_index) is not int or self.attempt_index != 1:
            raise ValueError(
                "Each independent whole-question attempt starts with attempt_index=1"
            )
        if type(self.text_observation_available) is not bool:
            raise ValueError("text_observation_available must be boolean")
        if self.text_observation_available and not self.current.strip():
            raise ValueError(
                "text observation route requires a current textual observation"
            )
        for name in ("experts", "history", "images"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.experts or any(
            not isinstance(item, ExpertDescriptor) for item in self.experts
        ):
            raise ValueError("experts must contain validated ExpertDescriptor objects")
        if len(set(self.expert_ids)) != len(self.experts):
            raise ValueError("expert IDs must be unique")
        if any(not isinstance(item, DecisionEvent) for item in self.history):
            raise ValueError("history must contain visible DecisionEvent objects")
        if any(not isinstance(item, ImageReference) for item in self.images):
            raise ValueError("images must contain validated ImageReference objects")
        if self.availability_mask is not None:
            object.__setattr__(self, "availability_mask", tuple(self.availability_mask))
            _validate_mask(self.availability_mask, len(self.experts))
        self.feasible_mask()  # Fail explicitly before inference if no expert can receive state.

    @property
    def expert_ids(self) -> tuple[str, ...]:
        return tuple(expert.expert_id for expert in self.experts)

    def feasible_mask(self) -> tuple[bool, ...]:
        available = self.availability_mask or (True,) * len(self.experts)
        mask = tuple(
            expert.available
            and allowed
            and (
                (self.text_observation_available and "text" in expert.modalities)
                or (bool(self.images) and "image" in expert.modalities)
            )
            for expert, allowed in zip(self.experts, available)
        )
        if not any(mask):
            raise NoFeasibleExpertError(
                "no available expert can receive the current observation"
            )
        return mask


def _validate_mask(mask: tuple[bool, ...], count: int) -> None:
    if len(mask) != count or any(type(value) is not bool for value in mask):
        raise ValueError("mask must contain one boolean per expert in the same order")


@dataclass(frozen=True)
class DecisionResult:
    """Explicitly separate selection, outcome forecasts, and expected utility.

    policy_probs is a normalized choice distribution. It is *not* the probability
    each expert succeeds, nor the probability it is globally optimal. Optional
    forecasts stay None until a real prediction source exists; tuple entries may
    individually be None. Success and remaining cost are conditioned on taking
    this expert now and then following continuation_policy_id. No conversion
    from confidence or a selection readout to an outcome forecast is performed.

    metadata may record a provider's raw distribution and any explicit rounding
    correction. Constructor validation never silently normalizes inputs.
    """

    expert_ids: tuple[str, ...]
    policy_probs: tuple[float, ...]
    feasible_mask: tuple[bool, ...]
    provenance: Provenance
    success_probs: tuple[float | None, ...] | None = None
    future_costs: tuple[float | None, ...] | None = None
    expected_utilities: tuple[float | None, ...] | None = None
    continuation_policy_id: str | None = None
    future_cost_unit: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("expert_ids", "policy_probs", "feasible_mask"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        count = len(self.expert_ids)
        if count == 0 or len(set(self.expert_ids)) != count:
            raise ValueError("result requires nonempty unique expert IDs")
        for expert_id in self.expert_ids:
            _nonempty(expert_id, "result expert_id")
        _validate_mask(self.feasible_mask, count)
        if not any(self.feasible_mask):
            raise NoFeasibleExpertError("result has no feasible expert")
        if self.provenance not in {"raw_selection_readout", "trained_policy"}:
            raise ValueError(
                "provenance must distinguish raw selection readout from trained policy"
            )
        if len(self.policy_probs) != count:
            raise ValueError("policy probabilities must align with expert IDs")
        for probability, allowed in zip(self.policy_probs, self.feasible_mask):
            value = _finite(probability, "policy probability")
            if not 0 <= value <= 1:
                raise ValueError("policy probabilities must be in [0, 1]")
            if not allowed and value != 0:
                raise ValueError(
                    "masked expert must have exactly zero policy probability"
                )
        if not math.isclose(
            math.fsum(self.policy_probs), 1.0, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                "policy probabilities must sum to one; normalization must be explicit"
            )

        has_forecast = False
        for name in ("success_probs", "future_costs", "expected_utilities"):
            values = getattr(self, name)
            if values is None:
                continue
            values = tuple(values)
            object.__setattr__(self, name, values)
            if len(values) != count:
                raise ValueError(f"{name} must align with expert IDs")
            for value in values:
                if value is None:
                    continue
                has_forecast = True
                number = _finite(value, name)
                if name == "success_probs" and not 0 <= number <= 1:
                    raise ValueError(
                        "success probabilities must independently be in [0, 1]"
                    )
                if name == "future_costs" and number < 0:
                    raise ValueError("future costs must be nonnegative")
        if has_forecast:
            _nonempty(self.continuation_policy_id, "forecast continuation_policy_id")
        if self.future_costs is not None and any(
            v is not None for v in self.future_costs
        ):
            _nonempty(self.future_cost_unit, "future_cost_unit")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def validate_for(self, request: DecisionRequest) -> None:
        if self.expert_ids != request.expert_ids:
            raise ValueError("result expert IDs/order do not match the request")
        if self.feasible_mask != request.feasible_mask():
            raise ValueError("result feasibility mask does not match the request")


class DecisionBackend(Protocol):
    def decide(self, request: DecisionRequest) -> DecisionResult: ...


def masked_policy_from_logits(
    expert_ids: tuple[str, ...],
    logits: tuple[float, ...],
    mask: tuple[bool, ...],
    provenance: Provenance,
) -> DecisionResult:
    """Stable masked softmax for adapters/tests; creates no outcome predictions."""
    expert_ids, logits, mask = tuple(expert_ids), tuple(logits), tuple(mask)
    _validate_mask(mask, len(expert_ids))
    if not any(mask):
        raise NoFeasibleExpertError("cannot normalize a policy with no feasible expert")
    if len(logits) != len(expert_ids):
        raise ValueError("logits must align with expert IDs")
    # Even masked NaNs are rejected, exposing upstream numerical corruption.
    values = [_finite(value, "logit") for value in logits]
    maximum = max(value for value, allowed in zip(values, mask) if allowed)
    weights = [
        math.exp(value - maximum) if allowed else 0.0
        for value, allowed in zip(values, mask)
    ]
    total = math.fsum(weights)
    return DecisionResult(
        expert_ids, tuple(value / total for value in weights), mask, provenance
    )
