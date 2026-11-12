"""Full-parameter closed-pool training, calibration and final refit.

Only the five observed expert qualities and user-priced token costs form targets.
The native StartLux Instant architecture is reused for the API expert pool.
Training uses offline labels and explicit bounded router observations. Complete
expert inputs are preserved; no expert API is called and no adapter is trained.
"""

from __future__ import annotations
import argparse, collections, contextlib, functools, hashlib, json, math, os
from pathlib import Path
import random, shutil, sys, time

PACKAGE = Path(__file__).resolve().parent
BASE_PATH = Path(
    os.environ.get("ADAPTROUTER_API_BASE", Path.cwd() / ".artifacts/startlux-base")
)
SOURCE_DIR = PACKAGE / "_vendor/startlux"
import torch
from .native.decision import DecisionRequest, ExpertDescriptor, ImageReference
from .reward import (
    fit_cost_scale,
    quality_first_target,
    LAMBDA_GRID,
    select_validation_candidate,
)
from .data import read_rows, validate_schema
from .objective import routing_loss
from .runtime import load_training_policy, load_release_policy

EXPERTS = (
    "gpt-6.1-terra",
    "deepseek-flash",
    "claude-sonnet-5-5",
    "gemini-4-argon",
    "kimi-k3.1-pro",
)
PRICES = ((1.0, 5.0), (0.15, 0.60), (2.0, 10.0), (2.0, 10.0), (2.0, 10.0))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def checkpoint(path, value):
    path = Path(path)
    if shutil.disk_usage(path.parent).free < 18 * 1024**3:
        raise RuntimeError("At least 18 GiB checkpoint headroom required")
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def router_view(row, policy):
    from .observation import observation

    return observation(
        row["question"],
        row.get("routing_question") or "",
        row.get("document_char_count"),
        policy,
    )


def request(row, lam, order=None, input_policy="full"):

    order = tuple(range(5)) if order is None else tuple(order)
    experts = []
    for i in order:
        pi, po = PRICES[i]
        experts.append(
            ExpertDescriptor(
                EXPERTS[i],
                f"{EXPERTS[i]}; API token price: input USD {pi}/million, output USD {po}/million.",
                frozenset(("text", "image")),
                price_metadata={
                    "input_usd_per_million": pi,
                    "output_usd_per_million": po,
                },
            )
        )
    raw_images = row.get("images", [])
    unique_indices = []
    seen_images = set()
    for index, image in enumerate(raw_images):
        identity = image.get("sha256") or image["uri"]
        if identity not in seen_images:
            unique_indices.append(index)
            seen_images.add(identity)
    image_indices = (
        unique_indices
        if len(unique_indices) <= 16
        else [
            unique_indices[round(i * (len(unique_indices) - 1) / 15)] for i in range(16)
        ]
    )
    images = tuple(
        ImageReference(
            raw_images[i]["uri"], raw_images[i]["mime_type"], authorized=True
        )
        for i in image_indices
    )
    # No task ID, dataset label, answer, target score, future output length or observed
    # per-question expert cost is exposed to the native routing architecture.
    return DecisionRequest(
        task=router_view(row, input_policy),
        current=json.dumps(
            {
                "cost_penalty_weight": lam,
                "objective": "Maximize answer quality; use token-priced USD cost when quality is comparable.",
                "original_characters": len(row["question"]),
                "original_utf8_bytes": len(row["question"].encode()),
                "rough_input_tokens_utf8_div4": math.ceil(
                    len(row["question"].encode()) / 4
                ),
                "context_size": "long" if len(row["question"]) > 32000 else "standard",
                "router_image_policy": "deduplicate_sha_then_all_up_to_16_else_evenly_spaced_originals",
                "original_image_count": len(raw_images),
                "unique_image_count": len(unique_indices),
                "selected_image_indices": image_indices,
            }
        ),
        experts=tuple(experts),
        text_observation_available=True,
        images=images,
    )


def configure_memory(device, fraction):
    if not 0 < fraction <= 1:
        raise ValueError("Memory fraction must be in (0,1]")
    if not device.startswith("cuda"):
        raise ValueError("Explicit CUDA device required")
    torch.cuda.set_per_process_memory_fraction(fraction, torch.device(device))
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    free, total = torch.cuda.mem_get_info(torch.device(device))
    return {
        "memory_fraction": fraction,
        "allocator_limit_bytes": int(total * fraction),
        "free_bytes_before_model": free,
        "total_bytes": total,
    }


def memory_report(device):
    return {
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def load_policy(family, device, max_input_tokens, seed):
    if family != "startlux":
        raise ValueError("This release contains the Table 3 StartLux API trainer")
    return load_training_policy(BASE_PATH, SOURCE_DIR, device, max_input_tokens)


def install_checkpointing(policy):
    """Recompute native decoder layers, preserving the native attention/readout.

    Use non-reentrant checkpointing because native layers accept structured kwargs.
    This leaves every weight in the optimizer and never caches detached features.
    """
    from torch.utils.checkpoint import checkpoint as rematerialize
    from torch.nn.attention import sdpa_kernel, SDPBackend

    layers = [
        m for m in policy.modules() if m.__class__.__name__ == "Qwen3_5DecoderLayer"
    ]
    if not layers:
        raise RuntimeError("Expected native Qwen3.5 decoder layers")

    def wrap(original):
        @functools.wraps(original)
        def forward(*args, **kwargs):
            if not torch.is_grad_enabled():
                return original(*args, **kwargs)

            def inner(*a, **kw):
                with (
                    sdpa_kernel(
                        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
                    )
                    if next(policy.parameters()).is_cuda
                    else contextlib.nullcontext()
                ):
                    return original(*a, **kw)

            return rematerialize(inner, *args, use_reentrant=False, **kwargs)

        return forward

    for layer in layers:
        layer.forward = wrap(layer.forward)
    return len(layers)


def source_binding():
    return {str(p.relative_to(PACKAGE)): sha(p) for p in sorted(PACKAGE.rglob("*.py"))}


def metrics(rows, decisions):
    groups = collections.defaultdict(list)
    for row, idx in zip(rows, decisions):
        groups[row["dataset"]].append((row["qualities"][idx], row["costs_usd"][idx]))
    by = {}
    for ds, vs in groups.items():
        known = [c for _, c in vs if c is not None]
        by[ds] = {
            "n": len(vs),
            "quality": sum(q for q, _ in vs) / len(vs),
            "cost_mean_usd": sum(known) / len(known) if len(known) == len(vs) else None,
            "cost_known_sum_usd": sum(known),
            "cost_unknown": len(vs) - len(known),
        }
    known_all = all(x["cost_mean_usd"] is not None for x in by.values())
    return {
        "quality_macro": sum(x["quality"] for x in by.values()) / len(by),
        "cost_macro_usd": (
            sum(x["cost_mean_usd"] for x in by.values()) / len(by)
            if known_all
            else None
        ),
        "by_dataset": by,
        "n": len(rows),
        "cost_unknown": sum(x["cost_unknown"] for x in by.values()),
    }


@torch.no_grad()
def evaluate(policy, rows, lam, out=None, input_policy="full"):
    policy.eval()
    choices = []
    records = []
    for i, row in enumerate(rows):
        f = policy.forward_request(request(row, lam, input_policy=input_policy))
        probs = f["probabilities"][0].float().cpu()
        idx = int(probs.argmax())
        choices.append(idx)
        records.append(
            {
                "task_id": row["task_id"],
                "dataset": row["dataset"],
                "expert_id": EXPERTS[idx],
                "probabilities": probs.tolist(),
                "router_input_tokens": f["usage"]["input_tokens"],
            }
        )
        if (i + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "event": "validation",
                        "done": i + 1,
                        "total": len(rows),
                        "lambda": lam,
                    }
                ),
                flush=True,
            )
    result = metrics(rows, choices)
    if out:
        atomic(out, {"lambda_cost": lam, "metrics": result, "decisions": records})
    return result


def state_payload(
    policy, opt, identity, epoch, next_index, update, best, initial_probe
):
    return {
        "identity": identity,
        "state": policy.state_dict(),
        "optimizer": opt.state_dict(),
        "epoch": epoch,
        "next_index": next_index,
        "update": update,
        "best": best,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "initial_probe": initial_probe,
    }


def train(args):
    rows = read_rows(args.data)
    validate_schema(rows, args.phase, args.expected_datasets, args.expected_refit_rows)
    train_rows = [
        r
        for r in rows
        if r["split"] == "train"
        or (args.phase == "refit" and r["split"] == "validation")
    ]
    valid = [r for r in rows if r["split"] == "validation"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lam_grid = args.lambda_grid
    epochs = args.epochs
    if args.phase == "refit":
        selected = json.loads(Path(args.selection).read_text())
        if (
            selected["data_sha256"] != sha(args.data)
            or selected["family"] != args.family
            or selected["router_input_policy"] != args.router_input_policy
        ):
            raise ValueError("Calibration identity mismatch")
        lam_grid = [selected["lambda_cost"]]
        epochs = selected["epoch"]
    cost_scale = (
        selected["cost_scale"]
        if args.phase == "refit"
        else fit_cost_scale(
            [
                {
                    "split": "train",
                    "dataset": r["dataset"],
                    "arms": [{"cost_usd": c} for c in r["costs_usd"]],
                }
                for r in train_rows
            ]
        )
    )
    identity = {
        "schema": "closed_router_full_parameter_v1",
        "family": args.family,
        "phase": args.phase,
        "data_sha256": sha(args.data),
        "source_sha256": source_binding(),
        "epochs": epochs,
        "lambda_grid": lam_grid,
        "seed": args.seed,
        "max_input_tokens": args.max_input_tokens,
        "learning_rate": args.learning_rate,
        "questions_per_update": args.questions_per_update,
        "temperature": args.temperature,
        "cost_scale": cost_scale,
        "expert_input_preserved": True,
        "router_input_policy": args.router_input_policy,
        "router_text_token_budget": 4096,
        "literal_image_marker_escape": "[image attachment]",
        "router_image_policy": "deduplicate_sha_then_all_up_to_16_else_evenly_spaced_originals",
        "training_tasks": len(train_rows),
        "dataset_count": len({r["dataset"] for r in train_rows}),
        "decision_pretraining": args.family == "startlux",
        "test_used_for_selection": False,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
    }
    if (out / "COMPLETED.json").exists():
        done = json.loads((out / "COMPLETED.json").read_text())
        if (
            any(done["identity"].get(k) != v for k, v in identity.items())
            or sha(out / "model.pt") != done["model_sha256"]
        ):
            raise ValueError("Completed output identity differs")
        print(json.dumps({"status": "already_completed", "out": str(out)}))
        return
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Real CUDA full training required")
    memory = configure_memory(args.device, args.memory_fraction)
    policy = load_policy(args.family, args.device, args.max_input_tokens, args.seed)
    numel = sum(p.numel() for p in policy.parameters())
    n_layers = install_checkpointing(policy)
    identity.update(
        trainable_parameters=numel,
        total_parameters=numel,
        checkpointed_layers=n_layers,
        memory_limit={"memory_fraction": args.memory_fraction},
        publisher_identity=getattr(
            policy, "publisher_identity", getattr(policy, "download_provenance", None)
        ),
    )
    identity = json.loads(json.dumps(identity))
    # Store the exact runtime identity after native hash/revision verification.
    atomic(out / "RUN_IDENTITY.json", identity)
    atomic(out / "RUNTIME_MEMORY_ADMISSION.json", memory)
    preflight = out / "NATIVE_INPUT_PREFLIGHT.json"
    if preflight.exists():
        native = json.loads(preflight.read_text())
        if native["identity"] != identity or native["status"] != "passed":
            raise ValueError("Native input preflight identity differs")
    else:
        token_rows = []
        for row in train_rows + (valid if args.phase == "calibrate" else []):
            _, _, usage = policy.prepare(
                request(row, max(lam_grid), input_policy=args.router_input_policy)
            )
            token_rows.append(
                {
                    "task_id": row["task_id"],
                    "dataset": row["dataset"],
                    "input_tokens": usage["input_tokens"],
                }
            )
        atomic(
            preflight,
            {
                "status": "passed",
                "identity": identity,
                "rows": token_rows,
                "max_tokens": max(r["input_tokens"] for r in token_rows),
                "actual_training_forwards": 0,
                "full_expert_inputs_preserved": True,
            },
        )
    opt = torch.optim.AdamW(
        policy.parameters(), lr=args.learning_rate, weight_decay=0.01, foreach=False
    )
    best = None
    epoch0 = 0
    next0 = 0
    update = 0
    initial_probe = {
        n: p.detach().flatten()[:: max(1, p.numel() // 32)][:32].cpu().clone()
        for n, p in policy.named_parameters()
    }
    latest = out / "latest.pt"
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Refuse to resume changed data/source/config")
        policy.load_state_dict(saved["state"])
        opt.load_state_dict(saved["optimizer"])
        epoch0 = saved["epoch"]
        next0 = saved["next_index"]
        update = saved["update"]
        best = saved["best"]
        initial_probe = saved["initial_probe"]
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in saved["cuda_rng"]])
        del saved
    counts = collections.Counter(r["dataset"] for r in train_rows)
    history = (
        json.loads((out / "VALIDATION_HISTORY.json").read_text())
        if (out / "VALIDATION_HISTORY.json").exists()
        else []
    )
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    for epoch in range(epoch0, epochs):
        policy.train()
        order = list(range(len(train_rows)))
        random.Random(args.seed + epoch).shuffle(order)
        start = next0 if epoch == epoch0 else 0
        for begin in range(start, len(order), args.questions_per_update):
            batch = order[begin : begin + args.questions_per_update]
            loss_sum = 0.0
            cost_masked = 0
            tokens_max = 0
            for idx in batch:
                row = train_rows[idx]
                r = random.Random(
                    f'{args.seed}:{epoch}:{row["dataset"]}:{row["task_id"]}'
                )
                permutation = list(range(5))
                r.shuffle(permutation)
                lam = r.choice(lam_grid)
                q = [row["qualities"][k] for k in permutation]
                c = [row["costs_usd"][k] for k in permutation]
                target = quality_first_target(
                    q,
                    c,
                    lambda_cost=lam,
                    cost_scale=cost_scale,
                    temperature=args.temperature,
                )
                output = policy.forward_request(
                    request(
                        row, lam, permutation, input_policy=args.router_input_policy
                    )
                )
                logits = output["logits"][0].float()
                loss = routing_loss(logits, target)
                weight = len(train_rows) / (len(counts) * counts[row["dataset"]])
                (loss * weight / len(batch)).backward()
                loss_sum += float(loss.detach())
                cost_masked += any(value is None for value in c)
                tokens_max = max(tokens_max, output["usage"]["input_tokens"])
                del output, logits, loss
            grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0))
            if not math.isfinite(grad_norm):
                raise RuntimeError("Nonfinite gradient; no optimizer update")
            opt.step()
            opt.zero_grad(set_to_none=True)
            update += 1
            policy.policy_version = update
            policy.training_kind = "closed_pool_full_information_reward"
            progress = {
                "status": "training",
                "family": args.family,
                "phase": args.phase,
                "epoch": epoch + 1,
                "epochs": epochs,
                "processed": min(begin + len(batch), len(order)),
                "tasks_per_epoch": len(order),
                "updates": update,
                "loss": loss_sum / len(batch),
                "grad_norm": grad_norm,
                "cost_incomplete_batch": cost_masked,
                "max_input_tokens_batch": tokens_max,
                "elapsed_seconds": time.time() - t0,
                "real_cuda": True,
                "trainable_parameters": numel,
                **memory_report(args.device),
            }
            atomic(out / "PROGRESS.json", progress)
            print(json.dumps(progress), flush=True)
            if update % args.checkpoint_every == 0:
                checkpoint(
                    latest,
                    state_payload(
                        policy,
                        opt,
                        identity,
                        epoch,
                        begin + len(batch),
                        update,
                        best,
                        initial_probe,
                    ),
                )
        if args.phase == "calibrate":
            for lam in lam_grid:
                m = evaluate(
                    policy,
                    valid,
                    lam,
                    out / f"validation_epoch{epoch+1}_lambda{lam:g}.json",
                    input_policy=args.router_input_policy,
                )
                candidate = {
                    "family": args.family,
                    "epoch": epoch + 1,
                    "lambda_cost": lam,
                    "metrics": m,
                    "data_sha256": sha(args.data),
                    "cost_scale": cost_scale,
                    "router_input_policy": args.router_input_policy,
                }
                candidate.update(
                    id=f"epoch{epoch+1}_lambda{lam:g}",
                    split="validation",
                    quality_complete=True,
                    validation_ids_sha256=hashlib.sha256(
                        json.dumps(
                            sorted((r["dataset"], r["task_id"]) for r in valid)
                        ).encode()
                    ).hexdigest(),
                    dataset_names=sorted({r["dataset"] for r in valid}),
                    macro_quality=m["quality_macro"],
                    macro_cost_usd=m["cost_macro_usd"],
                    cost_complete=m["cost_unknown"] == 0,
                )
                history = [c for c in history if c["id"] != candidate["id"]] + [
                    candidate
                ]
                selection = select_validation_candidate(history)
                winner = selection["selected"]
                if best is None or winner["id"] != best["id"]:
                    # Final models are refit from the publisher; no historic checkpoint is
                    # needed to choose a previous epoch/lambda by its immutable validation log.
                    best = winner
                    atomic(
                        out / "SELECTION.json", {**best, "selection_rule": selection}
                    )
            atomic(out / "VALIDATION_HISTORY.json", history)
        checkpoint(
            latest,
            state_payload(
                policy, opt, identity, epoch + 1, 0, update, best, initial_probe
            ),
        )
    changed = []
    for n, p in policy.named_parameters():
        cur = p.detach().flatten()[:: max(1, p.numel() // 32)][:32].cpu()
        if not torch.equal(cur, initial_probe[n].cpu()):
            changed.append(n)
    if not changed:
        raise RuntimeError("No real model parameter updates")
    checkpoint(
        out / "model.pt",
        {
            "identity": identity,
            "state": policy.state_dict(),
            "policy_version": update,
            "training_kind": "closed_pool_full_information_reward",
        },
    )
    report = {
        "status": "completed",
        "identity": identity,
        "model_sha256": sha(out / "model.pt"),
        "optimizer_updates": update,
        "real_cuda": True,
        "sampled_changed_parameter_tensors": len(changed),
        "sampled_changed_names": changed,
        "all_parameters_trainable": True,
        "expert_api_calls": 0,
        "judge_api_calls": 0,
        **memory_report(args.device),
        "selected_calibration": best,
        "elapsed_seconds": time.time() - t0,
    }
    atomic(out / "COMPLETED.json", report)
    atomic(out / "PROGRESS.json", report)


def route(args):
    if args.device.startswith("cuda"):
        configure_memory(args.device, args.memory_fraction)
    else:
        torch.set_num_threads(4)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        identity = saved["identity"]
        if identity["phase"] != "refit":
            raise ValueError("Formal evaluation requires full-data final refit")
        if identity["source_sha256"] != source_binding():
            raise ValueError("Inference source differs from training source")
        p = load_policy(
            identity["family"],
            args.device,
            identity["max_input_tokens"],
            identity["seed"],
        )
        p.load_state_dict(saved["state"])
        del saved
        model_sha = sha(args.checkpoint)
    else:
        p = load_release_policy(Path(args.model_dir), SOURCE_DIR, args.device)
        identity = {
            "family": "startlux",
            "max_input_tokens": 8192,
            "router_input_policy": "question_length_head_tail",
            "lambda_grid": [0.02],
        }
        model_sha = hashlib.sha256(
            json.dumps(p.download_provenance, sort_keys=True).encode()
        ).hexdigest()
    p.policy_version = 1
    p.training_kind = "closed_pool_full_information_reward"
    data_sha = sha(args.data)
    rows = read_rows(args.data, require_labels=False)
    rows = [r for r in rows if r["split"] == "test"]
    if not rows:
        raise ValueError("Formal test rows absent")
    # Routing consumes only public inputs. The metrics helper is deliberately not
    # used here: test qualities/costs must come from subsequent selected API calls.
    output = []
    if Path(args.out).exists():
        previous = json.loads(Path(args.out).read_text())
        if (
            previous["model_sha256"] != model_sha
            or previous.get("data_sha256") != data_sha
        ):
            raise ValueError("Route checkpoint differs")
        output = previous["rows"]
        if [(r["dataset"], r["task_id"]) for r in output] != [
            (r["dataset"], r["task_id"]) for r in rows[: len(output)]
        ]:
            raise ValueError("Route prefix differs")
    with torch.no_grad():
        for row in rows[len(output) :]:
            f = p.forward_request(
                request(
                    row,
                    identity["lambda_grid"][0],
                    input_policy=identity["router_input_policy"],
                )
            )
            probs = f["probabilities"][0].float().cpu()
            idx = int(probs.argmax())
            output.append(
                {
                    "task_id": row["task_id"],
                    "dataset": row["dataset"],
                    "expert_id": EXPERTS[idx],
                    "probabilities": probs.tolist(),
                    "input_path": row.get("input_path"),
                    "input_sha256": row.get("input_sha256"),
                    "router_usage": f["usage"],
                }
            )
            if len(output) % 10 == 0:
                atomic(
                    args.out,
                    {
                        "status": "routing",
                        "model_sha256": model_sha,
                        "data_sha256": data_sha,
                        "rows": output,
                    },
                )
    atomic(
        args.out,
        {
            "status": "completed",
            "model_sha256": model_sha,
            "data_sha256": data_sha,
            "rows": output,
            "expert_calls": 0,
        },
    )


def smoke(args):
    rows = read_rows(args.data)
    if (
        len(rows) != 2
        or not any(r.get("images") for r in rows)
        or not any(not r.get("images") for r in rows)
    ):
        raise ValueError(
            "Smoke requires exactly one real text and one real image feedback row"
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(37)
    memory = configure_memory(args.device, args.memory_fraction)
    policy = load_policy(args.family, args.device, 8192, 37)
    numel = sum(p.numel() for p in policy.parameters())
    layers = install_checkpointing(policy)
    initial = {
        n: p.detach().flatten()[:: max(1, p.numel() // 32)][:32].cpu().clone()
        for n, p in policy.named_parameters()
    }
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-6, foreach=False)
    steps = []
    policy.train()
    for row in rows:
        opt.zero_grad(set_to_none=True)
        target = quality_first_target(
            row["qualities"], row["costs_usd"], lambda_cost=0.01, cost_scale=0.01
        )
        output = policy.forward_request(
            request(row, 0.01, input_policy="question_length_head_tail")
        )
        logits = output["logits"][0].float()
        y = torch.tensor(target["target_probs"], device=logits.device)
        loss = -(y * logits.log_softmax(-1)).sum()
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0))
        if not math.isfinite(float(loss)) or not math.isfinite(norm):
            raise RuntimeError("Nonfinite smoke")
        opt.step()
        steps.append(
            {
                "dataset": row["dataset"],
                "task_id": row["task_id"],
                "loss": float(loss.detach()),
                "grad_norm": norm,
                "input_tokens": output["usage"]["input_tokens"],
                "images": len(row.get("images", [])),
                **memory_report(args.device),
            }
        )
    changed = [
        n
        for n, p in policy.named_parameters()
        if not torch.equal(
            p.detach().flatten()[:: max(1, p.numel() // 32)][:32].cpu(), initial[n]
        )
    ]
    if not changed:
        raise RuntimeError("No native parameters changed")
    report = {
        "status": "passed",
        "family": args.family,
        "real_cuda": True,
        "all_parameters_trainable": True,
        "trainable_parameters": numel,
        "total_parameters": numel,
        "checkpointed_layers": layers,
        "sampled_changed_tensors": len(changed),
        "steps": steps,
        **memory_report(args.device),
        "memory": memory,
        "data_sha256": sha(args.data),
        "source_sha256": source_binding(),
        "engineering_only": True,
        "formal_training_tasks": 0,
        "input_feedback_is_user_supplied": True,
        "optimizer_steps_discarded_after_smoke": 2,
        "api_calls": 0,
        "checkpoints_saved": 0,
    }
    atomic(out / "SMOKE.json", report)
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--family", choices=["startlux"], default="startlux")
    t.add_argument("--phase", choices=["calibrate", "refit"], required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--selection")
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--memory-fraction", type=float, default=0.25)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--seed", type=int, default=37)
    t.add_argument("--lambda-grid", type=float, nargs="+", default=[0.02])
    t.add_argument(
        "--router-input-policy",
        choices=["full", "question_length_head_tail"],
        required=True,
    )
    t.add_argument("--max-input-tokens", type=int, default=8192)
    t.add_argument("--learning-rate", type=float, default=1e-6)
    t.add_argument("--questions-per-update", type=int, default=16)
    t.add_argument("--temperature", type=float, default=0.05)
    t.add_argument("--checkpoint-every", type=int, default=20)
    r = sub.add_parser("route")
    selection = r.add_mutually_exclusive_group(required=True)
    selection.add_argument("--checkpoint")
    selection.add_argument("--model-dir")
    r.add_argument("--data", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--memory-fraction", type=float, default=0.25)
    sm = sub.add_parser("smoke")
    sm.add_argument("--family", choices=["startlux"], default="startlux")
    sm.add_argument("--data", required=True)
    sm.add_argument("--out", required=True)
    sm.add_argument("--device", default="cuda:0")
    sm.add_argument("--memory-fraction", type=float, default=0.25)
    t.add_argument("--expected-datasets", type=int, default=10)
    t.add_argument("--expected-refit-rows", type=int, default=1328)
    for command in (t, r, sm):
        command.add_argument("--base-model-path", type=Path, default=BASE_PATH)
        command.add_argument("--startlux-source", type=Path, default=SOURCE_DIR)
    a = p.parse_args()
    if a.command == "train" and a.phase == "refit" and not a.selection:
        p.error("--selection is required for refit; first run prepare-refit")
    global_configure(a.base_model_path, a.startlux_source)
    {"train": train, "route": route, "smoke": smoke}[a.command](a)


def global_configure(base_path, source_dir):
    global BASE_PATH, SOURCE_DIR
    BASE_PATH, SOURCE_DIR = Path(base_path), Path(source_dir)
    from .observation import configure_tokenizer

    # Safetensors release assets also contain the required tokenizer.
    if BASE_PATH.is_dir():
        configure_tokenizer(BASE_PATH)


if __name__ == "__main__":
    main()
