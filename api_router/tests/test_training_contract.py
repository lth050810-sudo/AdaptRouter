import copy
import json
import math
import subprocess
import sys
from pathlib import Path
import pytest
import torch
from adaptrouter_api.data import read_rows, validate_schema
from adaptrouter_api.objective import routing_loss
from adaptrouter_api.reward import (
    EXPERTS,
    fit_cost_scale,
    quality_first_target,
    reward_vector,
    select_validation_candidate,
)
from adaptrouter_api import train as training
from adaptrouter_api import observation


def example(dataset="a", split="train", group="a-train", task="1"):
    return {
        "dataset": dataset,
        "task_id": task,
        "group_id": group,
        "split": split,
        "question": "What is two plus two?",
        "expert_ids": list(EXPERTS),
        "qualities": [1.0, 0.5, 0.9, 0.7, 0.8],
        "costs_usd": [0.02, 0.001, 0.03, 0.025, 0.03],
        "images": [],
    }


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(ids)


@pytest.fixture
def character_tokenizer(monkeypatch):
    monkeypatch.setattr(observation, "tokenizer", lambda: CharacterTokenizer())
    observation.observation.cache_clear()


def test_unknown_cost_is_quality_only_for_the_entire_question():
    q = [1, 0.5, 0.9, 0.7, 0.8]
    target = quality_first_target(q, [0.01, None, 0.03, 0.02, 0.03], 0.02, 0.01, 0.05)
    assert target["rewards"] == q and not target["cost_applied"]
    assert target["cost_mask"] == [True, False, True, True, True]


def test_fixed_positive_cost_prefers_cheaper_equal_quality():
    target = quality_first_target(
        [1] * 5, [0.1, 0.001, 0.1, 0.1, 0.1], 0.02, 0.01, 0.05
    )
    assert max(range(5), key=target["target_probs"].__getitem__) == 1
    assert sum(target["target_probs"]) == pytest.approx(1)


@pytest.mark.parametrize("bad", [None, math.nan, math.inf, -0.1, 1.1, True])
def test_invalid_feedback_is_rejected(bad):
    with pytest.raises(ValueError):
        reward_vector([bad, 0, 0, 0, 0], [0.1] * 5, 0.02, 0.1)


def test_cost_scale_is_train_only_and_dataset_balanced():
    a = [
        {"dataset": "a", "split": "train", "arms": [{"cost_usd": 1.0}] * 5}
        for _ in range(10)
    ]
    b = [{"dataset": "b", "split": "train", "arms": [{"cost_usd": 3.0}] * 5}]
    assert fit_cost_scale(a + b) == 2.0
    with pytest.raises(ValueError):
        fit_cost_scale([dict(a[0], split="validation")])


def test_loss_value_and_gradients_match_the_archived_formula():
    target = quality_first_target(
        [1, 0.5, 0.9, 0.7, 0.8], [0.02, 0.001, 0.03, 0.025, 0.03], 0.02, 0.04, 0.05
    )
    a = torch.tensor([0.3, -0.8, 1.2, 0.4, 0.1], requires_grad=True)
    b = a.detach().clone().requires_grad_()
    actual = routing_loss(a, target)
    y = torch.tensor(target["target_probs"], dtype=torch.float32)
    u = torch.tensor(target["rewards"], dtype=torch.float32)
    logp = b.log_softmax(-1)
    expected = -(y * logp).sum() + 0.1 * (u.max() - (logp.exp() * u).sum())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    assert torch.isfinite(a.grad).all()


def test_true_optimizer_update_on_synthetic_cpu_logits():
    logits = torch.nn.Parameter(torch.zeros(5))
    optimizer = torch.optim.AdamW([logits], lr=1e-3, weight_decay=0.01, foreach=False)
    target = quality_first_target([0, 1, 0, 0, 0], [0.1] * 5, 0.02, 0.1, 0.05)
    routing_loss(logits, target).backward()
    torch.nn.utils.clip_grad_norm_([logits], 1.0)
    optimizer.step()
    assert logits[1] > logits[0] and not torch.equal(logits.detach(), torch.zeros(5))


def test_candidate_permutation_preserves_target_alignment(character_tokenizer):
    row = example(dataset="SECRET_DATASET_LABEL", task="SECRET_TASK_LABEL")
    order = [4, 1, 3, 0, 2]
    request = training.request(row, 0.02, order, "question_length_head_tail")
    assert request.expert_ids == tuple(EXPERTS[i] for i in order)
    q = [row["qualities"][i] for i in order]
    c = [row["costs_usd"][i] for i in order]
    target = quality_first_target(q, c, 0.02, 0.04, 0.05)
    assert target["quality"] == q and target["cost_usd"] == c
    assert "qualities" not in request.current and "costs_usd" not in request.current
    assert row["dataset"] not in request.current and row["task_id"] not in request.task


def test_duplicate_ids_and_family_leakage_are_rejected(tmp_path):
    p = tmp_path / "rows.jsonl"
    with pytest.raises(ValueError, match="Duplicate"):
        read_rows(write(p, [example(), example()]))
    with pytest.raises(ValueError, match="crosses split"):
        read_rows(write(p, [example(), example("b", "validation", "a-train")]))
    assert len(read_rows(write(p, [example(), example("b", "train", "b-train")]))) == 2


def test_unlabeled_test_questions_are_accepted(tmp_path):
    row = example(split="test")
    row.pop("qualities")
    row.pop("costs_usd")
    row.pop("expert_ids")
    assert (
        len(read_rows(write(tmp_path / "test.jsonl", [row]), require_labels=False)) == 1
    )


def test_paper_cohort_guard_and_explicit_custom_cohort():
    rows = [example(), example("a", "validation", "a-val", "2")]
    with pytest.raises(ValueError, match="10"):
        validate_schema(rows, "refit")
    validate_schema(rows, "refit", expected_datasets=1, expected_refit_rows=2)


def test_long_question_preserves_original_and_stays_bounded(character_tokenizer):
    row = example()
    row["question"] = "A" * 50000 + "FINAL QUESTION"
    row["routing_question"] = "FINAL QUESTION"
    original = copy.deepcopy(row)
    result = json.loads(training.router_view(row, "question_length_head_tail"))
    assert result["question"] == "FINAL QUESTION" and result[
        "original_characters"
    ] == len(row["question"])
    assert len(json.dumps(result, ensure_ascii=False)) <= 4096
    assert result["tail"].endswith("FINAL QUESTION") and row == original


def test_image_dedup_and_even_sampling_preserve_expert_input(character_tokenizer):
    row = example()
    row["images"] = [
        {"uri": f"/tmp/example-{i}.png", "mime_type": "image/png"} for i in range(35)
    ]
    original = copy.deepcopy(row)
    request = training.request(row, 0.02, input_policy="question_length_head_tail")
    meta = json.loads(request.current)
    assert len(request.images) == 16
    assert (
        meta["selected_image_indices"][0] == 0
        and meta["selected_image_indices"][-1] == 34
        and row == original
    )
    row["images"] = [
        {
            "uri": f"/tmp/example-{i%7}.png",
            "mime_type": "image/png",
            "sha256": str(i % 7),
        }
        for i in range(35)
    ]
    assert (
        len(
            training.request(row, 0.02, input_policy="question_length_head_tail").images
        )
        == 7
    )


def test_validation_selection_excludes_test_and_unknown_cost():
    base = {
        "split": "validation",
        "quality_complete": True,
        "validation_ids_sha256": "same",
        "dataset_names": ["a"],
        "cost_complete": True,
    }
    a = dict(base, id="a", macro_quality=1.0, macro_cost_usd=0.1)
    b = dict(base, id="b", macro_quality=0.996, macro_cost_usd=0.01)
    c = dict(base, id="c", macro_quality=0.994, macro_cost_usd=0.001)
    assert select_validation_candidate([a, b, c])["selected"]["id"] == "b"
    c.update(split="test")
    with pytest.raises(ValueError):
        select_validation_candidate([a, c])
    b.update(cost_complete=False, macro_cost_usd=None)
    assert select_validation_candidate([a, b])["selected"]["id"] == "a"


def test_bundled_upstream_is_hash_checked(tmp_path):
    from adaptrouter_api.native.startlux import load_upstream

    loaded = load_upstream(training.SOURCE_DIR)
    assert loaded.record["commit"] == "cb1d11f1ac9e9b43b95ef3a6468fa29b607c40fd"
    assert len(loaded.record["sha256"]) == 3
    import shutil

    copied = tmp_path / "native"
    shutil.copytree(training.SOURCE_DIR, copied)
    (copied / "startlux_decide/prompt.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="changed"):
        load_upstream(copied)


def test_schema_command_and_fixed_recipe_prepare(tmp_path):
    from adaptrouter_api.tools import main

    path = write(
        tmp_path / "data.jsonl", [example(), example("a", "validation", "a-val", "2")]
    )
    main(["validate-data", "--data", str(path)])
    output = tmp_path / "selection.json"
    main(
        [
            "prepare-refit",
            "--data",
            str(path),
            "--out",
            str(output),
            "--expected-datasets",
            "1",
            "--expected-refit-rows",
            "2",
        ]
    )
    value = json.loads(output.read_text())
    assert value["epoch"] == 3 and value["lambda_cost"] == 0.02
    assert value["cost_scale"] == pytest.approx(sum(example()["costs_usd"]) / 5)
