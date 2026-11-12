"""Deterministic, label-free routing observation; expert inputs remain untouched.

Native Qwen token budget is for the routing observation only. Long documents are
never split into expert requests. Reference answers and benchmark names are not
accepted by this transform.
"""

from functools import lru_cache
from pathlib import Path
import json

TOKENIZER = None
TEXT_BUDGET = 4096


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer

    if TOKENIZER is None:
        raise ValueError(
            "Set the local base/tokenizer directory before building routing observations"
        )
    return AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)


def token_excerpt(text, head, tail):
    tok = tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= head + tail:
        return text
    return (
        tok.decode(ids[:head], skip_special_tokens=False)
        + "\n[router observation excerpt]\n"
        + tok.decode(ids[-tail:], skip_special_tokens=False)
    )


@lru_cache(maxsize=16384)
def observation(text, question, document_chars, policy):
    text = text.replace("<image>", "[image attachment]")
    if policy == "full":
        return text
    if policy != "question_length_head_tail":
        raise ValueError("Unregistered router input policy")
    tok = tokenizer()
    # Do not tokenize megabyte source documents merely to choose an expert.
    if (
        len(text) <= 8000
        and len(tok.encode(text, add_special_tokens=False)) <= TEXT_BUDGET
    ):
        return text
    head_chars, tail_chars = 1024, 2048
    q = question.replace("<image>", "[image attachment]") if question else ""
    q = token_excerpt(q, 1024, 512)
    head = token_excerpt(text[:head_chars], 256, 128)
    tail = token_excerpt(text[-tail_chars:], 256, 512)
    for stage in range(8):
        view = json.dumps(
            {
                "router_input_policy": policy,
                "question": q,
                "original_characters": len(text),
                "original_utf8_bytes": len(text.encode()),
                "document_characters": document_chars,
                "head": head,
                "tail": tail,
                "routing_text_token_budget": TEXT_BUDGET,
                "expert_receives_complete_original": True,
            },
            ensure_ascii=False,
        )
        if len(tok.encode(view, add_special_tokens=False)) <= TEXT_BUDGET:
            return view
        # A symbolic/question-only item can itself be long; preserve its opening and
        # final question, explicitly within the same budget used at train and test.
        n = max(16, 512 // (2**stage))
        q = token_excerpt(q, n, n)
        head = token_excerpt(head, n // 2, n // 2)
        tail = token_excerpt(tail, n, n)
    raise ValueError(
        "Routing observation metadata cannot fit its registered token budget"
    )


def configure_tokenizer(path):
    global TOKENIZER
    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError("Tokenizer directory does not exist")
    TOKENIZER = path
    tokenizer.cache_clear()
    observation.cache_clear()
