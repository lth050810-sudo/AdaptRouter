"""Small random CPU model: exercise the real native attention and gradients.

No pretrained assets, benchmark questions, GPU, or network are used here.
"""

import copy
import pytest
import torch
from torch import nn
import transformers

from adaptrouter_api.native.startlux import StartLuxTrainable
from adaptrouter_api.train import SOURCE_DIR, install_checkpointing
from adaptrouter_api.objective import routing_loss
from adaptrouter_api.reward import quality_first_target


@pytest.mark.skipif(
    not transformers.__version__.startswith("5.8."),
    reason="Native upstream requires transformers 5.8.x",
)
def test_native_letter_readout_checkpointing_and_optimizer_update():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    class TinyModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = Qwen3_5TextModel(config)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        def get_output_embeddings(self):
            return self.lm_head

    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["linear_attention", "full_attention"],
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(13)
    base = TinyModel(config).eval()
    remat = copy.deepcopy(base)
    kwargs = dict(
        tokenizer=None,
        processor=None,
        source_dir=SOURCE_DIR,
        decision_config={
            "prompt_attention": "bidirectional",
            "temperature_by_type": {"choice": 1.0},
            "letter_token_ids": list(range(26)),
        },
        base_id="synthetic-tiny-model",
        base_revision="random-test",
        max_input_tokens=64,
    )
    a = StartLuxTrainable(base, **kwargs)
    b = StartLuxTrainable(remat, **kwargs)
    assert a.bidirectional_check["late_token_reach"] > 0
    assert a.bidirectional_check["padding_invariance_max_diff"] < 1e-6
    assert install_checkpointing(b) == 2
    encoded = {"input_ids": torch.tensor([[1, 4, 7, 9, 2, 8]])}
    target = quality_first_target([0.2, 1, 0.1, 0.4, 0.3], [0.1] * 5, 0.02, 0.1, 0.05)
    logits_a = a.forward_encoded(encoded, 5)
    logits_b = b.forward_encoded(encoded, 5)
    torch.testing.assert_close(logits_a, logits_b, rtol=0, atol=0)
    routing_loss(logits_a, target).backward()
    routing_loss(logits_b, target).backward()
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert pa.requires_grad and pb.requires_grad
        assert pa.grad is not None, name
        torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-5, atol=1e-6)
        assert torch.isfinite(pb.grad).all()
    before = b.model.get_output_embeddings().weight.detach().clone()
    optimizer = torch.optim.AdamW(
        b.parameters(), lr=1e-3, weight_decay=0.01, foreach=False
    )
    torch.nn.utils.clip_grad_norm_(b.parameters(), 1.0)
    optimizer.step()
    assert not torch.equal(before, b.model.get_output_embeddings().weight)
