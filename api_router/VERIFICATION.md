# Release checks · 2026-11-11

- Python 3.11, PyTorch 2.7.1+cu128, Transformers 5.8.1, tokenizers 0.22.2.
- `python -m pytest -q`: **21 passed** on CPU.
- Random tiny Qwen3.5: native bidirectional attention and letter readout, checkpointed / uncheckpointed forward and gradient agreement, and a real optimizer update passed.
- Published global FP32 weights: verified local release files and routed two synthetic questions on CPU with network access disabled; both returned finite five-expert probability distributions. Expert API calls: **0**.
- Installable wheel built successfully; package includes the pinned upstream helpers and their license.

These checks verify code and packaging. Full CUDA training, original benchmark feedback collection, specialized refits, and final Table 3 metrics were not rerun.
