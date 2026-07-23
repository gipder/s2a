"""Shared embedding-extraction recipe for zero-shot retrieval.

Ported as-is from noise_aware_slu/retriever/src/embedding_utils.py (STOP
retriever project): PromptEOL-wrapped + last-token pooling, found there to
beat plain mean-pooling by a wide margin on zero-shot retrieval (domain_acc
48.3% -> 82.0%, intent_acc 14.3% -> 49.3% with Qwen3-0.6B). Reused verbatim
here since it's a generic sentence-embedding recipe, not STOP-specific.
"""
from __future__ import annotations

import torch

PROMPTEOL_TEMPLATE = 'This sentence: "{text}" means in one word:"'


def wrap_prompteol(texts: list[str]) -> list[str]:
    return [PROMPTEOL_TEMPLATE.format(text=t) for t in texts]


def pool_last(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Assumes right-padding, so per row the real tokens are a contiguous
    prefix [0, seq_len) and the last real token is at seq_len - 1."""
    seq_lens = attention_mask.sum(dim=1)
    idx = (seq_lens - 1).clamp(min=0)
    return hidden[torch.arange(hidden.size(0), device=hidden.device), idx, :]


def encode_texts(
    texts: list[str], tokenizer, model, batch_size: int, max_length: int, no_grad: bool = True,
) -> torch.Tensor:
    """PromptEOL-wrapped + last-token pooling, L2-normalized.

    no_grad=True (default, all inference callers: eval_zeroshot_retriever.py,
    filter_synthetic_utterances.py) detaches the graph and moves each chunk to
    CPU as it goes, which is what you want when just embedding a big corpus
    once. no_grad=False (train_retriever.py) keeps everything on-device with
    gradients attached instead -- callers there always pass a single
    same-size chunk (batch_size == len(texts)) so there's no cross-chunk
    graph-fragmentation concern to worry about."""
    wrapped = wrap_prompteol(texts)
    device = next(model.parameters()).device

    def _encode_chunk(batch: list[str]) -> torch.Tensor:
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        out = model(**enc)
        pooled = pool_last(out.last_hidden_state, enc["attention_mask"])
        return torch.nn.functional.normalize(pooled.float(), dim=-1)

    if not no_grad:
        return _encode_chunk(wrapped)

    vecs = []
    with torch.no_grad():
        for i in range(0, len(wrapped), batch_size):
            vecs.append(_encode_chunk(wrapped[i:i + batch_size]).cpu())
    return torch.cat(vecs, dim=0)
