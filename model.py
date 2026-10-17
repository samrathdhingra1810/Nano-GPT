"""
GPT language model (nanoGPT architecture).

State-dict keys match karpathy/nanoGPT exactly, so checkpoints written by
`train.py config/train_shakespeare_char.py` load without conversion.

Additions over the reference implementation:
  * `forward(..., return_attn=True)` returns per-layer attention maps
    (computed explicitly instead of via the fused SDPA kernel) for visualisation.
  * `generate_with_trace` returns the top-k candidate distribution at every
    sampling step, used by the token-probability panel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False  # nanoGPT default for shakespeare_char


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias (PyTorch's lacks `bias=False` on older versions)."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        att = None
        if return_attn:
            # Explicit path so the attention matrix can be inspected.
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
            mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = self.attn_dropout(att) @ v
        else:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, att


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm transformer block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        a, att = self.attn(self.ln_1(x), return_attn=return_attn)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, att


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None and config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: output layer shares weights with the token embedding.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Parameter count; position embeddings excluded by default, as in nanoGPT."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        """
        Returns (logits, loss) or, with return_attn=True, (logits, loss, attn)
        where attn is a list of n_layer tensors shaped (B, n_head, T, T).
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        attn_maps: List[torch.Tensor] = []
        for block in self.transformer.h:
            x, att = block(x, return_attn=return_attn)
            if return_attn:
                attn_maps.append(att)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])  # only the last position is needed
            loss = None

        if return_attn:
            return logits, loss, attn_maps
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        out, _ = self.generate_with_trace(idx, max_new_tokens, temperature, top_k, trace_k=0)
        return out

    @torch.no_grad()
    def generate_with_trace(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        trace_k: int = 5,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Autoregressive sampling (temperature + top-k), identical to nanoGPT's
        `generate`. Also records, for every step, the `trace_k` most likely
        next tokens and their probabilities under the *sampling* distribution
        (after temperature and top-k), plus the id actually chosen.

        Returns (idx, trace) where trace is a list of dicts:
            {"top_ids": [...], "top_probs": [...], "chosen": int, "chosen_prob": float}
        """
        temperature = max(float(temperature), 1e-5)
        trace = []
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1, generator=generator)
            if trace_k:
                p, ids = torch.topk(probs[0], trace_k)
                chosen = int(idx_next[0, 0])
                trace.append({
                    "top_ids": ids.tolist(),
                    "top_probs": p.tolist(),
                    "chosen": chosen,
                    "chosen_prob": float(probs[0, chosen]),
                })
            idx = torch.cat((idx, idx_next), dim=1)
        return idx, trace
