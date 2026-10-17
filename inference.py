"""
Checkpoint loading, character tokenizer, sampling and attention extraction.

Checkpoints are looked up in this order for each run:
  1. ./checkpoints/<filename>                       (committed via Git LFS)
  2. Hugging Face Hub model repo in $HF_MODEL_REPO   (downloaded on first use)
If neither exists the model is built with random weights and flagged
`trained=False`, so the UI stays usable and says clearly that it is untrained.
"""
from __future__ import annotations

import io
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from model import GPT, GPTConfig

ROOT = Path(__file__).parent
CKPT_DIR = ROOT / "checkpoints"

# The 65 distinct characters of Tiny Shakespeare, in sorted order. This is exactly
# what nanoGPT's data/shakespeare_char/prepare.py writes to meta.pkl, so it is a
# safe fallback when meta.pkl is not shipped alongside the checkpoints.
SHAKESPEARE_CHARS = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

RUNS: Dict[str, dict] = {
    "Dropout 0.2": {
        "file": "dropout_0.2.pt",
        "dropout": 0.2,
        "tagline": "Regularised baseline · best val 1.468 (iter 2,000)",
    },
    "Dropout 0.0": {
        "file": "dropout_0.0.pt",
        "dropout": 0.0,
        "tagline": "Ablated · best val 1.559 (iter 1,000) → 2.414 at iter 2,000",
    },
}


class CharTokenizer:
    def __init__(self, chars: str = SHAKESPEARE_CHARS, stoi: Optional[dict] = None, itos: Optional[dict] = None):
        if stoi is None or itos is None:
            stoi = {ch: i for i, ch in enumerate(chars)}
            itos = {i: ch for i, ch in enumerate(chars)}
        self.stoi, self.itos = stoi, itos

    @classmethod
    def load(cls) -> "CharTokenizer":
        meta = CKPT_DIR / "meta.pkl"
        if meta.exists():
            try:
                with open(meta, "rb") as f:
                    m = pickle.load(f)
                return cls(stoi=m["stoi"], itos=m["itos"])
            except Exception:
                pass
        return cls()

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def unknown_chars(self, s: str) -> List[str]:
        return sorted({c for c in s if c not in self.stoi})

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)


@dataclass
class LoadedModel:
    name: str
    model: GPT
    trained: bool
    source: str
    iter_num: Optional[int] = None
    best_val_loss: Optional[float] = None
    n_params: int = 0
    warnings: List[str] = field(default_factory=list)


def _resolve_checkpoint(filename: str) -> Optional[Path]:
    local = CKPT_DIR / filename
    if local.exists() and local.stat().st_size > 1024:  # skip un-pulled LFS pointers
        return local
    repo = os.environ.get("HF_MODEL_REPO", "").strip()
    if repo:
        try:
            from huggingface_hub import hf_hub_download

            return Path(hf_hub_download(repo_id=repo, filename=filename))
        except Exception:
            return None
    return None


def _clean_state_dict(sd: dict) -> dict:
    # torch.compile prefixes keys with "_orig_mod."; the causal-mask buffer
    # ("attn.bias") exists only in nanoGPT's non-flash path and is not a parameter.
    out = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.endswith(".attn.bias"):
            continue
        out[k] = v
    return out


def load_model(name: str, vocab_size: int, device: str = "cpu") -> LoadedModel:
    spec = RUNS[name]
    warnings: List[str] = []
    path = _resolve_checkpoint(spec["file"])

    if path is not None:
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            args = dict(ckpt.get("model_args", {}))
            cfg = GPTConfig(**{k: v for k, v in args.items() if k in GPTConfig.__dataclass_fields__})
            model = GPT(cfg)
            missing, unexpected = model.load_state_dict(_clean_state_dict(ckpt["model"]), strict=False)
            if missing or unexpected:
                warnings.append(f"State-dict mismatch: missing={missing[:3]} unexpected={unexpected[:3]}")
            model.to(device).eval()
            return LoadedModel(
                name=name,
                model=model,
                trained=True,
                source=str(path.relative_to(ROOT)) if ROOT in path.parents else f"HF Hub ({path.name})",
                iter_num=ckpt.get("iter_num"),
                best_val_loss=float(ckpt["best_val_loss"]) if "best_val_loss" in ckpt else None,
                n_params=model.get_num_params(),
                warnings=warnings,
            )
        except Exception as e:  # corrupt file, wrong format, etc.
            warnings.append(f"Could not load {spec['file']}: {e}")

    # Fallback: same architecture, random weights, fixed seed.
    torch.manual_seed(1337)
    model = GPT(GPTConfig(vocab_size=vocab_size, dropout=spec["dropout"])).to(device).eval()
    warnings.append(
        f"No trained weights found for '{name}' (expected checkpoints/{spec['file']} "
        "or $HF_MODEL_REPO). Running an untrained model with random weights."
    )
    return LoadedModel(name=name, model=model, trained=False, source="random init",
                       n_params=model.get_num_params(), warnings=warnings)


@dataclass
class Generation:
    prompt: str
    completion: str
    ids: List[int]
    trace: List[dict]


def generate(lm: LoadedModel, tok: CharTokenizer, prompt: str, max_new_tokens: int,
             temperature: float, top_k: int, seed: int) -> Generation:
    ids = tok.encode(prompt) or tok.encode("\n")
    device = next(lm.model.parameters()).device
    x = torch.tensor([ids], dtype=torch.long, device=device)
    g = torch.Generator(device=device).manual_seed(int(seed))
    out, trace = lm.model.generate_with_trace(
        x, max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=int(top_k), trace_k=5, generator=g,
    )
    all_ids = out[0].tolist()
    return Generation(prompt=tok.decode(ids), completion=tok.decode(all_ids[len(ids):]),
                      ids=all_ids, trace=trace)


@torch.no_grad()
def attention_maps(lm: LoadedModel, ids: List[int]) -> torch.Tensor:
    """Attention for the last block_size tokens: tensor (n_layer, n_head, T, T)."""
    ids = ids[-lm.model.config.block_size:]
    device = next(lm.model.parameters()).device
    x = torch.tensor([ids], dtype=torch.long, device=device)
    _, _, att = lm.model(x, return_attn=True)
    return torch.stack([a[0] for a in att]).cpu()


def display_char(c: str) -> str:
    return {"\n": "↵", " ": "␣"}.get(c, c)


def synthesize_speech(text: str) -> Optional[bytes]:
    """MP3 bytes via gTTS (needs outbound internet). Returns None on failure."""
    text = text.strip()
    if not text:
        return None
    try:
        from gtts import gTTS

        buf = io.BytesIO()
        gTTS(text=text[:3000], lang="en", tld="co.uk").write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None
