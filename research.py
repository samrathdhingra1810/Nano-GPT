"""
Single source of truth for the paper's numbers, text and citations.

Every figure here is transcribed from the manuscript (assets/paper.pdf).
Charts, the results table and the chatbot's grounding prompt all read from this
module, so the dashboard cannot drift from the paper.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
PAPER_PDF = ROOT / "assets" / "paper.pdf"
LOG_DIR = ROOT / "logs"

# Set this to your GitHub "owner/repo" once the project is pushed; it drives the
# Colab badge and the "code" links.
GITHUB_REPO = "your-username/nanogpt-dropout-study"
COLAB_NOTEBOOK_PATH = "notebooks/reproduce.ipynb"

TITLE = "Dropout and Overfitting in a Character-Level Transformer Trained on Tiny Shakespeare"
AUTHOR = "Samrath Singh Dhingra"
DATE = "July 2026"

UNIFORM_BASELINE = math.log(65)  # 4.174 nats/char

# Table 1: evaluated loss (nats/char), mean over 200 random batches, dropout off.
EVAL = pd.DataFrame(
    {
        "iter": [0, 500, 1000, 1500, 2000] * 2,
        "run": ["Dropout 0.2"] * 5 + ["Dropout 0.0"] * 5,
        "train": [4.287, 1.528, 1.269, 1.150, 1.054, 4.287, 1.396, 1.105, 0.759, 0.396],
        "val": [4.282, 1.728, 1.530, 1.476, 1.468, 4.282, 1.617, 1.559, 1.829, 2.414],
    }
)
# Gap as printed in the paper (computed from unrounded losses, so it can differ from
# val - train above in the last digit, e.g. -0.01 at iteration 0).
EVAL["gap"] = [-0.01, 0.20, 0.26, 0.33, 0.41, -0.01, 0.22, 0.45, 1.07, 2.02]

BEST = {
    "Dropout 0.2": {"val": 1.468, "iter": 2000},
    "Dropout 0.0": {"val": 1.559, "iter": 1000},
}

KEY_METRICS = {
    "params": "10.65M",
    "corpus_chars": "1,115,394",
    "train_tokens": "1,003,854",
    "val_tokens": "111,540",
    "vocab": 65,
    "best_val_d02": 1.468,
    "best_val_d00": 1.559,
    "final_gap_d02": 0.41,
    "final_gap_d00": 2.02,
    "bpc_d02": 2.12,
    "ppl_d02": 4.34,
    "ppl_d00_final": 11.2,
    "word_overlap_d02": "97.1% (933/961)",
    "word_overlap_d00": "93.8% (880/938)",
}

COMPUTE = [
    ("GPU", "1 × NVIDIA T4 (free Google Colab)"),
    ("Wall-clock per run", "≈ 25 min"),
    ("Step time", "0.55 s / iteration"),
    ("Evaluation cost", "79 s per evaluation (200 batches × 2 splits)"),
    ("Iterations", "2,000 of 5,000 configured (LR still 7.05 × 10⁻⁴)"),
    ("Tokens per iteration", "64 × 256 = 16,384"),
    ("Tokens seen per run", "32.8M  (32.6 epochs of the training split)"),
    ("Optimiser", "AdamW, β = (0.9, 0.99), weight decay 0.1, grad-clip 1.0"),
    ("Learning-rate schedule", "Linear warm-up to 10⁻³ over 100 iters, cosine to 10⁻⁴ over 5,000"),
    ("Seed", "1337 (shared by both runs)"),
]

ARCHITECTURE = [
    ("Type", "Decoder-only transformer (nanoGPT)"),
    ("Layers × heads", "6 × 6"),
    ("Embedding width", "384"),
    ("Context length", "256 characters"),
    ("Feed-forward", "4× width, GELU"),
    ("Normalisation", "Pre-LayerNorm"),
    ("Embeddings", "Token + learned position; output layer tied to token embedding"),
    ("Dropout sites", "Embeddings, attention weights, each residual branch output"),
    ("Parameters", "10.65M"),
]

ABSTRACT = (
    "I train a 10.65M-parameter GPT-style transformer to predict the next character of the "
    "Tiny Shakespeare corpus (1.1M characters) and ask how much its built-in dropout of 0.2 "
    "matters. Two runs of 2,000 iterations are compared that are identical in data, seed, "
    "architecture and optimiser, and differ only in dropout (0.2 or 0.0). With dropout the "
    "validation loss falls to 1.468 nats per character and is still improving slightly at the "
    "end, while the gap to the training loss grows to 0.41. Without dropout the model learns "
    "faster at first, reaches its best validation loss of 1.559 at iteration 1,000, and then "
    "overfits sharply: by iteration 2,000 the training loss is 0.40 and the validation loss has "
    "risen to 2.41. In text sampled from the best checkpoint of each run, 97.1% of generated "
    "words occur in the training text with dropout against 93.8% without. On a corpus this "
    "small, dropout is what keeps this model from memorising its training data. The findings "
    "rest on one seed per setting and loss evaluated only every 500 iterations."
)

CONTRIBUTIONS = [
    ("Controlled ablation",
     "Two runs identical in data, seed (1337), architecture and optimiser; only dropout differs. "
     "Identical iteration-0 losses (4.287 / 4.282) confirm a shared initialisation."),
    ("Overfitting quantified",
     "Without dropout the generalisation gap reaches 2.02 nats, 4.9× the baseline's 0.41, and "
     "validation loss rises from 1.559 to 2.414."),
    ("Sample-level evidence",
     "A vocabulary check over 10 × 500-char samples: 97.1% of words are real training-set words "
     "with dropout vs 93.8% without; the ablated model invents about twice as many non-words."),
    ("Honest limitations",
     "One seed per setting, evaluation every 500 iterations, a truncated LR schedule and only "
     "two dropout values; each is stated with its consequence."),
]

LIMITATIONS = [
    "One seed per setting, so there are no error bars; the 0.091 best-checkpoint difference could be partly noise.",
    "Loss evaluated only every 500 iterations; the true minimum of the no-dropout run lies between 500 and 1,500.",
    "Training stopped at 2,000 of 5,000 configured iterations; the baseline had not converged.",
    "Only dropout 0.0 and 0.2 tested; this shows some dropout helps, not that 0.2 is optimal.",
    "The validation split both selects the checkpoint and reports its loss (slightly optimistic).",
    "Word check measures spelling only; sampled checkpoints differ in training time; one small corpus.",
]

SAMPLES = {
    "Dropout 0.2": (
        "MENENIUS:\nHe is battle, for the rogues of a grave instruments\n"
        "From their envy; the bear-harded betwixt them; who at\n"
        "The time that was begon at himself: he was soldier"
    ),
    "Dropout 0.0": (
        "MENENIUS:\nHe is the content: stay is before him:\n"
        "The lords are pleased, and when he now his noble was\n"
        "our required, and then the hand be cruel and dew"
    ),
}

CITATIONS = [
    {
        "key": "vaswani2017attention",
        "short": "Vaswani et al., 2017",
        "title": "Attention Is All You Need",
        "venue": "NeurIPS 30",
        "role": "The transformer architecture this model instantiates.",
        "bibtex": """@inproceedings{vaswani2017attention,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and
               Jones, Llion and Gomez, Aidan N. and Kaiser, {\\L}ukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {30},
  year      = {2017}
}""",
    },
    {
        "key": "srivastava2014dropout",
        "short": "Srivastava et al., 2014",
        "title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
        "venue": "JMLR 15(56)",
        "role": "The regulariser under study.",
        "bibtex": """@article{srivastava2014dropout,
  title   = {Dropout: A Simple Way to Prevent Neural Networks from Overfitting},
  author  = {Srivastava, Nitish and Hinton, Geoffrey and Krizhevsky, Alex and
             Sutskever, Ilya and Salakhutdinov, Ruslan},
  journal = {Journal of Machine Learning Research},
  volume  = {15},
  number  = {56},
  pages   = {1929--1958},
  year    = {2014}
}""",
    },
    {
        "key": "loshchilov2019decoupled",
        "short": "Loshchilov & Hutter, 2019",
        "title": "Decoupled Weight Decay Regularization",
        "venue": "ICLR 2019",
        "role": "AdamW, the optimiser used for both runs.",
        "bibtex": """@inproceedings{loshchilov2019decoupled,
  title     = {Decoupled Weight Decay Regularization},
  author    = {Loshchilov, Ilya and Hutter, Frank},
  booktitle = {International Conference on Learning Representations},
  year      = {2019}
}""",
    },
    {
        "key": "karpathy2022nanogpt",
        "short": "Karpathy, 2022",
        "title": "nanoGPT",
        "venue": "GitHub",
        "role": "Reference implementation and character-level configuration.",
        "bibtex": """@misc{karpathy2022nanogpt,
  author       = {Karpathy, Andrej},
  title        = {nanoGPT},
  year         = {2022},
  howpublished = {\\url{https://github.com/karpathy/nanoGPT}}
}""",
    },
]

PAPER_BIBTEX = """@misc{dhingra2026dropout,
  title  = {Dropout and Overfitting in a Character-Level Transformer
            Trained on Tiny Shakespeare},
  author = {Dhingra, Samrath Singh},
  year   = {2026},
  month  = jul,
  note   = {Independent research report}
}"""


def load_minibatch_logs() -> dict:
    """
    Optional dense training traces: logs/dropout_0.2.csv and logs/dropout_0.0.csv
    with columns `iter,loss` (written by the reproduction notebook). Returns
    {run_name: DataFrame} for whichever files exist.
    """
    out = {}
    for run, fname in (("Dropout 0.2", "dropout_0.2.csv"), ("Dropout 0.0", "dropout_0.0.csv")):
        p = LOG_DIR / fname
        if p.exists():
            try:
                df = pd.read_csv(p)
                if {"iter", "loss"} <= set(df.columns):
                    out[run] = df
            except Exception:
                pass
    return out


def paper_context() -> str:
    """Grounding document for the research-assistant chatbot."""
    t = EVAL
    rows = "\n".join(
        f"  iter {r.iter:>4} | {r.run} | train {r.train:.3f} | val {r.val:.3f} | gap {r.gap:+.2f}"
        for r in t.itertuples()
    )
    comp = "\n".join(f"  - {k}: {v}" for k, v in COMPUTE)
    arch = "\n".join(f"  - {k}: {v}" for k, v in ARCHITECTURE)
    lims = "\n".join(f"  - {x}" for x in LIMITATIONS)
    refs = "\n".join(f"  [{i + 1}] {c['short']}: {c['title']} ({c['venue']}). {c['role']}"
                     for i, c in enumerate(CITATIONS))
    return f"""PAPER: "{TITLE}", {AUTHOR}, {DATE}.

ABSTRACT
{ABSTRACT}

DATA
Tiny Shakespeare (char-rnn distribution): 1,115,394 characters, one token per character,
vocabulary of 65 symbols. First 90% = training (1,003,854 tokens), last 10% = validation
(111,540 tokens). No separate test set. About ten parameters per training character.

ARCHITECTURE
{arch}

TRAINING & COMPUTE
{comp}
Evaluation: every 500 iterations, mean cross-entropy over 200 random batches per split with
dropout switched off. A checkpoint is written only when validation loss improves, so each
run's saved model is its best at this resolution (iter 2,000 for dropout 0.2; iter 1,000 for
dropout 0.0).

RESULTS TABLE (nats/char; gap = val - train)
{rows}

KEY RESULTS
- Iteration-0 loss 4.28 for both splits, close to ln(65) = 4.17 nats, the entropy of a uniform
  guess over the 65-character vocabulary. (Not stated in the paper, general background: the small
  excess arises because randomly initialised logits are not perfectly uniform.)
- Dropout 0.2: val 1.728 at 500 -> 1.468 at 2,000 = 2.12 bits/char, perplexity 4.34. Over the
  last 500 iters val dropped only 0.008 while train dropped 0.096. Gap widens 0.20 -> 0.41: the
  baseline has begun to overfit but its validation loss has not turned upwards.
- Dropout 0.0: ahead at iter 500 (val 0.11 lower). Best val 1.559 at iter 1,000 (vs 1.530 for
  baseline at the same point). Then train falls almost linearly to 0.396 while val rises to
  1.829 then 2.414 (perplexity 11.2), worse than at iter 500. Final gap 2.02 nats = 4.9x the
  baseline gap.
- Best-checkpoint difference: 0.091 nats (1.468 vs 1.559). At iter 2,000: 0.95 nats.
- Mini-batch loss (dropout active) sits above evaluated train loss in the baseline (1.12-1.17
  vs 1.05); without dropout they coincide (0.39-0.42 vs 0.40): nothing handicaps the network,
  so it uses its full capacity to fit text it has seen ~33 times.
- Samples (10 x 500 chars, temperature 0.8, top-k 200, same seed): both reproduce play format
  (capitalised speaker names, verse line breaks, mostly plausible spelling), neither coherent.
  Words found among the 11,676 distinct training words: 933/961 (97.1%) with dropout vs
  880/938 (93.8%) without. No-dropout model produces ~2x as many non-training words (58 vs 28),
  e.g. "blushion", "betwered", "accunning".

LIMITATIONS
{lims}

REFERENCES
{refs}
  [5] Karpathy 2015, "The Unreasonable Effectiveness of Recurrent Neural Networks" (source of
      the Tiny Shakespeare file).
"""
