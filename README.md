---
title: nanoGPT Dropout Study
emoji: 🎭
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: true
license: mit
short_description: Dropout vs overfitting in a 10.65M char-level transformer
tags:
  - nanogpt
  - transformer
  - regularization
  - dropout
  - shakespeare
  - streamlit
---

# Dropout and Overfitting in a Character-Level Transformer

An interactive companion to my research report on how dropout regularises a 10.65M-parameter
nanoGPT model trained on Tiny Shakespeare.

| | Dropout 0.2 | Dropout 0.0 |
|---|---|---|
| Best validation loss | **1.468** (iter 2,000) | 1.559 (iter 1,000) |
| Validation loss at iter 2,000 | 1.468 | 2.414 |
| Generalisation gap at iter 2,000 | 0.41 | **2.02** |
| Generated words found in training text | 97.1% | 93.8% |

## What's inside

| Tab | Features |
|---|---|
| 🎭 **Generation Workbench** | Switch checkpoints · temperature / top-k / length / seed · side-by-side comparison · top-5 next-token probabilities at any step · per-layer, per-head attention heatmaps · read-aloud (gTTS with a browser-voice fallback) |
| 📈 **Results** | Interactive train/val loss curves with the ln 65 ≈ 4.17 baseline · validation zoom · generalisation gap · Table 1 with best values highlighted |
| 💬 **Research Assistant** | Gemini 2.5 Flash, grounded in the full paper, with suggested questions |
| 📄 **Manuscript** | Rendered PDF reader · abstract · contributions · citations · compute specs |
| 🛠️ **Architecture · Cite · Reproduce** | Model/compute tables · BibTeX · Colab badge · reproduction commands |

## Project layout

```
app.py                 Streamlit UI (five tabs, caching, session state)
model.py               GPT, GPTConfig, Block, CausalSelfAttention, MLP (nanoGPT-compatible keys)
inference.py           checkpoint loading + fallbacks, tokenizer, sampling, attention, TTS
research.py            every number, sample and citation from the paper
charts.py              Plotly figures
assistant.py           Gemini client and grounded system prompt
notebooks/reproduce.ipynb   retrains both runs on a Colab T4 and exports assets
assets/paper.pdf       the manuscript
checkpoints/           dropout_0.2.pt, dropout_0.0.pt, meta.pkl (Git LFS)
logs/                  optional mini-batch traces: dropout_0.2.csv, dropout_0.0.csv
Dockerfile             Hugging Face Spaces container
```

## 1 · Get the checkpoints

Run [`notebooks/reproduce.ipynb`](notebooks/reproduce.ipynb) in Colab (T4, about 50 min for both runs).
It writes slim checkpoints (weights only, about 43 MB each) plus loss logs. Then either:

- **Commit them to the Space:** copy into `checkpoints/` and `logs/`. `.gitattributes` already routes
  `*.pt` through Git LFS.
- **Host them on the Hub:** upload to a model repo and set the Space variable
  `HF_MODEL_REPO=your-username/nanogpt-shakespeare-dropout`.

If you already have nanoGPT `out-*/ckpt.pt` files, they load as they are: rename them to
`dropout_0.2.pt` and `dropout_0.0.pt`. Optimizer state is ignored, and `_orig_mod.` prefixes from
`torch.compile` are stripped.

**No weights?** The app still runs: missing models fall back to random initialisation, and a banner
says so clearly.

## 2 · Deploy to Hugging Face Spaces

1. Create a Space → **Docker** → *Blank*, hardware **CPU basic** (free) is enough.
2. Push this folder:
   ```bash
   git lfs install
   git remote add space https://huggingface.co/spaces/<you>/nanogpt-dropout-study
   git push space main
   ```
3. **Settings → Variables and secrets**
   - Secret `GEMINI_API_KEY`: from <https://aistudio.google.com/apikey> (the chatbot turns read-only without it)
   - Variable `HF_MODEL_REPO` *(optional)*: if checkpoints live in a model repo
4. Edit `GITHUB_REPO` in `research.py` so the Colab badge points at your repository.

> Spaces now ships Streamlit through the Docker SDK. The YAML header above plus the `Dockerfile`
> is all it needs. `requirements.txt` pulls CPU-only PyTorch to keep the image small.

## 3 · Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your Gemini key
streamlit run app.py
```

## Error handling

| Situation | Behaviour |
|---|---|
| Checkpoint missing, an un-pulled LFS pointer, or corrupt | Random-init model of the same architecture, warning banner, 🟠 status in sidebar |
| `meta.pkl` missing | Built-in 65-character Tiny Shakespeare vocabulary (identical to nanoGPT's) |
| Prompt has out-of-vocabulary characters | They are listed and dropped before encoding |
| No Gemini key | Chat input disabled with setup instructions; key can also be pasted in the sidebar |
| Gemini auth / rate-limit / network errors | Readable error; the unanswered turn is removed so history stays valid |
| gTTS unreachable | Falls back to the browser's Web Speech API |
| PDF missing or unrenderable | Error message; download button when possible |

## Citation

```bibtex
@misc{dhingra2026dropout,
  title  = {Dropout and Overfitting in a Character-Level Transformer
            Trained on Tiny Shakespeare},
  author = {Dhingra, Samrath Singh},
  year   = {2026},
  month  = jul,
  note   = {Independent research report}
}
```

Model code adapted from [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (MIT).
