"""
nanoGPT Dropout Study: interactive research dashboard.

    streamlit run app.py
"""
from __future__ import annotations

import html
import json
import random
import time

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch

import charts
import research as R
from assistant import SUGGESTED_QUESTIONS, AssistantError, resolve_api_key, stream_reply
from inference import (RUNS, CharTokenizer, attention_maps, display_char, generate,
                       load_model, synthesize_speech)

st.set_page_config(
    page_title="nanoGPT Dropout Study · Samrath Singh Dhingra",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)
torch.set_num_threads(max(1, torch.get_num_threads()))

# --------------------------------------------------------------------------- styling
st.markdown(
    """
<style>
:root { --ink:#0b1220; --ink2:#475569; --line:#e2e8f0; --blue:#2a78d6; --orange:#eb6834; --card:#f8fafc; }
.block-container { padding-top: 2.5rem; max-width: 1280px; }
h1, h2, h3 { color: var(--ink); letter-spacing: -0.01em; }
.hero-title { font-size: 2.05rem; font-weight: 750; line-height: 1.35; margin: 0 0 .35rem 0; color: var(--ink); }
.hero-sub { color: var(--ink2); font-size: 1.02rem; margin-bottom: .2rem; }
.pill { display:inline-block; padding:.12rem .55rem; border-radius:999px; font-size:.78rem; font-weight:600;
        border:1px solid var(--line); background:#fff; color:var(--ink2); margin-right:.35rem; }
.pill.blue { border-color: var(--blue); color: var(--blue); }
.pill.orange { border-color: var(--orange); color: #b8461a; }
.card { border:1px solid var(--line); background:var(--card); border-radius:12px; padding:1rem 1.1rem; height:100%; }
.card h4 { margin:0 0 .35rem 0; font-size:1rem; color:var(--ink); }
.card p { margin:0; color:var(--ink2); font-size:.93rem; line-height:1.45; }
.gen-output { font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .92rem;
              line-height: 1.5; white-space: pre-wrap; word-break: break-word; background: #0b1220; color: #e2e8f0;
              border-radius: 10px; padding: 1rem 1.1rem; min-height: 120px; max-height: 520px; overflow-y: auto;
              border-left: 5px solid var(--accent, #2a78d6); }
.gen-output .prompt { color: #fbbf24; font-weight: 700; }
.ctx { font-family: "JetBrains Mono", ui-monospace, monospace; background:#f1f5f9; padding:.5rem .7rem; border-radius:8px;
       white-space: pre-wrap; color: var(--ink); font-size:.9rem; }
.ctx mark { background:#2a78d6; color:#fff; padding:0 .15rem; border-radius:3px; }
.callout { border-left: 4px solid #64748b; background:#f8fafc; padding:.75rem 1rem; border-radius:6px; color:var(--ink); }
div[data-testid="stMetricValue"] { font-weight: 700; color: var(--ink); }
div[data-testid="stMetricLabel"] p { color: var(--ink2); font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner=False)
def get_tokenizer() -> CharTokenizer:
    return CharTokenizer.load()


@st.cache_resource(show_spinner="Loading checkpoint…")
def get_model(name: str):
    return load_model(name, vocab_size=get_tokenizer().vocab_size)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_tts(text: str):
    return synthesize_speech(text)


@st.cache_data(show_spinner="Rendering manuscript…")
def render_pdf_pages(path: str, scale: float = 1.8):
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    return [pdf[i].render(scale=scale).to_pil() for i in range(len(pdf))]


tok = get_tokenizer()
st.session_state.setdefault("generations", {})  # run name -> Generation
st.session_state.setdefault("gen_settings", None)
st.session_state.setdefault("chat", [])
st.session_state.setdefault("pending_question", None)
st.session_state.setdefault("seed", 1337)


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙️ Sampling controls")
    model_choice = st.radio(
        "Checkpoint",
        list(RUNS),
        captions=[RUNS[r]["tagline"] for r in RUNS],
        help="Both checkpoints are the best saved model of each run (saved on validation improvement).",
    )
    compare = st.toggle("Side-by-side comparison", value=False,
                        help="Generate from both checkpoints with the same prompt, settings and seed.")
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1,
                            help="Divides the logits. <1 sharpens, >1 flattens the distribution. Paper: 0.8.")
    top_k = st.slider("Top-K", 1, 500, 200, 1,
                      help="Sample only among the K most likely characters. The vocabulary has 65, "
                           "so any K ≥ 65 means no truncation. Paper: 200.")
    max_new = st.slider("Max new tokens", 50, 1000, 200, 50, help="One token = one character.")
    seed_col, dice_col = st.columns([3, 1])
    with dice_col:
        st.write("")
        st.write("")
        if st.button("🎲", help="Random seed", width="stretch"):
            st.session_state["seed"] = random.randint(0, 2**31 - 1)
    with seed_col:
        seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, step=1, key="seed")

    st.divider()
    st.markdown("### 🔑 Gemini API key")
    user_key = st.text_input("Optional if set as a Space secret", type="password",
                             placeholder="GEMINI_API_KEY", label_visibility="collapsed")
    api_key = resolve_api_key(user_key)
    st.caption("✅ Key available" if api_key else "No key: the chatbot tab will be read-only.")

    st.divider()
    st.markdown("### 📦 Model status")
    for r in RUNS:
        lm = get_model(r)
        icon = "🟢" if lm.trained else "🟠"
        detail = f"iter {lm.iter_num}" if lm.iter_num is not None else lm.source
        st.caption(f"{icon} **{r}**: {lm.n_params / 1e6:.2f}M params · {detail}")


# --------------------------------------------------------------------------- hero
st.markdown(f'<div class="hero-title">{html.escape(R.TITLE)}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="hero-sub">{R.AUTHOR} · {R.DATE} · Independent research · '
    '<span class="pill blue">Dropout 0.2</span><span class="pill orange">Dropout 0.0</span>'
    '<span class="pill">nanoGPT · 10.65M params</span><span class="pill">NVIDIA T4</span></div>',
    unsafe_allow_html=True,
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Best val · dropout 0.2", "1.468", "-0.091 vs dropout 0.0", delta_color="inverse")
m2.metric("Best val · dropout 0.0", "1.559", "→ 2.414 at iter 2,000", delta_color="off")
m3.metric("Final gap · dropout 0.0", "2.02", "4.9× the 0.41 baseline", delta_color="off")
m4.metric("Compute per run", "≈ 25 min", "1 × NVIDIA T4", delta_color="off")

missing = [r for r in RUNS if not get_model(r).trained]
if missing:
    st.warning(
        "**Demo mode:** trained weights were not found for "
        + ", ".join(f"`{m}`" for m in missing)
        + ". Those models run with random weights, so their text is noise. Add `checkpoints/dropout_0.2.pt` "
        "and `checkpoints/dropout_0.0.pt`, or set `HF_MODEL_REPO`. See the README.",
        icon="⚠️",
    )

tab_gen, tab_res, tab_chat, tab_paper, tab_repro = st.tabs([
    "🎭 Generation Workbench",
    "📈 Results & Loss Curves",
    "💬 Research Assistant",
    "📄 Manuscript",
    "🛠️ Architecture · Cite · Reproduce",
])


# =========================================================================== 1. WORKBENCH
def render_output(gen, accent: str) -> None:
    st.markdown(
        f'<div class="gen-output" style="--accent:{accent}"><span class="prompt">{html.escape(gen.prompt)}</span>'
        f"{html.escape(gen.completion)}</div>",
        unsafe_allow_html=True,
    )


def read_aloud(text: str, key: str) -> None:
    c1, c2 = st.columns([1, 1])
    if c1.button("🔊 Read aloud", key=f"tts_{key}", width="stretch"):
        with st.spinner("Synthesising speech…"):
            audio = cached_tts(text)
        if audio:
            st.audio(audio, format="audio/mp3")
        else:
            st.info("Server-side speech failed (no internet?). Use the browser voice instead.")
    with c2:
        # Browser Web Speech API fallback: works offline, no server round-trip.
        components.html(
            f"""
<button id="b" style="width:100%;padding:.42rem;border-radius:8px;border:1px solid #cbd5e1;background:#fff;
 font:600 14px system-ui;color:#0b1220;cursor:pointer">🗣️ Browser voice</button>
<script>
const t = {json.dumps(text[:4000])}; let on = false;
document.getElementById('b').onclick = () => {{
  const s = window.speechSynthesis; if (!s) return;
  if (on) {{ s.cancel(); on = false; return; }}
  const u = new SpeechSynthesisUtterance(t.replace(/\\n+/g, '. ')); u.lang = 'en-GB'; u.rate = 0.95;
  u.onend = () => on = false; s.speak(u); on = true;
}};
</script>""",
            height=44,
        )


with tab_gen:
    left, right = st.columns([3, 2], gap="large")
    with left:
        prompt = st.text_area("Prompt", value="ROMEO:\n", height=130,
                              help="The model only knows the 65 characters of Tiny Shakespeare.")
        unknown = tok.unknown_chars(prompt)
        if unknown:
            st.caption(f"⚠️ Characters outside the vocabulary will be dropped: "
                       f"{' '.join(repr(c) for c in unknown[:12])}")
        go_btn = st.button("✨ Generate text", type="primary", width="stretch")
    with right:
        st.markdown("**Current settings**")
        st.markdown(
            f"`{'both checkpoints' if compare else model_choice}` · T = **{temperature}** · "
            f"top-k = **{top_k}** · **{max_new}** chars · seed **{seed}**"
        )
        st.markdown(
            '<div class="callout">Try the paper\'s setting (T 0.8, top-k 200), then compare both '
            "checkpoints. The no-dropout model invents about twice as many non-words "
            "(<i>blushion</i>, <i>betwered</i>, <i>accunning</i>). Because both runs share the "
            "sampling seed, outputs often open identically and then diverge.</div>",
            unsafe_allow_html=True,
        )

    if go_btn:
        targets = list(RUNS) if compare else [model_choice]
        st.session_state["generations"] = {}
        for r in targets:
            with st.spinner(f"Sampling {max_new} characters from {r}…"):
                t0 = time.perf_counter()
                try:
                    g = generate(get_model(r), tok, prompt, max_new, temperature, top_k, int(seed))
                    st.session_state["generations"][r] = g
                    st.session_state.setdefault("timings", {})[r] = time.perf_counter() - t0
                except Exception as e:
                    st.error(f"Generation failed for {r}: {e}")
        st.session_state["gen_settings"] = dict(T=temperature, k=top_k, n=max_new, seed=int(seed))

    gens = st.session_state["generations"]
    if gens:
        st.divider()
        cols = st.columns(len(gens), gap="large")
        for col, (r, g) in zip(cols, gens.items()):
            with col:
                lm = get_model(r)
                secs = st.session_state.get("timings", {}).get(r, 0)
                st.markdown(f"#### {r}" + ("" if lm.trained else "  ·  :orange[untrained weights]"))
                st.caption(f"{len(g.completion)} chars in {secs:.1f}s · {RUNS[r]['tagline']}")
                render_output(g, charts.COLORS[r])
                read_aloud(g.prompt + g.completion, key=r)
                st.download_button("⬇️ Download .txt", g.prompt + g.completion,
                                   file_name=f"sample_{r.replace(' ', '_').lower()}.txt",
                                   key=f"dl_{r}", width="stretch")

        # ---------------- token probabilities
        st.divider()
        st.subheader("🔢 Next-token probabilities")
        insp_run = st.selectbox("Inspect run", list(gens), key="insp_run") if len(gens) > 1 else next(iter(gens))
        g = gens[insp_run]
        if g.trace:
            n_steps = len(g.trace)
            step = st.slider("Generation step", 1, n_steps, n_steps, key="tp_step",
                             help="Defaults to the latest generated token.") - 1
            tr = g.trace[step]
            chars = [display_char(tok.itos[i]) for i in tr["top_ids"]]
            chosen_in_top = tr["chosen"] in tr["top_ids"]
            pc1, pc2 = st.columns([3, 2], gap="large")
            with pc1:
                st.plotly_chart(
                    charts.token_probs(chars, tr["top_probs"],
                                       tr["top_ids"].index(tr["chosen"]) if chosen_in_top else None),
                    config={"displayModeBar": False},
                )
            with pc2:
                pos = len(g.ids) - n_steps + step
                ctx = tok.decode(g.ids[max(0, pos - 60):pos])
                st.markdown("**Context → sampled character**")
                st.markdown(
                    f'<div class="ctx">{html.escape(ctx)}<mark>{html.escape(display_char(tok.itos[tr["chosen"]]))}</mark></div>',
                    unsafe_allow_html=True,
                )
                probs = np.array(tr["top_probs"])
                st.markdown(
                    f"- Sampled **'{display_char(tok.itos[tr['chosen']])}'** with p = **{tr['chosen_prob']:.1%}**\n"
                    f"- Top-5 hold **{probs.sum():.1%}** of the mass\n"
                    f"- Mean p(sampled) over all steps: **{np.mean([t['chosen_prob'] for t in g.trace]):.1%}**"
                )

        # ---------------- attention
        st.divider()
        st.subheader("🔍 Attention heatmap")
        lm = get_model(insp_run)
        cfg = lm.model.config
        a1, a2, a3 = st.columns(3)
        layer = a1.selectbox("Layer", list(range(1, cfg.n_layer + 1)), index=cfg.n_layer - 1, key="att_layer")
        head_opt = a2.selectbox("Head", ["Mean of all heads"] + [f"Head {h}" for h in range(1, cfg.n_head + 1)],
                                key="att_head")
        window = a3.slider("Characters shown (most recent)", 16, min(128, cfg.block_size), 48, 8, key="att_win")
        with st.spinner("Computing attention…"):
            # Run the model over its full context (last 256 chars) so the weights are the ones it
            # actually used, then show the most recent `window` × `window` corner.
            att = attention_maps(lm, g.ids)  # (L, H, T, T)
        window = min(window, att.shape[-1])
        mat = att[layer - 1].mean(0) if head_opt.startswith("Mean") else att[layer - 1, int(head_opt.split()[1]) - 1]
        mat = mat[-window:, -window:]
        labels = [display_char(tok.itos[i]) for i in g.ids[-window:]]
        st.plotly_chart(
            charts.attention_heatmap(mat.numpy(), labels, f"{insp_run} · layer {layer} · {head_opt.lower()}"),
        )
        st.caption("Each row is a query character and shows how it distributes attention over earlier "
                   "characters (causal mask: upper triangle empty). Weights come from the full 256-character "
                   "context, so a row can sum to less than 1 when attention falls outside the window. "
                   "↵ = newline, ␣ = space.")
    else:
        st.info("Enter a prompt and press **Generate text**. Turn on side-by-side comparison in the sidebar "
                "to sample both checkpoints with identical settings.", icon="🎭")


# =========================================================================== 2. RESULTS
with tab_res:
    st.subheader("Training curves")
    f1, f2, f3 = st.columns([2, 2, 1])
    runs_sel = f1.multiselect("Runs", list(RUNS), default=list(RUNS))
    splits_sel = f2.multiselect("Splits", ["train", "val"], default=["train", "val"],
                                format_func=lambda s: {"train": "Train (dashed)", "val": "Validation (solid)"}[s])
    show_base = f3.checkbox("ln 65 baseline", value=True)
    mb = R.load_minibatch_logs()
    st.plotly_chart(charts.loss_curves(runs_sel, splits_sel, mb, show_base))
    if not mb:
        st.caption("Markers are the losses evaluated every 500 iterations (mean of 200 batches, dropout off). "
                   "Drop `logs/dropout_0.2.csv` and `logs/dropout_0.0.csv` (columns `iter,loss`) into the "
                   "repo to overlay the 10-iteration mini-batch trace.")

    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.validation_zoom())
    c2.plotly_chart(charts.gap_chart())

    st.subheader("Table 1 · Evaluated loss (nats / char)")
    tbl = charts.results_table()

    def _highlight_best(df):
        styles = np.full(df.shape, "", dtype=object)
        for j, col in enumerate(df.columns):
            if col[1] == "val":
                i = int(np.argmin(df[col].values))
                styles[i, j] = "background-color:#dbeafe;font-weight:700;color:#0b1220"
            if col[1] == "gap":
                for i, v in enumerate(df[col].values):
                    if v >= 1.0:
                        styles[i, j] = "background-color:#ffedd5;font-weight:700;color:#7c2d12"
        return styles

    st.dataframe(
        tbl.style.format("{:.3f}").format("{:+.2f}", subset=[c for c in tbl.columns if c[1] == "gap"])
        .apply(_highlight_best, axis=None),
        width="stretch",
    )
    st.caption("Blue = best validation loss of each run (1.468 vs 1.559). Orange = generalisation gap ≥ 1 nat.")

    k1, k2 = st.columns([1, 1], gap="large")
    with k1:
        st.markdown(
            f"""<div class="callout"><b>Theoretical baseline.</b> A model that guesses uniformly over
the 65-character vocabulary has cross-entropy $\\ln 65 \\approx {R.UNIFORM_BASELINE:.2f}$ nats/char.
Both runs start at 4.28, just above it, as expected from a random initialisation. The best checkpoint
(1.468 nats) equals {1.468 / np.log(2):.2f} bits/char, a perplexity of {np.exp(1.468):.2f}: on average the
model is as uncertain as a choice among about 4 characters instead of 65.</div>""",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown("**Sample quality (word check)**")
        st.markdown(
            f"- Dropout 0.2: **{R.KEY_METRICS['word_overlap_d02']}** of generated words appear in training text\n"
            f"- Dropout 0.0: **{R.KEY_METRICS['word_overlap_d00']}** (58 vs 28 invented words)\n"
            "- 10 samples × 500 chars per model, T = 0.8, top-k = 200, same seed"
        )
    s1, s2 = st.columns(2)
    for col, r in zip((s1, s2), R.SAMPLES):
        with col:
            st.markdown(f"**{r}** (paper excerpt, iter {R.BEST[r]['iter']:,})")
            st.markdown(f'<div class="gen-output" style="--accent:{charts.COLORS[r]};min-height:0">'
                        f"{html.escape(R.SAMPLES[r])}</div>", unsafe_allow_html=True)


# =========================================================================== 3. CHATBOT
with tab_chat:
    st.subheader("Ask the paper")
    st.caption(f"Gemini 2.5 Flash, grounded in the full manuscript. It marks general background that is "
               "not from the paper and won't invent results.")
    if not api_key:
        st.warning("No Gemini API key found. Add `GEMINI_API_KEY` as a Space secret (Settings → Variables and "
                   "secrets) or paste a key in the sidebar. Get one free at aistudio.google.com.", icon="🔑")

    q_cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, q in zip(q_cols, SUGGESTED_QUESTIONS):
        if col.button(q, width="stretch", disabled=not api_key):
            st.session_state["pending_question"] = q

    chat_box = st.container(height=520, border=True)
    with chat_box:
        if not st.session_state["chat"]:
            st.markdown("<div style='color:#64748b;padding:1rem'>Ask about the method, the results, the "
                        "limitations, or how this compares to standard practice.</div>", unsafe_allow_html=True)
        for m in st.session_state["chat"]:
            with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "🎭"):
                st.markdown(m["content"])

    typed = st.chat_input("Ask a question about the paper…", disabled=not api_key)
    question = typed or st.session_state.pop("pending_question", None)
    if question and api_key:
        st.session_state["chat"].append({"role": "user", "content": question})
        with chat_box:
            with st.chat_message("user", avatar="🧑‍🎓"):
                st.markdown(question)
            with st.chat_message("assistant", avatar="🎭"):
                try:
                    reply = st.write_stream(stream_reply(api_key, st.session_state["chat"]))
                    st.session_state["chat"].append({"role": "assistant", "content": reply})
                except AssistantError as e:
                    st.session_state["chat"].pop()  # drop the unanswered turn so history stays valid
                    st.error(str(e))
    if st.session_state["chat"] and st.button("🧹 Clear conversation"):
        st.session_state["chat"] = []
        st.rerun()


# =========================================================================== 4. MANUSCRIPT
with tab_paper:
    left, right = st.columns([3, 2], gap="large")
    with right:
        st.markdown("#### Abstract")
        st.markdown(f"<div class='card'><p style='color:#0b1220'>{R.ABSTRACT}</p></div>", unsafe_allow_html=True)
        st.markdown("#### Core contributions")
        for title, body in R.CONTRIBUTIONS:
            st.markdown(f"<div class='card' style='margin-bottom:.6rem'><h4>{title}</h4><p>{body}</p></div>",
                        unsafe_allow_html=True)
        st.markdown("#### Key citations")
        for c in R.CITATIONS:
            with st.expander(f"{c['short']}: *{c['title']}*"):
                st.markdown(f"{c['venue']}. {c['role']}")
                st.code(c["bibtex"], language="bibtex")
        st.markdown("#### Compute environment")
        st.markdown(
            "<div class='card'><h4>🖥️ NVIDIA T4 · ≈ 25 min per run</h4><p>Free Google Colab. 0.55 s per "
            "iteration plus 79 s per evaluation; 2,000 iterations × 16,384 tokens = 32.8M tokens per run.</p></div>",
            unsafe_allow_html=True,
        )
        with st.expander("Stated limitations"):
            for x in R.LIMITATIONS:
                st.markdown(f"- {x}")
    with left:
        if R.PAPER_PDF.exists():
            st.download_button("⬇️ Download PDF", R.PAPER_PDF.read_bytes(), file_name="dhingra2026_dropout.pdf",
                               mime="application/pdf", type="primary")
            try:
                pages = render_pdf_pages(str(R.PAPER_PDF))
                page_view = st.radio("View", ["All pages", "Single page"], horizontal=True,
                                     label_visibility="collapsed")
                if page_view == "Single page":
                    p = st.select_slider("Page", options=list(range(1, len(pages) + 1)))
                    st.image(pages[p - 1], width="stretch")
                else:
                    with st.container(height=900, border=True):
                        for i, im in enumerate(pages):
                            st.image(im, caption=f"Page {i + 1} of {len(pages)}", width="stretch")
            except Exception as e:
                st.error(f"Could not render the PDF ({e}). Use the download button instead.")
        else:
            st.error("`assets/paper.pdf` is missing. Add the manuscript to enable the reader.")


# =========================================================================== 5. ARCH / CITE / REPRO
def md_table(rows, header) -> str:
    """Markdown table: wraps long values, unlike st.dataframe."""
    lines = [f"| {header[0]} | {header[1]} |", "|---|---|"]
    lines += [f"| **{k}** | {v.replace('|', '/')} |" for k, v in rows]
    return "\n".join(lines)


with tab_repro:
    a, b = st.columns(2, gap="large")
    with a:
        st.subheader("Model")
        st.markdown(md_table(R.ARCHITECTURE, ("Component", "Value")))
    with b:
        st.subheader("Training & compute")
        st.markdown(md_table(R.COMPUTE, ("Setting", "Value")))

    st.subheader("Data flow")
    st.graphviz_chart(
        """
digraph G { rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#94a3b8", fontname="Helvetica", fontsize=11];
  edge [color="#64748b"];
  in [label="chars (≤256)"]; emb [label="token + position\\nembedding (384)"];
  blk [label="6 × Block\\nLN → causal attn (6 heads)\\nLN → MLP 4× GELU", fillcolor="#dbeafe", color="#2a78d6"];
  ln [label="final LayerNorm"]; head [label="LM head\\n(tied to wte)"]; out [label="softmax over\\n65 chars"];
  d [label="dropout p\\n(0.2 or 0.0)", shape=note, fillcolor="#ffedd5", color="#eb6834"];
  in -> emb -> blk -> ln -> head -> out; d -> emb [style=dashed]; d -> blk [style=dashed];
}"""
    )

    st.subheader("📚 Cite this work")
    st.code(R.PAPER_BIBTEX, language="bibtex")
    with st.expander("BibTeX for all references"):
        st.code("\n\n".join(c["bibtex"] for c in R.CITATIONS), language="bibtex")
    st.download_button("⬇️ Download references.bib",
                       R.PAPER_BIBTEX + "\n\n" + "\n\n".join(c["bibtex"] for c in R.CITATIONS),
                       file_name="references.bib")

    st.subheader("🔁 Reproduce")
    colab_url = f"https://colab.research.google.com/github/{R.GITHUB_REPO}/blob/main/{R.COLAB_NOTEBOOK_PATH}"
    st.markdown(
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url}) &nbsp; "
        f"[![nanoGPT](https://img.shields.io/badge/code-nanoGPT-0b1220?logo=github)](https://github.com/karpathy/nanoGPT)"
    )
    if R.GITHUB_REPO.startswith("your-username"):
        st.caption("⚠️ Set `GITHUB_REPO` in `research.py` to your repository so the Colab badge resolves.")
    st.markdown("Runs both experiments end-to-end on a free T4 (≈ 50 min total), then exports slim "
                "checkpoints and loss logs that this dashboard loads directly.")
    st.code(
        """git clone https://github.com/karpathy/nanoGPT && cd nanoGPT
python data/shakespeare_char/prepare.py
# Baseline (dropout 0.2) and ablation (dropout 0.0): identical except dropout
python train.py config/train_shakespeare_char.py --max_iters=2000 --eval_interval=500 --dropout=0.2 --out_dir=out-d02
python train.py config/train_shakespeare_char.py --max_iters=2000 --eval_interval=500 --dropout=0.0 --out_dir=out-d00""",
        language="bash",
    )

    st.subheader("Project files")
    st.markdown(
        """
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: five tabs, caching, session state |
| `model.py` | nanoGPT `GPT`, `GPTConfig`, `Block`, `CausalSelfAttention`, `MLP` (+ attention capture, sampling trace) |
| `inference.py` | Checkpoint resolution (local → HF Hub → random-init fallback), tokenizer, generation, attention, TTS |
| `research.py` | Every number, sample, and citation from the paper; grounding text for the assistant |
| `charts.py` | Plotly figures |
| `assistant.py` | Gemini 2.5 Flash client and system prompt |
| `notebooks/reproduce.ipynb` | Colab notebook that retrains both runs and exports checkpoints |
"""
    )

st.markdown(
    "<hr style='margin-top:2rem'><div style='color:#64748b;font-size:.85rem'>© 2026 Samrath Singh Dhingra · "
    "Built with Streamlit, PyTorch and Plotly · Model architecture from nanoGPT (MIT).</div>",
    unsafe_allow_html=True,
)
