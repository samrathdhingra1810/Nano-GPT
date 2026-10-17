"""Plotly figures. Run identity is carried by colour; split by line style + legend."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from research import BEST, EVAL, UNIFORM_BASELINE

COLORS = {"Dropout 0.2": "#2a78d6", "Dropout 0.0": "#eb6834"}
INK = "#0b1220"
INK_2 = "#475569"
GRID = "#e5e7eb"
FONT = dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", color=INK, size=13)


def _base_layout(fig: go.Figure, height: int = 420, **kw) -> go.Figure:
    layout = dict(
        template="plotly_white",
        height=height,
        font=FONT,
        margin=dict(l=56, r=24, t=48, b=48),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=INK_2)),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    layout.update(kw)  # per-chart overrides win
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#cbd5e1", ticks="outside",
                     tickcolor="#cbd5e1", title_font=dict(color=INK_2))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="#cbd5e1",
                     title_font=dict(color=INK_2))
    return fig


def loss_curves(runs: List[str], splits: List[str], minibatch: Optional[dict] = None,
                show_baseline: bool = True) -> go.Figure:
    fig = go.Figure()
    minibatch = minibatch or {}

    for run in runs:
        if run in minibatch:
            mb = minibatch[run]
            fig.add_trace(go.Scatter(
                x=mb["iter"], y=mb["loss"], mode="lines", name=f"{run} · mini-batch",
                line=dict(color=COLORS[run], width=1), opacity=0.25,
                hovertemplate="%{y:.3f}<extra>" + run + " mini-batch</extra>",
            ))
        df = EVAL[EVAL.run == run]
        for split in splits:
            fig.add_trace(go.Scatter(
                x=df["iter"], y=df[split], mode="lines+markers",
                name=f"{run} · {split}",
                line=dict(color=COLORS[run], width=2.5, dash="solid" if split == "val" else "dash"),
                marker=dict(size=9, symbol="circle" if split == "val" else "diamond",
                            line=dict(color="white", width=2)),
                hovertemplate="%{y:.3f} nats<extra>" + f"{run} {split}" + "</extra>",
            ))
            # Direct label at the final point
            last = df.iloc[-1]
            fig.add_annotation(
                x=last["iter"], y=last[split], text=f"<b>{last[split]:.2f}</b>",
                xanchor="left", xshift=10, showarrow=False, font=dict(color=INK, size=12),
            )

    if show_baseline:
        fig.add_hline(
            y=UNIFORM_BASELINE, line=dict(color="#64748b", width=1.5, dash="dot"),
            annotation_text=f"uniform guess · ln 65 ≈ {UNIFORM_BASELINE:.2f}",
            annotation_position="top right", annotation_font=dict(color=INK_2, size=12),
        )

    _base_layout(fig, height=460)
    fig.update_xaxes(title="training iteration", range=[-60, 2180], dtick=500)
    fig.update_yaxes(title="cross-entropy loss (nats / char)", range=[0, 4.6])
    return fig


def validation_zoom() -> go.Figure:
    fig = go.Figure()
    for run in COLORS:
        df = EVAL[(EVAL.run == run) & (EVAL.iter >= 500)]
        fig.add_trace(go.Scatter(
            x=df["iter"], y=df["val"], mode="lines+markers", name=run,
            line=dict(color=COLORS[run], width=2.5),
            marker=dict(size=9, line=dict(color="white", width=2)),
            hovertemplate="%{y:.3f}<extra>" + run + "</extra>",
        ))
        b = BEST[run]
        fig.add_trace(go.Scatter(
            x=[b["iter"]], y=[b["val"]], mode="markers", showlegend=False,
            marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color=COLORS[run], width=2.5)),
            hoverinfo="skip",
        ))
        fig.add_annotation(x=b["iter"], y=b["val"], text=f"best {b['val']:.3f}",
                           yshift=-22 if run == "Dropout 0.2" else 22, showarrow=False,
                           font=dict(color=INK, size=12))
    _base_layout(fig, height=380, title=dict(text="Validation loss", font=dict(size=15)),
                 legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0, font=dict(color=INK_2)),
                 margin=dict(l=56, r=40, t=48, b=90))
    fig.update_xaxes(title="training iteration", dtick=500, range=[400, 2100])
    fig.update_yaxes(title="nats / char", range=[1.3, 2.55])
    return fig


def gap_chart() -> go.Figure:
    fig = go.Figure()
    for run in COLORS:
        df = EVAL[EVAL.run == run]
        fig.add_trace(go.Scatter(
            x=df["iter"], y=df["gap"], mode="lines+markers", name=run,
            line=dict(color=COLORS[run], width=2.5),
            marker=dict(size=9, line=dict(color="white", width=2)),
            hovertemplate="%{y:+.2f}<extra>" + run + "</extra>",
        ))
        last = df.iloc[-1]
        fig.add_annotation(x=last["iter"], y=last["gap"], text=f"<b>{last['gap']:.2f}</b>",
                           xanchor="left", xshift=10, showarrow=False, font=dict(color=INK))
    _base_layout(fig, height=380, title=dict(text="Generalisation gap (val − train)", font=dict(size=15)),
                 legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0, font=dict(color=INK_2)),
                 margin=dict(l=56, r=40, t=48, b=90))
    fig.update_xaxes(title="training iteration", dtick=500, range=[-60, 2200])
    fig.update_yaxes(title="nats / char", range=[-0.15, 2.25])
    return fig


def attention_heatmap(att: np.ndarray, labels: List[str], title: str) -> go.Figure:
    """att: (T, T) query × key weights; labels: display chars of length T."""
    T = len(labels)
    idx = list(range(T))
    hover = [[f"query #{q} '{labels[q]}' → key #{k} '{labels[k]}'" for k in idx] for q in idx]
    masked = np.where(np.tril(np.ones((T, T), dtype=bool)), att, np.nan)
    fig = go.Figure(go.Heatmap(
        z=masked, x=idx, y=idx, customdata=hover,
        colorscale=[[0, "#f1f5fe"], [0.15, "#bcd4f6"], [0.5, "#5b9be6"], [1, "#0b3d91"]],
        zmin=0, zmax=float(np.nanmax(masked)) if T else 1,
        hovertemplate="%{customdata}<br>weight %{z:.3f}<extra></extra>",
        colorbar=dict(title=dict(text="weight", font=dict(color=INK_2)), thickness=12),
        xgap=1, ygap=1,
    ))
    step = max(1, T // 48)
    ticks = idx[::step]
    _base_layout(fig, height=max(420, min(760, 14 * T + 140)), hovermode="closest",
                 title=dict(text=title, font=dict(size=15)))
    fig.update_xaxes(title="key (attended-to character)", tickvals=ticks,
                     ticktext=[labels[i] for i in ticks], tickangle=0, showgrid=False,
                     side="bottom", ticks="")
    fig.update_yaxes(title="query (current character)", tickvals=ticks,
                     ticktext=[labels[i] for i in ticks], autorange="reversed", showgrid=False)
    return fig


def token_probs(chars: List[str], probs: List[float], chosen_idx: Optional[int]) -> go.Figure:
    order = list(range(len(chars)))[::-1]  # most likely on top
    colors = ["#2a78d6" if i == chosen_idx else "#9db8de" for i in order]
    fig = go.Figure(go.Bar(
        x=[probs[i] for i in order], y=[f"'{chars[i]}'" for i in order], orientation="h",
        marker=dict(color=colors, line=dict(color="white", width=2)),
        text=[f"{probs[i]:.1%}" for i in order], textposition="outside",
        textfont=dict(color=INK), cliponaxis=False,
        hovertemplate="%{y}: %{x:.2%}<extra></extra>",
    ))
    _base_layout(fig, height=280, hovermode="closest", showlegend=False,
                 margin=dict(l=56, r=56, t=16, b=40))
    fig.update_xaxes(title="probability (after temperature & top-k)", range=[0, 1.1],
                     tickformat=".0%")
    fig.update_yaxes(showgrid=False, tickfont=dict(size=15, family="JetBrains Mono, monospace"))
    return fig


def results_table() -> pd.DataFrame:
    a = EVAL[EVAL.run == "Dropout 0.2"].set_index("iter")[["train", "val", "gap"]]
    b = EVAL[EVAL.run == "Dropout 0.0"].set_index("iter")[["train", "val", "gap"]]
    df = pd.concat({"Dropout 0.2": a, "Dropout 0.0": b}, axis=1)
    df.index.name = "Iteration"
    return df
