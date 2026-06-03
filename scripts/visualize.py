#!/usr/bin/env python3
"""
Generate two benchmark charts as a single PNG:
  1. Speed  — p95 latency line chart (with p50–p99 error bands)
  2. Recall — recall@K line chart

Usage:
    python scripts/visualize.py results/es_results.json
    python scripts/visualize.py es.json milvus.json --labels ES Milvus --output compare.png
"""

import json
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from config import TOP_K

SLA_LATENCY_MS = 1_000
SLA_RECALL = 0.95

PALETTE = ["#c0392b", "#2980b9", "#27ae60", "#e67e22", "#8e44ad"]
MARKERS = ["o", "s", "^", "D", "v"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load(path: str) -> list:
    with open(path) as f:
        rows = json.load(f)
    for row in rows:
        r = row.get("recall", "—")
        row["recall"] = float(r) if r not in ("—", "", None) else None
    return rows


def short_name(method: str) -> str:
    m = method.lower()
    if "script_score" in m:
        return "Exact (script_score)"
    if "flat" in m:
        return "Exact (FLAT)"
    for tok in method.split():
        if tok.startswith(("nc=", "ef=")):
            return f"HNSW {tok}"
    return method


def method_order(methods):
    exact, ann = [], []
    for m in methods:
        lm = m.lower()
        if "script_score" in lm or "flat" in lm or lm.startswith("exact"):
            exact.append(m)
        else:
            ann.append(m)
    ann.sort(
        key=lambda m: int(next((t.split("=")[1] for t in m.split() if "=" in t), "0"))
    )
    return exact + ann


def apply_style():
    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(name)
            break
        except Exception:
            pass
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlepad": 14,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#f9f9f9",
            "grid.color": "#dedede",
            "grid.linewidth": 0.9,
            "grid.linestyle": "--",
            "xtick.color": "#444",
            "ytick.color": "#444",
            "figure.facecolor": "#ffffff",
        }
    )


# ---------------------------------------------------------------------------
# Chart 1 — Speed (p95 latency)
# ---------------------------------------------------------------------------


def plot_speed(ax, lookup, scales, methods, colors):
    x = np.arange(len(scales))
    xlabels = [f"{s // 1000}k" for s in scales]

    for i, method in enumerate(methods):
        p50 = np.array(
            [lookup.get((s, method), {}).get("p50") for s in scales], dtype=float
        )
        p95 = np.array(
            [lookup.get((s, method), {}).get("p95") for s in scales], dtype=float
        )
        p99 = np.array(
            [lookup.get((s, method), {}).get("p99") for s in scales], dtype=float
        )

        mask = ~np.isnan(p95)
        if not mask.any():
            continue

        color = colors[i]
        marker = MARKERS[i % len(MARKERS)]
        label = short_name(method)

        # Shaded p50–p99 band
        if mask.sum() > 1:
            ax.fill_between(
                x[mask], p50[mask], p99[mask], alpha=0.12, color=color, zorder=2
            )

        # p95 main line
        ax.plot(
            x[mask],
            p95[mask],
            color=color,
            linewidth=2.5,
            marker=marker,
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=4,
            label=label,
        )

        # p50 / p99 dashed bounds
        if mask.sum() > 1:
            ax.plot(
                x[mask],
                p50[mask],
                color=color,
                linewidth=1,
                linestyle=":",
                alpha=0.6,
                zorder=3,
            )
            ax.plot(
                x[mask],
                p99[mask],
                color=color,
                linewidth=1,
                linestyle=":",
                alpha=0.6,
                zorder=3,
            )

        # Value labels above p95 markers
        for xi, yi in zip(x[mask], p95[mask]):
            ax.annotate(
                f"{yi:.0f} ms",
                xy=(xi, yi),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=color,
                fontweight="bold",
                zorder=5,
            )

    # SLA line
    ax.axhline(
        SLA_LATENCY_MS,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.8,
        alpha=0.85,
        zorder=3,
    )
    ax.text(
        x[-1] + 0.05,
        SLA_LATENCY_MS * 1.15,
        f"SLA = {SLA_LATENCY_MS:,} ms",
        color="#e74c3c",
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=12)
    ax.set_xlabel("Scale  (documents per org)", fontsize=11)
    ax.set_ylabel("p95 Latency  (ms)  —  lower is better", fontsize=11)
    ax.set_title("Speed  ·  p95 latency  ·  shaded = p50–p99 range", fontsize=13)
    _pad_xlim(ax, x)


# ---------------------------------------------------------------------------
# Chart 2 — Recall
# ---------------------------------------------------------------------------


def plot_recall(ax, lookup, scales, methods, colors):
    x = np.arange(len(scales))
    xlabels = [f"{s // 1000}k" for s in scales]

    has_any = False
    for i, method in enumerate(methods):
        rec = np.array(
            [lookup.get((s, method), {}).get("recall") for s in scales], dtype=float
        )
        mask = ~np.isnan(rec)
        if not mask.any():
            continue
        has_any = True

        color = colors[i]
        marker = MARKERS[i % len(MARKERS)]
        label = short_name(method)

        ax.plot(
            x[mask],
            rec[mask],
            color=color,
            linewidth=2.5,
            marker=marker,
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=4,
            label=label,
        )

        for xi, yi in zip(x[mask], rec[mask]):
            ax.annotate(
                f"{yi:.3f}",
                xy=(xi, yi),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=color,
                fontweight="bold",
                zorder=5,
            )

    if not has_any:
        ax.text(
            0.5,
            0.5,
            "No recall data\n(run without --no-exact)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="#888",
        )

    # SLA line
    ax.axhline(
        SLA_RECALL, color="#27ae60", linestyle="--", linewidth=1.8, alpha=0.85, zorder=3
    )
    ax.text(
        x[-1] + 0.05,
        SLA_RECALL - 0.035,
        f"SLA = {SLA_RECALL}",
        color="#27ae60",
        fontsize=9,
        fontweight="bold",
        va="top",
    )

    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=12)
    ax.set_xlabel("Scale  (documents per org)", fontsize=11)
    ax.set_ylabel(f"Recall @ {TOP_K}  —  higher is better", fontsize=11)
    ax.set_title(f"Recall @ {TOP_K}", fontsize=13)
    _pad_xlim(ax, x)


def _pad_xlim(ax, x):
    pad = 0.55 if len(x) == 1 else 0.35
    ax.set_xlim(x[0] - pad, x[-1] + pad)


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------


def build_figure(datasets: list, title: str) -> plt.Figure:
    apply_style()

    all_rows = [row for _, rows in datasets for row in rows]
    scales = sorted({r["scale"] for r in all_rows})
    raw_methods = list(dict.fromkeys(r["method"] for r in all_rows))
    methods = method_order(raw_methods)

    # Build lookup — prefix method with dataset label when comparing
    lookup: dict = {}
    if len(datasets) > 1:
        methods = method_order(
            list(
                dict.fromkeys(
                    f"{lbl} · {row['method']}" for lbl, rows in datasets for row in rows
                )
            )
        )
        for lbl, rows in datasets:
            for row in rows:
                lookup[(row["scale"], f"{lbl} · {row['method']}")] = row
    else:
        for row in all_rows:
            lookup[(row["scale"], row["method"])] = row

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(methods))]

    fig, (ax_spd, ax_rec) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        dpi=150,
        facecolor="white",
    )
    fig.suptitle(title, fontsize=15, fontweight="bold", color="#1a1a2e", y=1.01)

    plot_speed(ax_spd, lookup, scales, methods, colors)
    plot_recall(ax_rec, lookup, scales, methods, colors)

    # Shared legend below both charts
    handles = [
        Line2D(
            [0],
            [0],
            color=colors[i],
            linewidth=2.5,
            marker=MARKERS[i % len(MARKERS)],
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.2,
            label=short_name(m.split(" · ", 1)[-1] if " · " in m else m),
        )
        for i, m in enumerate(methods)
    ]
    handles += [
        Line2D(
            [0],
            [0],
            color="#e74c3c",
            linestyle="--",
            linewidth=1.8,
            label=f"Latency SLA ({SLA_LATENCY_MS:,} ms)",
        ),
        Line2D(
            [0],
            [0],
            color="#27ae60",
            linestyle="--",
            linewidth=1.8,
            label=f"Recall SLA ({SLA_RECALL})",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 5),
        bbox_to_anchor=(0.5, -0.07),
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=10,
        handlelength=1.8,
        columnspacing=1.4,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", metavar="FILE")
    p.add_argument("--labels", nargs="+", metavar="LABEL")
    p.add_argument("--output", "-o", default="benchmark_report.png")
    p.add_argument("--title", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    labels = list(args.labels or [])
    while len(labels) < len(args.inputs):
        labels.append(Path(args.inputs[len(labels)]).stem.upper())

    datasets = []
    for lbl, path in zip(labels, args.inputs):
        rows = load(path)
        datasets.append((lbl, rows))
        print(f"Loaded {len(rows)} rows  ←  {path}")

    if args.title:
        title = args.title
    elif len(datasets) == 1:
        title = (
            f"{labels[0]}  —  Vector Search Benchmark  (4096-dim · HNSW m=16 efC=100)"
        )
    else:
        title = "  vs  ".join(labels) + "  —  Vector Search Benchmark"

    fig = build_figure(datasets, title)
    out = Path(args.output)
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    main()
