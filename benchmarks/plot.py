"""Charts for the README, drawn from the benchmark JSON.

    python benchmarks/plot.py docs/bench/sweep_n1000000_d128_m32_final.json

Produces, in light and dark variants:

* ``recall_vs_qps[_dark].png`` -- the operating curve. Recall on x, throughput
  on y (log), one line per index backend. This is the standard way the ANN
  literature presents an index, because neither axis means anything alone: an
  index is a *curve*, and the only fair comparison is at equal recall.
* ``memory[_dark].png`` -- what each backend costs in RAM for the same corpus.

Both variants are generated deliberately rather than by inverting the light
one: the dark series colours are their own steps, chosen against the dark
surface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display in CI
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

# Validated categorical slots 1-3 (see the project's chart notes). Three series
# is the all-pairs-safe cap for this palette, which is exactly what is needed.
THEME: dict[str, dict[str, Any]] = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#dededa",
        "series": {"ivf": "#2a78d6", "ivfpq": "#eb6834", "refine": "#1baf7a"},
        "suffix": "",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#333330",
        "series": {"ivf": "#3987e5", "ivfpq": "#d95926", "refine": "#199e70"},
        "suffix": "_dark",
    },
}

LABELS = {
    "ivf": "IVF (full precision)",
    "ivfpq": "IVF-PQ (32 B/vector)",
    "refine": "Two-stage (PQ + exact re-rank)",
    "flat": "Exact search",
}


def load(paths: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Merge one or more sweep files, keeping the first row per config label."""
    meta: dict[str, Any] = {}
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        meta = {**payload["meta"], **meta} if meta else payload["meta"]
        for row in payload["rows"]:
            label = row.get("config", "")
            # A row without a latency measurement cannot be plotted; prefer a
            # later file's version of the same config if it has one.
            if label not in rows or _plottable(row) and not _plottable(rows[label]):
                rows[label] = row
    return meta, list(rows.values())


def _plottable(row: dict[str, Any]) -> bool:
    qps = row.get("qps")
    return isinstance(qps, (int, float)) and qps == qps and qps > 0


def style_axes(ax: plt.Axes, theme: dict[str, Any]) -> None:
    ax.set_facecolor(theme["surface"])
    ax.figure.set_facecolor(theme["surface"])
    # Recessive chrome: the data should be the only thing with weight.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=theme["muted"], labelsize=9, length=3, width=1.0)
    ax.grid(True, color=theme["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def plot_operating_curve(
    meta: dict[str, Any], rows: list[dict[str, Any]], out_dir: Path, mode: str
) -> Path:
    theme = THEME[mode]
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=200)
    style_axes(ax, theme)

    exact = next((r for r in rows if r["index"] == "flat" and _plottable(r)), None)

    for kind in ("ivf", "ivfpq", "refine"):
        series = sorted(
            (r for r in rows if r["index"] == kind and _plottable(r)),
            key=lambda r: r["recall"],
        )
        if not series:
            continue
        recalls = [r["recall"] for r in series]
        qps = [r["qps"] for r in series]
        colour = theme["series"][kind]
        ax.plot(
            recalls,
            qps,
            color=colour,
            linewidth=2.0,
            marker="o",
            markersize=6.5,
            markeredgecolor=theme["surface"],  # 2px surface ring where marks overlap
            markeredgewidth=1.6,
            label=LABELS[kind],
            zorder=3,
        )
        # Direct label anchored at the series' *leftmost* point: identity is
        # never colour-alone, the light-mode aqua needs the relief a visible
        # label provides, and anchoring left keeps every label inside the axes
        # instead of overflowing the right edge.
        ax.annotate(
            LABELS[kind],
            xy=(recalls[0], qps[0]),
            xytext=(7, 9),
            textcoords="offset points",
            color=theme["text"],
            fontsize=8.5,
            fontweight="medium",
            ha="left",
            zorder=5,
        )

    if exact:
        ax.axhline(exact["qps"], color=theme["muted"], linewidth=1.2, linestyle=(0, (4, 3)))
        ax.annotate(
            f"exact search — {exact['qps']:.0f} QPS, {exact['p50_ms']:.0f} ms, perfect recall",
            xy=(0.93, exact["qps"]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            color=theme["muted"],
            fontsize=8.5,
        )

    ax.set_xscale("logit")
    ax.set_yscale("log")
    ax.set_xlim(0.3, 0.9995)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3g}"))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    ax.set_xlabel(
        "recall@10  (fraction of the exact top-10 returned)", color=theme["muted"], fontsize=9.5
    )
    ax.set_ylabel("queries per second  (single query, p50)", color=theme["muted"], fontsize=9.5)
    ax.set_title(
        f"Recall against throughput — {meta['n']:,} × {meta['dim']}-d vectors",
        color=theme["text"],
        fontsize=12.5,
        fontweight="semibold",
        loc="left",
        pad=14,
    )
    ax.text(
        0.0,
        1.012,
        "up and to the right is better · measured against exact search on the same data",
        transform=ax.transAxes,
        color=theme["muted"],
        fontsize=8.5,
    )

    legend = ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=8.5,
        labelcolor=theme["text"],
        handlelength=1.6,
    )
    legend.set_zorder(4)

    fig.tight_layout()
    target = out_dir / f"recall_vs_qps{theme['suffix']}.png"
    fig.savefig(target, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(fig)
    return target


def plot_memory(meta: dict[str, Any], rows: list[dict[str, Any]], out_dir: Path, mode: str) -> Path:
    theme = THEME[mode]
    fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=200)
    style_axes(ax, theme)
    ax.grid(axis="y", visible=False)

    picks: list[tuple[str, float, str]] = []
    for kind, label in (
        ("flat", "Exact / IVF\n(float32)"),
        ("ivfpq", f"IVF-PQ\n({meta['m']} B/vector)"),
    ):
        row = next((r for r in rows if r["index"] == kind), None)
        if row:
            picks.append((label, float(row["memory_mb"]), kind))

    colour = {"flat": theme["muted"], "ivfpq": theme["series"]["ivfpq"]}
    positions = range(len(picks))
    # Bar width leaves a surface gap between adjacent bars.
    bars = ax.barh(
        list(positions),
        [value for _, value, _ in picks],
        height=0.42,
        color=[colour[kind] for _, _, kind in picks],
        zorder=3,
    )
    ax.set_yticks(
        list(positions), [label for label, _, _ in picks], color=theme["text"], fontsize=9
    )
    ax.invert_yaxis()
    ax.set_xlim(0, max(value for _, value, _ in picks) * 1.22)

    for bar, (_, value, _) in zip(bars, picks, strict=True):
        ax.annotate(
            f"{value:,.0f} MB",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            color=theme["text"],
            fontsize=9.5,
            fontweight="semibold",
        )

    ratio = picks[0][1] / picks[1][1] if len(picks) == 2 and picks[1][1] else 0.0
    ax.set_xlabel("resident memory for the whole index (MB)", color=theme["muted"], fontsize=9.5)
    ax.set_title(
        f"Memory for {meta['n']:,} × {meta['dim']}-d vectors — {ratio:.1f}× smaller compressed",
        color=theme["text"],
        fontsize=12.5,
        fontweight="semibold",
        loc="left",
        pad=12,
    )

    fig.tight_layout()
    target = out_dir / f"memory{theme['suffix']}.png"
    fig.savefig(target, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(fig)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="sweep JSON files written by bench_sweep.py")
    parser.add_argument("--out", default="docs/bench", help="where to write the PNGs")
    args = parser.parse_args(argv)

    meta, rows = load(args.inputs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in ("light", "dark"):
        for target in (
            plot_operating_curve(meta, rows, out_dir, mode),
            plot_memory(meta, rows, out_dir, mode),
        ):
            print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
