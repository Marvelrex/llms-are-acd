#!/usr/bin/env python3
"""
Render an action-graph JSON (edge score export) to a PNG with minimal overlap.

The script avoids importing the full CybORG stack; it only needs matplotlib.
Edges default to min_visits=1 to hide never-seen transitions. Node colors:
- Defender nodes (id startswith 'defender_'): blue
- Attacker nodes (otherwise): red

Usage:
    python scripts/render_action_graph.py path/to/action_graph_scores_live.json
Optional:
    --output path/to/out.png          (default: alongside input)
    --min-visits N                    (default: 1)
    --dpi 300                         (default: 300)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_edges(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Expected a list of edge dicts in the JSON export.")
    return data


def build_positions(defenders: List[str], attackers: List[str]) -> Dict[str, Tuple[float, float]]:
    """Place defenders on the left and attackers on the right on a simple grid."""
    pos: Dict[str, Tuple[float, float]] = {}
    x_left, x_right = 0.0, 4.0
    spacing = 1.8
    for i, node in enumerate(defenders):
        pos[node] = (x_left, i * spacing)
    for i, node in enumerate(attackers):
        pos[node] = (x_right, i * spacing)
    return pos


def render_graph(edges: List[Dict], output: Path, min_visits: int, dpi: int, label_jitter: float) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib import colors, colormaps
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required. Install with: pip install \"matplotlib<4\""
        ) from exc

    defenders: List[str] = []
    attackers: List[str] = []
    filtered_edges = []
    for e in edges:
        u, v = e.get("from"), e.get("to")
        if not u or not v:
            continue
        if int(e.get("visit_count", 0) or 0) < min_visits:
            continue
        for node in (u, v):
            bucket = defenders if str(node).startswith("defender") else attackers
            if node not in bucket:
                bucket.append(node)
        filtered_edges.append(e)

    defenders.sort()
    attackers.sort()

    if not filtered_edges:
        raise SystemExit("No edges passed the min_visits filter; nothing to render.")

    pos = build_positions(defenders, attackers)
    fig_height = max(10, (len(defenders) + len(attackers)) * 1.0)
    fig_width = 18
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    text_color = "#111111"
    node_text_color = "#111111"
    # Draw nodes (larger for readability)
    for node in defenders:
        x, y = pos[node]
        ax.scatter([x], [y], s=1400, color="#1f77b4")
        ax.text(x, y, node, ha="center", va="center", color=node_text_color, fontsize=10, fontweight="bold")
    for node in attackers:
        x, y = pos[node]
        ax.scatter([x], [y], s=1400, color="#d62728")
        ax.text(x, y, node, ha="center", va="center", color=node_text_color, fontsize=10, fontweight="bold")

    visits = [int(e.get("visit_count", 0) or 0) for e in filtered_edges]
    vmax = max(visits) if visits else min_visits
    vmin = min_visits
    if vmax == vmin:
        vmax = vmin + 1  # avoid zero range
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = colormaps.get_cmap("Blues")

    # Draw edges and labels
    jitter_cycle = 17  # spread labels across a repeating band to reduce overlap
    for idx, e in enumerate(filtered_edges):
        u, v = e["from"], e["to"]
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        visit_ct = int(e.get("visit_count", 0) or 0)
        # Shift the colormap input so edges stay visible on white background.
        mapped_val = 0.25 + 0.75 * norm(visit_ct)
        edge_color = cmap(mapped_val)
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=edge_color, lw=2.0),
        )
        t = 0.35  # place label closer to source
        mx, my = (1 - t) * x1 + t * x2, (1 - t) * y1 + t * y2
        # deterministically offset labels in Y to avoid stacking
        offset = ((idx % jitter_cycle) - jitter_cycle // 2) * label_jitter
        label = f"s={float(e.get('score', 0.0) or 0.0):.1f}\\nv={int(e.get('visit_count', 0) or 0)}"
        ax.text(mx, my + offset, label, fontsize=7, ha="center", va="center", color=text_color)

    # Add padding so node labels are not clipped
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if xs and ys:
        x_pad = 1.5
        y_pad = max(2.0, (max(ys) - min(ys)) * 0.05)
        ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
        ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    # Colorbar for visit_count
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("visit_count", fontsize=8, color=text_color)
    cbar.ax.tick_params(labelsize=7, colors=text_color)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"Wrote: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render action graph scores JSON to PNG.")
    parser.add_argument("json_path", type=Path, help="Path to action_graph_scores_live.json (edge list).")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: same directory, name action_graph_rendered.png).",
    )
    parser.add_argument("--min-visits", type=int, default=1, help="Only render edges with visit_count >= this (default: 1).")
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI (default: 300).")
    parser.add_argument("--label-jitter", type=float, default=0.12, help="Vertical offset step to de-overlap edge labels (default: 0.12).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path: Path = args.json_path
    if not json_path.exists():
        raise SystemExit(f"Input file not found: {json_path}")
    out_path = args.output or json_path.with_name("action_graph_rendered.png")
    edges = load_edges(json_path)
    render_graph(edges, out_path, min_visits=args.min_visits, dpi=args.dpi, label_jitter=args.label_jitter)


if __name__ == "__main__":
    main()
