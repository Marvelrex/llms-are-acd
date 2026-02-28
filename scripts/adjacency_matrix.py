#!/usr/bin/env python3
"""
Generate a simple adjacency matrix CSV from an edge-score JSON export.

Input JSON format: list of dicts with at least "from" and "to" keys.
Value written to the matrix can be chosen via --field:
  - score (default)
  - visit_count
  - presence (writes 1 if edge exists, else 0)

Usage:
    python scripts/adjacency_matrix.py path/to/action_graph_scores_live.json
    python scripts/adjacency_matrix.py path/to/edges.json --field visit_count --output adj.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_edges(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Expected a list of edge dicts in the JSON export.")
    return data


def _is_defender(node_id: str) -> bool:
    return str(node_id).startswith("defender")


def build_node_sets(edges: List[Dict]) -> Tuple[List[str], List[str]]:
    defenders = set()
    attackers = set()
    for e in edges:
        u, v = e.get("from"), e.get("to")
        if u:
            (defenders if _is_defender(u) else attackers).add(u)
        if v:
            (defenders if _is_defender(v) else attackers).add(v)
    return sorted(defenders), sorted(attackers)


def build_matrix(
    edges: List[Dict], row_nodes: List[str], col_nodes: List[str], field: str
) -> List[List[float]]:
    row_idx = {n: i for i, n in enumerate(row_nodes)}
    col_idx = {n: i for i, n in enumerate(col_nodes)}
    matrix = [[0.0 for _ in range(len(col_nodes))] for _ in range(len(row_nodes))]
    for e in edges:
        u, v = e.get("from"), e.get("to")
        if u not in row_idx or v not in col_idx:
            continue
        r, c = row_idx[u], col_idx[v]
        if field == "presence":
            matrix[r][c] = 1.0
        else:
            matrix[r][c] = float(e.get(field, 0.0) or 0.0)
    return matrix


def write_csv(output: Path, row_nodes: List[str], col_nodes: List[str], matrix: List[List[float]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["defender \\ attacker"] + col_nodes)
        for name, row in zip(row_nodes, matrix):
            writer.writerow([name] + row)


def render_heatmap(
    output: Path, row_nodes: List[str], col_nodes: List[str], matrix: List[List[float]], title: str, cmap_name: str = "Blues"
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "matplotlib and numpy are required for plotting. Install with: pip install \"matplotlib<4\" numpy"
        ) from exc

    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(
        figsize=(max(8, len(col_nodes) * 0.5), max(6, len(row_nodes) * 0.5))
    )
    cmap = plt.get_cmap(cmap_name)
    im = ax.imshow(arr, cmap=cmap)

    # Tick labels
    ax.set_xticks(range(len(col_nodes)))
    ax.set_yticks(range(len(row_nodes)))
    ax.set_xticklabels(col_nodes, rotation=90, fontsize=7)
    ax.set_yticklabels(row_nodes, fontsize=7)
    ax.set_title(title, fontsize=10)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.ax.tick_params(labelsize=7)

    # Layout padding
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Wrote heatmap: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adjacency matrix CSV from edge JSON.")
    parser.add_argument("json_path", type=Path, help="Path to edge JSON (e.g., action_graph_scores_live.json).")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path (default: alongside input, name adjacency.csv).")
    parser.add_argument(
        "--field",
        choices=["score", "visit_count", "presence"],
        default="score",
        help="Edge value to place in the matrix (default: score).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also render a heatmap PNG next to the CSV (or next to input if no CSV path provided).",
    )
    parser.add_argument(
        "--plot-visits",
        action="store_true",
        help="Render an additional heatmap using visit_count values, saved as *_visits.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path: Path = args.json_path
    if not json_path.exists():
        raise SystemExit(f"Input file not found: {json_path}")
    out_path = args.output or json_path.with_name("adjacency.csv")
    edges = load_edges(json_path)
    defenders, attackers = build_node_sets(edges)
    matrix = build_matrix(edges, defenders, attackers, field=args.field)
    write_csv(out_path, defenders, attackers, matrix)
    print(f"Wrote: {out_path}")
    if args.plot:
        png_path = out_path.with_suffix(".png")
        render_heatmap(png_path, defenders, attackers, matrix, title=f"Adjacency ({args.field})", cmap_name="Blues")

    if args.plot_visits:
        visits_matrix = build_matrix(edges, defenders, attackers, field="visit_count")
        png_visits = out_path.with_name(out_path.stem + "_visits.png")
        render_heatmap(png_visits, defenders, attackers, visits_matrix, title="Adjacency (visit_count)", cmap_name="Oranges")


if __name__ == "__main__":
    main()
