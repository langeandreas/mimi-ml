"""One-off utility: plot a single minimal XGBoost tree from a saved analysis artifact.

This script is intentionally standalone and safe to delete after use.
It renders only split feature names and split conditions for internal nodes.
Leaf nodes are shown as plain "leaf" labels without values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import joblib
import matplotlib.pyplot as plt


def _latest_analysis(analyses_dir: Path) -> Path:
    candidates = sorted(analyses_dir.glob("analysis_*.joblib"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No analysis artifacts found in: {analyses_dir}")
    return candidates[0]


def _resolve_booster(model: Any):
    """Return underlying booster from GridSearchCV or direct estimator."""
    estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    if estimator is None or not hasattr(estimator, "get_booster"):
        raise TypeError("Model object does not expose get_booster().")
    return estimator.get_booster()


def _collect_positions(
    node: Dict[str, Any],
    depth: int,
    x_cursor: list[float],
    positions: Dict[int, Tuple[float, float]],
) -> float:
    """Compute tidy x/y positions by assigning leaves left-to-right."""
    node_id = int(node["nodeid"])

    if "leaf" in node:
        x = x_cursor[0]
        x_cursor[0] += 1.0
        positions[node_id] = (x, -float(depth))
        return x

    children = {str(child.get("nodeid")): child for child in node.get("children", [])}
    yes_id = str(node.get("yes"))
    no_id = str(node.get("no"))
    missing_id = str(node.get("missing"))

    yes_child = children.get(yes_id)
    no_child = children.get(no_id)

    child_x_values = []
    if yes_child is not None:
        child_x_values.append(_collect_positions(yes_child, depth + 1, x_cursor, positions))
    if no_child is not None and no_child is not yes_child:
        child_x_values.append(_collect_positions(no_child, depth + 1, x_cursor, positions))

    for child in node.get("children", []):
        cid = str(child.get("nodeid"))
        if cid in {yes_id, no_id}:
            continue
        child_x_values.append(_collect_positions(child, depth + 1, x_cursor, positions))

    if child_x_values:
        x = sum(child_x_values) / len(child_x_values)
    else:
        x = x_cursor[0]
        x_cursor[0] += 1.0

    positions[node_id] = (x, -float(depth))

    if missing_id not in {yes_id, no_id} and missing_id in children:
        _collect_positions(children[missing_id], depth + 1, x_cursor, positions)

    return x


def _draw_tree(
    ax: plt.Axes,
    node: Dict[str, Any],
    positions: Dict[int, Tuple[float, float]],
    feature_names: Optional[Iterable[str]],
    font_size: int,
    line_width: float,
    show_leaves: bool,
) -> None:
    """Draw nodes and edges with minimal labels."""
    features = list(feature_names) if feature_names is not None else []

    def feature_label(raw: Any) -> str:
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            text = str(raw)
            if text.startswith("f") and text[1:].isdigit() and features:
                i = int(text[1:])
                if 0 <= i < len(features):
                    return str(features[i])
            return text
        if 0 <= idx < len(features):
            return str(features[idx])
        return str(raw)

    def walk(cur: Dict[str, Any]) -> None:
        node_id = int(cur["nodeid"])
        x, y = positions[node_id]

        if "leaf" in cur:
            if not show_leaves:
                return
            label = "leaf"
            facecolor = "#efefef"
        else:
            split = feature_label(cur.get("split"))
            cond = cur.get("split_condition")
            label = f"{split} < {cond}"
            facecolor = "#ffffff"

        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=font_size,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": facecolor, "edgecolor": "#111111", "linewidth": 1.2},
        )

        for child in cur.get("children", []):
            if "leaf" in child and not show_leaves:
                continue
            cid = int(child["nodeid"])
            cx, cy = positions[cid]
            ax.plot([x, cx], [y, cy], color="#222222", linewidth=line_width)
            walk(child)

    walk(node)


def _label_for_node(
    node: Dict[str, Any],
    feature_names: Optional[Iterable[str]],
    show_leaves: bool,
) -> Optional[str]:
    """Return the rendered label for a node, or None if it should be hidden."""
    features = list(feature_names) if feature_names is not None else []

    def feature_label(raw: Any) -> str:
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            text = str(raw)
            if text.startswith("f") and text[1:].isdigit() and features:
                i = int(text[1:])
                if 0 <= i < len(features):
                    return str(features[i])
            return text
        if 0 <= idx < len(features):
            return str(features[idx])
        return str(raw)

    if "leaf" in node:
        return "leaf" if show_leaves else None

    split = feature_label(node.get("split"))
    cond = node.get("split_condition")
    return f"{split} < {cond}"


def _collect_depths(node: Dict[str, Any], depth: int, out: Dict[int, int]) -> None:
    """Collect depth for each node id."""
    out[int(node["nodeid"])] = depth
    for child in node.get("children", []):
        _collect_depths(child, depth + 1, out)


def _spread_positions_by_level(
    positions: Dict[int, Tuple[float, float]],
    depths: Dict[int, int],
    labels: Dict[int, str],
    font_size: int,
) -> Dict[int, Tuple[float, float]]:
    """Reduce node-label overlaps by enforcing a minimum horizontal gap per depth."""
    out = dict(positions)
    depth_to_nodes: Dict[int, list[int]] = {}
    for node_id, depth in depths.items():
        if node_id not in labels:
            continue
        depth_to_nodes.setdefault(depth, []).append(node_id)

    # Approximate label width in plot units from character count and font size.
    char_unit = 0.012 * float(font_size)
    min_base_gap = max(0.55, 0.05 * float(font_size))

    for depth, node_ids in depth_to_nodes.items():
        node_ids.sort(key=lambda nid: out[nid][0])
        prev_id: Optional[int] = None
        for nid in node_ids:
            if prev_id is None:
                prev_id = nid
                continue

            prev_x, prev_y = out[prev_id]
            cur_x, cur_y = out[nid]
            prev_half = max(0.3, len(labels[prev_id]) * char_unit * 0.5)
            cur_half = max(0.3, len(labels[nid]) * char_unit * 0.5)
            required_gap = prev_half + cur_half + min_base_gap
            actual_gap = cur_x - prev_x

            if actual_gap < required_gap:
                shift = required_gap - actual_gap
                out[nid] = (cur_x + shift, cur_y)

            prev_id = nid

        # Re-center each level to preserve the original layout balance.
        old_center = sum(positions[nid][0] for nid in node_ids) / float(len(node_ids))
        new_center = sum(out[nid][0] for nid in node_ids) / float(len(node_ids))
        recenter_delta = old_center - new_center
        for nid in node_ids:
            x, y = out[nid]
            out[nid] = (x + recenter_delta, y)

    return out


def _plot_single_tree(
    tree_json: Dict[str, Any],
    out_path: Path,
    feature_names: Optional[Iterable[str]],
    title: str,
    font_size: int,
    line_width: float,
    x_spacing: float,
    y_spacing: float,
    show_leaves: bool,
) -> None:
    positions: Dict[int, Tuple[float, float]] = {}
    _collect_positions(tree_json, depth=0, x_cursor=[0.0], positions=positions)

    depths: Dict[int, int] = {}
    _collect_depths(tree_json, depth=0, out=depths)
    labels: Dict[int, str] = {}
    for node_id, depth in depths.items():
        del depth  # depth already encoded in y-position; keep node iteration explicit.
        node_lookup = None
        # lightweight DFS lookup for current node id
        stack = [tree_json]
        while stack:
            cur = stack.pop()
            if int(cur["nodeid"]) == node_id:
                node_lookup = cur
                break
            stack.extend(cur.get("children", []))
        if node_lookup is None:
            continue
        label = _label_for_node(node_lookup, feature_names, show_leaves)
        if label is not None:
            labels[node_id] = label

    positions = _spread_positions_by_level(positions, depths, labels, font_size)

    scaled_positions = {
        node_id: (xy[0] * x_spacing, xy[1] * y_spacing)
        for node_id, xy in positions.items()
    }

    max_x = max((xy[0] for xy in scaled_positions.values()), default=0.0)
    min_y = min((xy[1] for xy in scaled_positions.values()), default=0.0)

    fig_w = max(12.0, (max_x + 1.0) * 1.1)
    fig_h = max(8.0, abs(min_y) * 1.15 + 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _draw_tree(ax, tree_json, scaled_positions, feature_names, font_size, line_width, show_leaves)

    ax.set_title(title)
    ax.set_axis_off()
    ax.set_xlim(-1.0, max_x + 1.0)
    ax.set_ylim(min_y - 1.0, 1.0)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot one minimal XGBoost tree from an analysis_*.joblib artifact."
    )
    parser.add_argument(
        "--analysis",
        type=str,
        default="",
        help="Path to analysis_*.joblib. Defaults to latest in data/results/analyses.",
    )
    parser.add_argument(
        "--tree-index",
        type=int,
        default=0,
        help="Tree index to render (default: 0).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/results/single_tree_minimal.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=12,
        help="Node label font size (default: 12).",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=1.6,
        help="Edge line width (default: 1.6).",
    )
    parser.add_argument(
        "--x-spacing",
        type=float,
        default=0.7,
        help="Horizontal spacing multiplier. Lower values make layout denser.",
    )
    parser.add_argument(
        "--y-spacing",
        type=float,
        default=1.1,
        help="Vertical spacing multiplier.",
    )
    parser.add_argument(
        "--show-leaves",
        action="store_true",
        help="Render leaf nodes. Default is hidden for cleaner split readability.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    analyses_dir = project_root / "data" / "results" / "analyses"

    analysis_path = Path(args.analysis) if args.analysis else _latest_analysis(analyses_dir)
    analysis_obj = joblib.load(analysis_path)

    if not hasattr(analysis_obj, "model"):
        raise AttributeError("Loaded object has no 'model' attribute.")

    booster = _resolve_booster(analysis_obj.model)
    feature_names = getattr(booster, "feature_names", None)

    dumps = booster.get_dump(dump_format="json", with_stats=False)
    if not dumps:
        raise ValueError("Booster contains no trees.")
    if args.tree_index < 0 or args.tree_index >= len(dumps):
        raise IndexError(f"tree-index out of bounds: 0..{len(dumps)-1}")

    tree_json = json.loads(dumps[args.tree_index])
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = project_root / out_path

    title = f"Minimal XGBoost Tree #{args.tree_index} ({analysis_path.name})"
    _plot_single_tree(
        tree_json,
        out_path,
        feature_names,
        title,
        font_size=max(6, int(args.font_size)),
        line_width=max(0.5, float(args.line_width)),
        x_spacing=max(0.2, float(args.x_spacing)),
        y_spacing=max(0.4, float(args.y_spacing)),
        show_leaves=bool(args.show_leaves),
    )

    print(f"Analysis: {analysis_path}")
    print(f"Rendered tree index: {args.tree_index}")
    print(f"Output image: {out_path}")


if __name__ == "__main__":
    main()
