"""Generate judge-friendly SVG plots from evaluation comparison CSV.

This module intentionally avoids matplotlib to keep plotting deterministic
in restricted CI/sandbox environments.
"""

from __future__ import annotations

import csv
from pathlib import Path

ARTIFACT_DIR = Path("artifacts/evals")
COMPARISON_CSV = ARTIFACT_DIR / "comparison.csv"


def _load_rows() -> list[dict[str, str]]:
    with COMPARISON_CSV.open() as f:
        return list(csv.DictReader(f))


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
    ]


def _svg_footer() -> list[str]:
    return ["</svg>"]


def plot_reward_by_task(rows: list[dict[str, str]]) -> None:
    tasks = [row["task_id"] for row in rows]
    baseline = [float(row["baseline_reward"]) for row in rows]
    improved = [float(row["improved_reward"]) for row in rows]

    width, height = 1360, 520
    left, right, top, bottom = 80, 40, 70, 110
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / max(len(tasks), 1)
    bar_w = max(group_w * 0.32, 10)

    lines = _svg_header(width, height)
    lines.append('<text x="80" y="35" font-size="22" font-family="Arial" fill="#111827">Baseline vs Improved Reward by Task</text>')
    lines.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#374151" stroke-width="1"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#374151" stroke-width="1"/>')

    for tick in range(0, 6):
        value = tick / 5
        y = top + plot_h - (value * plot_h)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#E5E7EB" stroke-width="1"/>')
        lines.append(f'<text x="{left-38}" y="{y+5:.2f}" font-size="12" font-family="Arial" fill="#374151">{value:.1f}</text>')

    for idx, task in enumerate(tasks):
        gx = left + (idx * group_w) + (group_w * 0.5)
        b_h = baseline[idx] * plot_h
        i_h = improved[idx] * plot_h
        b_x = gx - bar_w - 2
        i_x = gx + 2
        b_y = top + plot_h - b_h
        i_y = top + plot_h - i_h
        lines.append(f'<rect x="{b_x:.2f}" y="{b_y:.2f}" width="{bar_w:.2f}" height="{b_h:.2f}" fill="#9CA3AF"/>')
        lines.append(f'<rect x="{i_x:.2f}" y="{i_y:.2f}" width="{bar_w:.2f}" height="{i_h:.2f}" fill="#2563EB"/>')
        lines.append(
            f'<text x="{gx:.2f}" y="{top+plot_h+22}" font-size="10" text-anchor="middle" '
            f'font-family="Arial" fill="#374151" transform="rotate(25 {gx:.2f},{top+plot_h+22})">{task}</text>'
        )

    legend_y = 52
    lines.append(f'<rect x="{width-300}" y="{legend_y-10}" width="12" height="12" fill="#9CA3AF"/>')
    lines.append(f'<text x="{width-282}" y="{legend_y}" font-size="12" font-family="Arial" fill="#111827">Baseline</text>')
    lines.append(f'<rect x="{width-210}" y="{legend_y-10}" width="12" height="12" fill="#2563EB"/>')
    lines.append(f'<text x="{width-192}" y="{legend_y}" font-size="12" font-family="Arial" fill="#111827">Improved</text>')
    lines.extend(_svg_footer())

    (ARTIFACT_DIR / "reward_by_task.svg").write_text("\n".join(lines))


def plot_violation_before_after(rows: list[dict[str, str]]) -> None:
    tasks = [row["task_id"] for row in rows]
    baseline = [int(row["baseline_violations"]) for row in rows]
    improved = [int(row["improved_violations"]) for row in rows]
    max_v = max(max(baseline, default=0), max(improved, default=0), 1)

    width, height = 1360, 500
    left, right, top, bottom = 80, 40, 70, 100
    plot_w = width - left - right
    plot_h = height - top - bottom

    def point_x(idx: int) -> float:
        return left + (idx / max(len(tasks) - 1, 1)) * plot_w

    def point_y(value: int) -> float:
        return top + plot_h - ((value / max_v) * plot_h)

    lines = _svg_header(width, height)
    lines.append('<text x="80" y="35" font-size="22" font-family="Arial" fill="#111827">Commitment Violations Before vs After</text>')
    lines.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#374151" stroke-width="1"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#374151" stroke-width="1"/>')

    for tick in range(max_v + 1):
        y = point_y(tick)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#E5E7EB" stroke-width="1"/>')
        lines.append(f'<text x="{left-24}" y="{y+5:.2f}" font-size="12" font-family="Arial" fill="#374151">{tick}</text>')

    baseline_points = " ".join(f"{point_x(i):.2f},{point_y(v):.2f}" for i, v in enumerate(baseline))
    improved_points = " ".join(f"{point_x(i):.2f},{point_y(v):.2f}" for i, v in enumerate(improved))
    lines.append(f'<polyline points="{baseline_points}" fill="none" stroke="#DC2626" stroke-width="2"/>')
    lines.append(f'<polyline points="{improved_points}" fill="none" stroke="#059669" stroke-width="2"/>')

    for i, task in enumerate(tasks):
        x = point_x(i)
        lines.append(f'<circle cx="{x:.2f}" cy="{point_y(baseline[i]):.2f}" r="3" fill="#DC2626"/>')
        lines.append(f'<circle cx="{x:.2f}" cy="{point_y(improved[i]):.2f}" r="3" fill="#059669"/>')
        lines.append(
            f'<text x="{x:.2f}" y="{top+plot_h+20}" font-size="10" text-anchor="middle" '
            f'font-family="Arial" fill="#374151" transform="rotate(25 {x:.2f},{top+plot_h+20})">{task}</text>'
        )

    legend_y = 52
    lines.append(f'<line x1="{width-320}" y1="{legend_y-5}" x2="{width-300}" y2="{legend_y-5}" stroke="#DC2626" stroke-width="2"/>')
    lines.append(f'<text x="{width-295}" y="{legend_y}" font-size="12" font-family="Arial" fill="#111827">Baseline</text>')
    lines.append(f'<line x1="{width-220}" y1="{legend_y-5}" x2="{width-200}" y2="{legend_y-5}" stroke="#059669" stroke-width="2"/>')
    lines.append(f'<text x="{width-195}" y="{legend_y}" font-size="12" font-family="Arial" fill="#111827">Improved</text>')
    lines.extend(_svg_footer())

    (ARTIFACT_DIR / "violations_before_after.svg").write_text("\n".join(lines))


def main() -> None:
    rows = _load_rows()
    plot_reward_by_task(rows)
    plot_violation_before_after(rows)
    print("Wrote SVG plots to", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
