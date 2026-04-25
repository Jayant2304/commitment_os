"""Render SVG visuals for LLM checkpoint comparison."""

from __future__ import annotations

import csv
from pathlib import Path

ARTIFACT_DIR = Path("artifacts/evals_llm")
COMPARISON_CSV = ARTIFACT_DIR / "llm_comparison.csv"


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
    ]


def _svg_footer() -> list[str]:
    return ["</svg>"]


def _rows() -> list[dict[str, str]]:
    with COMPARISON_CSV.open() as f:
        return list(csv.DictReader(f))


def plot_reward(rows: list[dict[str, str]]) -> None:
    tasks = [r["task_id"] for r in rows]
    base = [float(r["baseline_reward"]) for r in rows]
    trained = [float(r["trained_reward"]) for r in rows]

    width, height = 1360, 520
    left, right, top, bottom = 80, 40, 70, 110
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / max(len(tasks), 1)
    bar_w = max(group_w * 0.32, 10)

    lines = _svg_header(width, height)
    lines.append('<text x="80" y="35" font-size="22" font-family="Arial" fill="#111827">Base vs Trained LLM Reward by Task</text>')
    lines.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#374151" stroke-width="1"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#374151" stroke-width="1"/>')

    for tick in range(0, 6):
        value = tick / 5
        y = top + plot_h - (value * plot_h)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#E5E7EB" stroke-width="1"/>')
        lines.append(f'<text x="{left-38}" y="{y+5:.2f}" font-size="12" font-family="Arial" fill="#374151">{value:.1f}</text>')

    for idx, task in enumerate(tasks):
        gx = left + (idx * group_w) + (group_w * 0.5)
        b_h = base[idx] * plot_h
        t_h = trained[idx] * plot_h
        b_x = gx - bar_w - 2
        t_x = gx + 2
        b_y = top + plot_h - b_h
        t_y = top + plot_h - t_h
        lines.append(f'<rect x="{b_x:.2f}" y="{b_y:.2f}" width="{bar_w:.2f}" height="{b_h:.2f}" fill="#9CA3AF"/>')
        lines.append(f'<rect x="{t_x:.2f}" y="{t_y:.2f}" width="{bar_w:.2f}" height="{t_h:.2f}" fill="#2563EB"/>')
        lines.append(
            f'<text x="{gx:.2f}" y="{top+plot_h+22}" font-size="10" text-anchor="middle" '
            f'font-family="Arial" fill="#374151" transform="rotate(25 {gx:.2f},{top+plot_h+22})">{task}</text>'
        )

    legend_y = 52
    lines.append(f'<rect x="{width-310}" y="{legend_y-10}" width="12" height="12" fill="#9CA3AF"/>')
    lines.append(f'<text x="{width-292}" y="{legend_y}" font-size="12" font-family="Arial" fill="#111827">Base</text>')
    lines.append(f'<rect x="{width-230}" y="{legend_y-10}" width="12" height="12" fill="#2563EB"/>')
    lines.append(f'<text x="{width-212}" y="{legend_y}" font-size="12" font-family="Arial" fill="#111827">Trained</text>')
    lines.extend(_svg_footer())
    (ARTIFACT_DIR / "llm_reward_by_task.svg").write_text("\n".join(lines))


def plot_violations(rows: list[dict[str, str]]) -> None:
    tasks = [r["task_id"] for r in rows]
    base = [int(r["baseline_violations"]) for r in rows]
    trained = [int(r["trained_violations"]) for r in rows]
    max_v = max(max(base, default=0), max(trained, default=0), 1)

    width, height = 1360, 500
    left, right, top, bottom = 80, 40, 70, 100
    plot_w = width - left - right
    plot_h = height - top - bottom

    def point_x(i: int) -> float:
        return left + (i / max(len(tasks) - 1, 1)) * plot_w

    def point_y(v: int) -> float:
        return top + plot_h - ((v / max_v) * plot_h)

    lines = _svg_header(width, height)
    lines.append('<text x="80" y="35" font-size="22" font-family="Arial" fill="#111827">Base vs Trained LLM Commitment Violations</text>')
    lines.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#374151" stroke-width="1"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#374151" stroke-width="1"/>')

    for tick in range(max_v + 1):
        y = point_y(tick)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#E5E7EB" stroke-width="1"/>')
        lines.append(f'<text x="{left-24}" y="{y+5:.2f}" font-size="12" font-family="Arial" fill="#374151">{tick}</text>')

    base_points = " ".join(f"{point_x(i):.2f},{point_y(v):.2f}" for i, v in enumerate(base))
    tr_points = " ".join(f"{point_x(i):.2f},{point_y(v):.2f}" for i, v in enumerate(trained))
    lines.append(f'<polyline points="{base_points}" fill="none" stroke="#DC2626" stroke-width="2"/>')
    lines.append(f'<polyline points="{tr_points}" fill="none" stroke="#059669" stroke-width="2"/>')

    for i, task in enumerate(tasks):
        x = point_x(i)
        lines.append(f'<circle cx="{x:.2f}" cy="{point_y(base[i]):.2f}" r="3" fill="#DC2626"/>')
        lines.append(f'<circle cx="{x:.2f}" cy="{point_y(trained[i]):.2f}" r="3" fill="#059669"/>')
        lines.append(
            f'<text x="{x:.2f}" y="{top+plot_h+20}" font-size="10" text-anchor="middle" '
            f'font-family="Arial" fill="#374151" transform="rotate(25 {x:.2f},{top+plot_h+20})">{task}</text>'
        )

    legend_y = 52
    lines.append(f'<line x1="{width-320}" y1="{legend_y-5}" x2="{width-300}" y2="{legend_y-5}" stroke="#DC2626" stroke-width="2"/>')
    lines.append(f'<text x="{width-295}" y="{legend_y}" font-size="12" font-family="Arial" fill="#111827">Base</text>')
    lines.append(f'<line x1="{width-230}" y1="{legend_y-5}" x2="{width-210}" y2="{legend_y-5}" stroke="#059669" stroke-width="2"/>')
    lines.append(f'<text x="{width-205}" y="{legend_y}" font-size="12" font-family="Arial" fill="#111827">Trained</text>')
    lines.extend(_svg_footer())
    (ARTIFACT_DIR / "llm_violations_before_after.svg").write_text("\n".join(lines))


def main() -> None:
    rows = _rows()
    plot_reward(rows)
    plot_violations(rows)
    print("Wrote checkpoint comparison SVG plots to", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
