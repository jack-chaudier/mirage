"""Render candidate-pool scatter plots as standalone HTML+SVG figures.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.plot_candidate_scatter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_INPUT_JSON = OUTPUT_DIR / "candidate_pool_data.json"
DEFAULT_INVALID_HTML = OUTPUT_DIR / "candidate_scatter_invalid.html"
DEFAULT_VALID_HTML = OUTPUT_DIR / "candidate_scatter_valid.html"

COLOR_DEV0 = "#d62728"
COLOR_DEV1P = "#1f77b4"


def _resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _safe_label(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _points_for_seeds(
    *,
    candidate_rows: list[dict[str, Any]],
    seed_set: set[int],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in candidate_rows:
        seed = int(row["seed"])
        if seed not in seed_set:
            continue
        tp_position = row.get("tp_position")
        if tp_position is None:
            continue
        points.append(
            {
                "seed": seed,
                "x": float(tp_position),
                "y": float(row.get("q_score", 0.0) or 0.0),
                "dev_beat_count": int(row.get("dev_beat_count", 0) or 0),
                "is_winner": bool(row.get("is_search_winner", False)),
                "strict_valid": bool(row.get("strict_valid", False)),
                "tp_event_type": str(row.get("tp_event_type") or ""),
                "candidate_index": int(row.get("candidate_index", -1)),
                "anchor_index": int(row.get("anchor_index", -1)),
            }
        )
    return points


def _svg_scatter(
    *,
    title: str,
    subtitle: str,
    points: list[dict[str, Any]],
    annotate_invalid: bool,
) -> str:
    width = 1180
    height = 760
    margin_left = 90
    margin_right = 40
    margin_top = 90
    margin_bottom = 95
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    if points:
        y_values = [float(point["y"]) for point in points]
        y_min_raw = min(y_values)
        y_max_raw = max(y_values)
    else:
        y_min_raw = 0.0
        y_max_raw = 1.0

    y_span = y_max_raw - y_min_raw
    pad = max(0.02, y_span * 0.12)
    y_min = max(0.0, y_min_raw - pad)
    y_max = min(1.0, y_max_raw + pad)
    if y_max - y_min < 0.05:
        mid = (y_max + y_min) / 2.0
        y_min = max(0.0, mid - 0.05)
        y_max = min(1.0, mid + 0.05)

    def x_to_px(x_value: float) -> float:
        return margin_left + (max(0.0, min(1.0, x_value)) * plot_width)

    def y_to_px(y_value: float) -> float:
        if y_max - y_min <= 1e-12:
            return margin_top + plot_height / 2.0
        ratio = (y_value - y_min) / (y_max - y_min)
        return margin_top + (1.0 - ratio) * plot_height

    def clamp_x(px: float) -> float:
        return max(float(margin_left + 8), min(float(margin_left + plot_width - 8), float(px)))

    def clamp_y(py: float) -> float:
        return max(float(margin_top + 16), min(float(margin_top + plot_height - 8), float(py)))

    svg: list[str] = []
    svg.append(
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg">'
    )
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    svg.append(
        f'<text x="{margin_left}" y="42" font-family="Helvetica, Arial, sans-serif" '
        'font-size="26" font-weight="700" fill="#111111">'
        f"{_safe_label(title)}</text>"
    )
    svg.append(
        f'<text x="{margin_left}" y="66" font-family="Helvetica, Arial, sans-serif" '
        'font-size="15" fill="#444444">'
        f"{_safe_label(subtitle)}</text>"
    )

    for tick in range(11):
        x_value = tick / 10.0
        x_px = x_to_px(x_value)
        svg.append(
            f'<line x1="{x_px:.2f}" y1="{margin_top}" x2="{x_px:.2f}" y2="{margin_top + plot_height}" '
            'stroke="#ececec" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{x_px:.2f}" y="{margin_top + plot_height + 28}" '
            'font-family="Helvetica, Arial, sans-serif" font-size="12" text-anchor="middle" fill="#555555">'
            f"{x_value:.1f}</text>"
        )

    y_ticks = 7
    for tick in range(y_ticks + 1):
        y_value = y_min + (tick / y_ticks) * (y_max - y_min)
        y_px = y_to_px(y_value)
        svg.append(
            f'<line x1="{margin_left}" y1="{y_px:.2f}" x2="{margin_left + plot_width}" y2="{y_px:.2f}" '
            'stroke="#ececec" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{margin_left - 12}" y="{y_px + 4:.2f}" '
            'font-family="Helvetica, Arial, sans-serif" font-size="12" text-anchor="end" fill="#555555">'
            f"{y_value:.3f}</text>"
        )

    svg.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" '
        f'y2="{margin_top + plot_height}" stroke="#333333" stroke-width="1.5"/>'
    )
    svg.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" '
        'stroke="#333333" stroke-width="1.5"/>'
    )

    boundary_x = x_to_px(0.25)
    svg.append(
        f'<line x1="{boundary_x:.2f}" y1="{margin_top}" x2="{boundary_x:.2f}" y2="{margin_top + plot_height}" '
        'stroke="#7f7f7f" stroke-width="2" stroke-dasharray="8,7"/>'
    )
    svg.append(
        f'<text x="{boundary_x + 8:.2f}" y="{margin_top + 18}" font-family="Helvetica, Arial, sans-serif" '
        'font-size="12" fill="#666666">TP=0.25 boundary</text>'
    )

    for point in points:
        x_px = x_to_px(float(point["x"]))
        y_px = y_to_px(float(point["y"]))
        color = COLOR_DEV0 if int(point["dev_beat_count"]) == 0 else COLOR_DEV1P
        tooltip = (
            f"seed={point['seed']} "
            f"tp={float(point['x']):.3f} "
            f"q={float(point['y']):.3f} "
            f"dev={int(point['dev_beat_count'])} "
            f"winner={str(bool(point['is_winner'])).lower()} "
            f"strict_valid={str(bool(point['strict_valid'])).lower()} "
            f"tp_type={point['tp_event_type']}"
        )
        if bool(point["is_winner"]):
            size = 8.0
            triangle = (
                f"{x_px:.2f},{y_px - size:.2f} "
                f"{x_px - size * 0.9:.2f},{y_px + size * 0.75:.2f} "
                f"{x_px + size * 0.9:.2f},{y_px + size * 0.75:.2f}"
            )
            svg.append(
                f'<polygon points="{triangle}" fill="{color}" stroke="#111111" stroke-width="1.2">'
                f"<title>{_safe_label(tooltip)}</title></polygon>"
            )
        else:
            svg.append(
                f'<circle cx="{x_px:.2f}" cy="{y_px:.2f}" r="4.1" fill="{color}" fill-opacity="0.78" '
                'stroke="#ffffff" stroke-width="0.8">'
                f"<title>{_safe_label(tooltip)}</title></circle>"
            )

    legend_x = margin_left + plot_width - 290
    legend_y = margin_top + 12
    svg.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="266" height="108" rx="8" '
        'fill="#fafafa" stroke="#dddddd"/>'
    )
    svg.append(
        f'<circle cx="{legend_x + 18}" cy="{legend_y + 25}" r="5" fill="{COLOR_DEV0}"/>'
        f'<text x="{legend_x + 32}" y="{legend_y + 29}" font-family="Helvetica, Arial, sans-serif" '
        'font-size="13" fill="#222222">dev_beat_count = 0</text>'
    )
    svg.append(
        f'<circle cx="{legend_x + 18}" cy="{legend_y + 49}" r="5" fill="{COLOR_DEV1P}"/>'
        f'<text x="{legend_x + 32}" y="{legend_y + 53}" font-family="Helvetica, Arial, sans-serif" '
        'font-size="13" fill="#222222">dev_beat_count >= 1</text>'
    )
    tri = (
        f"{legend_x + 18:.2f},{legend_y + 66:.2f} "
        f"{legend_x + 10:.2f},{legend_y + 80:.2f} "
        f"{legend_x + 26:.2f},{legend_y + 80:.2f}"
    )
    svg.append(
        f'<polygon points="{tri}" fill="#555555" stroke="#111111" stroke-width="1"/>'
        f'<text x="{legend_x + 32}" y="{legend_y + 78}" font-family="Helvetica, Arial, sans-serif" '
        'font-size="13" fill="#222222">search-selected winner</text>'
    )

    if annotate_invalid and points:
        exploit_points = [
            point for point in points
            if int(point["dev_beat_count"]) == 0 and float(point["x"]) < 0.25
        ]
        valid_points = [
            point for point in points
            if int(point["dev_beat_count"]) >= 1 and float(point["x"]) >= 0.30
        ]
        winner_points = [point for point in points if bool(point["is_winner"])]

        if exploit_points:
            cx = _mean([float(point["x"]) for point in exploit_points])
            cy = _mean([float(point["y"]) for point in exploit_points])
            target_x = x_to_px(cx)
            target_y = y_to_px(cy)
            text_x = clamp_x(x_to_px(0.07))
            text_y = clamp_y(y_to_px(min(1.0, cy + 0.08)))
            svg.append(
                f'<line x1="{text_x:.2f}" y1="{text_y + 4:.2f}" x2="{target_x:.2f}" y2="{target_y:.2f}" '
                'stroke="#b22222" stroke-width="1.5"/>'
            )
            svg.append(
                f'<text x="{text_x:.2f}" y="{text_y:.2f}" font-family="Helvetica, Arial, sans-serif" '
                'font-size="13" font-weight="600" fill="#b22222">'
                "Exploit basin: high-Q, zero development beats</text>"
            )

        if valid_points:
            cx = _mean([float(point["x"]) for point in valid_points])
            cy = _mean([float(point["y"]) for point in valid_points])
            target_x = x_to_px(cx)
            target_y = y_to_px(cy)
            text_x = clamp_x(x_to_px(0.57))
            text_y = clamp_y(y_to_px(min(1.0, cy + 0.11)))
            svg.append(
                f'<line x1="{text_x:.2f}" y1="{text_y + 4:.2f}" x2="{target_x:.2f}" y2="{target_y:.2f}" '
                'stroke="#1f4e8c" stroke-width="1.5"/>'
            )
            svg.append(
                f'<text x="{text_x:.2f}" y="{text_y:.2f}" font-family="Helvetica, Arial, sans-serif" '
                'font-size="13" font-weight="600" fill="#1f4e8c">'
                "Valid alternatives: lower Q, structurally sound</text>"
            )
        else:
            text_x = clamp_x(x_to_px(0.57))
            text_y = clamp_y(y_to_px(y_min + 0.40 * (y_max - y_min)))
            target_x = x_to_px(0.62)
            target_y = y_to_px(y_min + 0.30 * (y_max - y_min))
            svg.append(
                f'<line x1="{text_x:.2f}" y1="{text_y + 4:.2f}" x2="{target_x:.2f}" y2="{target_y:.2f}" '
                'stroke="#1f4e8c" stroke-width="1.5"/>'
            )
            svg.append(
                f'<text x="{text_x:.2f}" y="{text_y:.2f}" font-family="Helvetica, Arial, sans-serif" '
                'font-size="13" font-weight="600" fill="#1f4e8c">'
                "Valid alternatives: none in this search pool</text>"
            )

        if winner_points:
            cx = _mean([float(point["x"]) for point in winner_points])
            cy = _mean([float(point["y"]) for point in winner_points])
            target_x = x_to_px(cx)
            target_y = y_to_px(cy)
            text_x = clamp_x(x_to_px(0.35))
            text_y = clamp_y(y_to_px(min(1.0, cy + 0.16)))
            svg.append(
                f'<line x1="{text_x:.2f}" y1="{text_y + 4:.2f}" x2="{target_x:.2f}" y2="{target_y:.2f}" '
                'stroke="#111111" stroke-width="1.4"/>'
            )
            svg.append(
                f'<text x="{text_x:.2f}" y="{text_y:.2f}" font-family="Helvetica, Arial, sans-serif" '
                'font-size="13" font-weight="700" fill="#111111">'
                "Search selects here (greedy Q-max)</text>"
            )

    svg.append(
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 24}" '
        'font-family="Helvetica, Arial, sans-serif" font-size="15" text-anchor="middle" fill="#222222">'
        "Turning point global position</text>"
    )
    svg.append(
        f'<text transform="translate(24,{margin_top + plot_height / 2:.2f}) rotate(-90)" '
        'font-family="Helvetica, Arial, sans-serif" font-size="15" text-anchor="middle" fill="#222222">'
        "Q-score</text>"
    )

    svg.append("</svg>")
    return "\n".join(svg)


def _wrap_html(*, title: str, svg_markup: str) -> str:
    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\" />\n"
        f"  <title>{_safe_label(title)}</title>\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
        "  <style>\n"
        "    body { margin: 0; font-family: Helvetica, Arial, sans-serif; background: #f5f5f5; }\n"
        "    .container { max-width: 1240px; margin: 20px auto; background: #fff; padding: 12px; "
        "box-shadow: 0 8px 26px rgba(0,0,0,0.08); border-radius: 12px; overflow: auto; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <div class=\"container\">\n"
        f"{svg_markup}\n"
        "  </div>\n"
        "</body>\n"
        "</html>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render candidate scatter plots from candidate_pool_data.json.")
    parser.add_argument("--input-json", type=str, default=str(DEFAULT_INPUT_JSON))
    parser.add_argument("--output-invalid", type=str, default=str(DEFAULT_INVALID_HTML))
    parser.add_argument("--output-valid", type=str, default=str(DEFAULT_VALID_HTML))
    args = parser.parse_args()

    input_json_path = _resolve_path(args.input_json)
    output_invalid_path = _resolve_path(args.output_invalid)
    output_valid_path = _resolve_path(args.output_valid)

    payload = json.loads(input_json_path.read_text(encoding="utf-8"))
    candidate_rows = list((payload.get("diana") or {}).get("candidate_records") or [])
    invalid_seeds = [int(seed) for seed in ((payload.get("diana") or {}).get("invalid_seed_set_from_strict") or [])]
    valid_seed_sample = [int(seed) for seed in ((payload.get("diana") or {}).get("valid_seed_sample") or [])]

    invalid_points = _points_for_seeds(candidate_rows=candidate_rows, seed_set=set(invalid_seeds))
    valid_points = _points_for_seeds(candidate_rows=candidate_rows, seed_set=set(valid_seed_sample))

    invalid_svg = _svg_scatter(
        title="Candidate Pool Scatter: Diana Invalid Seeds",
        subtitle=f"Seeds={invalid_seeds} | points={len(invalid_points)} | x=TP position | y=Q-score",
        points=invalid_points,
        annotate_invalid=True,
    )
    invalid_html = _wrap_html(
        title="Candidate Scatter (Invalid Seeds)",
        svg_markup=invalid_svg,
    )

    valid_svg = _svg_scatter(
        title="Candidate Pool Scatter: Diana Valid Seed Sample",
        subtitle=f"Seeds={valid_seed_sample} | points={len(valid_points)} | x=TP position | y=Q-score",
        points=valid_points,
        annotate_invalid=False,
    )
    valid_html = _wrap_html(
        title="Candidate Scatter (Valid Seed Sample)",
        svg_markup=valid_svg,
    )

    output_invalid_path.parent.mkdir(parents=True, exist_ok=True)
    output_invalid_path.write_text(invalid_html, encoding="utf-8")

    output_valid_path.parent.mkdir(parents=True, exist_ok=True)
    output_valid_path.write_text(valid_html, encoding="utf-8")

    print(f"Wrote invalid plot: {output_invalid_path}")
    print(f"Wrote valid plot: {output_valid_path}")


if __name__ == "__main__":
    main()
