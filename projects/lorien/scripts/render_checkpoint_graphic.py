from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _catastrophes_from_runs(runs: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for run in runs:
        by_agent = run.get("catastrophes_by_agent")
        if not isinstance(by_agent, dict):
            continue
        for agent, count in by_agent.items():
            key = str(agent)
            out[key] = out.get(key, 0) + _as_int(count)
    return out


def _mean_location_share_from_runs(runs: list[dict[str, Any]]) -> dict[str, float]:
    all_locations: set[str] = set()
    for run in runs:
        loc = run.get("location_distribution")
        if isinstance(loc, dict):
            all_locations.update(str(k) for k in loc.keys())

    if not all_locations:
        return {}

    count = max(1, len(runs))
    out: dict[str, float] = {}
    for location in sorted(all_locations):
        total = 0.0
        for run in runs:
            loc = run.get("location_distribution")
            if not isinstance(loc, dict):
                continue
            total += _as_float(loc.get(location), 0.0)
        out[location] = total / count
    return out


def _extract_metrics(report: dict[str, Any]) -> dict[str, Any]:
    aggregate = report.get("aggregate")
    if not isinstance(aggregate, dict):
        aggregate = {}

    runs = report.get("runs")
    if not isinstance(runs, list):
        runs = []

    determinism_check = report.get("determinism_check")
    if not isinstance(determinism_check, dict):
        determinism_check = {}

    run_count = _as_int(aggregate.get("run_count"), len(runs))
    determinism_matches = bool(determinism_check.get("matches", False))
    determinism_seed = _as_int(determinism_check.get("seed"), 0)

    catastrophes_by_agent = aggregate.get("catastrophes_by_agent")
    if not isinstance(catastrophes_by_agent, dict):
        catastrophes_by_agent = _catastrophes_from_runs(runs)
    catastrophes_clean = {str(k): _as_int(v) for k, v in catastrophes_by_agent.items()}

    mean_location_share = aggregate.get("mean_location_share")
    if not isinstance(mean_location_share, dict):
        mean_location_share = _mean_location_share_from_runs(runs)
    mean_location_clean = {str(k): _as_float(v) for k, v in mean_location_share.items()}

    total_catastrophes = _as_int(aggregate.get("total_catastrophes"), sum(catastrophes_clean.values()))
    runs_with_catastrophe = _as_int(
        aggregate.get("runs_with_catastrophe"),
        sum(1 for run in runs if _as_int(run.get("catastrophe_count"), 0) > 0),
    )

    generated_at = str(report.get("generated_at") or "")
    scenario = ""
    params = report.get("parameters")
    if isinstance(params, dict):
        scenario = str(params.get("scenario") or "")

    return {
        "run_count": run_count,
        "determinism_matches": determinism_matches,
        "determinism_seed": determinism_seed,
        "total_catastrophes": total_catastrophes,
        "runs_with_catastrophe": runs_with_catastrophe,
        "catastrophes_by_agent": catastrophes_clean,
        "mean_location_share": mean_location_clean,
        "generated_at": generated_at,
        "scenario": scenario,
    }


def _sorted_catastrophes(catastrophes_by_agent: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(catastrophes_by_agent.items(), key=lambda kv: (-kv[1], kv[0]))


def _sorted_location_share(mean_location_share: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(mean_location_share.items(), key=lambda kv: (-kv[1], kv[0]))


def _svg_text(x: int, y: int, content: str, *, size: int = 16, weight: int = 400, fill: str = "#112A46") -> str:
    escaped = html.escape(content)
    return (
        f'<text x="{x}" y="{y}" font-family="Georgia, \'Times New Roman\', serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{escaped}</text>'
    )


def render_svg(report: dict[str, Any]) -> str:
    metrics = _extract_metrics(report)
    catastrophe_rows = _sorted_catastrophes(metrics["catastrophes_by_agent"])
    location_rows = _sorted_location_share(metrics["mean_location_share"])

    width = 1120
    section_gap = 52
    row_height = 34
    margin_top = 48
    margin_bottom = 48

    catastrophe_rows_count = max(1, len(catastrophe_rows))
    location_rows_count = max(1, len(location_rows))
    catastrophe_height = 72 + catastrophe_rows_count * row_height
    location_height = 102 + location_rows_count * row_height
    height = margin_top + 160 + section_gap + catastrophe_height + section_gap + location_height + margin_bottom

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    lines.append("<defs>")
    lines.append(
        '<linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#F8F5EF"/>'
        '<stop offset="100%" stop-color="#E9F0F8"/>'
        "</linearGradient>"
    )
    lines.append("</defs>")
    lines.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="url(#bg)"/>')

    x0 = 72
    y = margin_top

    lines.append(_svg_text(x0, y, "NarrativeField Checkpoint Overview", size=34, weight=700, fill="#0B1E34"))
    y += 36

    subtitle_parts = []
    if metrics["scenario"]:
        subtitle_parts.append(f"scenario={metrics['scenario']}")
    if metrics["generated_at"]:
        subtitle_parts.append(f"generated={metrics['generated_at']}")
    subtitle = " | ".join(subtitle_parts) if subtitle_parts else "checkpoint summary"
    lines.append(_svg_text(x0, y, subtitle, size=14, weight=400, fill="#35506D"))
    y += 36

    run_count = metrics["run_count"]
    det_label = "PASS" if metrics["determinism_matches"] else "FAIL"
    det_color = "#1B7A3F" if metrics["determinism_matches"] else "#A32424"
    lines.append(_svg_text(x0, y, f"Runs: {run_count}", size=26, weight=700))
    lines.append(_svg_text(x0 + 290, y, f"Determinism: {det_label}", size=26, weight=700, fill=det_color))
    y += 40
    lines.append(
        _svg_text(
            x0,
            y,
            f"Determinism probe seed: {metrics['determinism_seed']}  |  Runs with catastrophe: "
            f"{metrics['runs_with_catastrophe']}/{run_count}  |  Total catastrophes: {metrics['total_catastrophes']}",
            size=15,
            fill="#18344F",
        )
    )
    y += section_gap

    lines.append(_svg_text(x0, y, "Catastrophe Distribution by Agent", size=24, weight=700, fill="#0D2948"))
    y += 36

    bar_label_x = x0
    bar_x = x0 + 255
    bar_w = 720
    max_cat = max((count for _, count in catastrophe_rows), default=1)
    if max_cat <= 0:
        max_cat = 1

    if not catastrophe_rows:
        lines.append(_svg_text(x0, y + 8, "No catastrophe events recorded.", size=16, fill="#1B3858"))
        y += row_height
    else:
        for agent, count in catastrophe_rows:
            lines.append(_svg_text(bar_label_x, y + 22, f"Agent {agent}", size=15, fill="#18344F"))
            width_px = int(round((count / max_cat) * bar_w))
            lines.append(f'<rect x="{bar_x}" y="{y}" width="{bar_w}" height="22" fill="#DCE5EF" rx="4" ry="4"/>')
            lines.append(f'<rect x="{bar_x}" y="{y}" width="{width_px}" height="22" fill="#C6543B" rx="4" ry="4"/>')
            lines.append(_svg_text(bar_x + bar_w + 12, y + 18, str(count), size=14, weight=700, fill="#6A1F10"))
            y += row_height

    y += section_gap - 10

    lines.append(_svg_text(x0, y, "Mean Location Share", size=24, weight=700, fill="#0D2948"))
    y += 32

    lines.append(_svg_text(bar_x, y, "0%", size=12, fill="#48627E"))
    lines.append(_svg_text(bar_x + int(bar_w * 0.25), y, "25%", size=12, fill="#48627E"))
    lines.append(_svg_text(bar_x + int(bar_w * 0.50), y, "50%", size=12, fill="#48627E"))
    lines.append(_svg_text(bar_x + int(bar_w * 0.75), y, "75%", size=12, fill="#48627E"))
    lines.append(_svg_text(bar_x + bar_w - 20, y, "100%", size=12, fill="#48627E"))
    y += 16

    if not location_rows:
        lines.append(_svg_text(x0, y + 20, "No location-share data available.", size=16, fill="#1B3858"))
    else:
        for loc, share in location_rows:
            clamped = max(0.0, min(1.0, share))
            width_px = int(round(clamped * bar_w))
            lines.append(_svg_text(bar_label_x, y + 22, loc, size=15, fill="#18344F"))
            lines.append(f'<rect x="{bar_x}" y="{y}" width="{bar_w}" height="22" fill="#DCE5EF" rx="4" ry="4"/>')
            lines.append(f'<rect x="{bar_x}" y="{y}" width="{width_px}" height="22" fill="#2B6FA8" rx="4" ry="4"/>')
            lines.append(_svg_text(bar_x + bar_w + 12, y + 18, f"{clamped * 100:.1f}%", size=14, weight=700, fill="#173A59"))
            y += row_height

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="render_checkpoint_graphic")
    parser.add_argument("--input", required=True, help="Path to checkpoint JSON generated by science_harness.py")
    parser.add_argument("--output", required=True, help="Path to output SVG")
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.output)

    report = json.loads(in_path.read_text(encoding="utf-8"))
    svg = render_svg(report)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
