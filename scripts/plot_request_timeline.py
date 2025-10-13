#!/usr/bin/env python3
"""
Plot per-request timelines from benchmark JSON.

Input JSON format (example):
{
  "benchmark_start_perf_counter": 27248567.534459863,
  "unit": "ms",
  "events": [
    {"request_id": 0, "t_yield_ms": 0.01, "t_post_ms": 0.38, "t_ttft_ms": 2056.89, "t_done_ms": 10779.25, "success": true, "error": ""},
    ...
  ]
}

For each request, this script draws:
- Prefill/queue: from t_post_ms to t_ttft_ms (blue)
- Decode/stream: from t_ttft_ms to t_done_ms (green)
- Optional markers: t_yield_ms (▼) and t_ttft_ms (●)

Usage:
  python scripts/plot_request_timeline.py input.json --out timeline.png --show

Dependencies:
  pip install matplotlib
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_ms(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_events(obj: Any) -> Tuple[List[Dict[str, Any]], str]:
    """Return (events, unit). Accepts top-level dict with 'events' or a list."""
    if isinstance(obj, dict) and "events" in obj:
        unit = str(obj.get("unit", "ms"))
        events = obj.get("events", [])
    elif isinstance(obj, list):
        unit = "ms"
        events = obj
    else:
        raise ValueError("Input must be a dict with 'events' or a list of events")

    # Normalize and validate
    norm_events: List[Dict[str, Any]] = []
    for idx, e in enumerate(events):
        if not isinstance(e, dict):
            continue
        request_id = e.get("request_id", idx)
        t_post = _coerce_ms(e.get("t_post_ms"))
        t_ttft = _coerce_ms(e.get("t_ttft_ms"))
        t_done = _coerce_ms(e.get("t_done_ms"))
        t_yield = _coerce_ms(e.get("t_yield_ms"))
        success = bool(e.get("success", True))
        error = str(e.get("error", ""))

        norm_events.append(
            {
                "request_id": request_id,
                "t_post_ms": t_post,
                "t_ttft_ms": t_ttft,
                "t_done_ms": t_done,
                "t_yield_ms": t_yield,
                "success": success,
                "error": error,
            }
        )

    return norm_events, unit


def _compute_segments(e: Dict[str, Any]) -> Dict[str, Optional[Tuple[float, float]]]:
    """Return segment ranges (start, duration) in ms for prefill and decode."""
    t_post = e.get("t_post_ms")
    t_ttft = e.get("t_ttft_ms")
    t_done = e.get("t_done_ms")

    prefill: Optional[Tuple[float, float]] = None
    decode: Optional[Tuple[float, float]] = None

    if t_post is not None and t_ttft is not None and t_ttft >= t_post:
        prefill = (t_post, max(0.0, t_ttft - t_post))

    if t_ttft is not None and t_done is not None and t_done >= t_ttft:
        decode = (t_ttft, max(0.0, t_done - t_ttft))

    return {"prefill": prefill, "decode": decode}


def plot_timeline(
    events: List[Dict[str, Any]],
    *,
    out_path: str,
    show: bool = False,
    sort_by: str = "post",
    filter_success: bool = False,
    dpi: int = 200,
    title: Optional[str] = None,
) -> None:
    if filter_success:
        events = [e for e in events if e.get("success", True)]

    key_map = {
        "post": "t_post_ms",
        "yield": "t_yield_ms",
        "ttft": "t_ttft_ms",
        "done": "t_done_ms",
        "id": "request_id",
    }
    key = key_map.get(sort_by, "t_post_ms")
    events.sort(key=lambda e: (float("inf") if e.get(key) is None else e.get(key)))

    if not events:
        raise ValueError("No events to plot after filtering")

    num = len(events)
    bar_height = 0.8
    fig_height = max(3.0, min(0.35 * num + 1.5, 18.0))
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=dpi)

    prefill_color = "#6baed6"  # blue
    decode_color = "#31a354"   # green
    failed_alpha = 0.4          # fade failed requests if included

    y_ticks = []
    y_labels = []

    for i, e in enumerate(events):
        y = i
        rid = e.get("request_id", i)
        segments = _compute_segments(e)
        alpha = 1.0 if e.get("success", True) else failed_alpha

        # Draw segments
        if segments["prefill"] is not None:
            start, dur = segments["prefill"]
            ax.broken_barh([(start, dur)], (y - bar_height / 2, bar_height), facecolors=prefill_color, alpha=alpha)
        if segments["decode"] is not None:
            start, dur = segments["decode"]
            ax.broken_barh([(start, dur)], (y - bar_height / 2, bar_height), facecolors=decode_color, alpha=alpha)

        # Markers
        t_yield = e.get("t_yield_ms")
        if t_yield is not None:
            ax.plot([t_yield], [y], marker="v", color="#aa00ff", markersize=5, alpha=alpha, linestyle="none", label=None)
        t_ttft = e.get("t_ttft_ms")
        if t_ttft is not None:
            ax.plot([t_ttft], [y], marker="o", color="#000000", markersize=3.5, alpha=alpha, linestyle="none", label=None)

        y_ticks.append(y)
        y_labels.append(str(rid))

    # Axis formatting
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylabel("request_id")

    # X axis as ms with thousands separators, and an auxiliary top axis in seconds
    ax.set_xlabel("Time (ms)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))

    # Secondary top axis in seconds for convenience
    def _to_sec(x):
        return x / 1000.0

    def _to_ms(x):
        return x * 1000.0

    sec_ax = ax.secondary_xaxis("top", functions=(_to_sec, _to_ms))
    sec_ax.set_xlabel("Time (s)")
    sec_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}"))

    # Limits
    all_times: List[float] = []
    for e in events:
        for k in ("t_post_ms", "t_ttft_ms", "t_done_ms"):
            v = e.get(k)
            if isinstance(v, (int, float)):
                all_times.append(float(v))
    if all_times:
        xmin = max(0.0, min(all_times) - 0.02 * (max(all_times) - min(all_times) + 1))
        xmax = max(all_times) * 1.02 + 1
        ax.set_xlim(xmin, xmax)

    ax.set_ylim(-1, num)
    ax.grid(True, axis="x", alpha=0.3)

    # Legend
    legend_items = [
        Patch(facecolor=prefill_color, label="Post→TTFT (prefill/queue)"),
        Patch(facecolor=decode_color, label="TTFT→Done (decode/stream)"),
    ]
    ax.legend(handles=legend_items, loc="lower right")

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved timeline to: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-request timeline from benchmark JSON")
    parser.add_argument("input", type=str, help="Path to JSON file (dict with 'events' or list of events)")
    parser.add_argument("--out", type=str, default="request_timeline.png", help="Output image path (png/pdf/svg)")
    parser.add_argument("--show", action="store_true", help="Show interactive window after saving")
    parser.add_argument(
        "--sort-by",
        type=str,
        default="post",
        choices=["post", "yield", "ttft", "done", "id"],
        help="Sort order for requests",
    )
    parser.add_argument("--only-success", action="store_true", help="Only include success==true requests")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI for raster outputs")
    args = parser.parse_args()

    data = _read_json(args.input)
    events, unit = _extract_events(data)
    if unit.lower() != "ms":
        print(f"Warning: Expected 'ms' unit; got '{unit}'. Proceeding as milliseconds.")

    title = f"Request Timeline ({len(events)} requests)"
    plot_timeline(
        events,
        out_path=args.out,
        show=args.show,
        sort_by=args.sort_by,
        filter_success=bool(args.only_success),
        dpi=args.dpi,
        title=title,
    )


if __name__ == "__main__":
    main()
