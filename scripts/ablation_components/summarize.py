#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize component-ablation action-protocol metrics into paper CSV/JSON."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SEMANTIC_PROTOCOL_LABELS = {
    "1_30": "P1-IP",
    "31_35": "P2-SIP",
    "36_50": "P3-NIP",
}

SEMANTIC_GROUP_ALIASES = {
    "1_30": ("1_30", "mean_1_30", "P1-IP", "In-Place"),
    "31_35": ("31_35", "mean_31_35", "P2-SIP", "Sit-In-Place"),
    "36_50": ("36_50", "mean_36_50", "P3-NIP", "Non-In-Place"),
}


def _load_metrics(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if "test" in payload and isinstance(payload["test"], dict):
        return payload["test"]
    if "metrics" in payload and isinstance(payload["metrics"], dict):
        return payload["metrics"]
    return payload


def _metric_action_group_item(metrics: dict[str, Any]) -> dict[str, Any]:
    groups = metrics.get("m4human_action_groups", {})
    metric_item = groups.get("absolute_mpjpe") or groups.get("mpjpe")
    if not isinstance(metric_item, dict):
        raise KeyError("Metrics JSON does not contain m4human_action_groups.{absolute_mpjpe|mpjpe}")
    return metric_item


def _semantic_action_group_abs(metrics: dict[str, Any]) -> dict[str, float]:
    metric_item = _metric_action_group_item(metrics)
    if not isinstance(metric_item.get("by_group"), dict):
        raise KeyError("Metrics JSON does not contain m4human_action_groups.{absolute_mpjpe|mpjpe}.by_group")
    return {str(key): float(value) for key, value in metric_item["by_group"].items() if value is not None}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entry",
        action="append",
        default=[],
        metavar="SPLIT,VARIANT,PATH",
        help="One metrics entry, e.g. Cross-Subject,No body token,/path/metrics.json",
    )
    parser.add_argument("--output-csv", required=True, help="Output fig2-style CSV path.")
    parser.add_argument("--output-json", help="Optional output JSON path.")
    parser.add_argument(
        "--bucket-mode",
        choices=("semantic",),
        default="semantic",
        help="Use semantic M4Human paper protocols: P1/IP 1-30, P2/SIP 31-35, P3/NIP 36-50.",
    )
    parser.add_argument("--indices-path", help=argparse.SUPPRESS)
    parser.add_argument("--allow-missing", action="store_true", help="Write blank cells for missing action groups.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    rows: list[dict[str, Any]] = []
    for spec in args.entry:
        parts = [part.strip() for part in str(spec).split(",", 2)]
        if len(parts) != 3:
            raise ValueError(f"--entry must be SPLIT,VARIANT,PATH; got {spec!r}")
        split, variant, path = parts
        metrics = _load_metrics(path)
        by_group = _semantic_action_group_abs(metrics)
        protocol_labels = SEMANTIC_PROTOCOL_LABELS
        group_aliases = SEMANTIC_GROUP_ALIASES

        for group_key, protocol in protocol_labels.items():
            value = None
            for alias in group_aliases[group_key]:
                if alias in by_group:
                    value = by_group[alias]
                    break
            if value is None:
                if args.allow_missing:
                    rows.append(
                        {
                            "protocol": protocol,
                            "split": split,
                            "variant": variant,
                            "test_abs_mpjpe": "",
                            "source": str(Path(path).expanduser()),
                        }
                    )
                    continue
                raise KeyError(f"{path} missing action group {group_key}")
            rows.append(
                {
                    "protocol": protocol,
                    "split": split,
                    "variant": variant,
                    "test_abs_mpjpe": f"{value:.2f}",
                    "source": str(Path(path).expanduser()),
                }
            )

    out_csv = Path(args.output_csv).expanduser()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protocol", "split", "variant", "test_abs_mpjpe", "source"])
        writer.writeheader()
        writer.writerows(rows)
    if args.output_json:
        out_json = Path(args.output_json).expanduser()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
