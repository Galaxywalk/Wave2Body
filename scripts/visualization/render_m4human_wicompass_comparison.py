#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Render Wave2Body vs WiCompass/PT paired qualitative comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wave2body.visualization.rendering import (  # noqa: E402
    load_manifest,
    render_paired_case_variant_files,
    render_paired_comparison_file,
    render_paired_comparison_grid,
    render_paired_comparison_panel_files,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Manifest produced by select_m4human_wicompass_comparison_samples.py.")
    parser.add_argument("--output-prefix", help="Output prefix. Defaults to <manifest-dir>/comparison_grid.")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--pad-inches", type=float, default=0.05)
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--no-individual", action="store_true")
    parser.add_argument("--individual-dir", help="Directory for individual comparison files.")
    parser.add_argument("--individual-format", choices=("png", "pdf"), default="png")
    parser.add_argument("--panel-dir", help="Directory for separate per-panel files.")
    parser.add_argument("--panel-format", choices=("pdf", "png"), default="pdf")
    parser.add_argument("--panel-pad-inches", type=float, default=0.0)
    parser.add_argument("--split-panels", action="store_true", help="Render each case as three separate panel files.")
    parser.add_argument("--panel-titles", action="store_true", help="Include titles on separate panel files.")
    parser.add_argument(
        "--case-variants",
        action="store_true",
        help="Render each case as point-cloud-only and all-pose overlay files.",
    )
    parser.add_argument("--case-variant-dir", help="Directory for point-cloud-only and overlay case files.")
    parser.add_argument("--case-variant-format", choices=("pdf", "png"), default="pdf")
    parser.add_argument("--case-variant-pad-inches", type=float, default=0.0)
    parser.add_argument("--case-variant-titles", action="store_true", help="Include titles on case variant files.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    manifest_path = Path(args.manifest).expanduser()
    prefix = Path(args.output_prefix).expanduser() if args.output_prefix else manifest_path.parent / "comparison_grid"
    out_paths: list[Path] = []
    if not args.no_grid:
        out_paths.extend(
            render_paired_comparison_grid(
                manifest_path,
                prefix,
                dpi=args.dpi,
                pad_inches=args.pad_inches,
            )
        )

    needs_manifest = (not args.no_individual) or args.split_panels or args.case_variants
    manifest = load_manifest(manifest_path) if needs_manifest else None

    if not args.no_individual:
        individual_dir = Path(args.individual_dir).expanduser() if args.individual_dir else manifest_path.parent / "individual"
        suffix = f".{args.individual_format}"
        for idx, record in enumerate(manifest.get("samples", []), start=1):
            npz_path = Path(str(record["npz_path"]))
            if not npz_path.is_absolute():
                npz_path = manifest_path.parent / npz_path
            split = str(record.get("split_id", "split"))
            action_id = int(record.get("action_id", 0))
            frame_id = int(record.get("frame_id", 0))
            output = individual_dir / f"comparison_{idx:02d}_{split}_action_{action_id:02d}_frame_{frame_id:06d}{suffix}"
            out_paths.append(
                render_paired_comparison_file(
                    npz_path,
                    output,
                    dpi=args.dpi,
                    pad_inches=args.pad_inches,
                )
            )

    if args.split_panels:
        panel_dir = Path(args.panel_dir).expanduser() if args.panel_dir else manifest_path.parent / "panels"
        for idx, record in enumerate(manifest.get("samples", []), start=1):
            npz_path = Path(str(record["npz_path"]))
            if not npz_path.is_absolute():
                npz_path = manifest_path.parent / npz_path
            out_paths.extend(
                render_paired_comparison_panel_files(
                    npz_path,
                    panel_dir,
                    record=record,
                    sample_index=idx,
                    dpi=args.dpi,
                    pad_inches=args.panel_pad_inches,
                    file_format=args.panel_format,
                    include_titles=args.panel_titles,
                )
            )

    if args.case_variants:
        variant_dir = (
            Path(args.case_variant_dir).expanduser() if args.case_variant_dir else manifest_path.parent / "case_variants"
        )
        for idx, record in enumerate(manifest.get("samples", []), start=1):
            npz_path = Path(str(record["npz_path"]))
            if not npz_path.is_absolute():
                npz_path = manifest_path.parent / npz_path
            out_paths.extend(
                render_paired_case_variant_files(
                    npz_path,
                    variant_dir,
                    record=record,
                    sample_index=idx,
                    dpi=args.dpi,
                    pad_inches=args.case_variant_pad_inches,
                    file_format=args.case_variant_format,
                    include_titles=args.case_variant_titles,
                )
            )

    for path in out_paths:
        print(f"[Paired Comparison Render] Wrote {path}")


if __name__ == "__main__":
    main()
