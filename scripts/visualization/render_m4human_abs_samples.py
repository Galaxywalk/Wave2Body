#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Render exported M4Human qualitative samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wave2body.visualization.rendering import (  # noqa: E402
    RENDER_VARIANTS,
    load_manifest,
    render_manifest_grid,
    render_sample_file,
    render_sample_variant_files,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Manifest produced by select_m4human_abs_samples.py.")
    parser.add_argument("--output-prefix", help="Output prefix. Defaults to <manifest-dir>/qualitative_grid.")
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--pad-inches", type=float, default=0.05, help="White border used when cropping exports.")
    parser.add_argument("--no-individual", action="store_true", help="Only render the grid figure.")
    parser.add_argument("--no-grid", action="store_true", help="Only render individual sample files.")
    parser.add_argument(
        "--individual-dir",
        help="Directory for individual files. Defaults to <manifest-dir>/individual.",
    )
    parser.add_argument(
        "--individual-variants",
        default="pointcloud_gt_pred",
        help=(
            "Comma-separated individual variants. Use all for "
            f"{','.join(RENDER_VARIANTS)}."
        ),
    )
    parser.add_argument("--individual-format", choices=("png", "pdf"), default="png")
    return parser


def _parse_variants(value: str) -> tuple[str, ...]:
    raw = str(value).strip()
    if raw.lower() == "all":
        return RENDER_VARIANTS
    variants = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not variants:
        raise ValueError("Expected at least one render variant")
    invalid = [variant for variant in variants if variant not in RENDER_VARIANTS]
    if invalid:
        raise ValueError(f"Unknown render variants {invalid}; expected one of {RENDER_VARIANTS}")
    return variants


def main() -> None:
    args = _build_parser().parse_args()
    manifest_path = Path(args.manifest).expanduser()
    prefix = Path(args.output_prefix).expanduser() if args.output_prefix else manifest_path.parent / "qualitative_grid"
    out_paths = []
    if not args.no_grid:
        out_paths.extend(
            render_manifest_grid(
                manifest_path,
                prefix,
                columns=args.columns,
                dpi=args.dpi,
                pad_inches=args.pad_inches,
            )
        )

    if not args.no_individual:
        manifest = load_manifest(manifest_path)
        individual_dir = Path(args.individual_dir).expanduser() if args.individual_dir else manifest_path.parent / "individual"
        variants = _parse_variants(args.individual_variants)
        for idx, record in enumerate(manifest.get("samples", []), start=1):
            npz_path = Path(str(record["npz_path"]))
            if not npz_path.is_absolute():
                npz_path = manifest_path.parent / npz_path
            action_id = int(record["action_id"])
            frame_id = int(record.get("frame_id", 0))
            if args.individual_format == "pdf":
                out_paths.extend(
                    render_sample_variant_files(
                        npz_path,
                        individual_dir,
                        action_id=action_id,
                        frame_id=frame_id,
                        sample_index=idx,
                        variants=variants,
                        dpi=args.dpi,
                        pad_inches=args.pad_inches,
                    )
                )
            else:
                if len(variants) != 1:
                    raise ValueError("--individual-format png supports exactly one --individual-variants value")
                output = individual_dir / f"{variants[0]}_action_{action_id:02d}_sample_{idx:02d}_frame_{frame_id:06d}.png"
                render_sample_file(
                    npz_path,
                    output,
                    dpi=args.dpi,
                    pad_inches=args.pad_inches,
                    variant=variants[0],
                )
                out_paths.append(output)

    for path in out_paths:
        print(f"[Qualitative Render] Wrote {path}")


if __name__ == "__main__":
    main()
