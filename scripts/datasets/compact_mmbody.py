#!/usr/bin/env python3
"""Build Wave2Body's content-addressed paper-only mmBody dataset."""

from __future__ import annotations

import argparse
import json

from wave2body.dataset.radar_pose.mmbody_compact import compact_mmbody_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Full raw mmBody root")
    parser.add_argument("--output", required=True, help="Absent or empty compact output root")
    parser.add_argument("--compression-level", type=int, default=1, choices=range(0, 10))
    args = parser.parse_args()
    manifest = compact_mmbody_dataset(
        args.source,
        args.output,
        compression_level=args.compression_level,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
