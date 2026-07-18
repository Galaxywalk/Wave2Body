#!/usr/bin/env python3
"""Validate every canonical radar/pose sample in a compact mmBody dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import msgpack
import numpy as np

from wave2body.dataset.radar_pose.mmbody_compact import MMBodyCompactStore


def _source_paths(source: Path, record: dict):
    split = str(record["split"])
    sequence_name = str(record["sequence_name"])
    frame = int(record["frame"])
    if split == "train":
        sequence = source / split / sequence_name
    else:
        sequence = source / split / str(record["scenario"]) / sequence_name
    return (
        sequence / "radar" / f"frame_{frame}.npy",
        sequence / "mesh" / f"frame_{frame}.npz",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--compact", required=True)
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    store = MMBodyCompactStore(args.compact)
    canonical_hash = hashlib.sha256()
    seen_samples = set()
    try:
        for index, record in enumerate(store.samples, 1):
            identity = (
                str(record["split"]),
                str(record["scenario"]),
                str(record["sequence_name"]),
                int(record["frame"]),
            )
            if identity in seen_samples:
                raise RuntimeError(f"Duplicate compact sample identity: {identity}")
            seen_samples.add(identity)
            radar_path, mesh_path = _source_paths(source, record)
            raw_radar = np.load(radar_path).astype(np.float32)
            expected_radar = np.ascontiguousarray(raw_radar[:, [0, 1, 2, 3, 5]], dtype=np.float32)
            with np.load(mesh_path) as mesh:
                expected_pose = np.ascontiguousarray(mesh["joints"][:22], dtype=np.float32)
            actual_radar = store.read_radar(bytes(record["radar_key"]))
            actual_pose = store.read_pose(bytes(record["pose_key"]))
            if not np.array_equal(actual_radar, expected_radar, equal_nan=True):
                raise RuntimeError(f"Radar mismatch: {identity}")
            if not np.array_equal(actual_pose, expected_pose, equal_nan=True):
                raise RuntimeError(f"Pose mismatch: {identity}")
            meta = {
                "split": str(record["split"]),
                "scenario": str(record["scenario"]),
                "sequence": int(record["sequence"]),
                "sequence_name": str(record["sequence_name"]),
                "frame": int(record["frame"]),
            }
            canonical_hash.update(msgpack.packb(meta, use_bin_type=True))
            canonical_hash.update(expected_radar.tobytes(order="C"))
            canonical_hash.update(expected_pose.tobytes(order="C"))
            if args.progress_every > 0 and index % args.progress_every == 0:
                print(f"validated {index}/{len(store.samples)}", flush=True)
        expected_hash = str(store.manifest["canonical_samples_sha256"])
        actual_hash = canonical_hash.hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"Canonical hash mismatch: {actual_hash} != {expected_hash}")
        report = {
            "status": "ok",
            "sample_count": len(store.samples),
            "canonical_samples_sha256": actual_hash,
            "unique_sample_identities": len(seen_samples),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        store.close()


if __name__ == "__main__":
    main()
