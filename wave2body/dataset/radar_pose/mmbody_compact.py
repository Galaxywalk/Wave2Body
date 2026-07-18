# -*- coding: utf-8 -*-
"""Content-addressed, paper-only storage for mmBody radar and 22-joint poses."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any
import zlib

import lmdb
import msgpack
import numpy as np


FORMAT_NAME = "wave2body.mmbody.compact"
FORMAT_VERSION = 1
MANIFEST_NAME = "mmbody_compact.json"
INDEX_NAME = "samples.msgpack"
RADAR_DB_NAME = "radar.lmdb"
POSE_DB_NAME = "poses.lmdb"
RADAR_CHANNELS = 5
POSE_JOINTS = 22
_RADAR_HEADER = struct.Struct("<I")


def is_mmbody_compact(root: str | Path) -> bool:
    path = Path(root) / MANIFEST_NAME
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return payload.get("format") == FORMAT_NAME and int(payload.get("version", -1)) == FORMAT_VERSION


def encode_radar_points(points: np.ndarray, *, compression_level: int = 1) -> bytes:
    arr = np.ascontiguousarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != RADAR_CHANNELS:
        raise ValueError(f"Expected radar points [N,{RADAR_CHANNELS}], got {arr.shape}")
    return _RADAR_HEADER.pack(int(arr.shape[0])) + zlib.compress(arr.tobytes(order="C"), compression_level)


def decode_radar_points(blob: bytes | memoryview) -> np.ndarray:
    raw_blob = bytes(blob)
    if len(raw_blob) < _RADAR_HEADER.size:
        raise ValueError("Invalid compact mmBody radar blob")
    (n_points,) = _RADAR_HEADER.unpack_from(raw_blob)
    raw = zlib.decompress(raw_blob[_RADAR_HEADER.size :])
    expected = int(n_points) * RADAR_CHANNELS * np.dtype(np.float32).itemsize
    if len(raw) != expected:
        raise ValueError(f"Invalid compact radar payload: expected {expected} bytes, got {len(raw)}")
    return np.frombuffer(raw, dtype=np.float32).reshape(int(n_points), RADAR_CHANNELS).copy()


def decode_pose(blob: bytes | memoryview) -> np.ndarray:
    raw = bytes(blob)
    expected = POSE_JOINTS * 3 * np.dtype(np.float32).itemsize
    if len(raw) != expected:
        raise ValueError(f"Invalid compact pose payload: expected {expected} bytes, got {len(raw)}")
    return np.frombuffer(raw, dtype=np.float32).reshape(POSE_JOINTS, 3).copy()


class MMBodyCompactStore:
    """Lazy, multiprocessing-safe reader for a compact mmBody root."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not is_mmbody_compact(self.root):
            raise ValueError(f"Not a supported compact mmBody root: {self.root}")
        self.manifest = json.loads((self.root / MANIFEST_NAME).read_text(encoding="utf-8"))
        payload = msgpack.unpackb((self.root / INDEX_NAME).read_bytes(), raw=False)
        if payload.get("format") != FORMAT_NAME or int(payload.get("version", -1)) != FORMAT_VERSION:
            raise ValueError(f"Unsupported compact mmBody index: {self.root / INDEX_NAME}")
        self.samples = list(payload["samples"])
        self._owner_pid: int | None = None
        self._radar_env: lmdb.Environment | None = None
        self._pose_env: lmdb.Environment | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_owner_pid"] = None
        state["_radar_env"] = None
        state["_pose_env"] = None
        return state

    def close(self) -> None:
        for env in (self._radar_env, self._pose_env):
            if env is not None:
                env.close()
        self._radar_env = None
        self._pose_env = None
        self._owner_pid = None

    def _ensure_envs(self) -> tuple[lmdb.Environment, lmdb.Environment]:
        pid = os.getpid()
        if self._owner_pid != pid:
            self.close()
            self._owner_pid = pid
        if self._radar_env is None:
            self._radar_env = lmdb.open(
                str(self.root / RADAR_DB_NAME),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=256,
            )
        if self._pose_env is None:
            self._pose_env = lmdb.open(
                str(self.root / POSE_DB_NAME),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=256,
            )
        return self._radar_env, self._pose_env

    def read_radar(self, key: bytes) -> np.ndarray:
        radar_env, _ = self._ensure_envs()
        with radar_env.begin(buffers=True) as txn:
            blob = txn.get(bytes(key))
            if blob is None:
                raise KeyError(f"Missing compact mmBody radar key {bytes(key).hex()}")
            return decode_radar_points(blob)

    def read_pose(self, key: bytes) -> np.ndarray:
        _, pose_env = self._ensure_envs()
        with pose_env.begin(buffers=True) as txn:
            blob = txn.get(bytes(key))
            if blob is None:
                raise KeyError(f"Missing compact mmBody pose key {bytes(key).hex()}")
            return decode_pose(blob)


def _sample_metadata(source_root: Path, radar_path: Path) -> dict[str, Any]:
    rel = radar_path.relative_to(source_root)
    parts = rel.parts
    if len(parts) == 4 and parts[0] == "train":
        split, sequence_name = parts[0], parts[1]
        scenario = sequence_name
    elif len(parts) == 5 and parts[0] == "test":
        split, scenario, sequence_name = parts[0], parts[1], parts[2]
    else:
        raise ValueError(f"Unexpected mmBody radar path: {rel}")
    frame = int(radar_path.stem.split("_")[-1])
    sequence = int(sequence_name.split("_")[-1])
    return {
        "split": split,
        "scenario": scenario,
        "sequence": sequence,
        "sequence_name": sequence_name,
        "frame": frame,
    }


def _record_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if record["split"] == "train" else 1,
        str(record["scenario"]),
        int(record["sequence"]),
        int(record["frame"]),
    )


def compact_mmbody_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    compression_level: int = 1,
    transaction_batch_size: int = 512,
) -> dict[str, Any]:
    """Convert raw mmBody into content-addressed radar and pose LMDBs."""

    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Missing mmBody source root: {source}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Compact output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    radar_paths = sorted(source.glob("**/radar/frame_*.npy"), key=lambda p: _record_sort_key(_sample_metadata(source, p)))
    if not radar_paths:
        raise RuntimeError(f"No mmBody radar frames found under {source}")

    radar_env = lmdb.open(
        str(output / RADAR_DB_NAME),
        subdir=False,
        map_size=16 * 1024**3,
        lock=True,
        sync=True,
        metasync=True,
    )
    pose_env = lmdb.open(
        str(output / POSE_DB_NAME),
        subdir=False,
        map_size=1024**3,
        lock=True,
        sync=True,
        metasync=True,
    )
    radar_txn: lmdb.Transaction | None = radar_env.begin(write=True)
    pose_txn: lmdb.Transaction | None = pose_env.begin(write=True)
    records: list[dict[str, Any]] = []
    canonical_hash = hashlib.sha256()
    unique_radar: set[bytes] = set()
    unique_pose: set[bytes] = set()
    canonical_radar_bytes = 0
    canonical_pose_bytes = 0
    try:
        for index, radar_path in enumerate(radar_paths, 1):
            meta = _sample_metadata(source, radar_path)
            sequence_root = radar_path.parent.parent
            mesh_path = sequence_root / "mesh" / f"frame_{meta['frame']}.npz"
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Missing paired mmBody mesh: {mesh_path}")

            radar_raw = np.load(radar_path).astype(np.float32)
            if radar_raw.ndim != 2 or radar_raw.shape[1] < 6:
                raise ValueError(f"Expected raw mmBody radar [N,>=6], got {radar_raw.shape}: {radar_path}")
            radar = np.ascontiguousarray(radar_raw[:, [0, 1, 2, 3, 5]], dtype=np.float32)
            with np.load(mesh_path) as mesh:
                pose = np.ascontiguousarray(mesh["joints"][:POSE_JOINTS], dtype=np.float32)
            if pose.shape != (POSE_JOINTS, 3):
                raise ValueError(f"Expected mmBody pose [{POSE_JOINTS},3], got {pose.shape}: {mesh_path}")

            radar_bytes = radar.tobytes(order="C")
            pose_bytes = pose.tobytes(order="C")
            radar_key = hashlib.sha256(radar_bytes).digest()
            pose_key = hashlib.sha256(pose_bytes).digest()
            assert radar_txn is not None and pose_txn is not None
            if radar_key not in unique_radar:
                radar_txn.put(radar_key, encode_radar_points(radar, compression_level=compression_level), overwrite=False)
                unique_radar.add(radar_key)
                canonical_radar_bytes += len(radar_bytes)
            if pose_key not in unique_pose:
                pose_txn.put(pose_key, pose_bytes, overwrite=False)
                unique_pose.add(pose_key)
                canonical_pose_bytes += len(pose_bytes)

            record = {**meta, "radar_key": radar_key, "pose_key": pose_key}
            records.append(record)
            canonical_hash.update(msgpack.packb(meta, use_bin_type=True))
            canonical_hash.update(radar_bytes)
            canonical_hash.update(pose_bytes)

            if index % max(1, int(transaction_batch_size)) == 0:
                radar_txn.commit()
                pose_txn.commit()
                radar_txn = radar_env.begin(write=True)
                pose_txn = pose_env.begin(write=True)
        assert radar_txn is not None and pose_txn is not None
        radar_txn.commit()
        radar_txn = None
        pose_txn.commit()
        pose_txn = None
    except Exception:
        if radar_txn is not None:
            radar_txn.abort()
        if pose_txn is not None:
            pose_txn.abort()
        raise
    finally:
        radar_env.sync()
        pose_env.sync()
        radar_env.close()
        pose_env.close()
        (output / f"{RADAR_DB_NAME}-lock").unlink(missing_ok=True)
        (output / f"{POSE_DB_NAME}-lock").unlink(missing_ok=True)

    index_payload = {"format": FORMAT_NAME, "version": FORMAT_VERSION, "samples": records}
    (output / INDEX_NAME).write_bytes(msgpack.packb(index_payload, use_bin_type=True))
    counts_by_split = {
        split: sum(1 for record in records if record["split"] == split)
        for split in ("train", "test")
    }
    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "source_root": str(source),
        "sample_count": len(records),
        "counts_by_split": counts_by_split,
        "radar_channels": RADAR_CHANNELS,
        "pose_joints": POSE_JOINTS,
        "radar_dtype": "float32",
        "pose_dtype": "float32",
        "radar_compression": f"zlib:{int(compression_level)}",
        "unique_radar_count": len(unique_radar),
        "unique_pose_count": len(unique_pose),
        "canonical_unique_radar_bytes": canonical_radar_bytes,
        "canonical_unique_pose_bytes": canonical_pose_bytes,
        "canonical_samples_sha256": canonical_hash.hexdigest(),
    }
    (output / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "MANIFEST_NAME",
    "MMBodyCompactStore",
    "compact_mmbody_dataset",
    "decode_pose",
    "decode_radar_points",
    "encode_radar_points",
    "is_mmbody_compact",
]
