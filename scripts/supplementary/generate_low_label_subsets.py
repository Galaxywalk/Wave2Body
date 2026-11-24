#!/usr/bin/env python3
"""Generate deterministic stratified low-label training subsets for M4Human.

Produces JSON index files compatible with ``m4human.train_subset_indices_path``.
Stratifies across ``(subject_id, action_id)`` pairs so every subject and action class
remains represented at every supervision budget (e.g., 10%, 25%, 50%).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wave2body.train.common import load_config
from wave2body.train.radar_pose_data import create_radar_pose_dataset


def build_stratified_subset(
    data_list: np.ndarray,
    *,
    fraction: float,
    seed: int = 42,
) -> tuple[list[int], dict[str, int]]:
    """Select a deterministic ``(subject, action)``-stratified subset of row indices."""
    n = int(len(data_list))
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx in range(n):
        groups[(int(data_list[idx, 0]), int(data_list[idx, 1]))].append(idx)

    sorted_keys = sorted(groups.keys())
    target_total = int(round(n * fraction))
    raw_alloc: dict[tuple[int, int], int] = {}
    remainders: list[tuple[float, tuple[int, int]]] = []

    for key in sorted_keys:
        m = len(groups[key])
        exact = m * fraction
        base = int(np.floor(exact))
        if base == 0 and m > 0:
            base = 1
        raw_alloc[key] = base
        remainders.append((exact - base, key))

    diff = target_total - sum(raw_alloc.values())
    if diff > 0:
        remainders.sort(key=lambda item: (-item[0], item[1]))
        for _, key in remainders:
            if diff == 0:
                break
            if raw_alloc[key] < len(groups[key]):
                raw_alloc[key] += 1
                diff -= 1
    elif diff < 0:
        remainders.sort(key=lambda item: (item[0], item[1]))
        for _, key in remainders:
            if diff == 0:
                break
            if raw_alloc[key] > 1:
                raw_alloc[key] -= 1
                diff += 1

    rng = np.random.default_rng(int(seed))
    chosen: list[int] = []
    subjects: set[int] = set()
    actions: set[int] = set()
    for key in sorted_keys:
        idxs = np.asarray(groups[key], dtype=np.int64)
        take = raw_alloc[key]
        picked = rng.choice(idxs, size=take, replace=False)
        picked.sort()
        chosen.extend(int(i) for i in picked)
        subjects.add(key[0])
        actions.add(key[1])

    chosen.sort()
    return chosen, {
        "num_strata": len(groups),
        "subjects_covered": len(subjects),
        "actions_covered": len(actions),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default=str(REPO_ROOT / "wave2body/configs/hpe_token/m4human_s2_k96_xattn_trainonly_tokenizers.yaml"),
        help="Training YAML config identifying the M4Human split",
    )
    ap.add_argument("--output-dir", required=True, help="Directory to write subset index JSON files")
    ap.add_argument("--fractions", default="10,25,50", help="Comma-separated percentages (e.g. '10,25,50')")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(Path(args.config).expanduser())
    split_id = str(cfg.get("m4human", {}).get("split", "s2"))
    ds = create_radar_pose_dataset(cfg, split="train", device="cpu")
    try:
        arr = np.asarray(ds.data_list, dtype=np.int64)
    finally:
        if hasattr(ds, "close"):
            ds.close()

    n = int(len(arr))
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "split": split_id,
        "subset": "train",
        "seed": int(args.seed),
        "total_samples": n,
        "fractions": {},
    }

    for tok in [t.strip() for t in args.fractions.split(",") if t.strip()]:
        pct = float(tok)
        frac = pct / 100.0 if pct > 1.0 else pct
        pct_label = f"{int(round(frac * 100))}pct"
        chosen, stats = build_stratified_subset(arr, fraction=frac, seed=args.seed)
        out_path = out_dir / f"m4human_{split_id}_train_{pct_label}_seed{int(args.seed)}.json"
        out_path.write_text(json.dumps(chosen) + "\n", encoding="utf-8")
        summary["fractions"][pct_label] = {
            "count": len(chosen),
            "ratio": len(chosen) / max(n, 1),
            "path": str(out_path),
            **stats,
        }
        print(
            f"[subsets] {pct_label}: {len(chosen)}/{n} ({len(chosen) / max(n, 1) * 100:.2f}%), "
            f"subjects={stats['subjects_covered']}, actions={stats['actions_covered']} -> {out_path}"
        )

    summary_path = out_dir / f"summary_{split_id}_seed{int(args.seed)}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
