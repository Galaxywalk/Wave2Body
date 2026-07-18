#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the self-supervised mmWave tokenizer."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wave2body.train.mmwave_tokenizer_cli import run_mmwave_tokenizer_cli  # noqa: E402


def main() -> None:
    run_mmwave_tokenizer_cli(
        default_config="sota",
        default_mode="train",
        stage_name="mmwave_tokenizer",
        log_prefix="[MMWaveTokenizer Runtime]",
        tensorboard_namespace="mmwave_tokenizer",
        description=__doc__,
    )


if __name__ == "__main__":
    main()
