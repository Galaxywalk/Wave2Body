# -*- coding: utf-8 -*-
"""Config resolution helpers for HPE token scripts."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "wave2body" / "configs" / "hpe_token"
BASELINE_CONFIG_DIR = CONFIG_DIR / "baselines"

CANONICAL_CONFIG_ALIASES = {
    "mmbody_abs": CONFIG_DIR
    / "mmbody_k96_queryroot_pointiwmean_absolute_nosoftce_trainable_translator_trainonly_bodytoken.yaml",
    "mmbody_stage1": CONFIG_DIR
    / "mmbody_k96_xattn_fulltrain_softdecode_t1p25_noinput_rootnorm_nosoftce_trainonly_bodytoken.yaml",
    "mmbody_abs_trainonly_bodytoken": CONFIG_DIR
    / "mmbody_k96_queryroot_pointiwmean_absolute_nosoftce_trainable_translator_trainonly_bodytoken.yaml",
    "mmbody_stage1_trainonly_bodytoken": CONFIG_DIR
    / "mmbody_k96_xattn_fulltrain_softdecode_t1p25_noinput_rootnorm_nosoftce_trainonly_bodytoken.yaml",
    "m4human_abs_s2": CONFIG_DIR / "m4human_s2_k96_queryroot_pointiwmean_absolute_trainable_translator.yaml",
    "m4human_abs_s3": CONFIG_DIR / "m4human_s3_k96_queryroot_pointiwmean_absolute_trainable_translator.yaml",
    "m4human_stage1_s2": CONFIG_DIR / "m4human_s2_k96_xattn.yaml",
    "m4human_stage1_s3": CONFIG_DIR / "m4human_s3_k96_xattn.yaml",
}

# Backward-looking experiments must opt in by path/stem. Keep the public name
# for tests and importers, but no baseline shortcuts are included here.
CONFIG_ALIASES = CANONICAL_CONFIG_ALIASES


def _candidate_paths(value: str) -> list[Path]:
    path = Path(value).expanduser()
    candidates = [path]
    if path.suffix not in {".yaml", ".yml"}:
        candidates.append(path.with_suffix(".yaml"))
    if not path.is_absolute():
        candidates.extend(
            [
                REPO_ROOT / path,
                REPO_ROOT / path.with_suffix(".yaml") if path.suffix not in {".yaml", ".yml"} else REPO_ROOT / path,
                CONFIG_DIR / path,
                CONFIG_DIR / path.with_suffix(".yaml") if path.suffix not in {".yaml", ".yml"} else CONFIG_DIR / path,
                CONFIG_DIR / path.name,
            ]
        )
        if path.parts and path.parts[0] == "baselines":
            rel = Path(*path.parts[1:])
            candidates.extend(
                [
                    BASELINE_CONFIG_DIR / rel,
                    BASELINE_CONFIG_DIR / rel.with_suffix(".yaml")
                    if rel.suffix not in {".yaml", ".yml"}
                    else BASELINE_CONFIG_DIR / rel,
                    BASELINE_CONFIG_DIR / rel.name,
                ]
            )
    return candidates


def resolve_config(value: str) -> Path:
    value = str(value).strip()
    if value in CONFIG_ALIASES:
        return CONFIG_ALIASES[value]
    for candidate in _candidate_paths(value):
        if candidate.is_file():
            return candidate
    known = ", ".join(
        [
            *sorted(CONFIG_ALIASES),
            *sorted(path.stem for path in CONFIG_DIR.glob("*.yaml")),
            *[f"baselines/{path.stem}" for path in sorted(BASELINE_CONFIG_DIR.glob("*.yaml"))],
        ]
    )
    raise ValueError(f"Unknown HPE token config {value!r}; expected an alias, YAML stem, or YAML path. Known: {known}")


def print_configs() -> None:
    print("Canonical aliases:")
    for name, path in sorted(CONFIG_ALIASES.items()):
        print(f"{name}: {path}")
    print("\nCanonical YAML stems:")
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        print(f"{path.stem}: {path}")
    print("\nBaseline-only YAML stems (explicit opt-in):")
    for path in sorted(BASELINE_CONFIG_DIR.glob("*.yaml")):
        print(f"baselines/{path.stem}: {path}")
