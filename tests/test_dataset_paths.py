# -*- coding: utf-8 -*-

import yaml

from wave2body.utils.paths import resolve_dataset_paths, resolve_dataset_uri
from wave2body.train.common import load_config


def test_dataset_uri_resolution_uses_env_config(tmp_path, monkeypatch):
    path_config = tmp_path / "dataset_paths.yaml"
    path_config.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "mmbody": str(tmp_path / "mmbody_root"),
                    "amass_preproc": str(tmp_path / "amass_root"),
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WAVE2BODY_DATASET_PATHS", str(path_config))

    assert resolve_dataset_uri("dataset://mmbody") == str(tmp_path / "mmbody_root")
    assert resolve_dataset_uri("dataset://amass_preproc/KIT") == str(tmp_path / "amass_root" / "KIT")


def test_train_config_resolves_dataset_paths(tmp_path, monkeypatch):
    path_config = tmp_path / "dataset_paths.yaml"
    path_config.write_text(
        "datasets:\n  mmbody: /data/mmbody\n  m4human: /data/m4human\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WAVE2BODY_DATASET_PATHS", str(path_config))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "dataset_path: dataset://mmbody\nm4human:\n  smplx_model_root: dataset://m4human/models\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg["dataset_path"] == "/data/mmbody"
    assert cfg["m4human"]["smplx_model_root"] == "/data/m4human/models"


def test_vqvae_dataset_configs_resolve_dataset_paths(tmp_path, monkeypatch):
    path_config = tmp_path / "dataset_paths.yaml"
    path_config.write_text("datasets:\n  mmbody: /data/mmbody\n", encoding="utf-8")
    monkeypatch.setenv("WAVE2BODY_DATASET_PATHS", str(path_config))

    cfg = resolve_dataset_paths(
        {"data": {"dataset_configs": [{"name": "MMBody", "type": "mmbody", "path": "dataset://mmbody"}]}}
    )

    assert cfg["data"]["dataset_configs"][0]["path"] == "/data/mmbody"
