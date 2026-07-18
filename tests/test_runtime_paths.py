from pathlib import Path

import wave2body.utils.paths as runtime_paths
from wave2body.utils.paths import artifact_root, python_executable, resolve_runtime_uri, workspace_root


def test_artifact_root_prefers_wave2body_env(monkeypatch):
    monkeypatch.setenv("WAVE2BODY_ARTIFACT_ROOT", "/wave2body/artifacts")

    assert artifact_root() == Path("/wave2body/artifacts")


def test_artifact_root_expands_env_vars(monkeypatch):
    monkeypatch.setenv("W2B_TEST_ROOT", "/expanded")
    monkeypatch.setenv("WAVE2BODY_ARTIFACT_ROOT", "$W2B_TEST_ROOT/artifacts")

    assert artifact_root() == Path("/expanded/artifacts")


def test_workspace_root_prefers_wave2body_env(monkeypatch):
    monkeypatch.setenv("WAVE2BODY_WORKSPACE_ROOT", "/wave2body/workspace")
    monkeypatch.delenv("WAVE2BODY_ARTIFACT_ROOT", raising=False)

    assert workspace_root() == Path("/wave2body/workspace")
    assert artifact_root() == Path("/wave2body/workspace/wave2body_artifacts")


def test_runtime_uri_resolution(monkeypatch):
    monkeypatch.setenv("WAVE2BODY_WORKSPACE_ROOT", "/wave2body/workspace")
    monkeypatch.delenv("WAVE2BODY_ARTIFACT_ROOT", raising=False)

    assert resolve_runtime_uri("workspace://vqvae/checkpoints/best.pth") == "/wave2body/workspace/vqvae/checkpoints/best.pth"
    assert (
        resolve_runtime_uri("artifact://experiments/hpe_token/checkpoints/best.pth")
        == "/wave2body/workspace/wave2body_artifacts/experiments/hpe_token/checkpoints/best.pth"
    )


def test_workspace_root_uses_portable_repo_default(monkeypatch):
    monkeypatch.delenv("WAVE2BODY_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("WAVE2BODY_ARTIFACT_ROOT", raising=False)

    assert workspace_root() == runtime_paths.repo_root() / "workspace"


def test_python_executable_prefers_wave2body_env(monkeypatch):
    monkeypatch.setenv("WAVE2BODY_PYTHON", "/wave2body/python")

    assert python_executable() == "/wave2body/python"
