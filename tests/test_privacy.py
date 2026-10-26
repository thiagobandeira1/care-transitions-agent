from pathlib import Path

import pytest

from ctagent.privacy import (
    PrivacyError,
    assert_local_model,
    assert_no_cloud_tracing,
    assert_outside_repo,
)


@pytest.mark.parametrize("model", ["qwen3:14b", "qwen3:8b", "my-extractor:latest"])
def test_local_models_allowed(model: str) -> None:
    assert_local_model(model, "http://localhost:11434")
    assert_local_model(model, "http://127.0.0.1:11434")


@pytest.mark.parametrize(
    ("model", "url"),
    [
        ("gpt-oss:120b-cloud", "http://localhost:11434"),
        ("qwen3-coder:cloud", "http://localhost:11434"),
        ("qwen3:14b", "https://ollama.com"),
        ("qwen3:14b", "http://10.0.0.5:11434"),
    ],
)
def test_cloud_models_and_remote_endpoints_refused(model: str, url: str) -> None:
    with pytest.raises(PrivacyError):
        assert_local_model(model, url)


@pytest.mark.parametrize("value", ["true", "1", "TRUE", "yes"])
def test_tracing_on_is_refused(value: str) -> None:
    with pytest.raises(PrivacyError, match="LANGSMITH_TRACING"):
        assert_no_cloud_tracing({"LANGSMITH_TRACING": value})


@pytest.mark.parametrize("env", [{}, {"LANGCHAIN_TRACING_V2": "false"}, {"LANGSMITH_TRACING": ""}])
def test_tracing_off_is_allowed(env: dict[str, str]) -> None:
    assert_no_cloud_tracing(env)


def test_data_dir_inside_repo_is_refused(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    with pytest.raises(PrivacyError):
        assert_outside_repo(repo / "data", repo_root=repo)
    with pytest.raises(PrivacyError):
        assert_outside_repo(repo, repo_root=repo)


def test_data_dir_outside_repo_is_allowed(tmp_path: Path) -> None:
    repo, outside = tmp_path / "repo", tmp_path / "mimic"
    repo.mkdir()
    outside.mkdir()
    assert assert_outside_repo(outside, repo_root=repo) == outside.resolve()
