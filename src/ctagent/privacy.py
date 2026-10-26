"""Startup guards: no cloud tracing, and MIMIC paths must stay outside the repository."""

import os
from collections.abc import Mapping
from pathlib import Path

# Any of these set to a truthy value would ship prompts and outputs to a hosted service.
TRACING_VARS = ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGCHAIN_TRACING")
_FALSY = {"", "0", "false", "no", "off"}

REPO_ROOT = Path(__file__).resolve().parents[2]


class PrivacyError(RuntimeError):
    """A configuration would send data off this machine or into git."""


def assert_no_cloud_tracing(env: Mapping[str, str] | None = None) -> None:
    env = os.environ if env is None else env
    on = [v for v in TRACING_VARS if env.get(v, "").strip().lower() not in _FALSY]
    if on:
        raise PrivacyError(f"Cloud tracing must stay off; unset {', '.join(on)}")


def assert_outside_repo(path: Path, repo_root: Path = REPO_ROOT) -> Path:
    """MIMIC data directories must live outside the repository so they can never be committed."""
    resolved = path.resolve()
    if resolved == repo_root or repo_root in resolved.parents:
        raise PrivacyError(f"Data directory {resolved} is inside the repository")
    return resolved
