"""Claude Code config helpers for benchmarks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_BENCH_CONFIG_DIR = Path(__file__).parent.parent / ".claude-config"


def resolve_claude_config_dir(cli_value: Optional[str] = None) -> Optional[Path]:
    """Resolve the Claude Code config directory for benchmark runs.

    Priority:
    1) CLI-provided path
    2) CLAUDE_BENCH_CONFIG_DIR env
    3) CLAUDE_CONFIG_DIR env
    4) repo-local benchmarks/.claude-config
    """
    if cli_value:
        return Path(cli_value).expanduser()

    env_value = os.environ.get("CLAUDE_BENCH_CONFIG_DIR") or os.environ.get("CLAUDE_CONFIG_DIR")
    if env_value:
        return Path(env_value).expanduser()

    return DEFAULT_BENCH_CONFIG_DIR


def build_claude_env(config_dir: Optional[Path]) -> dict[str, str]:
    """Build env for Claude Code subprocess calls.

    Ensures CLAUDE_CONFIG_DIR points at the benchmark config dir so we do not
    touch the user's global Claude Code config.
    """
    env = os.environ.copy()

    if config_dir:
        config_dir.mkdir(parents=True, exist_ok=True)
        env["CLAUDE_CONFIG_DIR"] = str(config_dir)

    # Isolate Claude CLI data dir per benchmark process to avoid cross-run locks.
    # Can be overridden via CLAUDE_BENCH_DATA_DIR if you want a fixed location.
    data_dir = os.environ.get("CLAUDE_BENCH_DATA_DIR")
    if not data_dir and config_dir:
        data_dir = str(config_dir / f"xdg-data-{os.getpid()}")
    if data_dir:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        env["XDG_DATA_HOME"] = data_dir

    return env
