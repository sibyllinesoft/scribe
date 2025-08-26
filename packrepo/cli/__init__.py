"""
FastPath CLI integration for PackRepo.

Provides command-line interface for FastPath optimization:
- fastpack.py: Main CLI entry point with mode selection
"""

from __future__ import annotations

from .fastpack import FastPackCLI, create_cli

__all__ = [
    "FastPackCLI",
    "create_cli",
]
