"""Lightweight secret scanning helpers used by Scribe tooling."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


class SecretScanner:
    """Detect common credential patterns in text files."""

    DEFAULT_PATTERNS: Dict[str, Iterable[str]] = {
        "api_key": (
            r"api[_-]?key[\"\s]*[:=][\"\s]*([a-zA-Z0-9_\-]{20,})",
            r"apikey[\"\s]*[:=][\"\s]*([a-zA-Z0-9_\-]{20,})",
        ),
        "password": (
            r"password[\"\s]*[:=][\"\s]*[\"']([^\"']{8,})[\"']",
            r"passwd[\"\s]*[:=][\"\s]*[\"']([^\"']{8,})[\"']",
        ),
        "token": (
            r"token[\"\s]*[:=][\"\s]*[\"']([a-zA-Z0-9_\-]{20,})[\"']",
            r"auth[_-]?token[\"\s]*[:=][\"\s]*[\"']([a-zA-Z0-9_\-]{20,})[\"']",
        ),
        "secret": (
            r"secret[_-]?key[\"\s]*[:=][\"\s]*[\"']([a-zA-Z0-9_\-]{16,})[\"']",
            r"client[_-]?secret[\"\s]*[:=][\"\s]*[\"']([a-zA-Z0-9_\-]{16,})[\"']",
        ),
    }

    DEFAULT_FILE_PATTERNS: Iterable[str] = (
        r"\\.env$",
        r"\\.env\\.",
        r"secret",
        r"credential",
        r"\\.key$",
        r"\\.pem$",
    )

    def __init__(
        self,
        *,
        patterns: Dict[str, Iterable[str]] | None = None,
        file_patterns: Iterable[str] | None = None,
    ) -> None:
        self.patterns = patterns or dict(self.DEFAULT_PATTERNS)
        self.file_patterns = tuple(file_patterns or self.DEFAULT_FILE_PATTERNS)

    def _check_pattern_on_line(
        self, pattern: str, line: str, line_num: int, file_path: Path, secret_type: str
    ) -> List[Dict[str, Any]]:
        """Check a single pattern against a line and return matches."""
        return [
            {
                "type": secret_type,
                "file": str(file_path),
                "line": line_num,
                "pattern": pattern,
                "context": line.strip(),
                "severity": "high",
                "match": match.group(0),
            }
            for match in re.finditer(pattern, line, re.IGNORECASE)
        ]

    def _scan_line_for_secret_type(
        self, secret_type: str, patterns: Iterable[str], line: str, line_num: int, file_path: Path
    ) -> List[Dict[str, Any]]:
        """Scan a line for a specific secret type."""
        findings: List[Dict[str, Any]] = []
        for pattern in patterns:
            findings.extend(self._check_pattern_on_line(pattern, line, line_num, file_path, secret_type))
        return findings

    def _scan_line_for_patterns(
        self, line: str, line_num: int, file_path: Path
    ) -> List[Dict[str, Any]]:
        """Scan a single line for all secret patterns."""
        findings: List[Dict[str, Any]] = []
        for secret_type, patterns in self.patterns.items():
            findings.extend(self._scan_line_for_secret_type(secret_type, patterns, line, line_num, file_path))
        return findings

    def scan_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan a single file and return matches."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except (UnicodeDecodeError, PermissionError):
            return []

        findings: List[Dict[str, Any]] = []
        for line_num, line in enumerate(content.splitlines(), start=1):
            findings.extend(self._scan_line_for_patterns(line, line_num, file_path))
        return findings

    def _check_suspicious_filename(self, file_path: Path) -> List[Dict[str, Any]]:
        """Check if a filename matches suspicious patterns."""
        findings: List[Dict[str, Any]] = []
        lower_name = file_path.name.lower()
        for pattern in self.file_patterns:
            if re.search(pattern, lower_name):
                findings.append(
                    {
                        "type": "suspicious_filename",
                        "file": str(file_path),
                        "line": 0,
                        "pattern": pattern,
                        "context": f"Filename matches pattern: {pattern}",
                        "severity": "medium",
                    }
                )
        return findings

    def _scan_single_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan a single file including filename pattern checks."""
        findings = self._check_suspicious_filename(file_path)
        findings.extend(self.scan_file(file_path))
        return findings

    def scan_directory(self, directory: Path) -> List[Dict[str, Any]]:
        """Recursively scan a directory for potential secrets."""
        findings: List[Dict[str, Any]] = []
        ignore_names = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            ".venv",
            "venv",
            ".serena",
            "artifacts",
        }

        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in ignore_names]
            for filename in files:
                file_path = Path(root) / filename
                findings.extend(self._scan_single_file(file_path))
        return findings

    def summarize(self, findings: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Produce a compact summary for reporting."""
        findings_list = list(findings)
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for item in findings_list:
            by_type[item["type"]] = by_type.get(item["type"], 0) + 1
            by_severity[item["severity"]] = by_severity.get(item["severity"], 0) + 1

        return {
            "total_findings": len(findings_list),
            "findings_by_type": by_type,
            "findings_by_severity": by_severity,
            "files_affected": len({item["file"] for item in findings_list}),
            "findings": findings_list,
        }


__all__ = ["SecretScanner"]
