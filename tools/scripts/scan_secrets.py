#!/usr/bin/env python3
"""Scribe secret scanning CLI."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.support import SecretScanner


def main() -> None:
    parser = argparse.ArgumentParser(description="Secret scanner for Scribe repositories")
    parser.add_argument("--directory", "-d", type=Path, default=Path.cwd(), help="Directory to scan")
    parser.add_argument("--out", "-o", type=Path, help="Optional output JSON path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Include per-finding console output")
    args = parser.parse_args()

    scanner = SecretScanner()
    print(f"Scanning directory: {args.directory}")

    findings = scanner.scan_directory(args.directory)
    report = scanner.summarize(findings)
    report["timestamp"] = Path(__file__).stat().st_mtime

    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report saved to: {args.out}")

    print("Secret scan completed:")
    print(f"  Total findings: {report['total_findings']}")
    print(f"  Files affected: {report['files_affected']}")
    if report["findings_by_severity"]:
        print("  By severity:")
        for severity, count in report["findings_by_severity"].items():
            print(f"    {severity}: {count}")

    if args.verbose and report["findings"]:
        print("\nDetailed findings:")
        for finding in report["findings"]:
            print(f"  {finding['severity'].upper()}: {finding['type']} in {finding['file']}:{finding['line']}")
            print(f"    Context: {finding['context']}")

    if report["findings_by_severity"].get("high", 0) > 0:
        print("ERROR: High-severity potential secrets found")
        sys.exit(1)

    print("✓ Secret scan passed")


if __name__ == "__main__":
    main()
