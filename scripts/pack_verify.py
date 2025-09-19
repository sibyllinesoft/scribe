#!/usr/bin/env python3
"""CLI for validating Scribe bundle metadata."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.support import PackVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scribe pack verification and schema tooling")
    parser.add_argument("--schema", type=Path, help="Path to an existing schema to load")
    parser.add_argument("--write-schema", type=Path, help="Write the default schema to this path")
    parser.add_argument("--validate", type=Path, help="Validate the provided pack JSON file")
    return parser.parse_args()


def load_pack(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        print(f"Failed to parse JSON: {err}")
        sys.exit(2)


def main() -> None:
    args = parse_args()
    verifier = PackVerifier(schema_path=args.schema)

    if args.write_schema:
        verifier.write_schema(args.write_schema)
        print(f"✓ Default schema written to {args.write_schema}")

    if args.validate:
        pack = load_pack(args.validate)
        result = verifier.validate_pack(pack)
        if result["is_valid"]:
            print("✓ Pack passes validation")
        else:
            print("Validation failed:")
            for error in result["errors"]:
                print(f"  - {error}")
            sys.exit(1)


if __name__ == "__main__":
    main()
