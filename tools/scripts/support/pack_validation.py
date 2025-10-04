"""Pack bundle validation helpers for Scribe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from jsonschema import ValidationError, validate


class PackVerifier:
    """Validate bundle metadata and token accounting."""

    def __init__(self, schema_path: Optional[Path] = None) -> None:
        self.schema_path = Path(schema_path) if schema_path else None
        self.schema = (
            self._load_schema() if self.schema_path else self._generate_default_schema()
        )

    def _load_schema(self) -> Dict[str, Any]:
        if not self.schema_path or not self.schema_path.exists():
            return self._generate_default_schema()
        return json.loads(self.schema_path.read_text(encoding="utf-8"))

    def _generate_default_schema(self) -> Dict[str, Any]:
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Scribe Pack Format Schema",
            "description": "Validation schema for Scribe pack index and format",
            "type": "object",
            "required": [
                "metadata",
                "tokenizer_info",
                "budget_info",
                "chunks",
                "statistics",
            ],
            "properties": {
                "metadata": {
                    "type": "object",
                    "required": ["version", "created_at", "repo_info"],
                    "properties": {
                        "version": {"type": "string", "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$"},
                        "created_at": {"type": "string", "format": "date-time"},
                        "repo_info": {
                            "type": "object",
                            "required": ["path", "commit"],
                            "properties": {
                                "path": {"type": "string"},
                                "commit": {
                                    "type": "string",
                                    "minLength": 40,
                                    "maxLength": 40,
                                },
                                "branch": {"type": "string"},
                                "remote": {"type": "string"},
                            },
                        },
                    },
                },
                "tokenizer_info": {
                    "type": "object",
                    "required": ["name", "version"],
                    "properties": {
                        "name": {"type": "string", "enum": ["cl100k", "o200k", "gpt2"]},
                        "version": {"type": "string"},
                        "vocab_size": {"type": "integer", "minimum": 1000},
                    },
                },
                "budget_info": {
                    "type": "object",
                    "required": ["target_budget", "actual_tokens", "utilization"],
                    "properties": {
                        "target_budget": {"type": "integer", "minimum": 1000},
                        "actual_tokens": {"type": "integer", "minimum": 0},
                        "utilization": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1.005,
                        },
                        "underflow_allowed": {"type": "number", "maximum": 0.005},
                    },
                },
                "chunks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "id",
                            "file_path",
                            "start_line",
                            "end_line",
                            "tokens",
                            "content_hash",
                        ],
                        "properties": {
                            "id": {"type": "string"},
                            "file_path": {"type": "string"},
                            "start_line": {"type": "integer", "minimum": 1},
                            "end_line": {"type": "integer", "minimum": 1},
                            "tokens": {"type": "integer", "minimum": 0},
                            "content_hash": {
                                "type": "string",
                                "minLength": 64,
                                "maxLength": 64,
                            },
                            "language": {"type": "string"},
                            "chunk_type": {
                                "type": "string",
                                "enum": ["code", "markdown", "text", "binary"],
                            },
                            "importance_score": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        },
                    },
                },
                "statistics": {
                    "type": "object",
                    "required": [
                        "total_files",
                        "total_chunks",
                        "total_tokens",
                    ],
                    "properties": {
                        "total_files": {"type": "integer", "minimum": 0},
                        "total_chunks": {"type": "integer", "minimum": 0},
                        "total_tokens": {"type": "integer", "minimum": 0},
                        "language_distribution": {"type": "object"},
                        "ignored_files": {"type": "integer", "minimum": 0},
                        "duplicates_removed": {"type": "integer", "minimum": 0},
                    },
                },
            },
        }

    def write_schema(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self._generate_default_schema(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def validate_pack_structure(self, pack_data: Dict[str, Any]) -> List[str]:
        if not self.schema:
            return ["No schema available for validation"]
        errors: List[str] = []
        try:
            validate(instance=pack_data, schema=self.schema)
        except ValidationError as err:
            errors.append(f"Schema validation error: {err.message}")
            if err.path:
                errors.append(f"  Path: {' -> '.join(str(part) for part in err.path)}")
        return errors

    def validate_budget_constraints(self, pack_data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        budget_info = pack_data.get("budget_info", {})
        target_budget = budget_info.get("target_budget", 0)
        actual_tokens = budget_info.get("actual_tokens", 0)
        utilization = budget_info.get("utilization", 0.0)
        underflow_allowed = budget_info.get("underflow_allowed", 0.0)

        if target_budget <= 0:
            errors.append("Invalid target budget")
        if actual_tokens < 0:
            errors.append("Actual tokens must be non-negative")
        if utilization < 0 or utilization > 1.01:
            errors.append("Utilization outside valid range")
        if (target_budget - actual_tokens) / max(target_budget, 1) > underflow_allowed:
            errors.append("Token underflow exceeds allowed threshold")
        return errors

    def verify_chunk_hashes(self, pack_data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        for chunk in pack_data.get("chunks", []):
            expected = chunk.get("content_hash")
            content = chunk.get("content", "")
            if not expected:
                errors.append(f"Chunk {chunk.get('id')} missing content hash")
                continue
            computed = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if expected != computed:
                errors.append(
                    f"Chunk {chunk.get('id')} hash mismatch: expected {expected}, got {computed}"
                )
        return errors

    def validate_pack(self, pack_data: Dict[str, Any]) -> Dict[str, Any]:
        structure_errors = self.validate_pack_structure(pack_data)
        budget_errors = self.validate_budget_constraints(pack_data)
        hash_errors = self.verify_chunk_hashes(pack_data)
        errors = structure_errors + budget_errors + hash_errors
        return {
            "is_valid": not errors,
            "errors": errors,
        }


__all__ = ["PackVerifier"]
