#!/usr/bin/env python3
"""
PackRepo Pack Verification Script

Implements runtime oracles and contracts for PackRepo pack format validation.
Generates JSON schema, validates pack structure, and enforces invariants.

Usage:
    python scripts/pack_verify.py --write-schema spec/index.schema.json
    python scripts/pack_verify.py --packs logs/V1/ --schema spec/index.schema.json
    python scripts/pack_verify.py --validate pack_output.json
"""

import argparse
import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import jsonschema
from jsonschema import validate, ValidationError


class PackVerifier:
    """Comprehensive pack validation with runtime oracles."""
    
    def __init__(self, schema_path: Optional[Path] = None):
        self.schema_path = schema_path
        self.schema = self._load_schema() if schema_path else None
        
    def _load_schema(self) -> Dict[str, Any]:
        """Load JSON schema for pack validation."""
        if not self.schema_path or not self.schema_path.exists():
            return self._generate_default_schema()
        
        return json.loads(self.schema_path.read_text(encoding='utf-8'))
    
    def _generate_default_schema(self) -> Dict[str, Any]:
        """Generate default JSON schema for pack format."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "PackRepo Pack Format Schema",
            "description": "Validation schema for PackRepo pack index and format",
            "type": "object",
            "required": [
                "metadata",
                "tokenizer_info",
                "budget_info",
                "chunks",
                "statistics"
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
                                "commit": {"type": "string", "minLength": 40, "maxLength": 40},
                                "branch": {"type": "string"},
                                "remote": {"type": "string"}
                            }
                        }
                    }
                },
                "tokenizer_info": {
                    "type": "object",
                    "required": ["name", "version"],
                    "properties": {
                        "name": {"type": "string", "enum": ["cl100k", "o200k", "gpt2"]},
                        "version": {"type": "string"},
                        "vocab_size": {"type": "integer", "minimum": 1000}
                    }
                },
                "budget_info": {
                    "type": "object",
                    "required": ["target_budget", "actual_tokens", "utilization"],
                    "properties": {
                        "target_budget": {"type": "integer", "minimum": 1000},
                        "actual_tokens": {"type": "integer", "minimum": 0},
                        "utilization": {"type": "number", "minimum": 0, "maximum": 1.005},
                        "underflow_allowed": {"type": "number", "maximum": 0.005}
                    }
                },
                "chunks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "file_path", "start_line", "end_line", "tokens", "content_hash"],
                        "properties": {
                            "id": {"type": "string"},
                            "file_path": {"type": "string"},
                            "start_line": {"type": "integer", "minimum": 1},
                            "end_line": {"type": "integer", "minimum": 1},
                            "tokens": {"type": "integer", "minimum": 0},
                            "content_hash": {"type": "string", "minLength": 64, "maxLength": 64},
                            "language": {"type": "string"},
                            "chunk_type": {"type": "string", "enum": ["code", "markdown", "text", "binary"]},
                            "importance_score": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    }
                },
                "statistics": {
                    "type": "object",
                    "required": ["total_files", "total_chunks", "total_tokens"],
                    "properties": {
                        "total_files": {"type": "integer", "minimum": 0},
                        "total_chunks": {"type": "integer", "minimum": 0},
                        "total_tokens": {"type": "integer", "minimum": 0},
                        "language_distribution": {"type": "object"},
                        "ignored_files": {"type": "integer", "minimum": 0},
                        "duplicates_removed": {"type": "integer", "minimum": 0}
                    }
                }
            }
        }
    
    def write_schema(self, output_path: Path):
        """Write JSON schema to file."""
        schema = self._generate_default_schema()
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write schema with pretty formatting
        output_path.write_text(
            json.dumps(schema, indent=2, sort_keys=True),
            encoding='utf-8'
        )
        
        print(f"✓ Pack schema written to: {output_path}")
    
    def validate_pack_structure(self, pack_data: Dict[str, Any]) -> List[str]:
        """Validate pack against JSON schema."""
        errors = []
        
        if not self.schema:
            errors.append("No schema available for validation")
            return errors
        
        try:
            validate(instance=pack_data, schema=self.schema)
        except ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
            if e.path:
                errors.append(f"  Path: {' -> '.join(str(p) for p in e.path)}")
        
        return errors
    
    def validate_budget_constraints(self, pack_data: Dict[str, Any]) -> List[str]:
        """Validate budget constraints and token accounting."""
        errors = []
        
        budget_info = pack_data.get('budget_info', {})
        target_budget = budget_info.get('target_budget', 0)
        actual_tokens = budget_info.get('actual_tokens', 0)
        
        # Check hard budget cap (0 overflow allowed)
        if actual_tokens > target_budget:
            errors.append(f"Budget overflow: {actual_tokens} > {target_budget} tokens")
        
        # Check underflow constraint (≤0.5% allowed)
        if target_budget > 0:
            utilization = actual_tokens / target_budget
            if utilization < 0.995:  # Less than 99.5% utilization
                underflow = (target_budget - actual_tokens) / target_budget
                if underflow > 0.005:  # More than 0.5% underflow
                    errors.append(f"Excessive underflow: {underflow:.3f} > 0.5%")
        
        # Validate token sum consistency
        chunks = pack_data.get('chunks', [])
        chunk_token_sum = sum(chunk.get('tokens', 0) for chunk in chunks)
        
        if abs(chunk_token_sum - actual_tokens) > 1:  # Allow 1 token rounding difference
            errors.append(f"Token sum mismatch: chunks={chunk_token_sum}, actual={actual_tokens}")
        
        return errors
    
    def validate_chunk_constraints(self, pack_data: Dict[str, Any]) -> List[str]:
        """Validate chunk-level constraints and anchors."""
        errors = []
        
        chunks = pack_data.get('chunks', [])
        
        # Track file positions for overlap detection
        file_positions = {}
        chunk_ids = set()
        
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get('id', f'chunk_{i}')
            file_path = chunk.get('file_path', '')
            start_line = chunk.get('start_line', 0)
            end_line = chunk.get('end_line', 0)
            
            # Check chunk ID uniqueness
            if chunk_id in chunk_ids:
                errors.append(f"Duplicate chunk ID: {chunk_id}")
            chunk_ids.add(chunk_id)
            
            # Check line range validity
            if start_line > end_line:
                errors.append(f"Invalid line range in {chunk_id}: {start_line} > {end_line}")
            
            if start_line < 1:
                errors.append(f"Invalid start line in {chunk_id}: {start_line} < 1")
            
            # Check for overlaps within the same file
            if file_path not in file_positions:
                file_positions[file_path] = []
            
            # Check overlap with existing chunks in same file
            for existing_start, existing_end, existing_id in file_positions[file_path]:
                if (start_line <= existing_end and end_line >= existing_start):
                    errors.append(f"Chunk overlap: {chunk_id} and {existing_id} in {file_path}")
            
            file_positions[file_path].append((start_line, end_line, chunk_id))
            
            # Validate content hash format
            content_hash = chunk.get('content_hash', '')
            if not content_hash or len(content_hash) != 64 or not all(c in '0123456789abcdef' for c in content_hash.lower()):
                errors.append(f"Invalid content hash in {chunk_id}: {content_hash}")
        
        return errors
    
    def validate_determinism(self, pack_data: Dict[str, Any]) -> List[str]:
        """Validate deterministic properties."""
        errors = []
        
        chunks = pack_data.get('chunks', [])
        
        # Check chunk ordering determinism
        if len(chunks) > 1:
            # Chunks should be sorted by some deterministic criteria
            # For now, check that file_path is sorted and within file by start_line
            prev_file = ""
            prev_start = 0
            
            for chunk in chunks:
                file_path = chunk.get('file_path', '')
                start_line = chunk.get('start_line', 0)
                
                if file_path < prev_file:
                    errors.append("Chunks not sorted by file path")
                    break
                elif file_path == prev_file and start_line < prev_start:
                    errors.append(f"Chunks not sorted by line number in {file_path}")
                    break
                
                prev_file = file_path
                prev_start = start_line if file_path != prev_file else start_line
        
        # Check for stable chunk IDs (should be deterministic)
        chunk_ids = [chunk.get('id', '') for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            errors.append("Non-unique chunk IDs detected")
        
        return errors
    
    def validate_security_constraints(self, pack_data: Dict[str, Any]) -> List[str]:
        """Validate security-related constraints."""
        errors = []
        
        chunks = pack_data.get('chunks', [])
        
        # Check for potential secret exposure
        secret_patterns = [
            r'api[_-]?key',
            r'password',
            r'secret',
            r'token',
            r'credential'
        ]
        
        # This is a basic check - in practice would use more sophisticated secret detection
        for chunk in chunks:
            file_path = chunk.get('file_path', '').lower()
            
            # Check for common secret file patterns
            if any(pattern in file_path for pattern in ['.env', 'secret', 'key', 'credential']):
                errors.append(f"Potential secret file included: {chunk.get('file_path', '')}")
        
        return errors
    
    def validate_metamorphic_properties(self, pack_data: Dict[str, Any]) -> List[str]:
        """Validate metamorphic properties (placeholders for future implementation)."""
        errors = []
        
        # TODO: Implement metamorphic property checks
        # M1: append non-referenced duplicate file => ≤1% selection delta
        # M2: rename path, content unchanged => only path fields differ
        # M3: budget×2 => coverage score increases monotonically
        # M4: switch tokenizer cl100k→o200k with scaled budget => similarity ≥ 0.8 Jaccard
        # M5: inject large vendor folder => selection unaffected except index ignored counts
        # M6: remove selected chunk => budget reallocated without over-cap
        
        # For now, just add placeholder checks
        chunks = pack_data.get('chunks', [])
        if len(chunks) == 0:
            errors.append("No chunks selected - metamorphic properties cannot be validated")
        
        return errors
    
    def generate_pack_digest(self, pack_data: Dict[str, Any]) -> str:
        """Generate deterministic digest of pack content."""
        # Create canonical representation for hashing
        canonical_data = {
            'chunks': sorted(pack_data.get('chunks', []), key=lambda x: (x.get('file_path', ''), x.get('start_line', 0))),
            'budget_info': pack_data.get('budget_info', {}),
            'tokenizer_info': pack_data.get('tokenizer_info', {})
        }
        
        canonical_json = json.dumps(canonical_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
    
    def validate_pack(self, pack_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Run comprehensive pack validation."""
        all_errors = []
        
        # Run all validation checks
        all_errors.extend(self.validate_pack_structure(pack_data))
        all_errors.extend(self.validate_budget_constraints(pack_data))
        all_errors.extend(self.validate_chunk_constraints(pack_data))
        all_errors.extend(self.validate_determinism(pack_data))
        all_errors.extend(self.validate_security_constraints(pack_data))
        all_errors.extend(self.validate_metamorphic_properties(pack_data))
        
        return len(all_errors) == 0, all_errors
    
    def validate_pack_file(self, pack_file: Path) -> Tuple[bool, List[str]]:
        """Validate a pack file."""
        if not pack_file.exists():
            return False, [f"Pack file not found: {pack_file}"]
        
        try:
            pack_data = json.loads(pack_file.read_text(encoding='utf-8'))
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON in pack file: {e}"]
        
        return self.validate_pack(pack_data)
    
    def validate_pack_directory(self, pack_dir: Path) -> Dict[str, Tuple[bool, List[str]]]:
        """Validate all pack files in a directory."""
        results = {}
        
        if not pack_dir.exists():
            return {"error": (False, [f"Pack directory not found: {pack_dir}"])}
        
        # Find all JSON files in directory
        json_files = list(pack_dir.glob("*.json"))
        
        if not json_files:
            return {"error": (False, [f"No JSON pack files found in: {pack_dir}"])}
        
        for json_file in json_files:
            results[json_file.name] = self.validate_pack_file(json_file)
        
        return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='PackRepo Pack Verification and Oracle System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate and write JSON schema
  python scripts/pack_verify.py --write-schema spec/index.schema.json
  
  # Validate packs in directory
  python scripts/pack_verify.py --packs logs/V1/ --schema spec/index.schema.json
  
  # Validate single pack file
  python scripts/pack_verify.py --validate pack_output.json --schema spec/index.schema.json
  
  # Generate pack digest
  python scripts/pack_verify.py --digest pack_output.json
        """
    )
    
    parser.add_argument(
        '--write-schema',
        type=Path,
        help='Write JSON schema to specified file'
    )
    
    parser.add_argument(
        '--packs',
        type=Path,
        help='Directory containing pack files to validate'
    )
    
    parser.add_argument(
        '--validate',
        type=Path,
        help='Single pack file to validate'
    )
    
    parser.add_argument(
        '--schema',
        type=Path,
        help='JSON schema file for validation'
    )
    
    parser.add_argument(
        '--digest',
        action='store_true',
        help='Generate pack digest'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Create verifier instance
    verifier = PackVerifier(schema_path=args.schema)
    
    # Handle schema generation
    if args.write_schema:
        verifier.write_schema(args.write_schema)
        return
    
    # Handle single pack validation
    if args.validate:
        success, errors = verifier.validate_pack_file(args.validate)
        
        print(f"Validating pack: {args.validate}")
        
        if success:
            print("✓ Pack validation PASSED")
            if args.digest:
                pack_data = json.loads(args.validate.read_text(encoding='utf-8'))
                digest = verifier.generate_pack_digest(pack_data)
                print(f"Pack digest: {digest}")
        else:
            print("✗ Pack validation FAILED")
            for error in errors:
                print(f"  ERROR: {error}")
        
        sys.exit(0 if success else 1)
    
    # Handle directory validation
    if args.packs:
        results = verifier.validate_pack_directory(args.packs)
        
        print(f"Validating packs in directory: {args.packs}")
        print("=" * 50)
        
        total_files = 0
        total_errors = 0
        
        for filename, (success, errors) in results.items():
            total_files += 1
            
            if success:
                print(f"✓ {filename}: PASSED")
            else:
                print(f"✗ {filename}: FAILED")
                total_errors += len(errors)
                
                if args.verbose:
                    for error in errors:
                        print(f"    ERROR: {error}")
                else:
                    print(f"    {len(errors)} error(s) - use -v for details")
        
        print("=" * 50)
        print(f"Summary: {total_files} files, {total_errors} total errors")
        
        if total_errors == 0:
            print("✓ All pack validations PASSED")
            sys.exit(0)
        else:
            print("✗ Pack validation FAILED")
            sys.exit(1)
    
    # If no specific action, print help
    parser.print_help()


if __name__ == '__main__':
    main()