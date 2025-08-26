#!/usr/bin/env python3
"""
Basic Secret Scanning Script for PackRepo

Scans codebase for potential secrets and sensitive information.
Part of the security validation pipeline.
"""

import os
import re
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

class SecretScanner:
    """Basic secret detection for development use."""
    
    def __init__(self):
        self.patterns = {
            'api_key': [
                r'api[_-]?key["\s]*[:=]["\s]*([a-zA-Z0-9_\-]{20,})',
                r'apikey["\s]*[:=]["\s]*([a-zA-Z0-9_\-]{20,})',
            ],
            'password': [
                r'password["\s]*[:=]["\s]*["\']([^"\']{8,})["\']',
                r'passwd["\s]*[:=]["\s]*["\']([^"\']{8,})["\']',
            ],
            'token': [
                r'token["\s]*[:=]["\s]*["\']([a-zA-Z0-9_\-]{20,})["\']',
                r'auth[_-]?token["\s]*[:=]["\s]*["\']([a-zA-Z0-9_\-]{20,})["\']',
            ],
            'secret': [
                r'secret[_-]?key["\s]*[:=]["\s]*["\']([a-zA-Z0-9_\-]{16,})["\']',
                r'client[_-]?secret["\s]*[:=]["\s]*["\']([a-zA-Z0-9_\-]{16,})["\']',
            ],
        }
        
        self.file_patterns = [
            r'\.env$',
            r'\.env\.',
            r'secret',
            r'credential',
            r'\.key$',
            r'\.pem$',
        ]
    
    def scan_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan a single file for secrets."""
        findings = []
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                for secret_type, patterns in self.patterns.items():
                    for pattern in patterns:
                        matches = re.finditer(pattern, line, re.IGNORECASE)
                        for match in matches:
                            findings.append({
                                'type': secret_type,
                                'file': str(file_path),
                                'line': line_num,
                                'pattern': pattern,
                                'context': line.strip(),
                                'severity': 'high'
                            })
                            
        except (UnicodeDecodeError, PermissionError):
            pass  # Skip binary or inaccessible files
        
        return findings
    
    def scan_directory(self, directory: Path) -> List[Dict[str, Any]]:
        """Scan directory recursively for secrets."""
        all_findings = []
        
        # Skip common ignore patterns
        ignore_patterns = [
            '.git', '__pycache__', '.pytest_cache', 
            'node_modules', '.venv', 'venv',
            '.serena', 'artifacts'
        ]
        
        for root, dirs, files in os.walk(directory):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not any(ignore in d for ignore in ignore_patterns)]
            
            for file in files:
                file_path = Path(root) / file
                
                # Check for suspicious filenames
                filename_lower = file.lower()
                for pattern in self.file_patterns:
                    if re.search(pattern, filename_lower):
                        all_findings.append({
                            'type': 'suspicious_filename',
                            'file': str(file_path),
                            'line': 0,
                            'pattern': pattern,
                            'context': f'Filename matches pattern: {pattern}',
                            'severity': 'medium'
                        })
                
                # Scan file content
                file_findings = self.scan_file(file_path)
                all_findings.extend(file_findings)
        
        return all_findings
    
    def generate_report(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary report of findings."""
        report = {
            'timestamp': Path(__file__).stat().st_mtime,
            'total_findings': len(findings),
            'findings_by_type': {},
            'findings_by_severity': {},
            'files_affected': len(set(f['file'] for f in findings)),
            'findings': findings
        }
        
        # Group by type
        for finding in findings:
            finding_type = finding['type']
            report['findings_by_type'][finding_type] = report['findings_by_type'].get(finding_type, 0) + 1
        
        # Group by severity
        for finding in findings:
            severity = finding['severity']
            report['findings_by_severity'][severity] = report['findings_by_severity'].get(severity, 0) + 1
        
        return report


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Basic secret scanner for PackRepo')
    parser.add_argument('--directory', '-d', type=Path, default=Path.cwd(),
                       help='Directory to scan (default: current directory)')
    parser.add_argument('--out', '-o', type=Path, 
                       help='Output JSON file for results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize scanner
    scanner = SecretScanner()
    
    print(f"Scanning directory: {args.directory}")
    
    # Scan for secrets
    findings = scanner.scan_directory(args.directory)
    
    # Generate report
    report = scanner.generate_report(findings)
    
    # Output results
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"Report saved to: {args.out}")
    
    # Print summary
    print(f"Secret scan completed:")
    print(f"  Total findings: {report['total_findings']}")
    print(f"  Files affected: {report['files_affected']}")
    
    if report['findings_by_severity']:
        print("  By severity:")
        for severity, count in report['findings_by_severity'].items():
            print(f"    {severity}: {count}")
    
    if args.verbose and findings:
        print("\nDetailed findings:")
        for finding in findings:
            print(f"  {finding['severity'].upper()}: {finding['type']} in {finding['file']}:{finding['line']}")
            print(f"    Context: {finding['context']}")
    
    # Exit with error code if high severity findings
    high_severity = report['findings_by_severity'].get('high', 0)
    if high_severity > 0:
        print(f"ERROR: {high_severity} high-severity potential secrets found")
        sys.exit(1)
    
    print("✓ Secret scan passed")
    sys.exit(0)


if __name__ == '__main__':
    main()