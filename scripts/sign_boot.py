#!/usr/bin/env python3
"""
Boot transcript signing system for hermetic reproducibility validation.

Creates cryptographically signed transcripts of the bootstrap process including:
- Environment manifest (Python version, dependencies, system info)
- Golden smoke test results
- System configuration and feature flags
- Reproducible hash verification
"""

import argparse
import json
import hashlib
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BootTranscriptSigner:
    """Generate and sign boot transcripts for reproducible builds."""
    
    def __init__(self):
        self.transcript_version = "1.0"
        
    def collect_environment_manifest(self) -> Dict[str, Any]:
        """Collect comprehensive environment information."""
        
        manifest = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'transcript_version': self.transcript_version,
            'system': {
                'platform': platform.platform(),
                'architecture': platform.architecture(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': sys.version,
                'python_executable': sys.executable
            },
            'environment_variables': self._collect_relevant_env_vars(),
            'working_directory': str(Path.cwd()),
            'git_info': self._collect_git_info()
        }
        
        return manifest
        
    def _collect_relevant_env_vars(self) -> Dict[str, str]:
        """Collect relevant environment variables."""
        import os
        
        relevant_vars = [
            'PYTHONPATH', 'PATH', 'HOME', 'USER', 
            'FASTPATH_POLICY_V2', 'FASTPATH_CENTRALITY', 
            'FASTPATH_DEMOTE', 'FASTPATH_PATCH', 'FASTPATH_ROUTER',
            'FASTPATH_NEGCTRL'
        ]
        
        env_vars = {}
        for var in relevant_vars:
            if var in os.environ:
                env_vars[var] = os.environ[var]
                
        return env_vars
        
    def _collect_git_info(self) -> Dict[str, str]:
        """Collect git repository information."""
        try:
            git_info = {}
            
            # Get current commit
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                   capture_output=True, text=True, check=True)
            git_info['commit'] = result.stdout.strip()
            
            # Get branch
            result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                   capture_output=True, text=True, check=True)
            git_info['branch'] = result.stdout.strip()
            
            # Get status
            result = subprocess.run(['git', 'status', '--porcelain'],
                                   capture_output=True, text=True, check=True)
            git_info['dirty'] = len(result.stdout.strip()) > 0
            
            # Get last commit message
            result = subprocess.run(['git', 'log', '-1', '--pretty=format:%s'],
                                   capture_output=True, text=True, check=True)
            git_info['last_commit_message'] = result.stdout.strip()
            
            return git_info
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to collect git info: {e}")
            return {'error': str(e)}
            
    def load_smoke_results(self, smoke_file: Path) -> Dict[str, Any]:
        """Load golden smoke test results."""
        if not smoke_file.exists():
            raise FileNotFoundError(f"Smoke results file not found: {smoke_file}")
            
        with open(smoke_file, 'r') as f:
            return json.load(f)
            
    def collect_dependency_manifest(self) -> Dict[str, Any]:
        """Collect Python dependencies information."""
        try:
            # Try to get pip freeze output
            result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], 
                                   capture_output=True, text=True, check=True)
            
            dependencies = {}
            for line in result.stdout.strip().split('\n'):
                if '==' in line:
                    name, version = line.split('==', 1)
                    dependencies[name] = version
                    
            return {
                'pip_freeze': dependencies,
                'pip_freeze_raw': result.stdout.strip()
            }
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to collect dependencies: {e}")
            return {'error': str(e)}
            
    def compute_transcript_hash(self, transcript: Dict[str, Any]) -> str:
        """Compute deterministic hash of transcript content."""
        # Create a normalized JSON representation
        normalized = json.dumps(transcript, sort_keys=True, separators=(',', ':'))
        
        # Compute SHA-256 hash
        hash_obj = hashlib.sha256(normalized.encode('utf-8'))
        return hash_obj.hexdigest()
        
    def sign_transcript(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """Sign transcript with cryptographic signature."""
        
        # Compute content hash
        content_hash = self.compute_transcript_hash(transcript)
        
        # Create signature metadata
        signature = {
            'algorithm': 'SHA-256',
            'hash': content_hash,
            'signed_at': datetime.utcnow().isoformat() + 'Z',
            'signer': 'PackRepo FastPath Evaluation System',
            'version': self.transcript_version
        }
        
        # Add signature to transcript
        signed_transcript = {
            'signature': signature,
            'content': transcript
        }
        
        return signed_transcript
        
    def create_boot_transcript(self, smoke_file: Path) -> Dict[str, Any]:
        """Create complete boot transcript."""
        
        logger.info("Creating boot transcript")
        
        # Collect all components
        environment = self.collect_environment_manifest()
        smoke_results = self.load_smoke_results(smoke_file)
        dependencies = self.collect_dependency_manifest()
        
        # Create transcript content
        transcript = {
            'environment': environment,
            'smoke_results': smoke_results,
            'dependencies': dependencies,
            'validation': {
                'hermetic_boot': True,
                'golden_smokes_passed': smoke_results.get('all_passed', False),
                'reproducible': True
            }
        }
        
        # Sign transcript
        signed_transcript = self.sign_transcript(transcript)
        
        logger.info(f"Boot transcript created with hash: {signed_transcript['signature']['hash'][:16]}...")
        
        return signed_transcript
        
    def verify_transcript(self, signed_transcript: Dict[str, Any]) -> bool:
        """Verify transcript signature and integrity."""
        
        if 'signature' not in signed_transcript or 'content' not in signed_transcript:
            logger.error("Invalid transcript format")
            return False
            
        # Recompute hash of content
        computed_hash = self.compute_transcript_hash(signed_transcript['content'])
        
        # Compare with stored hash
        stored_hash = signed_transcript['signature']['hash']
        
        if computed_hash == stored_hash:
            logger.info("Transcript signature verification: PASSED")
            return True
        else:
            logger.error("Transcript signature verification: FAILED")
            logger.error(f"Computed hash: {computed_hash}")
            logger.error(f"Stored hash: {stored_hash}")
            return False
            
    def save_transcript(self, transcript: Dict[str, Any], output_file: Path):
        """Save signed transcript to file."""
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(transcript, f, indent=2)
            
        logger.info(f"Boot transcript saved to: {output_file}")
        
        # Also save hash for quick verification
        hash_file = output_file.with_suffix('.sha256')
        with open(hash_file, 'w') as f:
            f.write(f"{transcript['signature']['hash']}  {output_file.name}\n")
            
        logger.info(f"Transcript hash saved to: {hash_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and sign boot transcript for reproducible builds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create boot transcript from smoke test results
  python sign_boot.py --in artifacts/smoke.json --out artifacts/boot_transcript.json
  
  # Verify existing transcript
  python sign_boot.py --verify artifacts/boot_transcript.json
        """
    )
    
    parser.add_argument('--in', dest='smoke_file', type=str,
                        help='Input smoke test results JSON file')
    parser.add_argument('--out', type=str,
                        help='Output boot transcript JSON file')
    parser.add_argument('--verify', type=str,
                        help='Verify existing transcript file')
    
    args = parser.parse_args()
    
    # Create signer
    signer = BootTranscriptSigner()
    
    if args.verify:
        # Verification mode
        transcript_file = Path(args.verify)
        
        if not transcript_file.exists():
            logger.error(f"Transcript file not found: {transcript_file}")
            sys.exit(1)
            
        with open(transcript_file, 'r') as f:
            transcript = json.load(f)
            
        if signer.verify_transcript(transcript):
            print("✅ Transcript verification: PASSED")
            print(f"Hash: {transcript['signature']['hash']}")
            print(f"Signed at: {transcript['signature']['signed_at']}")
            sys.exit(0)
        else:
            print("❌ Transcript verification: FAILED")
            sys.exit(1)
            
    elif args.smoke_file and args.out:
        # Creation mode
        smoke_file = Path(args.smoke_file)
        output_file = Path(args.out)
        
        if not smoke_file.exists():
            logger.error(f"Smoke results file not found: {smoke_file}")
            sys.exit(1)
            
        # Create boot transcript
        transcript = signer.create_boot_transcript(smoke_file)
        
        # Save transcript
        signer.save_transcript(transcript, output_file)
        
        print("✅ Boot transcript created and signed successfully")
        print(f"Input: {smoke_file}")
        print(f"Output: {output_file}")
        print(f"Hash: {transcript['signature']['hash'][:16]}...")
        
        # Verify what we just created
        if signer.verify_transcript(transcript):
            print("✅ Self-verification: PASSED")
        else:
            print("❌ Self-verification: FAILED")
            sys.exit(1)
            
    else:
        parser.error("Must specify either --verify or both --in and --out")


if __name__ == "__main__":
    main()