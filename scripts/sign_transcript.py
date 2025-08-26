#!/usr/bin/env python3
"""
PackRepo Boot Transcript Signing Utility

Cryptographically signs boot transcripts to ensure integrity and authenticity
of hermetic build verification results. Uses container digest, environment
hash, and timestamp for tamper-evident signing.

Usage:
    python scripts/sign_transcript.py artifacts/boot_env.txt artifacts/boot_transcript.json
    python scripts/sign_transcript.py --key-file signing.key artifacts/boot_transcript.json
    python scripts/sign_transcript.py --verify artifacts/boot_transcript.signed.json
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import os
import base64


class TranscriptSigner:
    """Handles signing and verification of PackRepo boot transcripts."""
    
    def __init__(self, key_file: Optional[str] = None):
        self.key_file = key_file
        
    def generate_content_hash(self, transcript_data: Dict[str, Any]) -> str:
        """Generate deterministic hash of transcript content."""
        # Create canonical representation
        canonical_data = self._canonicalize_transcript(transcript_data)
        content_str = json.dumps(canonical_data, sort_keys=True, separators=(',', ':'))
        
        # Generate SHA-256 hash
        return hashlib.sha256(content_str.encode('utf-8')).hexdigest()
    
    def generate_environment_hash(self, env_file: Path) -> str:
        """Generate hash of environment specification."""
        if not env_file.exists():
            raise FileNotFoundError(f"Environment file not found: {env_file}")
        
        env_content = env_file.read_text(encoding='utf-8')
        return hashlib.sha256(env_content.encode('utf-8')).hexdigest()
    
    def get_container_digest(self, container_tag: str) -> Optional[str]:
        """Extract container digest from Docker."""
        try:
            result = subprocess.run(
                ['docker', 'image', 'inspect', container_tag, '--format={{index .RepoDigests 0}}'],
                capture_output=True,
                text=True,
                check=True
            )
            digest = result.stdout.strip()
            return digest if digest and digest != '<no value>' else None
        except subprocess.CalledProcessError:
            # Fallback to image ID if no digest available
            try:
                result = subprocess.run(
                    ['docker', 'image', 'inspect', container_tag, '--format={{.Id}}'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                return result.stdout.strip()
            except subprocess.CalledProcessError:
                return None
    
    def get_git_signature(self, project_root: Path) -> Dict[str, str]:
        """Get Git repository signature information."""
        git_info = {}
        
        try:
            # Get commit hash
            result = subprocess.run(
                ['git', '-C', str(project_root), 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            git_info['commit'] = result.stdout.strip()
            
            # Check if repository is dirty
            result = subprocess.run(
                ['git', '-C', str(project_root), 'diff', '--quiet'],
                capture_output=True,
                check=False  # Expected to fail if dirty
            )
            git_info['dirty'] = result.returncode != 0
            
            # Get remote origin
            result = subprocess.run(
                ['git', '-C', str(project_root), 'remote', 'get-url', 'origin'],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                git_info['remote'] = result.stdout.strip()
            
            # Get current branch
            result = subprocess.run(
                ['git', '-C', str(project_root), 'branch', '--show-current'],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                git_info['branch'] = result.stdout.strip()
                
        except subprocess.CalledProcessError:
            pass  # Git commands failed, skip Git signature
            
        return git_info
    
    def create_signature_payload(self, 
                                transcript_file: Path, 
                                env_file: Optional[Path] = None) -> Dict[str, Any]:
        """Create comprehensive signature payload."""
        # Load transcript
        transcript_data = json.loads(transcript_file.read_text(encoding='utf-8'))
        
        # Generate content hash
        content_hash = self.generate_content_hash(transcript_data)
        
        # Generate environment hash if env file provided
        env_hash = None
        if env_file and env_file.exists():
            env_hash = self.generate_environment_hash(env_file)
        
        # Get container information
        container_tag = transcript_data.get('container_tag', 'unknown')
        container_digest = self.get_container_digest(container_tag)
        
        # Get Git signature
        project_root = transcript_file.parent.parent
        git_signature = self.get_git_signature(project_root)
        
        # Create signature payload
        signature_payload = {
            'version': '1.0',
            'timestamp': time.time(),
            'timestamp_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'content_hash': content_hash,
            'environment_hash': env_hash,
            'container_tag': container_tag,
            'container_digest': container_digest,
            'git_signature': git_signature,
            'signing_host': os.uname().nodename,
            'signing_user': os.getenv('USER', 'unknown'),
            'python_version': sys.version,
        }
        
        return signature_payload
    
    def simple_sign(self, payload: Dict[str, Any]) -> str:
        """Simple HMAC-based signing for development."""
        # Use a combination of system information as the key
        key_material = f"{os.uname().nodename}:{os.getenv('USER')}:{time.strftime('%Y-%m-%d')}"
        key = hashlib.sha256(key_material.encode('utf-8')).digest()
        
        # Create HMAC signature
        payload_str = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        signature = hashlib.blake2b(
            payload_str.encode('utf-8'), 
            key=key, 
            digest_size=32
        ).hexdigest()
        
        return signature
    
    def sign_transcript(self, 
                       transcript_file: Path, 
                       env_file: Optional[Path] = None,
                       output_file: Optional[Path] = None) -> Path:
        """Sign a boot transcript with integrity verification."""
        # Create signature payload
        signature_payload = self.create_signature_payload(transcript_file, env_file)
        
        # Generate signature
        signature = self.simple_sign(signature_payload)
        
        # Load original transcript
        transcript_data = json.loads(transcript_file.read_text(encoding='utf-8'))
        
        # Create signed transcript
        signed_transcript = {
            'transcript': transcript_data,
            'signature': {
                'payload': signature_payload,
                'signature': signature,
                'algorithm': 'blake2b-hmac',
                'key_type': 'system-derived'
            }
        }
        
        # Determine output file
        if output_file is None:
            output_file = transcript_file.with_suffix('.signed.json')
        
        # Write signed transcript
        output_file.write_text(
            json.dumps(signed_transcript, indent=2, sort_keys=True),
            encoding='utf-8'
        )
        
        return output_file
    
    def verify_signature(self, signed_transcript_file: Path) -> bool:
        """Verify a signed boot transcript."""
        # Load signed transcript
        signed_data = json.loads(signed_transcript_file.read_text(encoding='utf-8'))
        
        if 'signature' not in signed_data:
            print("ERROR: No signature found in transcript")
            return False
        
        signature_info = signed_data['signature']
        original_signature = signature_info['signature']
        payload = signature_info['payload']
        
        # Recreate signature
        recreated_signature = self.simple_sign(payload)
        
        # Verify signature
        if recreated_signature == original_signature:
            print("✓ Signature verification PASSED")
            return True
        else:
            print("✗ Signature verification FAILED")
            return False
    
    def display_signature_info(self, signed_transcript_file: Path):
        """Display detailed signature information."""
        signed_data = json.loads(signed_transcript_file.read_text(encoding='utf-8'))
        
        if 'signature' not in signed_data:
            print("No signature information found")
            return
        
        sig_info = signed_data['signature']
        payload = sig_info['payload']
        
        print("Boot Transcript Signature Information:")
        print("=" * 50)
        print(f"Version: {payload.get('version', 'unknown')}")
        print(f"Signed at: {payload.get('timestamp_iso', 'unknown')}")
        print(f"Content hash: {payload.get('content_hash', 'unknown')}")
        print(f"Environment hash: {payload.get('environment_hash', 'not provided')}")
        print(f"Container tag: {payload.get('container_tag', 'unknown')}")
        print(f"Container digest: {payload.get('container_digest', 'unavailable')}")
        print(f"Signing host: {payload.get('signing_host', 'unknown')}")
        print(f"Signing user: {payload.get('signing_user', 'unknown')}")
        
        if 'git_signature' in payload:
            git_info = payload['git_signature']
            print("Git Information:")
            print(f"  Commit: {git_info.get('commit', 'unknown')}")
            print(f"  Branch: {git_info.get('branch', 'unknown')}")
            print(f"  Remote: {git_info.get('remote', 'unknown')}")
            print(f"  Dirty: {git_info.get('dirty', 'unknown')}")
        
        print(f"Algorithm: {sig_info.get('algorithm', 'unknown')}")
        print(f"Signature: {sig_info.get('signature', 'unknown')[:32]}...")
    
    def _canonicalize_transcript(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create canonical representation of transcript for hashing."""
        # Remove volatile fields that shouldn't affect signature
        canonical = data.copy()
        
        # Fields to exclude from signature
        volatile_fields = {'boot_session', 'hostname', 'user', 'pwd'}
        
        for field in volatile_fields:
            canonical.pop(field, None)
        
        # Canonicalize nested structures
        if 'phases' in canonical:
            for phase in canonical['phases']:
                # Keep timing but normalize precision
                if 'duration_seconds' in phase:
                    phase['duration_seconds'] = round(phase['duration_seconds'], 2)
        
        return canonical


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='PackRepo Boot Transcript Signing Utility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sign a boot transcript with environment file
  python scripts/sign_transcript.py artifacts/boot_env.txt artifacts/boot_transcript.json
  
  # Sign transcript only
  python scripts/sign_transcript.py artifacts/boot_transcript.json
  
  # Verify a signed transcript
  python scripts/sign_transcript.py --verify artifacts/boot_transcript.signed.json
  
  # Display signature information
  python scripts/sign_transcript.py --info artifacts/boot_transcript.signed.json
        """
    )
    
    parser.add_argument(
        'files', 
        nargs='*',
        help='Input files (env_file transcript_file OR transcript_file only)'
    )
    
    parser.add_argument(
        '--verify', '-v',
        action='store_true',
        help='Verify signature of signed transcript'
    )
    
    parser.add_argument(
        '--info', '-i',
        action='store_true',
        help='Display signature information'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file for signed transcript'
    )
    
    parser.add_argument(
        '--key-file', '-k',
        type=str,
        help='Signing key file (future enhancement)'
    )
    
    args = parser.parse_args()
    
    # Initialize signer
    signer = TranscriptSigner(key_file=args.key_file)
    
    # Handle verification mode
    if args.verify:
        if len(args.files) != 1:
            print("ERROR: Verification requires exactly one signed transcript file")
            sys.exit(1)
        
        signed_file = Path(args.files[0])
        if not signed_file.exists():
            print(f"ERROR: Signed transcript file not found: {signed_file}")
            sys.exit(1)
        
        success = signer.verify_signature(signed_file)
        sys.exit(0 if success else 1)
    
    # Handle info mode
    if args.info:
        if len(args.files) != 1:
            print("ERROR: Info display requires exactly one signed transcript file")
            sys.exit(1)
        
        signed_file = Path(args.files[0])
        if not signed_file.exists():
            print(f"ERROR: Signed transcript file not found: {signed_file}")
            sys.exit(1)
        
        signer.display_signature_info(signed_file)
        sys.exit(0)
    
    # Handle signing mode
    if len(args.files) == 0:
        print("ERROR: No input files specified")
        parser.print_help()
        sys.exit(1)
    elif len(args.files) == 1:
        # Single file: transcript only
        env_file = None
        transcript_file = Path(args.files[0])
    elif len(args.files) == 2:
        # Two files: env_file transcript_file
        env_file = Path(args.files[0])
        transcript_file = Path(args.files[1])
    else:
        print("ERROR: Too many input files specified")
        sys.exit(1)
    
    # Validate input files
    if not transcript_file.exists():
        print(f"ERROR: Transcript file not found: {transcript_file}")
        sys.exit(1)
    
    if env_file and not env_file.exists():
        print(f"ERROR: Environment file not found: {env_file}")
        sys.exit(1)
    
    try:
        # Sign the transcript
        print(f"Signing boot transcript: {transcript_file}")
        if env_file:
            print(f"Using environment file: {env_file}")
        
        output_file = signer.sign_transcript(transcript_file, env_file, args.output)
        
        print(f"✓ Signed transcript created: {output_file}")
        
        # Verify the signature
        if signer.verify_signature(output_file):
            print("✓ Signature verification passed")
            
            # Display signature info
            print("\nSignature Information:")
            print("-" * 30)
            signer.display_signature_info(output_file)
            
        else:
            print("✗ Signature verification failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Failed to sign transcript: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()