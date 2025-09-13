#!/usr/bin/env python3
"""
Build script for Scribe-RS Python bindings.

This script provides a convenient way to build and install the Python bindings
for development and distribution.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, cwd=None, check=True):
    """Run a shell command and return the result."""
    print(f"🔨 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


def check_requirements():
    """Check that all build requirements are available."""
    print("🔍 Checking build requirements...")
    
    # Check for Rust
    try:
        result = run_command(["cargo", "--version"], check=False)
        if result.returncode == 0:
            print(f"✅ Rust: {result.stdout.strip()}")
        else:
            print("❌ Rust (cargo) not found. Please install Rust from https://rustup.rs/")
            return False
    except FileNotFoundError:
        print("❌ Rust (cargo) not found. Please install Rust from https://rustup.rs/")
        return False
    
    # Check for Python
    print(f"✅ Python: {sys.version.split()[0]}")
    
    # Check for maturin
    try:
        result = run_command(["maturin", "--version"], check=False)
        if result.returncode == 0:
            print(f"✅ Maturin: {result.stdout.strip()}")
        else:
            print("❌ Maturin not found. Installing...")
            install_maturin()
    except FileNotFoundError:
        print("❌ Maturin not found. Installing...")
        install_maturin()
    
    return True


def install_maturin():
    """Install maturin for building Python wheels."""
    try:
        run_command([sys.executable, "-m", "pip", "install", "maturin[patchelf]"])
        print("✅ Maturin installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install maturin: {e}")
        sys.exit(1)


def clean_build():
    """Clean previous build artifacts."""
    print("🧹 Cleaning build artifacts...")
    
    # Clean Cargo artifacts
    run_command(["cargo", "clean"], check=False)
    
    # Clean Python artifacts
    for pattern in ["build", "dist", "*.egg-info", "**/__pycache__", "**/*.pyc"]:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                import shutil
                shutil.rmtree(path, ignore_errors=True)
                print(f"🗑️ Removed directory: {path}")
            else:
                path.unlink(missing_ok=True)
                print(f"🗑️ Removed file: {path}")


def build_development():
    """Build for development with debug symbols."""
    print("🔨 Building for development...")
    run_command(["maturin", "develop", "--release"])
    print("✅ Development build complete")


def build_wheel():
    """Build wheel for distribution.""" 
    print("🎡 Building wheel...")
    run_command(["maturin", "build", "--release"])
    print("✅ Wheel build complete")


def build_and_install():
    """Build and install in the current environment."""
    print("📦 Building and installing...")
    run_command(["maturin", "develop", "--release"])
    print("✅ Build and install complete")


def run_tests():
    """Run Python tests."""
    print("🧪 Running tests...")
    
    # Check if pytest is available
    try:
        run_command([sys.executable, "-m", "pytest", "--version"], check=False)
    except subprocess.CalledProcessError:
        print("📥 Installing pytest...")
        run_command([sys.executable, "-m", "pip", "install", "pytest", "pytest-asyncio"])
    
    # Run tests
    test_paths = ["tests", "python/tests"]
    for test_path in test_paths:
        if Path(test_path).exists():
            print(f"🧪 Running tests in {test_path}...")
            run_command([sys.executable, "-m", "pytest", test_path, "-v"])


def run_example():
    """Run the basic usage example.""" 
    print("🎯 Running basic usage example...")
    
    example_path = Path("examples/basic_usage.py")
    if example_path.exists():
        # Use current directory as the repository to analyze
        run_command([sys.executable, str(example_path), "."])
    else:
        print(f"❌ Example not found: {example_path}")


def main():
    """Main build script entry point."""
    parser = argparse.ArgumentParser(description="Build Scribe-RS Python bindings")
    parser.add_argument("command", choices=[
        "check", "clean", "develop", "wheel", "install", "test", "example", "all"
    ], help="Build command to execute")
    
    args = parser.parse_args()
    
    print("🦀 Scribe-RS Python Bindings Build Script")
    print("=" * 50)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if args.command == "check":
        check_requirements()
    
    elif args.command == "clean":
        clean_build()
    
    elif args.command == "develop":
        if check_requirements():
            build_development()
    
    elif args.command == "wheel":
        if check_requirements():
            build_wheel()
    
    elif args.command == "install":
        if check_requirements():
            build_and_install()
    
    elif args.command == "test":
        run_tests()
    
    elif args.command == "example":
        run_example()
    
    elif args.command == "all":
        if check_requirements():
            clean_build()
            build_and_install()
            run_tests()
            run_example()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()