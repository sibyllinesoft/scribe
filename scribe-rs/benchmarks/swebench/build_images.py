#!/usr/bin/env python3
"""Build SWE-bench Docker images for benchmark tasks.

Usage:
    ./build_images.py                    # Build images for first 10 tasks
    ./build_images.py --max-tasks 50     # Build images for first 50 tasks
    ./build_images.py --all              # Build images for all tasks
"""

import argparse
import sys
from pathlib import Path

# Must run from a directory where swebench doesn't conflict with local package
sys.path.insert(0, "/home/nathan/.local/share/mise/installs/python/3.14.0/lib/python3.14/site-packages")

import docker
from datasets import load_dataset
from swebench import build_instance_images
from swebench.harness.test_spec.test_spec import make_test_spec


def main():
    parser = argparse.ArgumentParser(description="Build SWE-bench Docker images")
    parser.add_argument("--max-tasks", "-n", type=int, default=10,
                        help="Maximum number of tasks to build images for")
    parser.add_argument("--all", action="store_true",
                        help="Build images for all tasks")
    parser.add_argument("--max-workers", "-w", type=int, default=4,
                        help="Number of parallel workers for building")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force rebuild even if images exist")
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite",
                        help="Dataset to build images for")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split="test")

    tasks = list(ds)
    if not args.all and args.max_tasks:
        tasks = tasks[:args.max_tasks]

    print(f"Building images for {len(tasks)} tasks...")

    # Create test specs for the tasks
    test_specs = []
    for task in tasks:
        try:
            spec = make_test_spec(task)
            test_specs.append(spec)
        except Exception as e:
            print(f"  Warning: Could not create test spec for {task.get('instance_id', '?')}: {e}")

    if not test_specs:
        print("Error: No valid test specs created")
        sys.exit(1)

    print(f"Created {len(test_specs)} test specs")

    # Build images
    client = docker.from_env()

    print(f"Building images with {args.max_workers} workers...")
    build_instance_images(
        client=client,
        dataset=test_specs,
        force_rebuild=args.force,
        max_workers=args.max_workers,
    )

    print("Done!")


if __name__ == "__main__":
    main()
