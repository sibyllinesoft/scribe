"""Unified result storage and loading for benchmarks."""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def get_results_dir(benchmark_name: Optional[str] = None) -> Path:
    """Get the results directory for a benchmark.

    Args:
        benchmark_name: Name of the benchmark (e.g., "agent-efficiency", "swebench").
                       If None, returns the root results directory.

    Returns:
        Path to the results directory.
    """
    benchmarks_dir = Path(__file__).parent.parent
    results_dir = benchmarks_dir / "results"

    if benchmark_name:
        return results_dir / benchmark_name
    return results_dir


def save_results(
    benchmark_name: str,
    data: Any,
    prefix: str = "results",
    timestamp: Optional[str] = None,
) -> Path:
    """Save benchmark results to JSON file.

    Args:
        benchmark_name: Name of the benchmark.
        data: Data to save (dict, dataclass, or list of dataclasses).
        prefix: Filename prefix (e.g., "results", "summary", "raw_data").
        timestamp: Optional timestamp string. If None, uses current time.

    Returns:
        Path to the saved file.
    """
    results_dir = get_results_dir(benchmark_name)
    results_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{prefix}_{timestamp}.json"
    filepath = results_dir / filename

    # Convert dataclasses to dicts
    if is_dataclass(data) and not isinstance(data, type):
        data = asdict(data)
    elif isinstance(data, list):
        data = [asdict(item) if is_dataclass(item) and not isinstance(item, type) else item for item in data]

    filepath.write_text(json.dumps(data, indent=2, default=str))
    return filepath


def save_report(
    benchmark_name: str,
    content: str,
    prefix: str = "report",
    timestamp: Optional[str] = None,
) -> Path:
    """Save a markdown report.

    Args:
        benchmark_name: Name of the benchmark.
        content: Markdown content.
        prefix: Filename prefix.
        timestamp: Optional timestamp string.

    Returns:
        Path to the saved file.
    """
    results_dir = get_results_dir(benchmark_name)
    results_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{prefix}_{timestamp}.md"
    filepath = results_dir / filename
    filepath.write_text(content)
    return filepath


def load_results(
    benchmark_name: str,
    prefix: str = "results",
    timestamp: Optional[str] = None,
) -> Optional[dict]:
    """Load benchmark results from JSON file.

    Args:
        benchmark_name: Name of the benchmark.
        prefix: Filename prefix to match.
        timestamp: Specific timestamp to load. If None, loads most recent.

    Returns:
        Parsed JSON data, or None if not found.
    """
    results_dir = get_results_dir(benchmark_name)

    if not results_dir.exists():
        return None

    if timestamp:
        filepath = results_dir / f"{prefix}_{timestamp}.json"
        if filepath.exists():
            return json.loads(filepath.read_text())
        return None

    # Find most recent file matching prefix
    pattern = f"{prefix}_*.json"
    files = sorted(results_dir.glob(pattern), reverse=True)

    if not files:
        return None

    return json.loads(files[0].read_text())


def list_results(benchmark_name: str, prefix: str = "*") -> list[dict]:
    """List available result files for a benchmark.

    Args:
        benchmark_name: Name of the benchmark.
        prefix: Filename prefix to filter (default: all).

    Returns:
        List of dicts with 'filename', 'timestamp', 'path' keys.
    """
    results_dir = get_results_dir(benchmark_name)

    if not results_dir.exists():
        return []

    pattern = f"{prefix}_*.json"
    files = sorted(results_dir.glob(pattern), reverse=True)

    results = []
    for f in files:
        # Extract timestamp from filename
        parts = f.stem.split("_")
        if len(parts) >= 3:
            # Format: prefix_YYYYMMDD_HHMMSS
            timestamp = f"{parts[-2]}_{parts[-1]}"
        else:
            timestamp = "unknown"

        results.append({
            "filename": f.name,
            "timestamp": timestamp,
            "path": str(f),
        })

    return results


def list_benchmarks() -> list[str]:
    """List available benchmarks (directories in results/).

    Returns:
        List of benchmark names.
    """
    results_dir = get_results_dir()

    if not results_dir.exists():
        return []

    return [d.name for d in results_dir.iterdir() if d.is_dir()]
