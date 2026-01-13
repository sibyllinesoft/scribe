"""Shared report generation utilities for benchmarks."""

from datetime import datetime
from typing import Any, Optional


def generate_markdown_table(
    headers: list[str],
    rows: list[list[Any]],
    alignments: Optional[list[str]] = None,
) -> str:
    """Generate a markdown table.

    Args:
        headers: List of column headers.
        rows: List of rows, each row is a list of cell values.
        alignments: Optional list of alignments ('left', 'right', 'center').
                   Defaults to left alignment.

    Returns:
        Markdown table string.
    """
    if not headers or not rows:
        return ""

    # Convert all values to strings
    str_headers = [str(h) for h in headers]
    str_rows = [[str(cell) for cell in row] for row in rows]

    # Calculate column widths
    widths = [len(h) for h in str_headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    # Build alignment row
    if alignments is None:
        alignments = ["left"] * len(headers)

    align_row = []
    for i, align in enumerate(alignments):
        width = widths[i] if i < len(widths) else 3
        if align == "right":
            align_row.append("-" * (width - 1) + ":")
        elif align == "center":
            align_row.append(":" + "-" * (width - 2) + ":")
        else:
            align_row.append("-" * width)

    # Build table
    lines = []

    # Header row
    header_cells = [h.ljust(widths[i]) for i, h in enumerate(str_headers)]
    lines.append("| " + " | ".join(header_cells) + " |")

    # Alignment row
    lines.append("| " + " | ".join(align_row) + " |")

    # Data rows
    for row in str_rows:
        cells = []
        for i, cell in enumerate(row):
            width = widths[i] if i < len(widths) else len(cell)
            align = alignments[i] if i < len(alignments) else "left"
            if align == "right":
                cells.append(cell.rjust(width))
            elif align == "center":
                cells.append(cell.center(width))
            else:
                cells.append(cell.ljust(width))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def generate_summary_section(
    title: str,
    metrics: dict[str, Any],
    description: Optional[str] = None,
) -> str:
    """Generate a summary section with key metrics.

    Args:
        title: Section title.
        metrics: Dict of metric name -> value.
        description: Optional description text.

    Returns:
        Markdown section string.
    """
    lines = [f"## {title}", ""]

    if description:
        lines.append(description)
        lines.append("")

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")

    for name, value in metrics.items():
        # Format value based on type
        if isinstance(value, float):
            if abs(value) < 0.01:
                formatted = f"{value:.4f}"
            elif abs(value) < 1:
                formatted = f"{value:.3f}"
            else:
                formatted = f"{value:.2f}"
        elif isinstance(value, int):
            formatted = f"{value:,}"
        else:
            formatted = str(value)

        lines.append(f"| **{name}** | {formatted} |")

    return "\n".join(lines)


def generate_comparison_section(
    title: str,
    baseline_name: str,
    comparison_name: str,
    baseline_metrics: dict[str, float],
    comparison_metrics: dict[str, float],
) -> str:
    """Generate a comparison section between two approaches.

    Args:
        title: Section title.
        baseline_name: Name of baseline approach.
        comparison_name: Name of comparison approach.
        baseline_metrics: Baseline metric values.
        comparison_metrics: Comparison metric values.

    Returns:
        Markdown section string.
    """
    lines = [f"## {title}", ""]

    lines.append(f"| Metric | {baseline_name} | {comparison_name} | Change |")
    lines.append("|--------|-------|-------|--------|")

    for name in baseline_metrics:
        baseline = baseline_metrics[name]
        comparison = comparison_metrics.get(name, 0)

        # Calculate change
        if baseline != 0:
            change_pct = ((comparison - baseline) / baseline) * 100
            if change_pct > 0:
                change_str = f"+{change_pct:.1f}%"
            else:
                change_str = f"{change_pct:.1f}%"
        else:
            change_str = "N/A"

        # Format values
        if isinstance(baseline, float):
            baseline_str = f"{baseline:.2f}"
            comparison_str = f"{comparison:.2f}"
        else:
            baseline_str = f"{baseline:,}"
            comparison_str = f"{comparison:,}"

        lines.append(f"| {name} | {baseline_str} | {comparison_str} | {change_str} |")

    return "\n".join(lines)


def generate_report_header(
    title: str,
    metadata: dict[str, Any],
) -> str:
    """Generate a report header with metadata.

    Args:
        title: Report title.
        metadata: Dict of metadata key -> value.

    Returns:
        Markdown header string.
    """
    lines = [f"# {title}", ""]

    for key, value in metadata.items():
        lines.append(f"**{key}:** {value}")

    lines.append("")
    return "\n".join(lines)


def format_duration(ms: float) -> str:
    """Format a duration in milliseconds to human-readable string."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        minutes = int(ms // 60000)
        seconds = (ms % 60000) / 1000
        return f"{minutes}m {seconds:.0f}s"


def format_tokens(tokens: int) -> str:
    """Format token count with thousands separator."""
    if tokens >= 1_000_000:
        return f"{tokens/1_000_000:.1f}M"
    elif tokens >= 1_000:
        return f"{tokens/1_000:.1f}k"
    else:
        return str(tokens)
