#!/usr/bin/env python3
"""
Hook to remind agents about scribe when they use Read/Grep for exploration.

This hook intercepts Read and Grep tool calls and provides guidance about
using scribe to get complete dependency context instead of manual exploration.
"""

import json
import sys
import os

# Track exploration patterns per session
EXPLORATION_THRESHOLD = 3  # After N reads/greps, remind about scribe


def get_session_count_file(session_id: str) -> str:
    """Get path to session exploration count file."""
    return f"/tmp/claude_scribe_hook_{session_id}.json"


def get_exploration_count(session_id: str) -> dict:
    """Get exploration count for session."""
    path = get_session_count_file(session_id)
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"read_count": 0, "grep_count": 0, "reminded": False}


def save_exploration_count(session_id: str, counts: dict):
    """Save exploration count for session."""
    path = get_session_count_file(session_id)
    with open(path, "w") as f:
        json.dump(counts, f)


def should_remind(tool_name: str, tool_input: dict, counts: dict) -> tuple[bool, str]:
    """Determine if we should remind about scribe."""
    # Don't remind if already reminded this session
    if counts.get("reminded"):
        return False, ""

    total_exploration = counts.get("read_count", 0) + counts.get("grep_count", 0)

    # Remind after threshold exploration calls
    if total_exploration >= EXPLORATION_THRESHOLD:
        return True, f"""EFFICIENCY TIP: You've made {total_exploration} exploration calls (Read/Grep).

Instead of reading files one by one, use scribe to get complete context:

  # Get a function and ALL its dependencies in one call:
  scribe --covering-set "path/to/file.py:function_name" --stdout

  # Get all code related to current git changes:
  scribe --covering-set-diff --stdout

  # Get prioritized context for a directory:
  scribe --token-target 8000 path/to/dir --stdout

Scribe returns the complete dependency cone - the target plus all types,
functions, and constants it uses. This is more efficient than iterative discovery."""

    # Check for patterns that suggest exploring dependencies
    if tool_name == "Grep":
        pattern = tool_input.get("pattern", "")
        # Looking for imports, definitions, or references
        if any(kw in pattern.lower() for kw in ["import", "from", "class ", "def ", "function"]):
            return True, f"""TIP: Looking for "{pattern}"?

Use scribe to get the complete dependency graph:
  scribe --covering-set "file.py:entity_name" --stdout

This returns the entity AND all its dependencies in one call."""

    return False, ""


def main():
    # Read hook input from stdin
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    session_id = data.get("session_id", "unknown")
    tool_name = data.get("tool_name", "")
    hook_event = data.get("hook_event_name", "")
    tool_input = data.get("tool_input", {})

    # Only process PreToolUse for Read/Grep
    if hook_event != "PreToolUse" or tool_name not in ("Read", "Grep"):
        sys.exit(0)

    # Get and update exploration count
    counts = get_exploration_count(session_id)

    if tool_name == "Read":
        counts["read_count"] = counts.get("read_count", 0) + 1
    elif tool_name == "Grep":
        counts["grep_count"] = counts.get("grep_count", 0) + 1

    # Check if we should remind
    should_remind_user, reminder = should_remind(tool_name, tool_input, counts)

    if should_remind_user:
        counts["reminded"] = True
        save_exploration_count(session_id, counts)

        # Output reminder to stderr and exit with code 2 to show it to Claude
        # But allow the tool to proceed (don't block)
        # Actually, exit 2 blocks - so we use a different approach
        # We'll output JSON that allows but includes a reason
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "permissionDecisionReason": reminder
            }
        }
        print(json.dumps(output))
        sys.exit(0)

    save_exploration_count(session_id, counts)
    sys.exit(0)


if __name__ == "__main__":
    main()
