#!/usr/bin/env python3
"""
Hook to enforce scribe usage for code exploration.

This hook:
1. BLOCKS Read on code files - redirects to scribe command
2. BLOCKS Grep on code files - redirects to scribe command
3. NUDGES on all other Read/Grep with short reminder
4. BLOCKS Bash when piping scribe output (head/tail/grep)
"""

import json
import sys
import os
import re

# Code file extensions that scribe supports
CODE_EXTENSIONS = {
    ".py", ".pyi",  # Python
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",  # JavaScript/TypeScript
    ".go",  # Go
    ".rs",  # Rust
    ".java", ".kt", ".kts",  # JVM
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",  # C/C++
    ".rb",  # Ruby
    ".php",  # PHP
    ".swift",  # Swift
    ".scala",  # Scala
    ".cs",  # C#
    ".lua",  # Lua
    ".r", ".R",  # R
    ".jl",  # Julia
    ".ex", ".exs",  # Elixir
    ".erl", ".hrl",  # Erlang
    ".hs",  # Haskell
    ".ml", ".mli",  # OCaml
    ".clj", ".cljs", ".cljc",  # Clojure
    ".elm",  # Elm
    ".vue", ".svelte",  # Frontend frameworks
    ".sol",  # Solidity
}

# Non-code files that Read is allowed without nudge
ALLOWED_EXTENSIONS = {
    ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".xml", ".html", ".css", ".scss", ".less",
    ".env", ".gitignore", ".dockerignore",
    ".lock", ".sum",  # Lock files
}


def get_session_file(session_id: str) -> str:
    return f"/tmp/claude_scribe_hook_{session_id}.json"


def get_session_state(session_id: str) -> dict:
    try:
        with open(get_session_file(session_id)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"grep_count": 0, "reminded": False, "allowed_reads": []}


def save_session_state(session_id: str, state: dict):
    with open(get_session_file(session_id), "w") as f:
        json.dump(state, f)


def is_code_file(file_path: str) -> bool:
    """Check if file is a code file that scribe can handle."""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in CODE_EXTENSIONS


def is_allowed_file(file_path: str) -> bool:
    """Check if file is explicitly allowed for Read (non-code)."""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def get_scribe_command_for_file(file_path: str) -> str:
    """Generate a short scribe command suggestion."""
    return f'scribe --covering-set "{file_path}:ENTITY" --stdout'


# Short nudge message for non-code files
NUDGE_MESSAGE = "Tip: scribe --covering-set may be faster for code exploration."


def check_bash_for_scribe_pipe(command: str) -> tuple[bool, str]:
    """Check if bash command pipes scribe output."""
    # Patterns that indicate piping scribe output
    pipe_patterns = [
        r'scribe\s+.*\|\s*head',
        r'scribe\s+.*\|\s*tail',
        r'scribe\s+.*\|\s*grep',
        r'scribe\s+.*\|\s*awk',
        r'scribe\s+.*\|\s*sed',
        r'scribe\s+.*\|\s*cut',
        r'scribe\s+.*\|\s*wc',
        r'scribe\s+.*\|\s*less',
        r'scribe\s+.*\|\s*more',
        r'scribe\s+.*>\s*/dev/null',
    ]

    for pattern in pipe_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return True, "BLOCKED: Don't pipe scribe output. Use --token-target to limit size."

    return False, ""


def handle_read(file_path: str, session_id: str, state: dict) -> tuple[str, bool]:
    """Handle Read tool - block code files, nudge on others."""

    # Always allow config/doc files without nudge
    if is_allowed_file(file_path):
        return "", True

    # Check if this specific file was already allowed (e.g., after scribe)
    if file_path in state.get("allowed_reads", []):
        return "", True

    # Block code files - redirect to scribe
    if is_code_file(file_path):
        scribe_cmd = get_scribe_command_for_file(file_path)
        return f"BLOCKED: Use {scribe_cmd} instead.", False

    # Unknown extension - allow with short nudge
    return NUDGE_MESSAGE, True


def handle_grep(tool_input: dict, state: dict) -> tuple[str, bool]:
    """Handle Grep tool - block on code files, nudge on others."""
    pattern = tool_input.get("pattern", "")
    path = tool_input.get("path", "")
    glob_pattern = tool_input.get("glob", "")

    # Check if targeting code files via path or glob
    targets_code = False
    if path:
        targets_code = is_code_file(path)
    if glob_pattern:
        # Check if glob targets code extensions
        for ext in CODE_EXTENSIONS:
            if ext in glob_pattern or f"*{ext}" in glob_pattern:
                targets_code = True
                break

    # Block grep on code files
    if targets_code:
        return "BLOCKED: Use scribe --covering-set for code search.", False

    # Allow file discovery (files_with_matches mode)
    output_mode = tool_input.get("output_mode", "")
    if output_mode == "files_with_matches":
        return "", True  # Allow without nudge - this is legitimate discovery

    # Nudge on other grep usage
    return NUDGE_MESSAGE, True


def handle_bash(command: str) -> tuple[str, bool]:
    """Handle Bash tool - block piped scribe commands."""
    is_piped, message = check_bash_for_scribe_pipe(command)
    if is_piped:
        return message, False  # Block
    return "", True


def main():
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    session_id = data.get("session_id", "unknown")
    tool_name = data.get("tool_name", "")
    hook_event = data.get("hook_event_name", "")
    tool_input = data.get("tool_input", {})

    # Only process PreToolUse
    if hook_event != "PreToolUse":
        sys.exit(0)

    state = get_session_state(session_id)
    message = ""
    allow = True

    if tool_name == "Read":
        file_path = tool_input.get("file_path", "")
        message, allow = handle_read(file_path, session_id, state)

    elif tool_name == "Grep":
        message, allow = handle_grep(tool_input, state)

    elif tool_name == "Bash":
        command = tool_input.get("command", "")
        message, allow = handle_bash(command)

    # Save state
    save_session_state(session_id, state)

    # Output decision
    if message:
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow" if allow else "deny",
                "permissionDecisionReason": message
            }
        }
        print(json.dumps(output))

    sys.exit(0)


if __name__ == "__main__":
    main()
