#!/usr/bin/env bash
#
# Scribe enforcement hook for Claude Code
# Blocks Read/Grep on code files, redirects to scribe usage
#

set -e

# Read JSON input from stdin
INPUT=$(cat)

# Extract fields using basic parsing (works without jq)
TOOL_NAME=$(echo "$INPUT" | grep -o '"tool_name"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"tool_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
EVENT_NAME=$(echo "$INPUT" | grep -o '"hook_event_name"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"hook_event_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')

# Only handle PreToolUse events
if [ "$EVENT_NAME" != "PreToolUse" ]; then
    exit 0
fi

# Code file extensions
CODE_EXTS="\.py$|\.pyi$|\.js$|\.jsx$|\.ts$|\.tsx$|\.mjs$|\.cjs$|\.go$|\.rs$|\.java$|\.kt$|\.c$|\.h$|\.cpp$|\.hpp$|\.cc$|\.rb$|\.php$|\.swift$|\.scala$|\.cs$|\.fs$|\.ex$|\.exs$|\.erl$|\.hs$|\.ml$|\.mli$|\.clj$|\.cljs$|\.lua$|\.r$|\.jl$|\.zig$|\.nim$|\.cr$|\.v$|\.d$|\.pas$|\.pl$|\.pm$|\.vue$|\.svelte$"

# Helper to output deny decision
deny() {
    local reason="$1"
    cat << EOF
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "$reason"
  }
}
EOF
    exit 0
}

# Helper to output allow decision
allow() {
    cat << EOF
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "allow"
  }
}
EOF
    exit 0
}

# Helper to output allow with context/reminder
allow_with_context() {
    local context="$1"
    cat << EOF
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "allow",
    "additionalContext": "$context"
  }
}
EOF
    exit 0
}

# Short nudge for non-code operations
NUDGE="Tip: scribe --covering-set may be faster for code."

# Config/doc extensions that don't need nudge
CONFIG_EXTS="\.md$|\.txt$|\.json$|\.ya?ml$|\.toml$|\.ini$|\.cfg$|\.xml$|\.html$|\.css$|\.env$|\.gitignore$|\.lock$"

# Check Grep tool - block on code files, nudge on others
if [ "$TOOL_NAME" = "Grep" ]; then
    GREP_PATH=$(echo "$INPUT" | grep -o '"path"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
    OUTPUT_MODE=$(echo "$INPUT" | grep -o '"output_mode"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"output_mode"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')

    # Allow file discovery without nudge
    if [ "$OUTPUT_MODE" = "files_with_matches" ]; then
        allow
    fi

    if [ -n "$GREP_PATH" ]; then
        if echo "$GREP_PATH" | grep -qE "$CODE_EXTS"; then
            deny "BLOCKED: Use scribe --covering-set instead."
        fi
    fi

    # Nudge on other grep
    allow_with_context "$NUDGE"
fi

# Check Read tool - block on code files, nudge on others
if [ "$TOOL_NAME" = "Read" ]; then
    FILE_PATH=$(echo "$INPUT" | grep -o '"file_path"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')

    if [ -n "$FILE_PATH" ]; then
        # Allow config/doc files without nudge
        if echo "$FILE_PATH" | grep -qE "$CONFIG_EXTS"; then
            allow
        fi
        # Block code files
        if echo "$FILE_PATH" | grep -qE "$CODE_EXTS"; then
            deny "BLOCKED: Use scribe --covering-set instead."
        fi
        # Nudge on unknown files
        allow_with_context "$NUDGE"
    fi
fi

# Allow everything else
allow
