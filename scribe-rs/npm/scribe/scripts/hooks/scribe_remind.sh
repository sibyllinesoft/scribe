#!/usr/bin/env bash
#
# Scribe reminder hook for Claude Code
# Warns about Read/Grep on code files but allows them
#
# Philosophy: Multiple small focused slices > few large dumps
#

# Don't use set -e as grep may return non-zero on some inputs
# set -e

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

# Helper to output allow with context
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

# Helper to output plain allow
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

# Short nudge with surgical guidance
NUDGE="Tip: Use scribe for surgical slices: scribe --covering-set 'file:func' --max-depth 1 --token-target 800 --stdout"

# Check Grep tool - nudge on code files
if [ "$TOOL_NAME" = "Grep" ]; then
    GREP_PATH=$(echo "$INPUT" | grep -o '"path"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
    OUTPUT_MODE=$(echo "$INPUT" | grep -o '"output_mode"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"output_mode"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')

    # Allow file discovery silently
    if [ "$OUTPUT_MODE" = "files_with_matches" ]; then
        allow
    fi

    if [ -n "$GREP_PATH" ]; then
        if echo "$GREP_PATH" | grep -qE "$CODE_EXTS"; then
            allow_with_context "$NUDGE"
        fi
    fi
fi

# Check Read tool - nudge on code files
if [ "$TOOL_NAME" = "Read" ]; then
    FILE_PATH=$(echo "$INPUT" | grep -o '"file_path"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')

    if [ -n "$FILE_PATH" ]; then
        if echo "$FILE_PATH" | grep -qE "$CODE_EXTS"; then
            allow_with_context "$NUDGE"
        fi
    fi
fi

# Allow everything else silently
allow
