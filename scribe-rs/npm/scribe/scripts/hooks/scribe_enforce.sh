#!/usr/bin/env bash
#
# Scribe enforcement hook for Claude Code
# Blocks Read/Grep on code files, guides toward surgical scribe usage
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

# Config/doc extensions that don't need scribe
CONFIG_EXTS="\.md$|\.txt$|\.json$|\.ya?ml$|\.toml$|\.ini$|\.cfg$|\.xml$|\.html$|\.css$|\.env$|\.gitignore$|\.lock$"

# Surgical scribe guidance message
GUIDANCE="Use scribe for surgical code slices with dependencies.

PATTERN: Start small, expand if needed
  1. First call: Get just the target function (--max-depth 1 --token-target 800)
  2. If you need more context: Get a specific dependency (--max-depth 1)
  3. Repeat for each piece you need

COMMAND FORMAT:
  scribe --covering-set \"FILE:FUNCTION\" --max-depth 1 --token-target 800 --stdout

EXAMPLES:
  # Get just the handler function
  scribe --covering-set \"api/handler.go:HandleRequest\" --max-depth 1 --token-target 800 --stdout

  # Then get a helper it calls
  scribe --covering-set \"api/validate.go:ValidateInput\" --max-depth 1 --token-target 800 --stdout

KEY PRINCIPLES:
- Use --max-depth 1 for tight focus (only direct dependencies)
- Use --token-target 800 for small slices
- Multiple small calls > one large call
- Target specific functions, never whole files"

# Helper to output deny decision
deny() {
    local reason="$1"
    # Escape special characters for JSON
    reason=$(echo "$reason" | sed 's/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')
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
            deny "BLOCKED: $GUIDANCE"
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
        # Block code files with surgical guidance
        if echo "$FILE_PATH" | grep -qE "$CODE_EXTS"; then
            deny "BLOCKED: $GUIDANCE"
        fi
        # Nudge on unknown files
        allow_with_context "$NUDGE"
    fi
fi

# Check Bash tool for scribe piping anti-pattern
if [ "$TOOL_NAME" = "Bash" ]; then
    COMMAND=$(echo "$INPUT" | grep -o '"command"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"command"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/' || echo "")

    # Detect scribe output being piped to other tools
    if [ -n "$COMMAND" ] && echo "$COMMAND" | grep -qE "scribe.*\|"; then
        PIPE_WARNING="WARNING: Piping scribe output loses context structure.\\n\\nInstead of: scribe ... | head/grep/etc\\n\\nUse scribe's built-in options:\\n  --token-target 800    # Limit output size\\n  --max-depth 1         # Limit dependency depth\\n\\nThese preserve scribe's intelligent context ordering."
        allow_with_context "$PIPE_WARNING"
    fi
fi

# Allow everything else
allow
