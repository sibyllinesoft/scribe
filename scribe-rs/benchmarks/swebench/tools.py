"""Tool definitions for SWE-bench benchmark.

Defines two tool sets:
- STANDARD_TOOLS: Basic tools without scribe
- SCRIBE_TOOLS: Standard tools + scribe for code understanding
"""

# Standard tools available to all agents
STANDARD_TOOLS = [
    {
        "name": "bash",
        "description": """Execute a bash command in the repository. Use this to:
- Run tests: `python -m pytest tests/`
- Search for files: `find . -name "*.py" | head -20`
- Check file contents: `cat path/to/file.py`
- Make edits: Use sed, or write with heredoc

The repository is at /repo. Always cd /repo first if needed.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                }
            },
            "required": ["command"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file. Returns the file content with line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (relative to /repo)"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Start line (1-indexed, default: 1)"
                },
                "end_line": {
                    "type": "integer",
                    "description": "End line (default: entire file)"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file, creating or overwriting it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (relative to /repo)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "search_files",
        "description": "Search for a pattern in files. Returns matching file paths and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files (default: *.py)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 50)"
                }
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "edit_file",
        "description": "Edit a file by replacing text. Use this for targeted edits.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (relative to /repo)"
                },
                "old_text": {
                    "type": "string",
                    "description": "Text to find and replace"
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text"
                }
            },
            "required": ["path", "old_text", "new_text"]
        }
    },
    {
        "name": "submit_patch",
        "description": "Submit your solution. Call this when you've fixed the issue.",
        "input_schema": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "Brief explanation of your fix"
                }
            },
            "required": ["explanation"]
        }
    },
]

# Scribe-enhanced tools - adds scribe on top of standard tools
SCRIBE_TOOL = {
    "name": "scribe",
    "description": """Get a function/class and all its dependencies in a single call.

Use this to understand code before making changes. Returns the target entity
plus all types, functions, and constants it depends on.

Examples:
- scribe("src/utils.py:validate_input") - Get function and dependencies
- scribe("src/models.py:UserModel") - Get class and dependencies

This is much more efficient than iteratively reading files to understand code.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query in format 'path/to/file.py:entity_name'"
            }
        },
        "required": ["query"]
    }
}

# Full tool set with scribe
SCRIBE_TOOLS = STANDARD_TOOLS + [SCRIBE_TOOL]


def get_tool_names(tools: list) -> list[str]:
    """Get list of tool names from tool definitions."""
    return [t["name"] for t in tools]
