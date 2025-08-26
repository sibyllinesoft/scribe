# Code Style and Conventions

## Style Analysis from Existing Code

### Python Style
- **Type Hints**: Uses `from __future__ import annotations` for forward references
- **Typing**: Uses modern type hints including `List[str]`, `str | None` union syntax
- **Dataclasses**: Uses `@dataclass` for structured data (`RenderDecision`, `FileInfo`)
- **Docstrings**: Minimal docstrings present, follows standard format
- **Error Handling**: Uses try/except blocks with appropriate exception handling
- **File Operations**: Uses `pathlib.Path` instead of string paths
- **Subprocess**: Uses `subprocess.run` with proper argument handling

### Code Organization
- **Imports**: Organized with stdlib imports first, then external dependencies
- **Constants**: ALL_CAPS naming for module-level constants
- **Functions**: Snake_case naming, clear single-purpose functions
- **Classes**: PascalCase for class names
- **Variables**: Snake_case for variables and methods

### Project Structure Patterns
- Single main module approach (`rendergit.py`)
- Entry point defined in `pyproject.toml` as `rendergit:main`
- Minimal external dependencies
- Clean separation of concerns (git operations, file processing, HTML generation)

### Conventions to Follow for PackRepo
- Maintain the clean, minimal dependency approach
- Use dataclasses for structured data representations
- Implement proper type hints throughout
- Use pathlib for all file operations
- Keep functions focused and single-purpose
- Use descriptive variable names
- Implement proper error handling with meaningful messages