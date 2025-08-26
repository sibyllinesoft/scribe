# Suggested Development Commands

## Environment Setup
```bash
# Set up development environment
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies (preferred with uv)
uv tool install git+https://github.com/karpathy/rendergit

# Alternative pip install for development
pip install -e .
```

## Running the Main Application
```bash
# Basic usage
rendergit https://github.com/karpathy/nanogpt

# With manual pip install
python rendergit.py https://github.com/karpathy/nanogpt
```

## Development Workflow
```bash
# Run main module directly for testing
python rendergit.py --help

# Check Python version compatibility
python --version  # Should be >=3.10
```

## Testing and Quality Assurance
Based on the TODO.md requirements, the following commands will be needed:

```bash
# Set up PackRepo environment (when implemented)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run environment validation
python -c "import sys,platform,hashlib; open('env_manifest.txt','w').write(f'{platform.platform()}\\n{sys.version}\\n')"

# Run baseline tests
python cli/packrepo.py --mode baseline --budget 120000 --tokenizer cl100k --repos {{DATASET_REPOS}} --out logs/V0/

# Run deterministic mode tests
python cli/packrepo.py --mode comprehension --budget 120000 --tokenizer cl100k --variant V1 --no-llm --repos {{DATASET_REPOS}} --out logs/V1/
```

## System Utilities (Linux)
```bash
# File operations
ls -la                    # List files with details
find . -name "*.py"       # Find Python files
grep -r "pattern" .       # Search in files

# Git operations
git status               # Check repository status
git add .               # Stage changes
git commit -m "message" # Commit changes
git log --oneline       # View commit history

# Process management
ps aux                  # List running processes
top                     # Monitor system resources
```

## Package Management
```bash
# Using uv (preferred)
uv add dependency-name
uv lock

# Using pip (fallback)
pip install package-name
pip freeze > requirements.txt
pip install -r requirements.txt
```