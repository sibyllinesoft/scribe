# Scribe Interactive Bundle Editor

The Scribe Interactive Bundle Editor provides a web-based interface for fine-tuning repository bundles by allowing you to interactively add or remove files from the bundle.

## Overview

Instead of generating a static bundle, the editor shows you:
- **All files** in the repository organized by category
- **Interactive checkboxes** to include/exclude files
- **Live statistics** showing token count and file counts as you make changes
- **Export options** to save your customized bundle

## Quick Start

### Launch Editor Mode

```bash
# Launch editor for current directory
python scribe.py --editor

# Launch with specific token target and open in browser
python scribe.py --editor --token-target 20000 --open

# Use traditional filtering as starting point
python scribe.py --editor --force-traditional --max-bytes 100000

# Specify custom output filename
python scribe.py --editor -o my-project-editor.html
```

### Standalone Editor

You can also run the editor directly:

```bash
# Basic usage
python scribe_editor.py /path/to/repo

# With custom settings
python scribe_editor.py /path/to/repo --token-target 15000 --open
```

## Interface Features

### File Categories

Files are automatically organized into expandable categories:

- **📁 Included** - Files currently in the bundle (always open)
- **⏰ Didn't Fit** - Files excluded due to token/size constraints (open by default)
- **🖼️ Binary** - Binary files like images, executables (collapsed)
- **📊 Too Large** - Files exceeding size limits (collapsed)
- **👁️ Ignored** - Files ignored by .gitignore (collapsed)

### Interactive Controls

- **Click any file** to add/remove it from the bundle
- **Format selector** - Choose HTML, CXML, or Repomix export format
- **Live statistics** update automatically as you make changes
- **Bulk operations**: Select/deselect all visible files
- **Expandable categories** to manage hundreds of files efficiently

### Export Options

- **Export Bundle** - Export in your choice of format (HTML, CXML, or Repomix)
- **Export Config** - Generate scribe.config.json for future use  
- **Save JSON** - Export current selection as JSON file for analysis

## File Categories Explained

### Included Files
These are files currently selected for the bundle. They contribute to the token count and will be included in any exported bundle.

### Didn't Fit
These are files that would normally be included but were excluded due to:
- Token budget constraints (when using intelligent selection)
- Manual exclusion in previous configurations
- Size limitations

**This category is open by default** since these are likely the files you want to review for inclusion.

### Binary Files
Files detected as binary (images, executables, compiled files). These are usually excluded from LLM bundles but can be included if needed for context.

### Too Large Files  
Files exceeding the `--max-bytes` threshold. You can include these if you increase the size limit or if they're critical to understanding the codebase.

### Ignored Files
Files excluded by .gitignore patterns. These are typically build artifacts, dependencies, or temporary files.

## Multi-Format Export

The bundle editor now supports exporting your customized file selection in any of Scribe's output formats:

### Export Formats

- **HTML** - Interactive web page with syntax highlighting and navigation
- **CXML** - Optimized XML format for LLM consumption  
- **Repomix** - Plain text format compatible with Repomix tools

### Export Workflow

1. **Select Format** - Choose your desired output format from the dropdown
2. **Click Export Bundle** - Downloads export instructions and file list
3. **Follow Instructions** - Use the provided commands to generate your bundle

The export process provides multiple options:
- **Direct Export** - Command with explicit file list
- **Config-Based** - Export config first, then run standard scribe command
- **Token-Based** - Use the current token estimate as a target

### Export Instructions

When you click "Export Bundle", you'll get a JSON file containing:
- Step-by-step command instructions
- Complete file list for your selection
- Token estimates and statistics
- Multiple command options for flexibility

Example export file structure:
```json
{
  "instructions": {
    "format": "html",
    "description": "Instructions to export your customized bundle in HTML format",
    "steps": [
      "1. Save this file list to use with scribe",
      "2. Run one of the following commands:",
      "python scribe.py \"/path/to/repo\" --output-format html --explicit-includes \"file1.py\" \"file2.py\"",
      "..."
    ]
  },
  "file_list": ["src/main.py", "README.md", "..."],
  "export_data": { ... }
}
```

## Usage Patterns

### Fine-Tuning AI Code Analysis

1. Start with intelligent selection: `scribe.py --editor --token-target 25000`
2. Review files in the "Didn't Fit" category
3. Add important architecture files, documentation, tests
4. Remove generated files or less relevant utilities
5. Export the refined bundle

### Exploring Large Codebases

1. Use traditional mode: `scribe.py --editor --force-traditional`
2. Expand categories one by one
3. Focus on core business logic first
4. Add supporting files as needed
5. Monitor token count to stay within LLM limits

### Configuration Management

1. Create your ideal bundle in the editor
2. Export as configuration file
3. Use the config file for consistent future runs:
   ```bash
   # The exported config file will be used automatically
   scribe.py /path/to/repo
   ```

## Advanced Features

### Smart Defaults

- **Intelligent selection** automatically picks the most relevant files
- **"Didn't fit" category** is expanded to help you find missing important files
- **Binary and ignored files** are collapsed to reduce noise

### Performance Optimization

- **Lazy loading** - File contents are only loaded when needed
- **Efficient categorization** - Files are grouped by exclusion reason
- **Responsive design** - Works well with repositories containing thousands of files

### Integration with Main Scribe

The editor integrates seamlessly with the main `scribe.py` command:
- All existing flags (`--token-target`, `--max-bytes`, etc.) work with `--editor`
- Configuration files created by the editor work with regular scribe runs
- Same intelligent algorithms determine initial file selection

## Technical Details

### Bundle State Format

The editor exports bundle configurations in JSON format:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "included_files": [
    "src/main.py", 
    "src/utils.py",
    "README.md"
  ],
  "stats": {
    "file_count": 3,
    "token_estimate": 15420,
    "total_size": 45123
  }
}
```

### Configuration File Format

Exported configurations follow the scribe.config.json format:

```json
{
  "version": "1.0",
  "include_patterns": [],
  "exclude_patterns": [],
  "explicit_includes": ["src/main.py", "README.md"],
  "explicit_excludes": ["tests/test_large.py"],
  "max_tokens": 20000,
  "created": "2024-01-15T10:30:00Z"
}
```

## Keyboard Shortcuts

- **Space** - Toggle selected file (when file is focused)
- **Arrow Keys** - Navigate between files
- **Enter** - Expand/collapse category
- **Ctrl+A** - Select all visible files
- **Ctrl+D** - Deselect all visible files

## Best Practices

### For Code Review Preparation
1. Use intelligent selection as starting point
2. Add test files and documentation from "Didn't Fit"
3. Remove generated/minified files
4. Include configuration files that affect behavior

### For LLM Training Data
1. Start with traditional filtering to see all code
2. Remove examples and demo code
3. Include comprehensive test suites
4. Add architectural documentation

### For Architecture Analysis  
1. Focus on core business logic
2. Include interface definitions and schemas
3. Add architectural decision records (ADRs)
4. Include database migration files

## Troubleshooting

### Editor Shows 0 Excluded Files
This happens when all discovered files are already selected. Try:
- Use a smaller `--token-target` to see more excluded files
- Use `--force-traditional` to see all files before intelligent filtering

### Performance Issues with Large Repositories
- Categories are collapsed by default for performance
- Only expand categories you need to review
- Use the browser's search function (Ctrl+F) to find specific files

### JavaScript Not Working
- Ensure you have internet access (loads Lucide icons from CDN)
- Try refreshing the page
- Check browser console for errors

## Examples

### Preparing Code for Claude Analysis

```bash
# Create a focused bundle for a specific feature
scribe.py --editor --query-hint "authentication" --token-target 30000 --open

# Review and adjust files in the "Didn't Fit" category
# Add relevant tests and documentation  
# Export the refined bundle for Claude
```

### Creating Reusable Configuration

```bash
# Set up perfect bundle for your project
scribe.py --editor --token-target 50000

# Export configuration
# Use "Export Config" button to save scribe.config.json

# Future runs automatically use your configuration
scribe.py  # Will use the saved configuration
```

### Handling Different Project Types

```bash
# Web frontend project - include assets
scribe.py --editor --force-traditional
# Manually add important CSS/HTML from binary category

# API backend - focus on business logic  
scribe.py --editor --entry-points src/main.py api/routes.py
# Review and include relevant tests

# Library project - include examples
scribe.py --editor --token-target 15000
# Add example files from "Didn't Fit" category
```