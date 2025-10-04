# Scribe HTML Report Templates

This directory contains the external HTML templates used by the Scribe CLI for generating beautiful repository analysis reports.

## Templates

### `report.html`
The main HTML template for repository analysis reports. Features:
- Professional dark theme with CSS variables
- Responsive design with mobile support  
- Beautiful glassmorphism header with blur effects
- Lucide icons for file types and UI elements
- Interactive hover effects and transitions
- Syntax-highlighted code display
- Table of contents with anchor links
- File metadata display (size, tokens, importance score)
- Performance optimizations (max-height with scrolling)

## Template Variables

The template uses Handlebars syntax with the following variables:

### Global Variables
- `{{repository_name}}` - Name of the repository being analyzed
- `{{algorithm}}` - Selection algorithm used (e.g., "V5Integrated")
- `{{generated_time}}` - Timestamp when report was generated
- `{{selection_time_ms}}` - Time taken for file selection in milliseconds
- `{{total_files}}` - Number of files selected
- `{{total_tokens}}` - Formatted total token count
- `{{total_size}}` - Formatted total file size
- `{{coverage_percentage}}` - Coverage percentage (formatted)

### File Arrays
- `{{#each files}}` - Iterates over selected files with:
  - `{{relative_path}}` - HTML-escaped relative file path
  - `{{content}}` - HTML-escaped file content
  - `{{size}}` - Formatted file size
  - `{{estimated_tokens}}` - Formatted token count for file
  - `{{importance_score}}` - Formatted importance score (0.00-1.00)
  - `{{icon}}` - Lucide icon name for file type

### Custom Helpers
- `{{add @index 1}}` - Adds 1 to the current array index (for 1-based IDs)

## Customization

To customize the template:

1. Edit `report.html` directly
2. Modify CSS variables in the `:root` section for colors and styling
3. Update HTML structure as needed
4. Rebuild the CLI: `cargo build --package scribe-analyzer --bin scribe`

The template is embedded at compile time using `include_str!()`, so changes require rebuilding the binary.

## File Icons

File icons are automatically selected based on file extension and name patterns:
- Programming languages: `file-code`
- Images: `image`
- Documentation: `file-text`, `book-open` (README)
- Configuration: `settings`, `list` (YAML)
- Data: `braces` (JSON), `database` (SQL)
- Archives: `archive`
- Scripts: `terminal`
- And many more...

## Browser Compatibility

The template is designed to work in modern browsers with support for:
- CSS Grid and Flexbox
- CSS Variables (Custom Properties)
- CSS `backdrop-filter` for glassmorphism effects
- Modern viewport units and responsive design