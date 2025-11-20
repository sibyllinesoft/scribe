# Scribe HTML Report Templates

This directory contains the external HTML templates used by the Scribe CLI for generating beautiful repository analysis reports.

## Templates

### `report_cdn.html` (Default - Recommended)
**The current default template** - Uses CDN links for minimal output size. Features:
- **60% smaller file size** (8KB vs 20KB template)
- **No bundled JavaScript** - Uses CDN for highlight.js
- Professional dark theme with CSS variables
- Responsive design with mobile support
- Syntax-highlighted code display via highlight.js CDN
- File metadata display (size, tokens, importance score)
- Expand/collapse file sections
- Clean, minimal design optimized for performance

**CDN Dependencies:**
- highlight.js (CSS + JS) from cdnjs.cloudflare.com

### `report_bundled.html` (Legacy)
The older template with all assets embedded. Features:
- Self-contained (no external dependencies)
- React-based file tree visualization
- Larger file size due to bundled assets (20KB template + 268KB JS bundle)
- Full offline functionality

### `report.html`
Alternative template with different styling approach.

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