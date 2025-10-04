# Scribe Frontend Bundle

This directory contains the bundling setup for React Arborist components used in Scribe HTML reports.

## Overview

The bundle includes:
- React 18
- ReactDOM 18
- React Arborist 3.4.0
- Lucide React icons

## Build Process

### Quick Build
```bash
# From the scribe-rs directory
./build_frontend.sh
```

### Manual Build
```bash
cd frontend
npm install
npm run build
```

## Output

The build process creates:
- `../templates/assets/scribe-tree-bundle.js` - Self-contained bundle (~260KB)
- `../templates/assets/scribe-tree-bundle.js.LICENSE.txt` - License information

## Usage

### In Templates

Use the bundled template:
```html
<!-- templates/report_bundled.html -->
<script src="assets/scribe-tree-bundle.js"></script>
<script>
  const fileTree = new window.ScribeFileTree();
  fileTree.renderTree('container-id', fileData);
</script>
```

### API

The bundle exposes:
- `window.ScribeFileTree` - Main tree component class
- `window.React` - React library
- `window.ReactDOM` - ReactDOM with createRoot
- `window.LucideReact` - Lucide React icons

## Development

### Project Structure
```
frontend/
├── src/
│   └── index.js          # Main bundle entry point
├── package.json          # Dependencies and scripts
├── webpack.config.js     # Webpack configuration
└── README.md            # This file
```

### Dependencies

**Runtime:**
- react ^18.2.0
- react-dom ^18.2.0  
- react-arborist ^3.4.0
- lucide-react ^0.294.0

**Build:**
- webpack ^5.90.3
- babel-loader ^9.1.3
- @babel/preset-react ^7.23.3

## Testing

```bash
# install dependencies once (bun or npm both work)
bun install

# run the fast unit/mocked suite
bun test

# run the browser-level e2e check for the bundled tree
bunx playwright install
bunx playwright test
```

For a manual smoke test you can still open `templates/test_bundle.html` in a browser to check the UI.

## Integration with Rust

The Rust application should:
1. Generate HTML using `templates/report_bundled.html`
2. Ensure the `assets/` directory is accessible to the generated HTML
3. Pass file data to the JavaScript initialization code
