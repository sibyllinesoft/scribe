# Scribe Bundle Editor

Scribe can export an interactive HTML report that makes it easy to review the chosen files, tweak the bundle, and hand the result to teammates or downstream tools.

## Generating the Editor

```bash
# Analyse the current repository and emit the interactive HTML view
git status  # optional sanity check
scribe --style html --editor --output bundle.html
```

The CLI writes two things next to the requested `bundle.html` file:

- `bundle.html` – the generated report, including selection statistics and a tree view of the files.
- `assets/scribe-tree-bundle.js` – the lightweight JavaScript bundle that powers the interactive tree.

Open the HTML file in any modern browser to explore the bundle.

## Features

- **File Tree.** Inspect the relative paths included in the bundle and expand nodes to examine their contents.
- **Search.** Use the browser's search to jump to specific files or symbols.
- **Selection Metadata.** The header summarises total size, estimated tokens, and which algorithm produced the bundle.
- **Sharing.** The HTML is self-contained, so it can be committed to docs or sent to collaborators.

The HTML export is generated from Handlebars templates stored in `scribe-rs/templates/`. To customise the layout you can edit those templates and re-run the CLI.

## Editing Workflow

At the moment the HTML editor is read-only—the Rust CLI remains the source of truth for which files are included. To adjust the selection:

1. Modify the CLI flags (include/ignore patterns, token budget, algorithm).
2. Re-run `scribe` to produce a new bundle.
3. Refresh the browser view of `bundle.html`.

Future work could capture client-side changes and feed them back into the CLI, but that plumbing has been removed along with the legacy research harness.

## Troubleshooting

- If the file tree does not expand, make sure `assets/scribe-tree-bundle.js` sits next to the HTML file when you open it.
- When analysing very large repositories you may want to pass `--max-file-size` and `--token-budget` flags to keep the output manageable.
- Use `--verbose` for additional logging during bundle generation.
