# Output Formats

Scribe supports multiple output formats for different use cases.

## Format Overview

| Format | Use Case | Metadata | Structured |
|--------|----------|----------|------------|
| XML | AI agents | Full | Yes |
| JSON | Programmatic | Full | Yes |
| Markdown | Documentation | Partial | No |
| Text | Human reading | Minimal | No |
| HTML | Interactive | Full | Yes |

## XML Format

**Recommended for AI agents.** Structured with full metadata.

```bash
scribe --covering-set "main.rs:main" --output-format xml --stdout
```

### Covering Set Output

```xml
<?xml version="1.0" encoding="UTF-8"?>
<covering_set>
  <target>
    <path>src/main.rs</path>
    <entity>main</entity>
    <entity_type>function</entity_type>
  </target>

  <files count="4">
    <file>
      <path>src/main.rs</path>
      <distance>0</distance>
      <reason>TargetFile</reason>
      <language>Rust</language>
      <entities>
        <entity type="function" lines="10-25">main</entity>
      </entities>
      <content><![CDATA[
fn main() {
    // function content
}
]]></content>
    </file>

    <file>
      <path>src/config.rs</path>
      <distance>1</distance>
      <reason>DirectDependency</reason>
      <language>Rust</language>
      <entities>
        <entity type="struct" lines="5-15">Config</entity>
        <entity type="function" lines="17-30">load</entity>
      </entities>
      <content><![CDATA[
pub struct Config { ... }
pub fn load() -> Config { ... }
]]></content>
    </file>
  </files>

  <statistics>
    <files_examined>142</files_examined>
    <files_selected>4</files_selected>
    <entities_selected>7</entities_selected>
    <max_depth_reached>2</max_depth_reached>
    <tokens_used>2450</tokens_used>
  </statistics>
</covering_set>
```

### Repository Bundle Output

```xml
<?xml version="1.0" encoding="UTF-8"?>
<repository_bundle>
  <metadata>
    <name>my-project</name>
    <path>/path/to/repo</path>
    <generated_at>2025-01-07T12:00:00Z</generated_at>
    <token_budget>100000</token_budget>
    <tokens_used>85432</tokens_used>
  </metadata>

  <files count="25">
    <file>
      <path>src/lib.rs</path>
      <language>Rust</language>
      <centrality>0.456</centrality>
      <position>HEAD</position>
      <render_mode>FULL</render_mode>
      <content><![CDATA[...]]></content>
    </file>
    <!-- more files -->
  </files>
</repository_bundle>
```

## JSON Format

**For programmatic consumption.** Full metadata as JSON.

```bash
scribe --covering-set "main.rs:main" --output-format json --stdout
```

### Covering Set Output

```json
{
  "target": {
    "path": "src/main.rs",
    "entity": "main",
    "entity_type": "function"
  },
  "files": [
    {
      "path": "src/main.rs",
      "distance": 0,
      "reason": "TargetFile",
      "language": "Rust",
      "entities": [
        {
          "name": "main",
          "type": "function",
          "start_line": 10,
          "end_line": 25
        }
      ],
      "content": "fn main() {\n    // ...\n}"
    }
  ],
  "statistics": {
    "files_examined": 142,
    "files_selected": 4,
    "entities_selected": 7,
    "max_depth_reached": 2,
    "tokens_used": 2450
  }
}
```

## Markdown Format

**For documentation and human reading.**

```bash
scribe --style markdown --output bundle.md
```

### Output Structure

```markdown
# Repository Bundle: my-project

Generated: 2025-01-07 12:00:00
Token Budget: 100,000 | Used: 85,432 (85.4%)

## File Index

| File | Language | Lines | Reason |
|------|----------|-------|--------|
| src/lib.rs | Rust | 150 | High Centrality |
| src/main.rs | Rust | 45 | Entry Point |
| ... | ... | ... | ... |

---

## src/lib.rs

**Language:** Rust | **Centrality:** 0.456 | **Position:** HEAD

```rust
//! Main library module

pub mod config;
pub mod api;
// ...
```

---

## src/main.rs

**Language:** Rust | **Centrality:** 0.234 | **Position:** HEAD

```rust
fn main() {
    // ...
}
```
```

## Text Format

**Simple, human-readable output.**

```bash
scribe --covering-set "main.rs:main" --output-format text --stdout
```

### Output

```
Covering Set for: src/main.rs:main

Files (4):
  [0] src/main.rs (target)
  [1] src/config.rs (direct dependency)
  [1] src/utils.rs (direct dependency)
  [2] src/types.rs (transitive dependency)

--- src/main.rs ---
fn main() {
    let config = config::load();
    // ...
}

--- src/config.rs ---
pub struct Config { ... }
pub fn load() -> Config { ... }

--- src/utils.rs ---
pub fn format_output(data: &Data) -> String { ... }

--- src/types.rs ---
pub struct Data { ... }

Statistics:
  Files examined: 142
  Files selected: 4
  Max depth: 2
  Tokens: 2,450
```

## HTML Format

**Interactive bundle editor.**

```bash
scribe --style html --editor --output bundle.html
```

### Features

- File tree navigation
- Syntax highlighting
- Search and filter
- Expand/collapse sections
- Copy to clipboard
- Token usage visualization

### Static HTML

Without `--editor`, generates a static HTML document:

```bash
scribe --style html --output bundle.html
```

## Repomix-Compatible Format

**Compatible with Repomix consumers.**

```bash
scribe --style repomix --output bundle.xml
```

Follows Repomix XML schema for tool compatibility.

## Choosing a Format

| Scenario | Recommended Format |
|----------|-------------------|
| AI agent consumption | XML |
| Programmatic processing | JSON |
| Documentation | Markdown |
| Quick inspection | Text |
| Interactive review | HTML |
| Tool compatibility | Repomix |

## Format-Specific Options

### XML

```bash
--output-format xml
--include-metadata    # Include full metadata (default)
--no-metadata         # Minimal output
```

### JSON

```bash
--output-format json
--pretty              # Pretty-print JSON (default)
--compact             # Minified JSON
```

### Markdown

```bash
--style markdown
--include-toc         # Include table of contents
--no-toc              # Skip table of contents
```

### HTML

```bash
--style html
--editor              # Include interactive editor
--theme dark          # Dark theme
--theme light         # Light theme
```

## See Also

- [CLI Reference](cli-reference.md) - All output options
- [Covering Sets](covering-sets.md) - What gets included
