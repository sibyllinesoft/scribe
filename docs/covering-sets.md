# Covering Sets

Covering sets are Scribe's core feature: given a target entity, compute the minimal set of code needed to fully understand it.

## What is a Covering Set?

A covering set for a function `f` includes:

1. **The target**: `f` itself
2. **Direct dependencies**: Functions `f` calls, types `f` uses
3. **Transitive dependencies**: Dependencies of dependencies
4. **Configuration**: Constants, configs, and settings `f` reads

```
Target: authenticate_user()
        ├── verify_password()        (direct dependency)
        │   └── PasswordHash         (type dependency)
        ├── create_session()         (direct dependency)
        │   └── Session              (type dependency)
        ├── AuthConfig               (config dependency)
        └── AUTH_TIMEOUT             (constant dependency)
```

## Basic Usage

### Target a Function

```bash
# File:function syntax
scribe --covering-set "src/auth.rs:authenticate_user" --stdout
```

### Target a Class/Type

```bash
# Class with all its methods
scribe --covering-set "api/models.py:User" --stdout
```

### Target a File

```bash
# All public entities in the file
scribe --covering-set "src/lib.rs" --stdout
```

## Controlling Depth

Limit how deep to traverse the dependency graph:

```bash
# Only direct dependencies (depth 1)
scribe --covering-set "main.rs:main" --max-depth 1 --stdout

# Up to 3 levels
scribe --covering-set "main.rs:main" --max-depth 3 --stdout

# Unlimited (default)
scribe --covering-set "main.rs:main" --stdout
```

## Granularity

### Entity Granularity (Default)

Returns only the specific functions/classes needed:

```bash
scribe --covering-set "auth.rs:login" --granularity entity --stdout
```

Output includes:
- `login()` from `auth.rs`
- `verify_password()` from `crypto.rs` (not the whole file)
- `Session` type from `types.rs` (not other types)

### File Granularity

Returns whole files (faster, less precise):

```bash
scribe --covering-set "auth.rs" --granularity file --stdout
```

Output includes:
- All of `auth.rs`
- All of `crypto.rs`
- All of `types.rs`

## Impact Analysis

Include files that depend on your target (dependents):

```bash
scribe --covering-set "utils.rs:format_date" --include-dependents --stdout
```

This answers: "If I change `format_date`, what else might break?"

## Git Diff Covering Set

Get the covering set for your current changes:

```bash
# Analyze uncommitted changes
scribe --covering-set-diff --stdout
```

This is powerful for code review—it returns not just the changed code, but everything needed to understand the changes.

## Example Output (XML)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<covering_set>
  <target>
    <path>src/auth.rs</path>
    <entity>authenticate_user</entity>
    <entity_type>function</entity_type>
  </target>
  <files count="4">
    <file>
      <path>src/auth.rs</path>
      <distance>0</distance>
      <reason>TargetFile</reason>
      <entities>
        <entity type="function">authenticate_user</entity>
      </entities>
      <content><![CDATA[
pub fn authenticate_user(creds: &Credentials) -> Result<Session> {
    let user = lookup_user(&creds.username)?;
    verify_password(&creds.password, &user.password_hash)?;
    create_session(user.id)
}
]]></content>
    </file>
    <file>
      <path>src/crypto.rs</path>
      <distance>1</distance>
      <reason>DirectDependency</reason>
      <entities>
        <entity type="function">verify_password</entity>
        <entity type="type">PasswordHash</entity>
      </entities>
      <content><![CDATA[
pub fn verify_password(input: &str, hash: &PasswordHash) -> Result<()> {
    // verification logic
}

pub struct PasswordHash {
    pub algorithm: Algorithm,
    pub hash: Vec<u8>,
}
]]></content>
    </file>
    <!-- more files... -->
  </files>
  <statistics>
    <files_examined>142</files_examined>
    <files_selected>4</files_selected>
    <entities_selected>7</entities_selected>
    <max_depth_reached>2</max_depth_reached>
  </statistics>
</covering_set>
```

## How It Works

1. **Parse the target file** using tree-sitter AST parsing
2. **Extract the target entity** and its local dependencies
3. **Resolve imports** to find dependency files
4. **Recursively analyze** dependencies up to max depth
5. **Filter by relevance** based on centrality and usage
6. **Extract entities** at the configured granularity
7. **Order by importance** using graph centrality

## Language Support

Covering set analysis works best with these languages:

| Language | Import Resolution | Entity Extraction |
|----------|------------------|-------------------|
| Rust | Full | Full |
| Python | Full | Full |
| TypeScript | Full | Full |
| JavaScript | Full | Full |
| Go | Full | Full |
| Java | Partial | Full |
| C/C++ | Partial | Partial |

## Tips

1. **Start with depth 2-3** for most use cases
2. **Use entity granularity** for precision, file for speed
3. **Include dependents** when planning changes
4. **Combine with git diff** for code review context

## See Also

- [CLI Usage](cli-usage.md) - Full CLI examples
- [Context Positioning](context-positioning.md) - How output is ordered
- [Architecture](architecture.md) - How covering sets are computed
