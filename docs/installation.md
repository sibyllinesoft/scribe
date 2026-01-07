# Installation

Scribe can be installed via npm or Cargo.

## npm (Recommended)

The npm package includes pre-built binaries for all major platforms.

```bash
# Install globally
npm install -g @sibyllinesoft/scribe

# Verify installation
scribe --version
```

You can also use npx to run without installing:

```bash
npx @sibyllinesoft/scribe --help
```

### Supported Platforms

| Platform | Architecture |
|----------|-------------|
| macOS | arm64 (Apple Silicon) |
| macOS | x64 (Intel) |
| Linux | x64 |
| Linux | arm64 |
| Windows | x64 |

## Cargo

Install from crates.io or build from source.

### From crates.io

```bash
cargo install scribe-cli
```

### From Source

```bash
git clone https://github.com/sibyllinesoft/scribe
cd scribe
cargo install --path scribe-rs --locked
```

### Build Requirements

- Rust 1.75 or later
- A C compiler (for tree-sitter grammars)

## Verifying Installation

```bash
# Check version
scribe --version

# View help
scribe --help

# Test on a repository
cd your-project
scribe --covering-set "src/main.rs" --stdout
```

## Updating

### npm

```bash
npm update -g @sibyllinesoft/scribe
```

### Cargo

```bash
cargo install scribe-cli --force
```

## Troubleshooting

### Command Not Found

Ensure the installation directory is in your PATH:

- **npm**: Usually `~/.npm-global/bin` or similar
- **Cargo**: `~/.cargo/bin`

### Permission Errors (npm)

If you get permission errors with global npm install:

```bash
# Option 1: Use a different global directory
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
export PATH=~/.npm-global/bin:$PATH

# Option 2: Use npx instead
npx @sibyllinesoft/scribe --help
```

### Build Errors (Cargo)

If building from source fails, ensure you have:

1. Rust toolchain installed (`rustup` recommended)
2. A C compiler (gcc, clang, or MSVC)
3. Development libraries for your platform
