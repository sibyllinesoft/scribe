//! File-related types and utilities.
//!
//! Provides comprehensive file metadata structures, language detection,
//! and file classification utilities for the Scribe analysis pipeline.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::error::{Result, ScribeError};

/// Binary file extensions that should typically be excluded from text analysis
pub const BINARY_EXTENSIONS: &[&str] = &[
    // Images
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".ico", ".tiff",
    // Documents
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", // Archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", // Media
    ".mp3", ".mp4", ".mov", ".avi", ".mkv", ".wav", ".ogg", ".flac", // Fonts
    ".ttf", ".otf", ".eot", ".woff", ".woff2", // Executables and libraries
    ".so", ".dll", ".dylib", ".class", ".jar", ".exe", ".bin", ".app",
];

/// Markdown file extensions
pub const MARKDOWN_EXTENSIONS: &[&str] = &[".md", ".markdown", ".mdown", ".mkd", ".mkdn"];

/// MIME types under `application/*` that should be treated as plain text.
const TEXTUAL_APPLICATION_MIME_TYPES: &[&str] = &[
    "application/json",
    "application/ld+json",
    "application/graphql",
    "application/javascript",
    "application/x-javascript",
    "application/typescript",
    "application/x-typescript",
    "application/xml",
    "application/xhtml+xml",
    "application/x-sh",
    "application/x-shellscript",
    "application/x-bash",
    "application/x-zsh",
    "application/x-python",
    "application/x-ruby",
    "application/x-perl",
    "application/x-php",
    "application/x-httpd-php",
    "application/x-toml",
    "application/toml",
    "application/x-yaml",
    "application/yaml",
    "application/x-sql",
    "application/sql",
    "application/x-rust",
    "application/x-go",
    "application/x-java",
    "application/x-scala",
    "application/x-kotlin",
    "application/x-swift",
    "application/x-dart",
    "application/x-haskell",
    "application/x-clojure",
    "application/x-ocaml",
    "application/x-lisp",
    "application/x-r",
    "application/x-matlab",
    "application/x-tex",
    "application/x-empty",
];

/// Keywords inside MIME subtypes that indicate textual content even if the
/// top-level type is `application/*`.
const TEXTUAL_APPLICATION_KEYWORDS: &[&str] = &[
    "+json",
    "+xml",
    "json",
    "xml",
    "yaml",
    "yml",
    "toml",
    "graphql",
    "javascript",
    "typescript",
    "ecmascript",
    "shellscript",
    "shell",
    "bash",
    "zsh",
    "sh",
    "python",
    "ruby",
    "perl",
    "php",
    "rust",
    "go",
    "java",
    "scala",
    "kotlin",
    "swift",
    "dart",
    "haskell",
    "clojure",
    "ocaml",
    "lisp",
    "sql",
    "graphql",
    "tex",
    "rscript",
    "matlab",
];

/// Merged weight for a file, representing its priority/importance for selection.
/// Higher values indicate files that should be prioritized for full content inclusion.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct FileWeight(pub f64);

impl FileWeight {
    pub fn new(weight: f64) -> Self {
        Self(weight)
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}

/// Decision about whether to include a file in analysis
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderDecision {
    /// Whether to include the file in analysis
    pub include: bool,
    /// Human-readable reason for the decision
    pub reason: String,
    /// Optional additional context
    pub context: Option<String>,
}

impl RenderDecision {
    /// Create a decision to include the file
    pub fn include<S: Into<String>>(reason: S) -> Self {
        Self {
            include: true,
            reason: reason.into(),
            context: None,
        }
    }

    /// Create a decision to exclude the file
    pub fn exclude<S: Into<String>>(reason: S) -> Self {
        Self {
            include: false,
            reason: reason.into(),
            context: None,
        }
    }

    /// Add context to the decision
    pub fn with_context<S: Into<String>>(mut self, context: S) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Check if the file should be included
    pub fn should_include(&self) -> bool {
        self.include
    }

    /// Get the reason as a standard category
    pub fn reason_category(&self) -> RenderDecisionCategory {
        match self.reason.as_str() {
            "ok" => RenderDecisionCategory::Ok,
            "binary" => RenderDecisionCategory::Binary,
            "too_large" => RenderDecisionCategory::TooLarge,
            "ignored" => RenderDecisionCategory::Ignored,
            "empty" => RenderDecisionCategory::Empty,
            _ => RenderDecisionCategory::Other,
        }
    }
}

/// Standard categories for render decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderDecisionCategory {
    Ok,
    Binary,
    TooLarge,
    Ignored,
    Empty,
    Other,
}

/// Programming language classification
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Language {
    // Systems languages
    Rust,
    C,
    Cpp,
    Go,
    Zig,

    // Web languages
    JavaScript,
    TypeScript,
    HTML,
    CSS,
    SCSS,
    SASS,

    // Backend languages
    Python,
    Java,
    CSharp,
    Kotlin,
    Scala,
    Ruby,
    PHP,

    // Functional languages
    Haskell,
    OCaml,
    FSharp,
    Erlang,
    Elixir,
    Clojure,

    // Configuration and markup
    JSON,
    YAML,
    TOML,
    XML,
    Markdown,

    // Database
    SQL,

    // Shell and scripts
    Bash,
    PowerShell,
    Batch,

    // Data science
    R,
    Julia,
    Matlab,

    // Mobile
    Swift,
    ObjectiveC,
    Dart,

    // Other
    Unknown,
}

impl Language {
    /// Detect language from file extension
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "rs" => Language::Rust,
            "c" | "h" => Language::C,
            "cpp" | "cxx" | "cc" | "hpp" | "hxx" => Language::Cpp,
            "go" => Language::Go,
            "zig" => Language::Zig,
            "js" | "mjs" | "cjs" => Language::JavaScript,
            "ts" | "mts" | "cts" => Language::TypeScript,
            "html" | "htm" => Language::HTML,
            "css" => Language::CSS,
            "scss" => Language::SCSS,
            "sass" => Language::SASS,
            "py" | "pyi" | "pyw" => Language::Python,
            "java" => Language::Java,
            "cs" => Language::CSharp,
            "kt" | "kts" => Language::Kotlin,
            "scala" | "sc" => Language::Scala,
            "rb" => Language::Ruby,
            "php" => Language::PHP,
            "hs" | "lhs" => Language::Haskell,
            "ml" | "mli" => Language::OCaml,
            "fs" | "fsi" | "fsx" => Language::FSharp,
            "erl" | "hrl" => Language::Erlang,
            "ex" | "exs" => Language::Elixir,
            "clj" | "cljs" | "cljc" => Language::Clojure,
            "json" => Language::JSON,
            "yaml" | "yml" => Language::YAML,
            "toml" => Language::TOML,
            "xml" => Language::XML,
            "md" | "markdown" | "mdown" | "mkd" | "mkdn" => Language::Markdown,
            "sql" => Language::SQL,
            "sh" | "bash" => Language::Bash,
            "ps1" | "psm1" | "psd1" => Language::PowerShell,
            "bat" | "cmd" => Language::Batch,
            "r" => Language::R,
            "jl" => Language::Julia,
            "swift" => Language::Swift,
            "dart" => Language::Dart,
            // Handle ambiguous .m extension - could be Matlab or Objective-C
            // Default to Objective-C as it's more common in modern development
            "m" | "mm" => Language::ObjectiveC,
            _ => Language::Unknown,
        }
    }

    /// Check if this language is typically used for documentation
    pub fn is_documentation(&self) -> bool {
        matches!(self, Language::Markdown | Language::HTML)
    }

    /// Check if this language is typically used for configuration
    pub fn is_configuration(&self) -> bool {
        matches!(
            self,
            Language::JSON | Language::YAML | Language::TOML | Language::XML
        )
    }

    /// Check if this is a programming language (not markup/config)
    pub fn is_programming(&self) -> bool {
        !matches!(
            self,
            Language::Markdown
                | Language::HTML
                | Language::JSON
                | Language::YAML
                | Language::TOML
                | Language::XML
                | Language::Unknown
        )
    }

    /// Display name used for user-facing messaging
    pub fn display_name(&self) -> &'static str {
        match self {
            Language::Rust => "Rust",
            Language::C => "C",
            Language::Cpp => "C++",
            Language::Go => "Go",
            Language::Zig => "Zig",
            Language::JavaScript => "JavaScript",
            Language::TypeScript => "TypeScript",
            Language::HTML => "HTML",
            Language::CSS => "CSS",
            Language::SCSS => "SCSS",
            Language::SASS => "SASS",
            Language::Python => "Python",
            Language::Java => "Java",
            Language::CSharp => "C#",
            Language::Kotlin => "Kotlin",
            Language::Scala => "Scala",
            Language::Ruby => "Ruby",
            Language::PHP => "PHP",
            Language::Haskell => "Haskell",
            Language::OCaml => "OCaml",
            Language::FSharp => "F#",
            Language::Erlang => "Erlang",
            Language::Elixir => "Elixir",
            Language::Clojure => "Clojure",
            Language::JSON => "JSON",
            Language::YAML => "YAML",
            Language::TOML => "TOML",
            Language::XML => "XML",
            Language::Markdown => "Markdown",
            Language::SQL => "SQL",
            Language::Bash => "Bash",
            Language::PowerShell => "PowerShell",
            Language::Batch => "Batch",
            Language::R => "R",
            Language::Julia => "Julia",
            Language::Matlab => "Matlab",
            Language::Swift => "Swift",
            Language::ObjectiveC => "Objective-C",
            Language::Dart => "Dart",
            Language::Bash => "Bash",
            Language::Unknown => "Unknown",
        }
    }

    /// Get the typical file extensions for this language
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            Language::Rust => &["rs"],
            Language::C => &["c", "h"],
            Language::Cpp => &["cpp", "cxx", "cc", "hpp", "hxx"],
            Language::Go => &["go"],
            Language::Zig => &["zig"],
            Language::JavaScript => &["js", "mjs", "cjs"],
            Language::TypeScript => &["ts", "mts", "cts"],
            Language::HTML => &["html", "htm"],
            Language::CSS => &["css"],
            Language::SCSS => &["scss"],
            Language::SASS => &["sass"],
            Language::Python => &["py", "pyi", "pyw"],
            Language::Java => &["java"],
            Language::CSharp => &["cs"],
            Language::Kotlin => &["kt", "kts"],
            Language::Scala => &["scala", "sc"],
            Language::Ruby => &["rb"],
            Language::PHP => &["php"],
            Language::Haskell => &["hs", "lhs"],
            Language::OCaml => &["ml", "mli"],
            Language::FSharp => &["fs", "fsi", "fsx"],
            Language::Erlang => &["erl", "hrl"],
            Language::Elixir => &["ex", "exs"],
            Language::Clojure => &["clj", "cljs", "cljc"],
            Language::JSON => &["json"],
            Language::YAML => &["yaml", "yml"],
            Language::TOML => &["toml"],
            Language::XML => &["xml"],
            Language::Markdown => &["md", "markdown", "mdown", "mkd", "mkdn"],
            Language::SQL => &["sql"],
            Language::Bash => &["sh", "bash"],
            Language::PowerShell => &["ps1", "psm1", "psd1"],
            Language::Batch => &["bat", "cmd"],
            Language::R => &["r"],
            Language::Julia => &["jl"],
            Language::Matlab => &["m"], // Note: .m conflicts with Objective-C
            Language::Swift => &["swift"],
            Language::ObjectiveC => &["m", "mm"],
            Language::Dart => &["dart"],
            Language::Unknown => &[],
        }
    }
}

/// File type classification for analysis purposes
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileType {
    /// Source code files
    Source { language: Language },
    /// Documentation files
    Documentation { format: DocumentationFormat },
    /// Configuration files
    Configuration { format: ConfigurationFormat },
    /// Test files
    Test { language: Language },
    /// Binary files that should be excluded
    Binary,
    /// Generated or built files
    Generated,
    /// Unknown or unclassified
    Unknown,
}

impl FileType {
    pub fn display_label(&self) -> &'static str {
        match self {
            FileType::Source { .. } => "Source",
            FileType::Documentation { .. } => "Documentation",
            FileType::Configuration { .. } => "Configuration",
            FileType::Test { .. } => "Test",
            FileType::Binary => "Binary",
            FileType::Generated => "Generated",
            FileType::Unknown => "Unknown",
        }
    }
}

/// Documentation format classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DocumentationFormat {
    Markdown,
    Html,
    PlainText,
    Rst,
    Asciidoc,
}

/// Configuration format classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConfigurationFormat {
    Json,
    Yaml,
    Toml,
    Xml,
    Ini,
    Dotenv,
}

/// Comprehensive file metadata structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    /// Absolute path to the file on disk
    pub path: PathBuf,

    /// Path relative to repository root (forward slash separated)
    pub relative_path: String,

    /// File size in bytes
    pub size: u64,

    /// File modification time
    pub modified: Option<SystemTime>,

    /// Analysis decision (include/exclude)
    pub decision: RenderDecision,

    /// Detected file type
    pub file_type: FileType,

    /// Detected programming language
    pub language: Language,

    /// File content (loaded on demand)
    pub content: Option<String>,

    /// Estimated token count for LLM processing
    pub token_estimate: Option<usize>,

    /// Line count (if text file)
    pub line_count: Option<usize>,

    /// Character count (if text file)
    pub char_count: Option<usize>,

    /// Whether the file is likely binary
    pub is_binary: bool,

    /// Git status information (if available)
    pub git_status: Option<GitStatus>,

    /// Merged weight for prioritization during selection (higher = more important)
    #[serde(default)]
    pub weight: FileWeight,

    /// PageRank centrality score (0.0-1.0, higher means more important)
    pub centrality_score: Option<f64>,
}

/// Git status information for a file
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitStatus {
    /// Working tree status
    pub working_tree: GitFileStatus,
    /// Index/staging area status
    pub index: GitFileStatus,
}

/// Git file status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GitFileStatus {
    Unmodified,
    Modified,
    Added,
    Deleted,
    Renamed,
    Copied,
    Unmerged,
    Untracked,
    Ignored,
}

impl FileInfo {
    /// Create a new FileInfo from a path
    pub fn new<P: AsRef<Path>>(
        path: P,
        relative_path: String,
        decision: RenderDecision,
    ) -> Result<Self> {
        let path = path.as_ref();
        let metadata = std::fs::metadata(path)
            .map_err(|e| ScribeError::path_with_source("Failed to read file metadata", path, e))?;

        let size = metadata.len();
        let modified = metadata.modified().ok();

        let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");

        let language = Language::from_extension(extension);
        let is_binary = Self::detect_binary_with_hint(path, extension);
        let file_type =
            Self::classify_file_type_with_binary(&relative_path, &language, extension, is_binary);

        Ok(Self {
            path: path.to_path_buf(),
            relative_path,
            size,
            modified,
            decision,
            file_type,
            language,
            content: None,
            token_estimate: None,
            line_count: None,
            char_count: None,
            is_binary,
            git_status: None,
            weight: FileWeight::default(),
            centrality_score: None,
        })
    }

    /// Load file content and compute statistics
    pub fn load_content(&mut self) -> Result<()> {
        if self.is_binary || !self.decision.should_include() {
            return Ok(());
        }

        let content = std::fs::read_to_string(&self.path).map_err(|e| {
            ScribeError::analysis(format!("Failed to read file content: {}", e), &self.path)
        })?;

        // Compute statistics
        let line_count = content.lines().count();
        let char_count = content.chars().count();
        let token_estimate = Self::estimate_tokens(&content);

        self.content = Some(content);
        self.line_count = Some(line_count);
        self.char_count = Some(char_count);
        self.token_estimate = Some(token_estimate);

        Ok(())
    }

    /// Estimate token count for LLM processing using tiktoken
    ///
    /// This method uses the shared global TokenCounter instance for optimal performance.
    /// If tiktoken fails, it falls back to the legacy character-based estimation.
    pub fn estimate_tokens(content: &str) -> usize {
        use crate::tokenization::{utils, TokenCounter};

        // Use the shared global instance for optimal performance
        match TokenCounter::global().count_tokens(content) {
            Ok(tokens) => tokens,
            Err(_) => {
                // Fall back to legacy estimation if tiktoken fails
                utils::estimate_tokens_legacy(content)
            }
        }
    }

    /// Estimate token count for LLM processing with file context
    ///
    /// This method uses the file path to apply language-specific multipliers
    /// for more accurate token estimation.
    pub fn estimate_tokens_with_path(content: &str, file_path: &std::path::Path) -> usize {
        use crate::tokenization::TokenCounter;

        // Use the shared global instance for optimal performance
        match TokenCounter::global().estimate_file_tokens(content, file_path) {
            Ok(tokens) => tokens,
            Err(_) => Self::estimate_tokens(content), // Fall back to basic estimation
        }
    }

    /// Detect whether a file is binary using libmagic-compatible signatures with
    /// sensible fallbacks for small or unknown files.
    pub fn detect_binary(path: &Path) -> bool {
        let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
        Self::detect_binary_with_hint(path, extension)
    }

    /// Detect whether a file is binary, allowing the caller to provide an
    /// extension hint for fallback heuristics.
    pub fn detect_binary_with_hint(path: &Path, extension: &str) -> bool {
        if let Some(mime) = tree_magic_mini::from_filepath(path) {
            if !Self::is_textual_mime(mime) {
                return true;
            }
            return false;
        }

        if let Ok(mut file) = File::open(path) {
            let mut buffer = [0u8; 8192];
            if let Ok(read) = file.read(&mut buffer) {
                if read == 0 {
                    return false;
                }

                let slice = &buffer[..read];
                let mime = tree_magic_mini::from_u8(slice);
                if !Self::is_textual_mime(mime) {
                    return true;
                }

                if slice.iter().any(|byte| *byte == 0) {
                    return true;
                }
            }
        }

        Self::detect_binary_by_extension(extension)
    }

    /// Detect whether in-memory content represents a binary file.
    pub fn detect_binary_from_bytes(bytes: &[u8], extension: Option<&str>) -> bool {
        if bytes.is_empty() {
            return false;
        }

        let mime = tree_magic_mini::from_u8(bytes);
        if !Self::is_textual_mime(mime) {
            return true;
        }

        if bytes.iter().any(|byte| *byte == 0) {
            return true;
        }

        extension
            .map(Self::detect_binary_by_extension)
            .unwrap_or(false)
    }

    /// Check if file extension indicates binary content (fallback heuristic).
    pub fn detect_binary_by_extension(extension: &str) -> bool {
        if extension.is_empty() {
            return false;
        }

        let trimmed = extension.trim_start_matches('.');
        let lower = trimmed.to_lowercase();
        let prefixed = format!(".{}", lower);

        BINARY_EXTENSIONS.contains(&prefixed.as_str())
    }

    #[inline]
    fn is_textual_mime(mime: &str) -> bool {
        let canonical = mime
            .split(';')
            .next()
            .unwrap_or(mime)
            .trim()
            .to_ascii_lowercase();
        let mime = canonical.as_str();

        if mime.starts_with("text/") || mime.starts_with("inode/") || mime.starts_with("message/") {
            return true;
        }

        if mime.starts_with("application/") {
            if TEXTUAL_APPLICATION_MIME_TYPES.contains(&mime) {
                return true;
            }

            if TEXTUAL_APPLICATION_KEYWORDS
                .iter()
                .any(|keyword| mime.contains(keyword))
            {
                return true;
            }
        }

        false
    }

    /// Classify file type based on path and language
    pub fn classify_file_type(path: &str, language: &Language, extension: &str) -> FileType {
        let is_binary = Self::detect_binary_by_extension(extension);
        Self::classify_file_type_with_binary(path, language, extension, is_binary)
    }

    /// Classify file type when the binary state is already known.
    pub fn classify_file_type_with_binary(
        path: &str,
        language: &Language,
        extension: &str,
        is_binary: bool,
    ) -> FileType {
        let path_lower = path.to_lowercase();

        if is_binary {
            return FileType::Binary;
        }

        // Test files
        if is_test_path(Path::new(path)) {
            return FileType::Test {
                language: language.clone(),
            };
        }

        // Documentation
        if language.is_documentation() {
            let format = match extension {
                "md" | "markdown" => DocumentationFormat::Markdown,
                "html" | "htm" => DocumentationFormat::Html,
                "rst" => DocumentationFormat::Rst,
                "txt" => DocumentationFormat::PlainText,
                _ => DocumentationFormat::Markdown,
            };
            return FileType::Documentation { format };
        }

        // Configuration
        if language.is_configuration() {
            let format = match extension {
                "json" => ConfigurationFormat::Json,
                "yaml" | "yml" => ConfigurationFormat::Yaml,
                "toml" => ConfigurationFormat::Toml,
                "xml" => ConfigurationFormat::Xml,
                "ini" => ConfigurationFormat::Ini,
                "env" => ConfigurationFormat::Dotenv,
                _ => ConfigurationFormat::Json,
            };
            return FileType::Configuration { format };
        }

        // Generated files (common patterns)
        if path_lower.contains("generated")
            || path_lower.contains("build")
            || path_lower.contains("dist")
            || path_lower.contains("target")
        {
            return FileType::Generated;
        }

        // Source code
        if language.is_programming() {
            return FileType::Source {
                language: language.clone(),
            };
        }

        FileType::Unknown
    }

    /// Get human-readable size
    pub fn human_size(&self) -> String {
        bytes_to_human(self.size)
    }

    /// Check if file should be included in analysis
    pub fn should_include(&self) -> bool {
        self.decision.should_include()
    }

    /// Get file name (last component of path)
    pub fn file_name(&self) -> Option<&str> {
        self.path.file_name()?.to_str()
    }

    /// Get file stem (name without extension)
    pub fn file_stem(&self) -> Option<&str> {
        self.path.file_stem()?.to_str()
    }

    /// Get file extension
    pub fn extension(&self) -> Option<&str> {
        self.path.extension()?.to_str()
    }
}

/// Convert bytes to human-readable format
pub fn bytes_to_human(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    const THRESHOLD: f64 = 1024.0;

    if bytes == 0 {
        return "0 B".to_string();
    }

    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= THRESHOLD && unit_idx < UNITS.len() - 1 {
        size /= THRESHOLD;
        unit_idx += 1;
    }

    if unit_idx == 0 {
        format!("{} {}", bytes, UNITS[unit_idx])
    } else {
        format!("{:.1} {}", size, UNITS[unit_idx])
    }
}

/// Detect language from a file path based on extension or special names
pub fn detect_language_from_path(path: &Path) -> Language {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(Language::from_extension)
        .unwrap_or(Language::Unknown)
}

/// Convenience helper returning the human-friendly language name
pub fn language_display_name(language: &Language) -> &'static str {
    language.display_name()
}

/// Heuristic test-file detection based on path segments and naming conventions
pub fn is_test_path(path: &Path) -> bool {
    let path_lower = path.to_string_lossy().to_lowercase();
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    if file_name == "output.md" || file_name.starts_with("output.") {
        return true;
    }

    let segments: Vec<&str> = path_lower
        .split(|c| c == '/' || c == '\\')
        .filter(|segment| !segment.is_empty())
        .collect();

    const TEST_DIR_MARKERS: &[&str] = &[
        "test",
        "tests",
        "testing",
        "__tests__",
        "integration-tests",
        "integration_test",
        "integrationtests",
        "e2e",
        "qa",
        "spec",
    ];

    if segments
        .iter()
        .any(|segment| TEST_DIR_MARKERS.contains(segment))
    {
        return true;
    }

    const TEST_PREFIXES: &[&str] = &["test_", "spec_", "itest_", "integration_"];
    if TEST_PREFIXES
        .iter()
        .any(|prefix| file_name.starts_with(prefix))
    {
        return true;
    }

    const TEST_SUFFIXES: &[&str] = &["_test", "_tests", "_spec", "_itest", "_integration", "_e2e"];
    if TEST_SUFFIXES
        .iter()
        .any(|suffix| file_name.strip_suffix(suffix).is_some())
    {
        return true;
    }

    if file_name.contains(".test.") || file_name.contains(".spec.") {
        return true;
    }

    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "rs" => file_name.ends_with("_test.rs") || segments.iter().any(|seg| *seg == "tests"),
        "py" => file_name.starts_with("test_") || file_name.ends_with("_test.py"),
        "go" => file_name.ends_with("_test.go"),
        "java" | "kt" => {
            file_name.ends_with("test.java")
                || file_name.ends_with("tests.java")
                || file_name.ends_with("test.kt")
                || file_name.ends_with("tests.kt")
        }
        "php" => file_name.ends_with("test.php"),
        "rb" => file_name.ends_with("_spec.rb") || file_name.ends_with("_test.rb"),
        "js" | "jsx" | "ts" | "tsx" => {
            file_name.contains(".test.")
                || file_name.contains(".spec.")
                || file_name.ends_with("_test.ts")
        }
        _ => false,
    }
}

/// Heuristic entrypoint detection based on common file names per language
pub fn is_entrypoint_path(path: &Path, language: &Language) -> bool {
    let path_lower = path.to_string_lossy().to_lowercase();
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    match language {
        Language::Rust => file_name == "main.rs" || file_name == "lib.rs",
        Language::Python => {
            file_name == "main.py"
                || path_lower.contains("/__main__.py")
                || path_lower.contains("/manage.py")
                || file_name == "app.py"
                || file_name == "__init__.py"
        }
        Language::JavaScript | Language::TypeScript => {
            file_name == "index.js"
                || file_name == "index.ts"
                || path_lower.contains("/app.js")
                || path_lower.contains("/server.js")
        }
        Language::Go => file_name == "main.go",
        Language::Java => file_name == "main.java" || path_lower.contains("/main.java"),
        _ => file_name.starts_with("main.") || file_name.starts_with("index."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_detection() {
        assert_eq!(Language::from_extension("rs"), Language::Rust);
        assert_eq!(Language::from_extension("py"), Language::Python);
        assert_eq!(Language::from_extension("js"), Language::JavaScript);
        assert_eq!(Language::from_extension("unknown"), Language::Unknown);
    }

    #[test]
    fn test_binary_detection() {
        assert!(FileInfo::detect_binary_by_extension("png"));
        assert!(FileInfo::detect_binary_by_extension("exe"));
        assert!(!FileInfo::detect_binary_by_extension("rs"));
        assert!(!FileInfo::detect_binary_by_extension("py"));
    }

    #[test]
    fn test_detect_binary_magic_on_files() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut text_file = NamedTempFile::new().unwrap();
        writeln!(text_file, "fn main() {{ println!(\"hi\"); }}").unwrap();

        assert!(!FileInfo::detect_binary(text_file.path()));

        let mut binary_file = NamedTempFile::new().unwrap();
        binary_file
            .write_all(&[0u8, 159, 146, 150, 0, 1, 2])
            .unwrap();

        assert!(FileInfo::detect_binary(binary_file.path()));
    }

    #[test]
    fn test_detect_binary_from_bytes() {
        let text_bytes = b"#!/usr/bin/env python3\nprint('hello')\n";
        assert!(!FileInfo::detect_binary_from_bytes(text_bytes, Some("py")));

        let binary_bytes = [0u8, 255, 1, 2, 3, 4, 5];
        assert!(FileInfo::detect_binary_from_bytes(&binary_bytes, None));
    }

    #[test]
    fn test_file_type_classification() {
        let rust_lang = Language::Rust;
        let py_lang = Language::Python;
        let md_lang = Language::Markdown;

        // Test Rust source files
        assert!(matches!(
            FileInfo::classify_file_type("src/lib.rs", &rust_lang, "rs"),
            FileType::Source { .. }
        ));

        assert!(matches!(
            FileInfo::classify_file_type("scribe-rs/src/lib.rs", &rust_lang, "rs"),
            FileType::Source { .. }
        ));

        // Test Python source files
        assert!(matches!(
            FileInfo::classify_file_type("script.py", &py_lang, "py"),
            FileType::Source { .. }
        ));

        // Test that is_programming works correctly
        assert!(rust_lang.is_programming());
        assert!(py_lang.is_programming());
        assert!(!md_lang.is_programming());
    }

    #[test]
    fn test_integration_file_classification() {
        // Test the full pipeline: extension -> language -> file_type

        // Test Rust files
        let rust_lang = Language::from_extension("rs");
        assert_eq!(rust_lang, Language::Rust);
        assert!(rust_lang.is_programming());

        let rust_file_type = FileInfo::classify_file_type("src/lib.rs", &rust_lang, "rs");
        assert!(matches!(rust_file_type, FileType::Source { .. }));

        // Test Python files
        let py_lang = Language::from_extension("py");
        assert_eq!(py_lang, Language::Python);
        assert!(py_lang.is_programming());

        let py_file_type = FileInfo::classify_file_type("script.py", &py_lang, "py");
        assert!(matches!(py_file_type, FileType::Source { .. }));

        // Test that Unknown language doesn't become Source
        let unknown_lang = Language::from_extension("xyz");
        assert_eq!(unknown_lang, Language::Unknown);
        assert!(!unknown_lang.is_programming());

        let unknown_file_type = FileInfo::classify_file_type("file.xyz", &unknown_lang, "xyz");
        assert!(matches!(unknown_file_type, FileType::Unknown));

        // Test Markdown files
        let md_lang = Language::from_extension("md");
        assert_eq!(md_lang, Language::Markdown);
        assert!(!md_lang.is_programming());

        let md_file_type = FileInfo::classify_file_type("README.md", &md_lang, "md");
        assert!(matches!(md_file_type, FileType::Documentation { .. }));
    }

    #[test]
    fn test_bytes_to_human() {
        assert_eq!(bytes_to_human(0), "0 B");
        assert_eq!(bytes_to_human(512), "512 B");
        assert_eq!(bytes_to_human(1024), "1.0 KiB");
        assert_eq!(bytes_to_human(1536), "1.5 KiB");
        assert_eq!(bytes_to_human(1048576), "1.0 MiB");
    }

    #[test]
    fn test_token_estimation() {
        let content = "Hello world, this is a test.";
        let tokens = FileInfo::estimate_tokens(content);
        assert!(tokens > 0);
        assert!(tokens < 20); // Should be reasonable estimate
    }

    #[test]
    fn test_render_decision() {
        let include = RenderDecision::include("valid file");
        assert!(include.should_include());
        assert_eq!(include.reason_category(), RenderDecisionCategory::Other);

        let exclude = RenderDecision::exclude("binary").with_context("detected by extension");
        assert!(!exclude.should_include());
        assert_eq!(exclude.reason_category(), RenderDecisionCategory::Binary);
        assert!(exclude.context.is_some());
    }
}
