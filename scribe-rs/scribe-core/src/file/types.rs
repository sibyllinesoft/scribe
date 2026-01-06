//! Type definitions for file analysis.

use serde::{Deserialize, Serialize};

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
