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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_weight() {
        let weight = FileWeight::new(0.8);
        assert_eq!(weight.value(), 0.8);
        assert_eq!(weight.0, 0.8);

        let default_weight = FileWeight::default();
        assert_eq!(default_weight.value(), 0.0);
    }

    #[test]
    fn test_render_decision_include() {
        let decision = RenderDecision::include("File is valid");
        assert!(decision.include);
        assert!(decision.should_include());
        assert_eq!(decision.reason, "File is valid");
        assert!(decision.context.is_none());
    }

    #[test]
    fn test_render_decision_exclude() {
        let decision = RenderDecision::exclude("File is binary");
        assert!(!decision.include);
        assert!(!decision.should_include());
        assert_eq!(decision.reason, "File is binary");
    }

    #[test]
    fn test_render_decision_with_context() {
        let decision = RenderDecision::include("ok").with_context("Passed all checks");
        assert!(decision.include);
        assert_eq!(decision.context, Some("Passed all checks".to_string()));
    }

    #[test]
    fn test_render_decision_category() {
        assert_eq!(
            RenderDecision::include("ok").reason_category(),
            RenderDecisionCategory::Ok
        );
        assert_eq!(
            RenderDecision::exclude("binary").reason_category(),
            RenderDecisionCategory::Binary
        );
        assert_eq!(
            RenderDecision::exclude("too_large").reason_category(),
            RenderDecisionCategory::TooLarge
        );
        assert_eq!(
            RenderDecision::exclude("ignored").reason_category(),
            RenderDecisionCategory::Ignored
        );
        assert_eq!(
            RenderDecision::exclude("empty").reason_category(),
            RenderDecisionCategory::Empty
        );
        assert_eq!(
            RenderDecision::exclude("custom reason").reason_category(),
            RenderDecisionCategory::Other
        );
    }

    #[test]
    fn test_language_from_extension_systems() {
        assert_eq!(Language::from_extension("rs"), Language::Rust);
        assert_eq!(Language::from_extension("c"), Language::C);
        assert_eq!(Language::from_extension("h"), Language::C);
        assert_eq!(Language::from_extension("cpp"), Language::Cpp);
        assert_eq!(Language::from_extension("hpp"), Language::Cpp);
        assert_eq!(Language::from_extension("go"), Language::Go);
        assert_eq!(Language::from_extension("zig"), Language::Zig);
    }

    #[test]
    fn test_language_from_extension_web() {
        assert_eq!(Language::from_extension("js"), Language::JavaScript);
        assert_eq!(Language::from_extension("mjs"), Language::JavaScript);
        assert_eq!(Language::from_extension("ts"), Language::TypeScript);
        assert_eq!(Language::from_extension("mts"), Language::TypeScript);
        assert_eq!(Language::from_extension("html"), Language::HTML);
        assert_eq!(Language::from_extension("css"), Language::CSS);
        assert_eq!(Language::from_extension("scss"), Language::SCSS);
        assert_eq!(Language::from_extension("sass"), Language::SASS);
    }

    #[test]
    fn test_language_from_extension_backend() {
        assert_eq!(Language::from_extension("py"), Language::Python);
        assert_eq!(Language::from_extension("pyi"), Language::Python);
        assert_eq!(Language::from_extension("java"), Language::Java);
        assert_eq!(Language::from_extension("cs"), Language::CSharp);
        assert_eq!(Language::from_extension("kt"), Language::Kotlin);
        assert_eq!(Language::from_extension("scala"), Language::Scala);
        assert_eq!(Language::from_extension("rb"), Language::Ruby);
        assert_eq!(Language::from_extension("php"), Language::PHP);
    }

    #[test]
    fn test_language_from_extension_functional() {
        assert_eq!(Language::from_extension("hs"), Language::Haskell);
        assert_eq!(Language::from_extension("ml"), Language::OCaml);
        assert_eq!(Language::from_extension("fs"), Language::FSharp);
        assert_eq!(Language::from_extension("erl"), Language::Erlang);
        assert_eq!(Language::from_extension("ex"), Language::Elixir);
        assert_eq!(Language::from_extension("clj"), Language::Clojure);
    }

    #[test]
    fn test_language_from_extension_config() {
        assert_eq!(Language::from_extension("json"), Language::JSON);
        assert_eq!(Language::from_extension("yaml"), Language::YAML);
        assert_eq!(Language::from_extension("yml"), Language::YAML);
        assert_eq!(Language::from_extension("toml"), Language::TOML);
        assert_eq!(Language::from_extension("xml"), Language::XML);
        assert_eq!(Language::from_extension("md"), Language::Markdown);
    }

    #[test]
    fn test_language_from_extension_scripts() {
        assert_eq!(Language::from_extension("sh"), Language::Bash);
        assert_eq!(Language::from_extension("bash"), Language::Bash);
        assert_eq!(Language::from_extension("ps1"), Language::PowerShell);
        assert_eq!(Language::from_extension("bat"), Language::Batch);
        assert_eq!(Language::from_extension("sql"), Language::SQL);
    }

    #[test]
    fn test_language_from_extension_data_science() {
        assert_eq!(Language::from_extension("r"), Language::R);
        assert_eq!(Language::from_extension("jl"), Language::Julia);
    }

    #[test]
    fn test_language_from_extension_mobile() {
        assert_eq!(Language::from_extension("swift"), Language::Swift);
        assert_eq!(Language::from_extension("dart"), Language::Dart);
        assert_eq!(Language::from_extension("m"), Language::ObjectiveC);
        assert_eq!(Language::from_extension("mm"), Language::ObjectiveC);
    }

    #[test]
    fn test_language_from_extension_unknown() {
        assert_eq!(Language::from_extension("xyz"), Language::Unknown);
        assert_eq!(Language::from_extension(""), Language::Unknown);
    }

    #[test]
    fn test_language_is_documentation() {
        assert!(Language::Markdown.is_documentation());
        assert!(Language::HTML.is_documentation());
        assert!(!Language::Rust.is_documentation());
        assert!(!Language::Python.is_documentation());
    }

    #[test]
    fn test_language_is_configuration() {
        assert!(Language::JSON.is_configuration());
        assert!(Language::YAML.is_configuration());
        assert!(Language::TOML.is_configuration());
        assert!(Language::XML.is_configuration());
        assert!(!Language::Rust.is_configuration());
    }

    #[test]
    fn test_language_is_programming() {
        assert!(Language::Rust.is_programming());
        assert!(Language::Python.is_programming());
        assert!(Language::Go.is_programming());
        assert!(!Language::Markdown.is_programming());
        assert!(!Language::JSON.is_programming());
        assert!(!Language::Unknown.is_programming());
    }

    #[test]
    fn test_language_display_name() {
        assert_eq!(Language::Rust.display_name(), "Rust");
        assert_eq!(Language::Cpp.display_name(), "C++");
        assert_eq!(Language::CSharp.display_name(), "C#");
        assert_eq!(Language::FSharp.display_name(), "F#");
        assert_eq!(Language::ObjectiveC.display_name(), "Objective-C");
        assert_eq!(Language::Unknown.display_name(), "Unknown");
    }

    #[test]
    fn test_language_extensions() {
        assert!(!Language::Rust.extensions().is_empty());
        assert!(Language::Rust.extensions().contains(&"rs"));
        assert!(Language::Python.extensions().contains(&"py"));
        assert!(Language::Unknown.extensions().is_empty());
    }

    #[test]
    fn test_file_type_display_label() {
        assert_eq!(
            FileType::Source {
                language: Language::Rust
            }
            .display_label(),
            "Source"
        );
        assert_eq!(
            FileType::Documentation {
                format: DocumentationFormat::Markdown
            }
            .display_label(),
            "Documentation"
        );
        assert_eq!(
            FileType::Configuration {
                format: ConfigurationFormat::Json
            }
            .display_label(),
            "Configuration"
        );
        assert_eq!(
            FileType::Test {
                language: Language::Python
            }
            .display_label(),
            "Test"
        );
        assert_eq!(FileType::Binary.display_label(), "Binary");
        assert_eq!(FileType::Generated.display_label(), "Generated");
        assert_eq!(FileType::Unknown.display_label(), "Unknown");
    }

    #[test]
    fn test_git_file_status_equality() {
        assert_eq!(GitFileStatus::Modified, GitFileStatus::Modified);
        assert_ne!(GitFileStatus::Modified, GitFileStatus::Added);
    }

    #[test]
    fn test_git_status() {
        let status = GitStatus {
            working_tree: GitFileStatus::Modified,
            index: GitFileStatus::Unmodified,
        };
        assert_eq!(status.working_tree, GitFileStatus::Modified);
        assert_eq!(status.index, GitFileStatus::Unmodified);
    }

    #[test]
    fn test_documentation_format_equality() {
        assert_eq!(DocumentationFormat::Markdown, DocumentationFormat::Markdown);
        assert_ne!(DocumentationFormat::Markdown, DocumentationFormat::Html);
    }

    #[test]
    fn test_configuration_format_equality() {
        assert_eq!(ConfigurationFormat::Json, ConfigurationFormat::Json);
        assert_ne!(ConfigurationFormat::Json, ConfigurationFormat::Yaml);
    }

    #[test]
    fn test_render_decision_clone() {
        let decision = RenderDecision::include("test").with_context("ctx");
        let cloned = decision.clone();
        assert_eq!(decision.include, cloned.include);
        assert_eq!(decision.reason, cloned.reason);
        assert_eq!(decision.context, cloned.context);
    }

    #[test]
    fn test_language_clone() {
        let lang = Language::Rust;
        let cloned = lang.clone();
        assert_eq!(lang, cloned);
    }

    #[test]
    fn test_file_type_clone() {
        let file_type = FileType::Source {
            language: Language::Go,
        };
        let cloned = file_type.clone();
        assert_eq!(file_type, cloned);
    }

    #[test]
    fn test_language_display_name_all() {
        // Test all display names
        assert_eq!(Language::Rust.display_name(), "Rust");
        assert_eq!(Language::C.display_name(), "C");
        assert_eq!(Language::Cpp.display_name(), "C++");
        assert_eq!(Language::Go.display_name(), "Go");
        assert_eq!(Language::Zig.display_name(), "Zig");
        assert_eq!(Language::JavaScript.display_name(), "JavaScript");
        assert_eq!(Language::TypeScript.display_name(), "TypeScript");
        assert_eq!(Language::HTML.display_name(), "HTML");
        assert_eq!(Language::CSS.display_name(), "CSS");
        assert_eq!(Language::SCSS.display_name(), "SCSS");
        assert_eq!(Language::SASS.display_name(), "SASS");
        assert_eq!(Language::Python.display_name(), "Python");
        assert_eq!(Language::Java.display_name(), "Java");
        assert_eq!(Language::CSharp.display_name(), "C#");
        assert_eq!(Language::Kotlin.display_name(), "Kotlin");
        assert_eq!(Language::Scala.display_name(), "Scala");
        assert_eq!(Language::Ruby.display_name(), "Ruby");
        assert_eq!(Language::PHP.display_name(), "PHP");
        assert_eq!(Language::Haskell.display_name(), "Haskell");
        assert_eq!(Language::OCaml.display_name(), "OCaml");
        assert_eq!(Language::FSharp.display_name(), "F#");
        assert_eq!(Language::Erlang.display_name(), "Erlang");
        assert_eq!(Language::Elixir.display_name(), "Elixir");
        assert_eq!(Language::Clojure.display_name(), "Clojure");
        assert_eq!(Language::JSON.display_name(), "JSON");
        assert_eq!(Language::YAML.display_name(), "YAML");
        assert_eq!(Language::TOML.display_name(), "TOML");
        assert_eq!(Language::XML.display_name(), "XML");
        assert_eq!(Language::Markdown.display_name(), "Markdown");
        assert_eq!(Language::SQL.display_name(), "SQL");
        assert_eq!(Language::Bash.display_name(), "Bash");
        assert_eq!(Language::PowerShell.display_name(), "PowerShell");
        assert_eq!(Language::Batch.display_name(), "Batch");
        assert_eq!(Language::R.display_name(), "R");
        assert_eq!(Language::Julia.display_name(), "Julia");
        assert_eq!(Language::Matlab.display_name(), "Matlab");
        assert_eq!(Language::Swift.display_name(), "Swift");
        assert_eq!(Language::ObjectiveC.display_name(), "Objective-C");
        assert_eq!(Language::Dart.display_name(), "Dart");
        assert_eq!(Language::Unknown.display_name(), "Unknown");
    }

    #[test]
    fn test_language_extensions_all() {
        // Test all extension lists
        assert!(Language::Rust.extensions().contains(&"rs"));
        assert!(Language::C.extensions().contains(&"c"));
        assert!(Language::C.extensions().contains(&"h"));
        assert!(Language::Cpp.extensions().contains(&"cpp"));
        assert!(Language::Cpp.extensions().contains(&"hpp"));
        assert!(Language::Go.extensions().contains(&"go"));
        assert!(Language::Zig.extensions().contains(&"zig"));
        assert!(Language::JavaScript.extensions().contains(&"js"));
        assert!(Language::JavaScript.extensions().contains(&"mjs"));
        assert!(Language::TypeScript.extensions().contains(&"ts"));
        assert!(Language::TypeScript.extensions().contains(&"mts"));
        assert!(Language::HTML.extensions().contains(&"html"));
        assert!(Language::CSS.extensions().contains(&"css"));
        assert!(Language::SCSS.extensions().contains(&"scss"));
        assert!(Language::SASS.extensions().contains(&"sass"));
        assert!(Language::Python.extensions().contains(&"py"));
        assert!(Language::Python.extensions().contains(&"pyi"));
        assert!(Language::Java.extensions().contains(&"java"));
        assert!(Language::CSharp.extensions().contains(&"cs"));
        assert!(Language::Kotlin.extensions().contains(&"kt"));
        assert!(Language::Scala.extensions().contains(&"scala"));
        assert!(Language::Ruby.extensions().contains(&"rb"));
        assert!(Language::PHP.extensions().contains(&"php"));
        assert!(Language::Haskell.extensions().contains(&"hs"));
        assert!(Language::OCaml.extensions().contains(&"ml"));
        assert!(Language::FSharp.extensions().contains(&"fs"));
        assert!(Language::Erlang.extensions().contains(&"erl"));
        assert!(Language::Elixir.extensions().contains(&"ex"));
        assert!(Language::Clojure.extensions().contains(&"clj"));
        assert!(Language::JSON.extensions().contains(&"json"));
        assert!(Language::YAML.extensions().contains(&"yaml"));
        assert!(Language::TOML.extensions().contains(&"toml"));
        assert!(Language::XML.extensions().contains(&"xml"));
        assert!(Language::Markdown.extensions().contains(&"md"));
        assert!(Language::SQL.extensions().contains(&"sql"));
        assert!(Language::Bash.extensions().contains(&"sh"));
        assert!(Language::PowerShell.extensions().contains(&"ps1"));
        assert!(Language::Batch.extensions().contains(&"bat"));
        assert!(Language::R.extensions().contains(&"r"));
        assert!(Language::Julia.extensions().contains(&"jl"));
        assert!(Language::Matlab.extensions().contains(&"m"));
        assert!(Language::Swift.extensions().contains(&"swift"));
        assert!(Language::ObjectiveC.extensions().contains(&"m"));
        assert!(Language::Dart.extensions().contains(&"dart"));
        assert!(Language::Unknown.extensions().is_empty());
    }

    #[test]
    fn test_language_ordering() {
        // Test that Language has PartialOrd and Ord
        use std::cmp::Ordering;
        let r1 = Language::Rust;
        let r2 = Language::Rust;
        assert_eq!(r1.cmp(&r2), Ordering::Equal);
    }

    #[test]
    fn test_language_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Language::Rust);
        set.insert(Language::Python);
        set.insert(Language::Rust); // duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(&Language::Rust));
        assert!(set.contains(&Language::Python));
    }

    #[test]
    fn test_render_decision_category_equality() {
        assert_eq!(RenderDecisionCategory::Ok, RenderDecisionCategory::Ok);
        assert_ne!(RenderDecisionCategory::Ok, RenderDecisionCategory::Binary);
    }

    #[test]
    fn test_git_file_status_clone() {
        let status = GitFileStatus::Modified;
        let cloned = status.clone();
        assert_eq!(status, cloned);
    }

    #[test]
    fn test_git_status_clone() {
        let status = GitStatus {
            working_tree: GitFileStatus::Added,
            index: GitFileStatus::Unmodified,
        };
        let cloned = status.clone();
        assert_eq!(status, cloned);
    }

    #[test]
    fn test_documentation_format_clone() {
        let format = DocumentationFormat::Markdown;
        let cloned = format.clone();
        assert_eq!(format, cloned);
    }

    #[test]
    fn test_configuration_format_clone() {
        let format = ConfigurationFormat::Toml;
        let cloned = format.clone();
        assert_eq!(format, cloned);
    }

    #[test]
    fn test_all_git_file_statuses() {
        let statuses = [
            GitFileStatus::Unmodified,
            GitFileStatus::Modified,
            GitFileStatus::Added,
            GitFileStatus::Deleted,
            GitFileStatus::Renamed,
            GitFileStatus::Copied,
            GitFileStatus::Unmerged,
            GitFileStatus::Untracked,
            GitFileStatus::Ignored,
        ];

        // All should be distinct
        for (i, s1) in statuses.iter().enumerate() {
            for (j, s2) in statuses.iter().enumerate() {
                if i == j {
                    assert_eq!(s1, s2);
                } else {
                    assert_ne!(s1, s2);
                }
            }
        }
    }

    #[test]
    fn test_all_documentation_formats() {
        let formats = [
            DocumentationFormat::Markdown,
            DocumentationFormat::Html,
            DocumentationFormat::PlainText,
            DocumentationFormat::Rst,
            DocumentationFormat::Asciidoc,
        ];

        for (i, f1) in formats.iter().enumerate() {
            for (j, f2) in formats.iter().enumerate() {
                if i == j {
                    assert_eq!(f1, f2);
                } else {
                    assert_ne!(f1, f2);
                }
            }
        }
    }

    #[test]
    fn test_all_configuration_formats() {
        let formats = [
            ConfigurationFormat::Json,
            ConfigurationFormat::Yaml,
            ConfigurationFormat::Toml,
            ConfigurationFormat::Xml,
            ConfigurationFormat::Ini,
            ConfigurationFormat::Dotenv,
        ];

        for (i, f1) in formats.iter().enumerate() {
            for (j, f2) in formats.iter().enumerate() {
                if i == j {
                    assert_eq!(f1, f2);
                } else {
                    assert_ne!(f1, f2);
                }
            }
        }
    }

    #[test]
    fn test_file_weight_copy() {
        let weight = FileWeight::new(0.5);
        let copied = weight; // Copy because FileWeight is Copy
        assert_eq!(weight.value(), copied.value());
    }

    #[test]
    fn test_render_decision_category_clone() {
        let category = RenderDecisionCategory::Ok;
        let cloned = category.clone();
        assert_eq!(category, cloned);
    }

    #[test]
    fn test_language_case_insensitive() {
        // from_extension should be case insensitive
        assert_eq!(Language::from_extension("RS"), Language::Rust);
        assert_eq!(Language::from_extension("Py"), Language::Python);
        assert_eq!(Language::from_extension("JSON"), Language::JSON);
    }

    #[test]
    fn test_file_type_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(FileType::Binary);
        set.insert(FileType::Generated);
        set.insert(FileType::Binary); // duplicate

        assert_eq!(set.len(), 2);
    }
}
