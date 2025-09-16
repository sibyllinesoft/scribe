//! # AST Language Support Definitions
//!
//! Defines comprehensive language support tiers and capabilities for 20+ programming languages.
//! Replaces the basic regex-based approach with proper AST analysis using tree-sitter.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Programming language support (focused on tree-sitter languages)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AstLanguage {
    // Currently supported with tree-sitter
    Python,
    JavaScript,
    TypeScript,
    Go,
    Rust,
    Html,

    // Future tree-sitter support (when dependencies added)
    Java,
    C,
    Cpp,
    Ruby,
    CSharp,
}

/// Language support tier indicating analysis depth
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LanguageTier {
    /// Full AST parsing with tree-sitter
    FullAst,
    /// Syntax-aware parsing for markup languages
    SyntaxAware,
    /// Future support (not yet implemented)
    Future,
}

/// Language-specific features and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageFeatures {
    /// Support tier
    pub tier: LanguageTier,
    /// Can extract functions/methods
    pub has_functions: bool,
    /// Can extract classes/types
    pub has_classes: bool,
    /// Has documentation conventions
    pub has_documentation: bool,
    /// Has import/dependency statements
    pub has_imports: bool,
    /// Language-specific complexity factors
    pub complexity_factors: Vec<String>,
    /// Common file extensions
    pub extensions: Vec<String>,
}

impl AstLanguage {
    /// Get the tree-sitter language for this language (Tier 1 and 2 only)
    #[cfg(feature = "tree-sitter")]
    pub fn tree_sitter_language(&self) -> Option<tree_sitter::Language> {
        match self {
            // Tier 1: Full AST languages
            AstLanguage::Python => Some(tree_sitter_python::language()),
            AstLanguage::JavaScript => Some(tree_sitter_javascript::language()),
            AstLanguage::TypeScript => Some(tree_sitter_typescript::language_typescript()),
            AstLanguage::Go => Some(tree_sitter_go::language()),
            AstLanguage::Rust => Some(tree_sitter_rust::language()),
            AstLanguage::Html => Some(tree_sitter_html::language()),

            // Future tree-sitter languages (when dependencies are added)
            AstLanguage::Java => None, // tree_sitter_java::language() when added
            AstLanguage::CSharp => None, // tree_sitter_c_sharp::language() when added
            AstLanguage::C => None,    // tree_sitter_c::language() when added
            AstLanguage::Cpp => None,  // tree_sitter_cpp::language() when added
            AstLanguage::PHP => None,  // tree_sitter_php::language() when added
            AstLanguage::Ruby => None, // tree_sitter_ruby::language() when added
            AstLanguage::Swift => None, // tree_sitter_swift::language() when added
            AstLanguage::Kotlin => None, // tree_sitter_kotlin::language() when added

            // Tier 2: Syntax-aware languages
            AstLanguage::Css => None, // tree_sitter_css::language() when added
            AstLanguage::Json => None, // tree_sitter_json::language() when added
            AstLanguage::Yaml => None, // tree_sitter_yaml::language() when added
            AstLanguage::Xml => None, // tree_sitter_xml::language() when added
            AstLanguage::Markdown => None, // tree_sitter_markdown::language() when added
            AstLanguage::Sql => None, // tree_sitter_sql::language() when added
            AstLanguage::Bash => None, // tree_sitter_bash::language() when added
            AstLanguage::PowerShell => None, // tree_sitter_powershell::language() when added
            AstLanguage::Dockerfile => None, // tree_sitter_dockerfile::language() when added

            // Tier 3: Basic structure languages (no tree-sitter)
            _ => None,
        }
    }

    /// Detect language from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            // Currently supported
            "py" | "pyi" | "pyw" => Some(AstLanguage::Python),
            "js" | "mjs" | "cjs" => Some(AstLanguage::JavaScript),
            "ts" | "mts" | "cts" | "tsx" => Some(AstLanguage::TypeScript),
            "go" => Some(AstLanguage::Go),
            "rs" => Some(AstLanguage::Rust),
            "html" | "htm" => Some(AstLanguage::Html),

            // Future support
            "java" => Some(AstLanguage::Java),
            "c" => Some(AstLanguage::C),
            "cpp" | "cc" | "cxx" | "c++" | "hpp" | "h" => Some(AstLanguage::Cpp),
            "rb" | "ruby" => Some(AstLanguage::Ruby),
            "cs" => Some(AstLanguage::CSharp),

            _ => None,
        }
    }

    /// Get language tier
    pub fn tier(&self) -> LanguageTier {
        match self {
            // Currently supported with tree-sitter
            AstLanguage::Python
            | AstLanguage::JavaScript
            | AstLanguage::TypeScript
            | AstLanguage::Go
            | AstLanguage::Rust => LanguageTier::FullAst,

            // Syntax-aware (markup)
            AstLanguage::Html => LanguageTier::SyntaxAware,

            // Future support
            AstLanguage::Java
            | AstLanguage::C
            | AstLanguage::Cpp
            | AstLanguage::Ruby
            | AstLanguage::CSharp => LanguageTier::Future,
        }
    }

    /// Get language features and capabilities
    pub fn features(&self) -> LanguageFeatures {
        match self {
            AstLanguage::Python => LanguageFeatures {
                tier: LanguageTier::FullAst,
                has_functions: true,
                has_classes: true,
                has_documentation: true,
                has_imports: true,
                complexity_factors: vec![
                    "list_comprehensions".to_string(),
                    "decorators".to_string(),
                    "async_await".to_string(),
                    "generators".to_string(),
                ],
                extensions: vec!["py".to_string(), "pyi".to_string(), "pyw".to_string()],
            },

            AstLanguage::JavaScript => LanguageFeatures {
                tier: LanguageTier::FullAst,
                has_functions: true,
                has_classes: true,
                has_documentation: true,
                has_imports: true,
                complexity_factors: vec![
                    "closures".to_string(),
                    "promises".to_string(),
                    "async_await".to_string(),
                    "prototypal_inheritance".to_string(),
                ],
                extensions: vec!["js".to_string(), "mjs".to_string(), "cjs".to_string()],
            },

            AstLanguage::TypeScript => LanguageFeatures {
                tier: LanguageTier::FullAst,
                has_functions: true,
                has_classes: true,
                has_documentation: true,
                has_imports: true,
                complexity_factors: vec![
                    "generic_types".to_string(),
                    "type_guards".to_string(),
                    "conditional_types".to_string(),
                    "mapped_types".to_string(),
                ],
                extensions: vec!["ts".to_string(), "tsx".to_string(), "mts".to_string()],
            },

            AstLanguage::Rust => LanguageFeatures {
                tier: LanguageTier::FullAst,
                has_functions: true,
                has_classes: false, // Rust has structs/traits instead
                has_documentation: true,
                has_imports: true,
                complexity_factors: vec![
                    "lifetimes".to_string(),
                    "borrowing".to_string(),
                    "pattern_matching".to_string(),
                    "macros".to_string(),
                ],
                extensions: vec!["rs".to_string()],
            },

            AstLanguage::Go => LanguageFeatures {
                tier: LanguageTier::FullAst,
                has_functions: true,
                has_classes: false, // Go has structs/interfaces instead
                has_documentation: true,
                has_imports: true,
                complexity_factors: vec![
                    "goroutines".to_string(),
                    "channels".to_string(),
                    "interfaces".to_string(),
                    "defer_statements".to_string(),
                ],
                extensions: vec!["go".to_string()],
            },

            AstLanguage::Java => LanguageFeatures {
                tier: LanguageTier::FullAst,
                has_functions: true,
                has_classes: true,
                has_documentation: true,
                has_imports: true,
                complexity_factors: vec![
                    "inheritance".to_string(),
                    "generics".to_string(),
                    "reflection".to_string(),
                    "annotations".to_string(),
                ],
                extensions: vec!["java".to_string()],
            },

            // Add more language features as needed...
            _ => LanguageFeatures {
                tier: self.tier(),
                has_functions: false,
                has_classes: false,
                has_documentation: false,
                has_imports: false,
                complexity_factors: vec![],
                extensions: vec![],
            },
        }
    }

    /// Get all supported languages
    pub fn all_supported() -> Vec<Self> {
        vec![
            // Currently supported
            AstLanguage::Python,
            AstLanguage::JavaScript,
            AstLanguage::TypeScript,
            AstLanguage::Go,
            AstLanguage::Rust,
            AstLanguage::Html,
            // Future support
            AstLanguage::Java,
            AstLanguage::C,
            AstLanguage::Cpp,
            AstLanguage::Ruby,
            AstLanguage::CSharp,
        ]
    }

    /// Get language name as string
    pub fn name(&self) -> &'static str {
        match self {
            AstLanguage::Python => "Python",
            AstLanguage::JavaScript => "JavaScript",
            AstLanguage::TypeScript => "TypeScript",
            AstLanguage::Go => "Go",
            AstLanguage::Rust => "Rust",
            AstLanguage::Html => "HTML",
            AstLanguage::Java => "Java",
            AstLanguage::C => "C",
            AstLanguage::Cpp => "C++",
            AstLanguage::Ruby => "Ruby",
            AstLanguage::CSharp => "C#",
        }
    }
}

/// Language statistics for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageStats {
    /// Total supported languages
    pub total_languages: usize,
    /// Languages by tier
    pub by_tier: HashMap<LanguageTier, usize>,
    /// Languages with AST support
    pub ast_supported: usize,
    /// Languages with tree-sitter support
    pub tree_sitter_available: usize,
}

impl LanguageStats {
    /// Calculate statistics for current language support
    pub fn calculate() -> Self {
        let all_languages = AstLanguage::all_supported();
        let total_languages = all_languages.len();

        let mut by_tier = HashMap::new();
        let mut ast_supported = 0;
        let mut tree_sitter_available = 0;

        for language in &all_languages {
            let tier = language.tier();
            *by_tier.entry(tier).or_insert(0) += 1;

            if tier == LanguageTier::FullAst || tier == LanguageTier::SyntaxAware {
                ast_supported += 1;
            }

            #[cfg(feature = "tree-sitter")]
            if language.tree_sitter_language().is_some() {
                tree_sitter_available += 1;
            }
        }

        Self {
            total_languages,
            by_tier,
            ast_supported,
            tree_sitter_available,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_detection() {
        assert_eq!(AstLanguage::from_extension("py"), Some(AstLanguage::Python));
        assert_eq!(
            AstLanguage::from_extension("js"),
            Some(AstLanguage::JavaScript)
        );
        assert_eq!(AstLanguage::from_extension("rs"), Some(AstLanguage::Rust));
        assert_eq!(AstLanguage::from_extension("go"), Some(AstLanguage::Go));
        assert_eq!(AstLanguage::from_extension("unknown"), None);
    }

    #[test]
    fn test_language_tiers() {
        assert_eq!(AstLanguage::Python.tier(), LanguageTier::FullAst);
        assert_eq!(AstLanguage::Html.tier(), LanguageTier::SyntaxAware);
        assert_eq!(AstLanguage::Java.tier(), LanguageTier::Future);
    }

    #[test]
    fn test_language_features() {
        let python_features = AstLanguage::Python.features();
        assert!(python_features.has_functions);
        assert!(python_features.has_classes);
        assert!(python_features.has_documentation);
        assert!(python_features.has_imports);
        assert!(!python_features.complexity_factors.is_empty());
    }

    #[test]
    fn test_language_count() {
        let all_languages = AstLanguage::all_supported();
        // Should have 11 languages (6 current + 5 future)
        assert_eq!(
            all_languages.len(),
            11,
            "Expected 11 languages, got {}",
            all_languages.len()
        );
    }

    #[test]
    fn test_language_stats() {
        let stats = LanguageStats::calculate();
        assert_eq!(stats.total_languages, 11);
        assert!(stats.by_tier.contains_key(&LanguageTier::FullAst));
        assert!(stats.by_tier.contains_key(&LanguageTier::SyntaxAware));
        assert!(stats.by_tier.contains_key(&LanguageTier::Future));
    }
}
