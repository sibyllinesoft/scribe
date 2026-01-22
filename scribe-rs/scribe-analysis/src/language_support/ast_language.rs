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
            AstLanguage::Ruby => None, // tree_sitter_ruby::language() when added
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

    #[test]
    fn test_python_extension_variants() {
        assert_eq!(AstLanguage::from_extension("py"), Some(AstLanguage::Python));
        assert_eq!(
            AstLanguage::from_extension("pyi"),
            Some(AstLanguage::Python)
        );
        assert_eq!(
            AstLanguage::from_extension("pyw"),
            Some(AstLanguage::Python)
        );
        assert_eq!(AstLanguage::from_extension("PY"), Some(AstLanguage::Python));
        // case insensitive
    }

    #[test]
    fn test_javascript_extension_variants() {
        assert_eq!(
            AstLanguage::from_extension("js"),
            Some(AstLanguage::JavaScript)
        );
        assert_eq!(
            AstLanguage::from_extension("mjs"),
            Some(AstLanguage::JavaScript)
        );
        assert_eq!(
            AstLanguage::from_extension("cjs"),
            Some(AstLanguage::JavaScript)
        );
    }

    #[test]
    fn test_typescript_extension_variants() {
        assert_eq!(
            AstLanguage::from_extension("ts"),
            Some(AstLanguage::TypeScript)
        );
        assert_eq!(
            AstLanguage::from_extension("tsx"),
            Some(AstLanguage::TypeScript)
        );
        assert_eq!(
            AstLanguage::from_extension("mts"),
            Some(AstLanguage::TypeScript)
        );
        assert_eq!(
            AstLanguage::from_extension("cts"),
            Some(AstLanguage::TypeScript)
        );
    }

    #[test]
    fn test_html_extension_variants() {
        assert_eq!(AstLanguage::from_extension("html"), Some(AstLanguage::Html));
        assert_eq!(AstLanguage::from_extension("htm"), Some(AstLanguage::Html));
    }

    #[test]
    fn test_cpp_extension_variants() {
        assert_eq!(AstLanguage::from_extension("cpp"), Some(AstLanguage::Cpp));
        assert_eq!(AstLanguage::from_extension("cc"), Some(AstLanguage::Cpp));
        assert_eq!(AstLanguage::from_extension("cxx"), Some(AstLanguage::Cpp));
        assert_eq!(AstLanguage::from_extension("c++"), Some(AstLanguage::Cpp));
        assert_eq!(AstLanguage::from_extension("hpp"), Some(AstLanguage::Cpp));
        assert_eq!(AstLanguage::from_extension("h"), Some(AstLanguage::Cpp));
    }

    #[test]
    fn test_ruby_extension_variants() {
        assert_eq!(AstLanguage::from_extension("rb"), Some(AstLanguage::Ruby));
        assert_eq!(AstLanguage::from_extension("ruby"), Some(AstLanguage::Ruby));
    }

    #[test]
    fn test_c_extension() {
        assert_eq!(AstLanguage::from_extension("c"), Some(AstLanguage::C));
    }

    #[test]
    fn test_java_extension() {
        assert_eq!(AstLanguage::from_extension("java"), Some(AstLanguage::Java));
    }

    #[test]
    fn test_csharp_extension() {
        assert_eq!(AstLanguage::from_extension("cs"), Some(AstLanguage::CSharp));
    }

    #[test]
    fn test_javascript_features() {
        let features = AstLanguage::JavaScript.features();
        assert!(features.has_functions);
        assert!(features.has_classes);
        assert!(features.has_documentation);
        assert!(features.has_imports);
        assert_eq!(features.tier, LanguageTier::FullAst);
        assert!(features
            .complexity_factors
            .contains(&"closures".to_string()));
        assert!(features
            .complexity_factors
            .contains(&"promises".to_string()));
        assert!(features.extensions.contains(&"js".to_string()));
    }

    #[test]
    fn test_typescript_features() {
        let features = AstLanguage::TypeScript.features();
        assert!(features.has_functions);
        assert!(features.has_classes);
        assert!(features.has_documentation);
        assert!(features.has_imports);
        assert_eq!(features.tier, LanguageTier::FullAst);
        assert!(features
            .complexity_factors
            .contains(&"generic_types".to_string()));
        assert!(features
            .complexity_factors
            .contains(&"type_guards".to_string()));
    }

    #[test]
    fn test_rust_features() {
        let features = AstLanguage::Rust.features();
        assert!(features.has_functions);
        assert!(!features.has_classes); // Rust has structs, not classes
        assert!(features.has_documentation);
        assert!(features.has_imports);
        assert_eq!(features.tier, LanguageTier::FullAst);
        assert!(features
            .complexity_factors
            .contains(&"lifetimes".to_string()));
        assert!(features
            .complexity_factors
            .contains(&"borrowing".to_string()));
    }

    #[test]
    fn test_go_features() {
        let features = AstLanguage::Go.features();
        assert!(features.has_functions);
        assert!(!features.has_classes); // Go has structs/interfaces, not classes
        assert!(features.has_documentation);
        assert!(features.has_imports);
        assert_eq!(features.tier, LanguageTier::FullAst);
        assert!(features
            .complexity_factors
            .contains(&"goroutines".to_string()));
        assert!(features
            .complexity_factors
            .contains(&"channels".to_string()));
    }

    #[test]
    fn test_java_features() {
        let features = AstLanguage::Java.features();
        assert!(features.has_functions);
        assert!(features.has_classes);
        assert!(features.has_documentation);
        assert!(features.has_imports);
        assert_eq!(features.tier, LanguageTier::FullAst);
        assert!(features
            .complexity_factors
            .contains(&"inheritance".to_string()));
        assert!(features
            .complexity_factors
            .contains(&"generics".to_string()));
    }

    #[test]
    fn test_html_features() {
        let features = AstLanguage::Html.features();
        assert!(!features.has_functions);
        assert!(!features.has_classes);
        assert_eq!(features.tier, LanguageTier::SyntaxAware);
    }

    #[test]
    fn test_default_features() {
        // Languages that use the fallthrough case
        let c_features = AstLanguage::C.features();
        assert!(!c_features.has_functions);
        assert!(!c_features.has_classes);
        assert_eq!(c_features.tier, LanguageTier::Future);

        let cpp_features = AstLanguage::Cpp.features();
        assert!(!cpp_features.has_functions);
        assert_eq!(cpp_features.tier, LanguageTier::Future);

        let ruby_features = AstLanguage::Ruby.features();
        assert!(!ruby_features.has_functions);
        assert_eq!(ruby_features.tier, LanguageTier::Future);

        let csharp_features = AstLanguage::CSharp.features();
        assert!(!csharp_features.has_functions);
        assert_eq!(csharp_features.tier, LanguageTier::Future);
    }

    #[test]
    fn test_language_names() {
        assert_eq!(AstLanguage::Python.name(), "Python");
        assert_eq!(AstLanguage::JavaScript.name(), "JavaScript");
        assert_eq!(AstLanguage::TypeScript.name(), "TypeScript");
        assert_eq!(AstLanguage::Go.name(), "Go");
        assert_eq!(AstLanguage::Rust.name(), "Rust");
        assert_eq!(AstLanguage::Html.name(), "HTML");
        assert_eq!(AstLanguage::Java.name(), "Java");
        assert_eq!(AstLanguage::C.name(), "C");
        assert_eq!(AstLanguage::Cpp.name(), "C++");
        assert_eq!(AstLanguage::Ruby.name(), "Ruby");
        assert_eq!(AstLanguage::CSharp.name(), "C#");
    }

    #[test]
    fn test_all_language_tiers() {
        // Full AST
        assert_eq!(AstLanguage::Python.tier(), LanguageTier::FullAst);
        assert_eq!(AstLanguage::JavaScript.tier(), LanguageTier::FullAst);
        assert_eq!(AstLanguage::TypeScript.tier(), LanguageTier::FullAst);
        assert_eq!(AstLanguage::Go.tier(), LanguageTier::FullAst);
        assert_eq!(AstLanguage::Rust.tier(), LanguageTier::FullAst);

        // Syntax Aware
        assert_eq!(AstLanguage::Html.tier(), LanguageTier::SyntaxAware);

        // Future
        assert_eq!(AstLanguage::Java.tier(), LanguageTier::Future);
        assert_eq!(AstLanguage::C.tier(), LanguageTier::Future);
        assert_eq!(AstLanguage::Cpp.tier(), LanguageTier::Future);
        assert_eq!(AstLanguage::Ruby.tier(), LanguageTier::Future);
        assert_eq!(AstLanguage::CSharp.tier(), LanguageTier::Future);
    }

    #[test]
    fn test_language_tier_equality() {
        assert_eq!(LanguageTier::FullAst, LanguageTier::FullAst);
        assert_ne!(LanguageTier::FullAst, LanguageTier::SyntaxAware);
        assert_ne!(LanguageTier::SyntaxAware, LanguageTier::Future);
    }

    #[test]
    fn test_language_tier_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(LanguageTier::FullAst);
        set.insert(LanguageTier::SyntaxAware);
        set.insert(LanguageTier::Future);
        set.insert(LanguageTier::FullAst); // duplicate

        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_language_tier_clone() {
        let tier = LanguageTier::FullAst;
        let cloned = tier.clone();
        assert_eq!(tier, cloned);
    }

    #[test]
    fn test_language_tier_debug() {
        let tier = LanguageTier::FullAst;
        let debug_str = format!("{:?}", tier);
        assert!(debug_str.contains("FullAst"));
    }

    #[test]
    fn test_language_features_clone() {
        let features = AstLanguage::Python.features();
        let cloned = features.clone();
        assert_eq!(features.tier, cloned.tier);
        assert_eq!(features.has_functions, cloned.has_functions);
        assert_eq!(features.complexity_factors, cloned.complexity_factors);
    }

    #[test]
    fn test_language_features_debug() {
        let features = AstLanguage::Python.features();
        let debug_str = format!("{:?}", features);
        assert!(debug_str.contains("LanguageFeatures"));
        assert!(debug_str.contains("FullAst"));
    }

    #[test]
    fn test_language_features_serialize() {
        let features = AstLanguage::Python.features();
        let json = serde_json::to_string(&features).unwrap();
        let deserialized: LanguageFeatures = serde_json::from_str(&json).unwrap();
        assert_eq!(features.has_functions, deserialized.has_functions);
        assert_eq!(features.tier, deserialized.tier);
    }

    #[test]
    fn test_language_stats_serialize() {
        let stats = LanguageStats::calculate();
        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: LanguageStats = serde_json::from_str(&json).unwrap();
        assert_eq!(stats.total_languages, deserialized.total_languages);
        assert_eq!(stats.ast_supported, deserialized.ast_supported);
    }

    #[test]
    fn test_language_stats_clone() {
        let stats = LanguageStats::calculate();
        let cloned = stats.clone();
        assert_eq!(stats.total_languages, cloned.total_languages);
        assert_eq!(stats.by_tier, cloned.by_tier);
    }

    #[test]
    fn test_language_stats_debug() {
        let stats = LanguageStats::calculate();
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("LanguageStats"));
        assert!(debug_str.contains("total_languages"));
    }

    #[test]
    fn test_ast_language_equality() {
        assert_eq!(AstLanguage::Python, AstLanguage::Python);
        assert_ne!(AstLanguage::Python, AstLanguage::JavaScript);
    }

    #[test]
    fn test_ast_language_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(AstLanguage::Python);
        set.insert(AstLanguage::Rust);
        set.insert(AstLanguage::Python); // duplicate

        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_ast_language_copy() {
        let lang = AstLanguage::Python;
        let copied = lang;
        assert_eq!(lang, copied);
    }

    #[test]
    fn test_ast_language_serialize() {
        let lang = AstLanguage::Python;
        let json = serde_json::to_string(&lang).unwrap();
        let deserialized: AstLanguage = serde_json::from_str(&json).unwrap();
        assert_eq!(lang, deserialized);
    }

    #[test]
    fn test_ast_language_debug() {
        let lang = AstLanguage::Python;
        let debug_str = format!("{:?}", lang);
        assert!(debug_str.contains("Python"));
    }

    #[test]
    fn test_all_supported_languages_unique() {
        let all = AstLanguage::all_supported();
        let mut unique: Vec<_> = all.clone();
        unique.sort_by_key(|l| l.name());
        unique.dedup();
        assert_eq!(all.len(), unique.len());
    }

    #[test]
    fn test_language_tier_serialize() {
        let tier = LanguageTier::FullAst;
        let json = serde_json::to_string(&tier).unwrap();
        let deserialized: LanguageTier = serde_json::from_str(&json).unwrap();
        assert_eq!(tier, deserialized);
    }

    #[test]
    fn test_language_stats_ast_supported_count() {
        let stats = LanguageStats::calculate();
        // AST supported = FullAst (5) + SyntaxAware (1) = 6
        assert_eq!(stats.ast_supported, 6);
    }

    #[test]
    fn test_language_stats_tier_breakdown() {
        let stats = LanguageStats::calculate();
        // 5 FullAst (Python, JavaScript, TypeScript, Go, Rust)
        assert_eq!(stats.by_tier.get(&LanguageTier::FullAst), Some(&5));
        // 1 SyntaxAware (Html)
        assert_eq!(stats.by_tier.get(&LanguageTier::SyntaxAware), Some(&1));
        // 5 Future (Java, C, Cpp, Ruby, CSharp)
        assert_eq!(stats.by_tier.get(&LanguageTier::Future), Some(&5));
    }
}
