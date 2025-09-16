//! # Multi-Language AST Analysis Support
//!
//! This module provides comprehensive language support for AST-based analysis,
//! significantly expanding beyond the basic regex-based parsing. Implements
//! proper tree-sitter AST parsing for 20+ programming languages.
//!
//! ## Supported Languages
//!
//! ### Tier 1: Full AST Support (Tree-sitter)
//! - Python, JavaScript, TypeScript, Go, Rust (existing)
//! - Java, C#, C, C++, PHP, Ruby, Swift, Kotlin (to be added)
//!
//! ### Tier 2: Syntax-Aware Support (Tree-sitter)
//! - HTML, CSS, JSON, YAML, XML, Markdown
//! - SQL, Bash, PowerShell, Dockerfile
//!
//! ### Tier 3: Basic Structure Analysis (Regex + Heuristics)
//! - Scala, Dart, Perl, R, Julia, Elixir, Lua, Haskell
//!
//! ## Key Features
//!
//! - **Function/Class Extraction**: Identify all functions, classes, and methods
//! - **Documentation Coverage**: Analyze docstrings, comments, and documentation
//! - **Language-Specific Complexity**: Tailored complexity metrics per language
//! - **Import/Dependency Analysis**: Accurate dependency graph construction
//! - **Symbol Resolution**: Variable, function, and class usage tracking

pub mod ast_language;
pub mod documentation_analysis;
pub mod function_extraction;
pub mod language_metrics;
pub mod symbol_analysis;

pub use ast_language::{AstLanguage, LanguageFeatures, LanguageTier};
pub use documentation_analysis::{DocumentationAnalyzer, DocumentationCoverage};
pub use function_extraction::{ClassInfo, FunctionExtractor, FunctionInfo};
pub use language_metrics::{LanguageMetrics, LanguageSpecificComplexity};
pub use symbol_analysis::{SymbolAnalyzer, SymbolType, SymbolUsage};

use scribe_core::Result;
use std::collections::HashMap;

/// Main language support coordinator
pub struct LanguageSupport {
    /// Function extractors by language
    function_extractors: HashMap<AstLanguage, FunctionExtractor>,
    /// Documentation analyzers by language
    doc_analyzers: HashMap<AstLanguage, DocumentationAnalyzer>,
    /// Symbol analyzers by language
    symbol_analyzers: HashMap<AstLanguage, SymbolAnalyzer>,
}

impl LanguageSupport {
    /// Create a new language support system
    pub fn new() -> Result<Self> {
        let mut support = Self {
            function_extractors: HashMap::new(),
            doc_analyzers: HashMap::new(),
            symbol_analyzers: HashMap::new(),
        };

        // Initialize extractors for all supported languages
        support.initialize_extractors()?;

        Ok(support)
    }

    /// Initialize extractors for all supported languages
    fn initialize_extractors(&mut self) -> Result<()> {
        for language in AstLanguage::all_supported() {
            self.function_extractors
                .insert(language, FunctionExtractor::new(language)?);
            self.doc_analyzers
                .insert(language, DocumentationAnalyzer::new(language)?);
            self.symbol_analyzers
                .insert(language, SymbolAnalyzer::new(language)?);
        }
        Ok(())
    }

    /// Get language features and capabilities
    pub fn get_language_features(&self, language: AstLanguage) -> LanguageFeatures {
        language.features()
    }

    /// Extract all functions and classes from source code
    pub fn extract_functions(
        &mut self,
        content: &str,
        language: AstLanguage,
    ) -> Result<Vec<FunctionInfo>> {
        if let Some(extractor) = self.function_extractors.get_mut(&language) {
            extractor.extract_functions(content)
        } else {
            Ok(Vec::new())
        }
    }

    /// Analyze documentation coverage
    pub fn analyze_documentation(
        &self,
        content: &str,
        language: AstLanguage,
    ) -> Result<DocumentationCoverage> {
        if let Some(analyzer) = self.doc_analyzers.get(&language) {
            analyzer.analyze_coverage(content)
        } else {
            Ok(DocumentationCoverage::default())
        }
    }

    /// Analyze symbol usage patterns
    pub fn analyze_symbols(
        &self,
        content: &str,
        language: AstLanguage,
    ) -> Result<Vec<SymbolUsage>> {
        if let Some(analyzer) = self.symbol_analyzers.get(&language) {
            analyzer.analyze_symbols(content)
        } else {
            Ok(Vec::new())
        }
    }

    /// Get language-specific complexity metrics
    pub fn calculate_language_metrics(
        &self,
        content: &str,
        language: AstLanguage,
    ) -> Result<LanguageMetrics> {
        LanguageMetrics::calculate(content, language)
    }

    /// Check if a language is supported
    pub fn is_supported(&self, language: AstLanguage) -> bool {
        self.function_extractors.contains_key(&language)
    }

    /// Get all supported languages
    pub fn supported_languages(&self) -> Vec<AstLanguage> {
        self.function_extractors.keys().copied().collect()
    }
}

impl Default for LanguageSupport {
    fn default() -> Self {
        Self::new().expect("Failed to create LanguageSupport")
    }
}

/// Comprehensive analysis result for a source file
#[derive(Debug, Clone)]
pub struct LanguageAnalysisResult {
    /// Detected programming language
    pub language: AstLanguage,
    /// Language tier and capabilities
    pub tier: LanguageTier,
    /// Extracted functions and classes
    pub functions: Vec<FunctionInfo>,
    /// Documentation coverage analysis
    pub documentation: DocumentationCoverage,
    /// Symbol usage patterns
    pub symbols: Vec<SymbolUsage>,
    /// Language-specific metrics
    pub metrics: LanguageMetrics,
}

/// Main entry point for language analysis
pub fn analyze_file_language(
    content: &str,
    file_extension: &str,
    language_support: &mut LanguageSupport,
) -> Result<Option<LanguageAnalysisResult>> {
    let language = match AstLanguage::from_extension(file_extension) {
        Some(lang) => lang,
        None => return Ok(None),
    };

    if !language_support.is_supported(language) {
        return Ok(None);
    }

    let functions = language_support.extract_functions(content, language)?;
    let documentation = language_support.analyze_documentation(content, language)?;
    let symbols = language_support.analyze_symbols(content, language)?;
    let metrics = language_support.calculate_language_metrics(content, language)?;

    Ok(Some(LanguageAnalysisResult {
        language,
        tier: language.tier(),
        functions,
        documentation,
        symbols,
        metrics,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_support_creation() {
        let support = LanguageSupport::new();
        if let Err(e) = &support {
            println!("Error creating LanguageSupport: {:?}", e);
        }
        assert!(support.is_ok());
    }

    #[test]
    fn test_supported_languages() {
        let support = LanguageSupport::new().unwrap();
        let languages = support.supported_languages();

        // Should have at least the core languages
        assert!(languages.contains(&AstLanguage::Python));
        assert!(languages.contains(&AstLanguage::JavaScript));
        assert!(languages.contains(&AstLanguage::Rust));
    }

    #[test]
    fn test_python_function_extraction() {
        let mut support = LanguageSupport::new().unwrap();
        let python_code = r#"
def hello_world():
    """A simple function."""
    print("Hello, World!")

class Calculator:
    """A simple calculator class."""
    
    def add(self, a, b):
        """Add two numbers."""
        return a + b
"#;

        let functions = support.extract_functions(python_code, AstLanguage::Python);
        assert!(functions.is_ok());
        let functions = functions.unwrap();
        assert!(!functions.is_empty());
    }

    #[test]
    fn test_language_analysis() {
        let mut support = LanguageSupport::new().unwrap();
        let result = analyze_file_language("def test(): pass", "py", &mut support);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().language, AstLanguage::Python);
    }
}
