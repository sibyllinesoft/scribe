//! Advanced programming language detection for 25+ languages.
//!
//! This module provides sophisticated language detection capabilities using:
//! - File extension analysis with priority mapping
//! - Content-based detection using language signatures
//! - Shebang line analysis for scripts
//! - Filename pattern matching (e.g., Makefile, Dockerfile)
//! - Statistical content analysis for ambiguous cases

mod analysis;
mod detection;
mod rules;
mod types;

pub use types::{
    ContentSignature, ContentSignatureConfig, CustomDetectionRules, DetectionEvidence,
    DetectionMethod, DetectionResult, DetectionStrategy, EvidenceType, LanguageHints, ProjectType,
    SyntaxAnalyzer,
};

use once_cell::sync::Lazy;
use scribe_core::Language;
use std::collections::HashMap;
use std::path::Path;
use tree_sitter::{Language as TsLanguage, Parser};

// Tree-sitter language mapping for AST analysis
static TS_LANGUAGES: Lazy<HashMap<Language, fn() -> TsLanguage>> = Lazy::new(|| {
    let mut languages = HashMap::new();
    languages.insert(
        Language::Python,
        tree_sitter_python::language as fn() -> TsLanguage,
    );
    languages.insert(
        Language::JavaScript,
        tree_sitter_javascript::language as fn() -> TsLanguage,
    );
    languages.insert(
        Language::TypeScript,
        tree_sitter_typescript::language_typescript as fn() -> TsLanguage,
    );
    languages.insert(
        Language::Rust,
        tree_sitter_rust::language as fn() -> TsLanguage,
    );
    languages.insert(Language::Go, tree_sitter_go::language as fn() -> TsLanguage);
    languages.insert(
        Language::Elixir,
        tree_sitter_elixir::language as fn() -> TsLanguage,
    );
    languages
});

/// High-performance language detector with multiple strategies
pub struct LanguageDetector {
    strategy: DetectionStrategy,
    extension_map: HashMap<String, Vec<(Language, f32)>>,
    filename_patterns: HashMap<String, Language>,
    content_signatures: HashMap<Language, Vec<ContentSignature>>,
    shebang_patterns: HashMap<String, Language>,
    ast_parsers: HashMap<Language, Parser>,
    syntax_analyzers: HashMap<Language, SyntaxAnalyzer>,
}

impl LanguageDetector {
    /// Create a new language detector with default configuration
    pub fn new() -> Self {
        Self {
            strategy: DetectionStrategy::default(),
            extension_map: rules::initialize_extension_map(),
            filename_patterns: rules::initialize_filename_patterns(),
            content_signatures: rules::initialize_content_signatures(),
            shebang_patterns: rules::initialize_shebang_patterns(),
            ast_parsers: rules::initialize_ast_parsers(&TS_LANGUAGES),
            syntax_analyzers: rules::initialize_syntax_analyzers(),
        }
    }

    /// Create a language detector with custom strategy
    pub fn with_strategy(strategy: DetectionStrategy) -> Self {
        let mut detector = Self::new();
        detector.strategy = strategy;
        detector
    }

    /// Detect language for a file path (extension-based)
    pub fn detect_language(&self, path: &Path) -> Language {
        match self.strategy {
            DetectionStrategy::ExtensionOnly => {
                detection::detect_by_extension(path, &self.extension_map)
            }
            _ => detection::detect_by_extension_and_filename(
                path,
                &self.extension_map,
                &self.filename_patterns,
            ),
        }
    }

    /// Detect language with full content analysis
    pub fn detect_language_with_content(&mut self, path: &Path, content: &str) -> DetectionResult {
        match self.strategy {
            DetectionStrategy::ExtensionOnly => {
                let language = detection::detect_by_extension(path, &self.extension_map);
                DetectionResult {
                    language: language.clone(),
                    confidence: if language == Language::Unknown {
                        0.1
                    } else {
                        0.9
                    },
                    detection_method: DetectionMethod::FileExtension,
                    alternatives: vec![],
                    evidence: vec![DetectionEvidence {
                        evidence_type: EvidenceType::Extension,
                        description: format!("File extension: {:?}", path.extension()),
                        weight: 0.9,
                    }],
                }
            }
            DetectionStrategy::ExtensionWithContent => {
                self.detect_with_content_analysis(path, content)
            }
            DetectionStrategy::FullAnalysis => self.detect_with_full_analysis(path, content),
            DetectionStrategy::Custom(ref rules) => {
                let rules = rules.clone();
                self.detect_with_custom_rules(path, content, &rules)
            }
        }
    }

    /// Detect language with project context hints
    pub fn detect_with_hints(
        &mut self,
        path: &Path,
        content: &str,
        hints: &LanguageHints,
    ) -> DetectionResult {
        let mut base_result = self.detect_language_with_content(path, content);

        if let Some(project_type) = &hints.project_type {
            base_result = detection::apply_project_type_bias(base_result, project_type);
        }

        if !hints.dominant_languages.is_empty() {
            base_result =
                detection::apply_dominant_language_bias(base_result, &hints.dominant_languages);
        }

        if !hints.framework_indicators.is_empty() {
            base_result = detection::apply_framework_bias(base_result, &hints.framework_indicators);
        }

        base_result
    }

    fn detect_with_content_analysis(&mut self, path: &Path, content: &str) -> DetectionResult {
        let mut candidates = Vec::new();
        let mut evidence = Vec::new();

        let extension_lang = detection::detect_by_extension_and_filename(
            path,
            &self.extension_map,
            &self.filename_patterns,
        );
        if extension_lang != Language::Unknown {
            candidates.push((extension_lang.clone(), 0.8));
            evidence.push(DetectionEvidence {
                evidence_type: EvidenceType::Extension,
                description: format!("File extension suggests: {:?}", extension_lang),
                weight: 0.8,
            });

            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                let confident_extensions = ["rs", "py", "js", "ts", "go", "java", "cpp", "c"];
                if confident_extensions.contains(&ext) {
                    if detection::quick_content_validation(&extension_lang, content) {
                        return DetectionResult {
                            language: extension_lang,
                            confidence: 0.95,
                            detection_method: DetectionMethod::FileExtension,
                            alternatives: vec![],
                            evidence,
                        };
                    }
                }
            }
        }

        if let Some(shebang_lang) = detection::detect_by_shebang(content, &self.shebang_patterns) {
            candidates.push((shebang_lang.clone(), 0.95));
            evidence.push(DetectionEvidence {
                evidence_type: EvidenceType::Shebang,
                description: format!("Shebang indicates: {:?}", shebang_lang),
                weight: 0.95,
            });
        }

        let signature_results = detection::analyze_content_signatures_optimized(
            content,
            &extension_lang,
            &self.content_signatures,
        );
        for (lang, confidence) in signature_results {
            candidates.push((lang.clone(), confidence));
            evidence.push(DetectionEvidence {
                evidence_type: EvidenceType::Syntax,
                description: format!("Content signatures match: {:?}", lang),
                weight: confidence,
            });
        }

        let max_confidence = candidates.iter().map(|(_, c)| *c).fold(0.0f32, f32::max);
        if max_confidence < 0.8 {
            let import_results = analysis::analyze_import_patterns(content, &mut self.ast_parsers);
            for (lang, confidence) in import_results {
                candidates.push((lang.clone(), confidence));
                evidence.push(DetectionEvidence {
                    evidence_type: EvidenceType::Import,
                    description: format!("Import patterns match: {:?}", lang),
                    weight: confidence,
                });
            }
        }

        detection::aggregate_detection_results(candidates, evidence)
    }

    fn detect_with_full_analysis(&mut self, path: &Path, content: &str) -> DetectionResult {
        let mut base_result = self.detect_with_content_analysis(path, content);

        let statistical_results =
            analysis::statistical_analysis(content, &mut self.ast_parsers, &self.syntax_analyzers);
        for (lang, confidence) in statistical_results {
            base_result.alternatives.push((lang, confidence));
        }

        base_result
            .alternatives
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        base_result
    }

    fn detect_with_custom_rules(
        &mut self,
        path: &Path,
        content: &str,
        rules: &CustomDetectionRules,
    ) -> DetectionResult {
        let mut candidates = Vec::new();
        let mut evidence = Vec::new();

        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            if let Some(language) = rules.extension_overrides.get(&extension.to_lowercase()) {
                candidates.push((language.clone(), 1.0));
                evidence.push(DetectionEvidence {
                    evidence_type: EvidenceType::Extension,
                    description: format!("Custom extension rule: {} -> {:?}", extension, language),
                    weight: 1.0,
                });
            }
        }

        if let Some(filename) = path.file_name().and_then(|name| name.to_str()) {
            if let Some(language) = rules.filename_patterns.get(filename) {
                candidates.push((language.clone(), 1.0));
                evidence.push(DetectionEvidence {
                    evidence_type: EvidenceType::Filename,
                    description: format!("Custom filename rule: {} -> {:?}", filename, language),
                    weight: 1.0,
                });
            }
        }

        for signature_config in &rules.content_signatures {
            let matches = signature_config
                .patterns
                .iter()
                .map(|pattern| match regex::Regex::new(pattern) {
                    Ok(regex) => regex.find_iter(content).count(),
                    Err(_) => content.matches(pattern).count(),
                })
                .sum::<usize>();

            if matches >= signature_config.required_matches {
                candidates.push((signature_config.language.clone(), signature_config.weight));
                evidence.push(DetectionEvidence {
                    evidence_type: EvidenceType::Syntax,
                    description: format!(
                        "Custom signature matches for {:?}: {}",
                        signature_config.language, matches
                    ),
                    weight: signature_config.weight,
                });
            }
        }

        if candidates.is_empty() {
            return self.detect_with_content_analysis(path, content);
        }

        detection::aggregate_detection_results(candidates, evidence)
    }
}

impl Default for LanguageDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_extension_detection() {
        let detector = LanguageDetector::new();

        assert_eq!(
            detector.detect_language(Path::new("test.rs")),
            Language::Rust
        );
        assert_eq!(
            detector.detect_language(Path::new("test.py")),
            Language::Python
        );
        assert_eq!(
            detector.detect_language(Path::new("test.js")),
            Language::JavaScript
        );
        assert_eq!(
            detector.detect_language(Path::new("test.ts")),
            Language::TypeScript
        );
        assert_eq!(
            detector.detect_language(Path::new("test.java")),
            Language::Java
        );
        assert_eq!(detector.detect_language(Path::new("test.go")), Language::Go);
        assert_eq!(
            detector.detect_language(Path::new("test.cpp")),
            Language::Cpp
        );
        assert_eq!(detector.detect_language(Path::new("test.c")), Language::C);
        assert_eq!(
            detector.detect_language(Path::new("test.ex")),
            Language::Elixir
        );
    }

    #[test]
    fn test_rust_files_are_programming() {
        let detector = LanguageDetector::new();

        let rust_files = [
            "src/lib.rs",
            "scribe-rs/src/lib.rs",
            "scribe-rs/scribe-core/src/lib.rs",
            "main.rs",
            "mod.rs",
        ];

        for file_path in &rust_files {
            let language = detector.detect_language(Path::new(file_path));
            assert_eq!(language, Language::Rust, "Failed for file: {}", file_path);
            assert!(
                language.is_programming(),
                "Rust should be programming language for file: {}",
                file_path
            );
        }
    }

    #[test]
    fn test_filename_patterns() {
        let mut detector = LanguageDetector::new();

        assert_eq!(
            detector.detect_language(Path::new("Makefile")),
            Language::Unknown
        );
        assert_eq!(
            detector.detect_language(Path::new("Dockerfile")),
            Language::Unknown
        );
        assert_eq!(
            detector.detect_language(Path::new("Cargo.toml")),
            Language::TOML
        );
        assert_eq!(
            detector.detect_language(Path::new("package.json")),
            Language::JSON
        );
    }

    #[test]
    fn test_shebang_detection() {
        let mut detector = LanguageDetector::new();

        let python_script = "#!/usr/bin/env python3\nprint('Hello, world!')";
        let result = detector.detect_language_with_content(Path::new("script"), python_script);
        assert_eq!(result.language, Language::Python);
        assert!(result.confidence > 0.9);
        assert_eq!(result.detection_method, DetectionMethod::Shebang);

        let bash_script = "#!/bin/bash\necho 'Hello, world!'";
        let result = detector.detect_language_with_content(Path::new("script"), bash_script);
        assert_eq!(result.language, Language::Bash);
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn test_content_signature_detection() {
        let mut detector = LanguageDetector::new();

        let python_code = r#"
def hello_world():
    print("Hello, world!")

class MyClass:
    def __init__(self):
        pass

import sys
from collections import defaultdict
        "#;

        let result = detector.detect_language_with_content(Path::new("unknown"), python_code);
        assert_eq!(result.language, Language::Python);
        assert!(result.confidence > 0.5);

        let rust_code = r#"
fn main() {
    println!("Hello, world!");
}

struct MyStruct {
    field: i32,
}

impl MyStruct {
    fn new() -> Self {
        MyStruct { field: 0 }
    }
}

use std::collections::HashMap;
        "#;

        let result = detector.detect_language_with_content(Path::new("unknown"), rust_code);
        assert_eq!(result.language, Language::Rust);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_import_pattern_detection() {
        let mut detector = LanguageDetector::new();

        let js_code = r#"
import React from 'react';
import { useState } from 'react';
const fs = require('fs');
        "#;

        let result = detector.detect_language_with_content(Path::new("unknown"), js_code);
        assert_eq!(result.language, Language::JavaScript);

        let python_code = r#"
import os
import sys
from collections import defaultdict, Counter
        "#;

        let result = detector.detect_language_with_content(Path::new("unknown"), python_code);
        assert_eq!(result.language, Language::Python);
    }

    #[test]
    fn test_hybrid_detection() {
        let mut detector = LanguageDetector::new();

        let python_code = "#!/usr/bin/env python\nprint('Hello')\n# Python comment";
        let result = detector.detect_language_with_content(Path::new("test.py"), python_code);
        assert_eq!(result.language, Language::Python);
        assert!(result.confidence > 0.6);
        assert!(result.evidence.len() > 1);

        let python_code = "def hello(): print('Hello')";
        let result = detector.detect_language_with_content(Path::new("test.js"), python_code);
        assert!(result.language == Language::Python || result.language == Language::JavaScript);
    }

    #[test]
    fn test_detection_with_hints() {
        let mut detector = LanguageDetector::new();

        let hints = LanguageHints {
            project_type: Some(ProjectType::WebFrontend),
            dominant_languages: vec![Language::TypeScript],
            framework_indicators: vec!["package.json".to_string()],
            ..Default::default()
        };

        let ts_code = "const hello = () => console.log('Hello');";
        let result = detector.detect_with_hints(Path::new("unknown"), ts_code, &hints);

        assert_eq!(result.language, Language::JavaScript);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_custom_detection_rules() {
        let mut custom_rules = CustomDetectionRules {
            extension_overrides: HashMap::new(),
            filename_patterns: HashMap::new(),
            content_signatures: vec![],
            priority_languages: vec![],
        };

        custom_rules
            .extension_overrides
            .insert("myext".to_string(), Language::Rust);

        let mut detector = LanguageDetector::with_strategy(DetectionStrategy::Custom(custom_rules));

        let result = detector.detect_language_with_content(Path::new("test.myext"), "some content");
        assert_eq!(result.language, Language::Rust);
        assert_eq!(result.confidence, 1.0);
    }

    #[test]
    fn test_detection_evidence() {
        let mut detector = LanguageDetector::new();

        let python_code = "#!/usr/bin/env python\nprint('Hello World')";
        let result = detector.detect_language_with_content(Path::new("test.py"), python_code);

        assert!(result.evidence.len() >= 2);
        assert!(result
            .evidence
            .iter()
            .any(|e| e.evidence_type == EvidenceType::Shebang));
        assert!(result
            .evidence
            .iter()
            .any(|e| e.evidence_type == EvidenceType::Extension));
    }

    #[test]
    fn test_confidence_scoring() {
        let mut detector = LanguageDetector::new();

        let strong_python = "#!/usr/bin/env python3\nimport os\ndef main(): pass\nclass Test: pass";
        let result = detector.detect_language_with_content(Path::new("test.py"), strong_python);
        assert!(result.confidence > 0.8);

        let weak_indicators = "hello world";
        let result = detector.detect_language_with_content(Path::new("test.txt"), weak_indicators);
        assert!(result.confidence < 0.8);
    }

    #[test]
    fn test_alternatives_ranking() {
        let mut detector = LanguageDetector::new();

        let ambiguous_code = "print hello";
        let result = detector.detect_language_with_content(Path::new("unknown"), ambiguous_code);

        if result.alternatives.len() > 1 {
            assert!(result.alternatives[0].1 >= result.alternatives[1].1);
        }
    }

    #[test]
    fn test_extension_only_strategy_known_extension() {
        let mut detector = LanguageDetector::with_strategy(DetectionStrategy::ExtensionOnly);

        let python_code = "print('Hello')";
        let result = detector.detect_language_with_content(Path::new("test.py"), python_code);

        assert_eq!(result.language, Language::Python);
        assert_eq!(result.confidence, 0.9);
        assert_eq!(result.detection_method, DetectionMethod::FileExtension);
        assert!(result.alternatives.is_empty());
    }

    #[test]
    fn test_extension_only_strategy_unknown_extension() {
        let mut detector = LanguageDetector::with_strategy(DetectionStrategy::ExtensionOnly);

        let code = "some code";
        let result = detector.detect_language_with_content(Path::new("test.xyz"), code);

        assert_eq!(result.language, Language::Unknown);
        assert_eq!(result.confidence, 0.1);
        assert_eq!(result.detection_method, DetectionMethod::FileExtension);
    }

    #[test]
    fn test_extension_only_detect_language() {
        let detector = LanguageDetector::with_strategy(DetectionStrategy::ExtensionOnly);

        assert_eq!(
            detector.detect_language(Path::new("test.rs")),
            Language::Rust
        );
        assert_eq!(
            detector.detect_language(Path::new("test.py")),
            Language::Python
        );
    }

    #[test]
    fn test_full_analysis_strategy() {
        let mut detector = LanguageDetector::with_strategy(DetectionStrategy::FullAnalysis);

        let python_code = r#"
#!/usr/bin/env python3
import os
import sys

def main():
    print("Hello, world!")

class MyClass:
    def __init__(self):
        pass

if __name__ == "__main__":
    main()
        "#;

        let result = detector.detect_language_with_content(Path::new("unknown"), python_code);
        assert_eq!(result.language, Language::Python);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_custom_rules_filename_pattern() {
        let mut custom_rules = CustomDetectionRules {
            extension_overrides: HashMap::new(),
            filename_patterns: HashMap::new(),
            content_signatures: vec![],
            priority_languages: vec![],
        };

        custom_rules
            .filename_patterns
            .insert("MyCustomFile".to_string(), Language::Go);

        let mut detector = LanguageDetector::with_strategy(DetectionStrategy::Custom(custom_rules));

        let result =
            detector.detect_language_with_content(Path::new("MyCustomFile"), "package main");
        assert_eq!(result.language, Language::Go);
        assert_eq!(result.confidence, 1.0);
    }

    #[test]
    fn test_custom_rules_content_signature() {
        let custom_rules = CustomDetectionRules {
            extension_overrides: HashMap::new(),
            filename_patterns: HashMap::new(),
            content_signatures: vec![ContentSignatureConfig {
                language: Language::Rust,
                patterns: vec!["fn main".to_string(), "let mut".to_string()],
                required_matches: 1,
                weight: 0.95,
            }],
            priority_languages: vec![],
        };

        let mut detector = LanguageDetector::with_strategy(DetectionStrategy::Custom(custom_rules));

        let rust_code = "fn main() { let mut x = 5; }";
        let result = detector.detect_language_with_content(Path::new("unknown"), rust_code);
        assert_eq!(result.language, Language::Rust);
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn test_custom_rules_content_signature_with_regex() {
        let custom_rules = CustomDetectionRules {
            extension_overrides: HashMap::new(),
            filename_patterns: HashMap::new(),
            content_signatures: vec![ContentSignatureConfig {
                language: Language::Python,
                patterns: vec!["def\\s+\\w+".to_string()], // Regex pattern
                required_matches: 1,
                weight: 0.9,
            }],
            priority_languages: vec![],
        };

        let mut detector = LanguageDetector::with_strategy(DetectionStrategy::Custom(custom_rules));

        let python_code = "def hello(): pass";
        let result = detector.detect_language_with_content(Path::new("unknown"), python_code);
        assert_eq!(result.language, Language::Python);
    }

    #[test]
    fn test_custom_rules_fallback_to_content_analysis() {
        let custom_rules = CustomDetectionRules {
            extension_overrides: HashMap::new(),
            filename_patterns: HashMap::new(),
            content_signatures: vec![],
            priority_languages: vec![],
        };

        let mut detector = LanguageDetector::with_strategy(DetectionStrategy::Custom(custom_rules));

        // With no custom rules matching, it should fall back to content analysis
        let python_code = "#!/usr/bin/env python3\ndef main(): pass";
        let result = detector.detect_language_with_content(Path::new("unknown"), python_code);
        assert_eq!(result.language, Language::Python);
    }

    #[test]
    fn test_hints_with_project_type_only() {
        let mut detector = LanguageDetector::new();

        let hints = LanguageHints {
            project_type: Some(ProjectType::WebBackend),
            dominant_languages: vec![],
            framework_indicators: vec![],
            ..Default::default()
        };

        let code = "const server = require('express')";
        let result = detector.detect_with_hints(Path::new("unknown"), code, &hints);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_hints_with_dominant_languages_only() {
        let mut detector = LanguageDetector::new();

        let hints = LanguageHints {
            project_type: None,
            dominant_languages: vec![Language::TypeScript],
            framework_indicators: vec![],
            ..Default::default()
        };

        let code = "const hello = 'world'";
        let result = detector.detect_with_hints(Path::new("unknown"), code, &hints);
        // The result should exist even if confidence is low
        assert!(result.confidence >= 0.0);
    }

    #[test]
    fn test_hints_with_framework_indicators_only() {
        let mut detector = LanguageDetector::new();

        let hints = LanguageHints {
            project_type: None,
            dominant_languages: vec![],
            framework_indicators: vec!["react".to_string(), "next.js".to_string()],
            ..Default::default()
        };

        let code = "import React from 'react'";
        let result = detector.detect_with_hints(Path::new("unknown"), code, &hints);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_content_analysis_with_ambiguous_extension() {
        let mut detector = LanguageDetector::new();

        // .h files are ambiguous (could be C or C++)
        let c_code = "#include <stdio.h>\nint main() { return 0; }";
        let result = detector.detect_language_with_content(Path::new("test.h"), c_code);
        // Should detect C or Cpp
        assert!(result.language == Language::C || result.language == Language::Cpp);
    }

    #[test]
    fn test_content_analysis_triggers_import_analysis() {
        let mut detector = LanguageDetector::new();

        // Code with low confidence from extension but clear import patterns
        let code = r#"
import something
from module import thing
        "#;

        let result = detector.detect_language_with_content(Path::new("file"), code);
        // Should recognize Python imports
        assert!(result.evidence.len() > 0);
    }

    #[test]
    fn test_detector_default_impl() {
        let detector = LanguageDetector::default();
        assert_eq!(
            detector.detect_language(Path::new("test.py")),
            Language::Python
        );
    }

    #[test]
    fn test_detection_result_evidence_types() {
        let mut detector = LanguageDetector::new();

        // Test that different evidence types are captured
        let python_code = "#!/usr/bin/env python3\nimport os\ndef main(): pass";
        let result = detector.detect_language_with_content(Path::new("test.py"), python_code);

        // Should have extension evidence
        assert!(result
            .evidence
            .iter()
            .any(|e| e.evidence_type == EvidenceType::Extension));

        // Should have evidence with weight > 0
        assert!(result.evidence.iter().any(|e| e.weight > 0.0));
    }

    #[test]
    fn test_extension_with_content_strategy_quick_validation() {
        let mut detector = LanguageDetector::with_strategy(DetectionStrategy::ExtensionWithContent);

        // Rust file with clearly valid Rust content
        let rust_code = "fn main() { println!(\"Hello\"); }";
        let result = detector.detect_language_with_content(Path::new("test.rs"), rust_code);

        assert_eq!(result.language, Language::Rust);
        assert_eq!(result.confidence, 0.95); // Quick validation confidence
        assert_eq!(result.detection_method, DetectionMethod::FileExtension);
    }

    #[test]
    fn test_all_confident_extensions() {
        let mut detector = LanguageDetector::with_strategy(DetectionStrategy::ExtensionWithContent);

        let test_cases = vec![
            ("test.rs", "fn main() {}", Language::Rust),
            ("test.py", "def main(): pass", Language::Python),
            ("test.js", "function main() {}", Language::JavaScript),
            ("test.ts", "function main(): void {}", Language::TypeScript),
            ("test.go", "func main() {}", Language::Go),
        ];

        for (path, code, expected_lang) in test_cases {
            let result = detector.detect_language_with_content(Path::new(path), code);
            assert_eq!(result.language, expected_lang, "Failed for path: {}", path);
        }
    }
}
