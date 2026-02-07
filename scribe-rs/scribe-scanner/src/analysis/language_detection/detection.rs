//! Core detection logic for language detection.

use super::types::{
    ContentSignature, CustomDetectionRules, DetectionEvidence, DetectionMethod, DetectionResult,
    EvidenceType, LanguageHints, ProjectType, SyntaxAnalyzer,
};
use scribe_core::Language;
use std::collections::HashMap;
use std::path::Path;
use tree_sitter::{Node, Parser};

/// Detect language by extension only
pub fn detect_by_extension(
    path: &Path,
    extension_map: &HashMap<String, Vec<(Language, f32)>>,
) -> Language {
    if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
        if let Some(languages) = extension_map.get(&extension.to_lowercase()) {
            return languages[0].0.clone();
        }
    }
    Language::Unknown
}

/// Detect language by extension and filename patterns
pub fn detect_by_extension_and_filename(
    path: &Path,
    extension_map: &HashMap<String, Vec<(Language, f32)>>,
    filename_patterns: &HashMap<String, Language>,
) -> Language {
    if let Some(filename) = path.file_name().and_then(|name| name.to_str()) {
        if let Some(language) = filename_patterns.get(filename) {
            return language.clone();
        }
    }
    detect_by_extension(path, extension_map)
}

/// Detect language from shebang line
pub fn detect_by_shebang(
    content: &str,
    shebang_patterns: &HashMap<String, Language>,
) -> Option<Language> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return None;
    }

    let first_line = lines[0];
    if first_line.starts_with("#!") {
        let shebang_path = &first_line[2..].trim();

        for (pattern, language) in shebang_patterns {
            if shebang_path.contains(pattern) {
                return Some(language.clone());
            }
        }
    }

    None
}

/// Quick content validation for extension-based detection
pub fn quick_content_validation(language: &Language, content: &str) -> bool {
    let markers = get_language_markers(language);
    if markers.is_empty() {
        return true;
    }
    markers.iter().any(|marker| content.contains(marker))
}

/// Get language-specific content markers for quick validation
pub fn get_language_markers(language: &Language) -> &'static [&'static str] {
    match language {
        Language::Rust => &["fn ", "use ", "struct "],
        Language::Python => &["def ", "import ", "class "],
        Language::JavaScript => &["function ", "const ", "var "],
        Language::TypeScript => &["interface ", "type ", ": "],
        Language::Go => &["func ", "package ", "import "],
        Language::Java => &["class ", "public ", "import "],
        Language::C => &["#include", "int main", "void "],
        Language::Cpp => &["#include", "class ", "namespace "],
        _ => &[],
    }
}

/// Count signature matches efficiently using pre-compiled regexes
pub fn count_signature_matches(signature: &ContentSignature, content: &str) -> usize {
    signature
        .patterns
        .iter()
        .map(|regex| regex.find_iter(content).count())
        .sum::<usize>()
}

/// Optimized content signature analysis that prioritizes the extension language
pub fn analyze_content_signatures_optimized(
    content: &str,
    extension_lang: &Language,
    content_signatures: &HashMap<Language, Vec<ContentSignature>>,
) -> Vec<(Language, f32)> {
    let mut results = Vec::new();

    // First try the extension language if available
    if *extension_lang != Language::Unknown {
        if let Some(signatures) = content_signatures.get(extension_lang) {
            for signature in signatures {
                let matches = count_signature_matches(signature, content);
                if matches >= signature.required_matches {
                    let confidence =
                        (matches as f32 / signature.patterns.len() as f32) * signature.weight;
                    results.push((extension_lang.clone(), confidence));

                    if confidence > 0.7 {
                        return results;
                    }
                }
            }
        }
    }

    // If extension language didn't match well, try others
    for (language, signatures) in content_signatures {
        if *language == *extension_lang {
            continue;
        }

        for signature in signatures {
            let matches = count_signature_matches(signature, content);
            if matches >= signature.required_matches {
                let confidence =
                    (matches as f32 / signature.patterns.len() as f32) * signature.weight;
                results.push((language.clone(), confidence));
            }
        }
    }

    results
}

/// Analyze content signatures
pub fn analyze_content_signatures(
    content: &str,
    content_signatures: &HashMap<Language, Vec<ContentSignature>>,
) -> Vec<(Language, f32)> {
    let mut results = Vec::new();

    for (language, signatures) in content_signatures {
        for signature in signatures {
            let matches = signature
                .patterns
                .iter()
                .map(|pattern| pattern.find_iter(content).count())
                .sum::<usize>();

            if matches >= signature.required_matches {
                let confidence =
                    (matches as f32 / signature.patterns.len() as f32) * signature.weight;
                results.push((language.clone(), confidence));
            }
        }
    }

    results
}

/// Get likely languages from quick content analysis (no AST parsing)
pub fn get_likely_languages_from_content(content: &str) -> Vec<Language> {
    let mut likely_languages = Vec::new();

    if content.contains("def ") || content.contains("import ") || content.contains("from ") {
        likely_languages.push(Language::Python);
    }
    if content.contains("fn ") || content.contains("use ") || content.contains("struct ") {
        likely_languages.push(Language::Rust);
    }
    if content.contains("function ") || content.contains("const ") || content.contains("let ") {
        likely_languages.push(Language::JavaScript);
    }
    if content.contains("interface ") || content.contains("type ") || content.contains(": string") {
        likely_languages.push(Language::TypeScript);
    }
    if content.contains("func ") || content.contains("package ") {
        likely_languages.push(Language::Go);
    }
    if content.contains("defmodule ")
        || content.contains("alias ")
        || content.contains("require ")
        || content.contains("import ")
    {
        likely_languages.push(Language::Elixir);
    }

    if likely_languages.is_empty() {
        likely_languages = vec![
            Language::JavaScript,
            Language::Python,
            Language::TypeScript,
            Language::Rust,
            Language::Go,
            Language::Elixir,
        ];
    }

    likely_languages
}

/// Aggregate detection results from multiple sources
pub fn aggregate_detection_results(
    candidates: Vec<(Language, f32)>,
    evidence: Vec<DetectionEvidence>,
) -> DetectionResult {
    if candidates.is_empty() {
        return DetectionResult {
            language: Language::Unknown,
            confidence: 0.0,
            detection_method: DetectionMethod::FileExtension,
            alternatives: vec![],
            evidence,
        };
    }

    let mut language_scores: HashMap<Language, f32> = HashMap::new();

    for (lang, confidence) in &candidates {
        *language_scores.entry(lang.clone()).or_insert(0.0) += confidence;
    }

    let (best_language, best_confidence) = language_scores
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(lang, conf)| (lang.clone(), *conf))
        .unwrap_or((Language::Unknown, 0.0));

    let normalized_confidence = best_confidence.min(1.0);

    let mut alternatives: Vec<(Language, f32)> = language_scores
        .into_iter()
        .filter(|(lang, _)| *lang != best_language)
        .collect();
    alternatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let detection_method = if evidence
        .iter()
        .any(|e| e.evidence_type == EvidenceType::Shebang)
    {
        DetectionMethod::Shebang
    } else if evidence
        .iter()
        .any(|e| e.evidence_type == EvidenceType::Syntax)
    {
        DetectionMethod::ContentSignature
    } else if evidence
        .iter()
        .any(|e| e.evidence_type == EvidenceType::Extension)
    {
        DetectionMethod::FileExtension
    } else {
        DetectionMethod::Hybrid
    };

    DetectionResult {
        language: best_language,
        confidence: normalized_confidence,
        detection_method,
        alternatives,
        evidence,
    }
}

/// Apply project type bias to detection results
pub fn apply_project_type_bias(
    mut result: DetectionResult,
    project_type: &ProjectType,
) -> DetectionResult {
    let bias_factor = 0.25;

    match project_type {
        ProjectType::WebFrontend => {
            if matches!(
                result.language,
                Language::JavaScript | Language::TypeScript | Language::HTML | Language::CSS
            ) {
                result.confidence += bias_factor;
            }
        }
        ProjectType::WebBackend => {
            if matches!(
                result.language,
                Language::Python
                    | Language::JavaScript
                    | Language::TypeScript
                    | Language::Java
                    | Language::Go
                    | Language::Rust
            ) {
                result.confidence += bias_factor;
            }
        }
        ProjectType::SystemsProgram => {
            if matches!(
                result.language,
                Language::Rust | Language::C | Language::Cpp | Language::Go
            ) {
                result.confidence += bias_factor;
            }
        }
        ProjectType::DataScience => {
            if matches!(
                result.language,
                Language::Python | Language::R | Language::SQL
            ) {
                result.confidence += bias_factor;
            }
        }
        _ => {}
    }

    result.confidence = result.confidence.min(1.0);
    result
}

/// Apply dominant language bias
pub fn apply_dominant_language_bias(
    mut result: DetectionResult,
    dominant_languages: &[Language],
) -> DetectionResult {
    if dominant_languages.contains(&result.language) {
        result.confidence += 0.15;
        result.confidence = result.confidence.min(1.0);
    }
    result
}

/// Apply framework bias based on indicators
pub fn apply_framework_bias(
    mut result: DetectionResult,
    framework_indicators: &[String],
) -> DetectionResult {
    for indicator in framework_indicators {
        match indicator.as_str() {
            "package.json" | "node_modules" => {
                if matches!(result.language, Language::JavaScript | Language::TypeScript) {
                    result.confidence += 0.1;
                }
            }
            "Cargo.toml" | "Cargo.lock" => {
                if result.language == Language::Rust {
                    result.confidence += 0.1;
                }
            }
            "requirements.txt" | "__pycache__" | ".pyc" => {
                if result.language == Language::Python {
                    result.confidence += 0.1;
                }
            }
            _ => {}
        }
    }

    result.confidence = result.confidence.min(1.0);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn create_extension_map() -> HashMap<String, Vec<(Language, f32)>> {
        let mut map = HashMap::new();
        map.insert("rs".to_string(), vec![(Language::Rust, 1.0)]);
        map.insert("py".to_string(), vec![(Language::Python, 1.0)]);
        map.insert("js".to_string(), vec![(Language::JavaScript, 1.0)]);
        map.insert("ts".to_string(), vec![(Language::TypeScript, 1.0)]);
        map.insert("go".to_string(), vec![(Language::Go, 1.0)]);
        map.insert("java".to_string(), vec![(Language::Java, 1.0)]);
        map.insert("c".to_string(), vec![(Language::C, 1.0)]);
        map.insert("cpp".to_string(), vec![(Language::Cpp, 1.0)]);
        map
    }

    fn create_filename_patterns() -> HashMap<String, Language> {
        let mut map = HashMap::new();
        map.insert("Makefile".to_string(), Language::Unknown);
        map.insert("Dockerfile".to_string(), Language::Unknown);
        map
    }

    fn create_shebang_patterns() -> HashMap<String, Language> {
        let mut map = HashMap::new();
        map.insert("python".to_string(), Language::Python);
        map.insert("python3".to_string(), Language::Python);
        map.insert("bash".to_string(), Language::Bash);
        map.insert("node".to_string(), Language::JavaScript);
        map
    }

    #[test]
    fn test_detect_by_extension() {
        let map = create_extension_map();

        assert_eq!(
            detect_by_extension(Path::new("test.rs"), &map),
            Language::Rust
        );
        assert_eq!(
            detect_by_extension(Path::new("test.py"), &map),
            Language::Python
        );
        assert_eq!(
            detect_by_extension(Path::new("test.js"), &map),
            Language::JavaScript
        );
        assert_eq!(
            detect_by_extension(Path::new("test.unknown"), &map),
            Language::Unknown
        );
        assert_eq!(
            detect_by_extension(Path::new("no_extension"), &map),
            Language::Unknown
        );
    }

    #[test]
    fn test_detect_by_extension_case_insensitive() {
        let map = create_extension_map();

        // Extensions should be converted to lowercase
        assert_eq!(
            detect_by_extension(Path::new("test.RS"), &map),
            Language::Rust
        );
        assert_eq!(
            detect_by_extension(Path::new("test.PY"), &map),
            Language::Python
        );
    }

    #[test]
    fn test_detect_by_extension_and_filename() {
        let ext_map = create_extension_map();
        let filename_patterns = create_filename_patterns();

        // Filename patterns take precedence
        assert_eq!(
            detect_by_extension_and_filename(Path::new("Makefile"), &ext_map, &filename_patterns),
            Language::Unknown
        );

        // Falls back to extension
        assert_eq!(
            detect_by_extension_and_filename(Path::new("test.rs"), &ext_map, &filename_patterns),
            Language::Rust
        );
    }

    #[test]
    fn test_detect_by_shebang() {
        let patterns = create_shebang_patterns();

        // Python shebang
        let python_script = "#!/usr/bin/env python3\nprint('hello')";
        assert_eq!(
            detect_by_shebang(python_script, &patterns),
            Some(Language::Python)
        );

        // Bash shebang
        let bash_script = "#!/bin/bash\necho hello";
        assert_eq!(
            detect_by_shebang(bash_script, &patterns),
            Some(Language::Bash)
        );

        // Node shebang
        let node_script = "#!/usr/bin/env node\nconsole.log('hi')";
        assert_eq!(
            detect_by_shebang(node_script, &patterns),
            Some(Language::JavaScript)
        );

        // No shebang
        let no_shebang = "print('hello')";
        assert_eq!(detect_by_shebang(no_shebang, &patterns), None);

        // Empty content
        assert_eq!(detect_by_shebang("", &patterns), None);

        // Unknown shebang
        let unknown = "#!/usr/bin/unknown\ncode";
        assert_eq!(detect_by_shebang(unknown, &patterns), None);
    }

    #[test]
    fn test_quick_content_validation() {
        // Rust markers
        assert!(quick_content_validation(&Language::Rust, "fn main() {}"));
        assert!(quick_content_validation(&Language::Rust, "use std::io;"));
        assert!(quick_content_validation(&Language::Rust, "struct Foo {}"));
        assert!(!quick_content_validation(
            &Language::Rust,
            "no rust markers here"
        ));

        // Python markers
        assert!(quick_content_validation(&Language::Python, "def foo():"));
        assert!(quick_content_validation(&Language::Python, "import os"));
        assert!(quick_content_validation(
            &Language::Python,
            "class MyClass:"
        ));
        assert!(!quick_content_validation(
            &Language::Python,
            "no python markers"
        ));

        // JavaScript markers
        assert!(quick_content_validation(
            &Language::JavaScript,
            "function foo() {}"
        ));
        assert!(quick_content_validation(
            &Language::JavaScript,
            "const x = 1;"
        ));
        assert!(quick_content_validation(
            &Language::JavaScript,
            "var y = 2;"
        ));

        // Unknown language has no markers, always returns true
        assert!(quick_content_validation(&Language::Unknown, "anything"));
    }

    #[test]
    fn test_get_language_markers() {
        assert!(!get_language_markers(&Language::Rust).is_empty());
        assert!(!get_language_markers(&Language::Python).is_empty());
        assert!(!get_language_markers(&Language::JavaScript).is_empty());
        assert!(!get_language_markers(&Language::TypeScript).is_empty());
        assert!(!get_language_markers(&Language::Go).is_empty());
        assert!(!get_language_markers(&Language::Java).is_empty());
        assert!(!get_language_markers(&Language::C).is_empty());
        assert!(!get_language_markers(&Language::Cpp).is_empty());

        // Unknown returns empty
        assert!(get_language_markers(&Language::Unknown).is_empty());
    }

    #[test]
    fn test_get_likely_languages_from_content() {
        // Python-like content
        let python_content = "def hello():\n    import os";
        let languages = get_likely_languages_from_content(python_content);
        assert!(languages.contains(&Language::Python));

        // Rust-like content
        let rust_content = "fn main() {\n    use std::io;\n}";
        let languages = get_likely_languages_from_content(rust_content);
        assert!(languages.contains(&Language::Rust));

        // JavaScript-like content
        let js_content = "function foo() {\n    const x = 1;\n}";
        let languages = get_likely_languages_from_content(js_content);
        assert!(languages.contains(&Language::JavaScript));

        // TypeScript-like content
        let ts_content = "interface Foo {\n    type Bar = string;\n}";
        let languages = get_likely_languages_from_content(ts_content);
        assert!(languages.contains(&Language::TypeScript));

        // Go-like content
        let go_content = "func main() {\n    package main\n}";
        let languages = get_likely_languages_from_content(go_content);
        assert!(languages.contains(&Language::Go));

        // Elixir-like content
        let elixir_content = "defmodule AppWeb.Router do\n  alias AppWeb.Endpoint\nend";
        let languages = get_likely_languages_from_content(elixir_content);
        assert!(languages.contains(&Language::Elixir));

        // Unknown content returns default set
        let unknown_content = "hello world";
        let languages = get_likely_languages_from_content(unknown_content);
        assert!(!languages.is_empty());
    }

    #[test]
    fn test_aggregate_detection_results_empty() {
        let candidates: Vec<(Language, f32)> = vec![];
        let evidence = vec![];

        let result = aggregate_detection_results(candidates, evidence);
        assert_eq!(result.language, Language::Unknown);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_aggregate_detection_results_single() {
        let candidates = vec![(Language::Rust, 0.9)];
        let evidence = vec![DetectionEvidence {
            evidence_type: EvidenceType::Extension,
            weight: 0.9,
            description: "Extension match".to_string(),
        }];

        let result = aggregate_detection_results(candidates, evidence);
        assert_eq!(result.language, Language::Rust);
        assert_eq!(result.confidence, 0.9);
        assert_eq!(result.detection_method, DetectionMethod::FileExtension);
    }

    #[test]
    fn test_aggregate_detection_results_multiple() {
        let candidates = vec![
            (Language::Python, 0.6),
            (Language::Python, 0.3),
            (Language::JavaScript, 0.4),
        ];
        let evidence = vec![];

        let result = aggregate_detection_results(candidates, evidence);
        assert_eq!(result.language, Language::Python); // 0.6 + 0.3 = 0.9 > 0.4
        assert_eq!(result.alternatives.len(), 1);
    }

    #[test]
    fn test_aggregate_detection_results_shebang_method() {
        let candidates = vec![(Language::Python, 0.95)];
        let evidence = vec![DetectionEvidence {
            evidence_type: EvidenceType::Shebang,
            weight: 0.95,
            description: "Shebang match".to_string(),
        }];

        let result = aggregate_detection_results(candidates, evidence);
        assert_eq!(result.detection_method, DetectionMethod::Shebang);
    }

    #[test]
    fn test_aggregate_detection_results_syntax_method() {
        let candidates = vec![(Language::Rust, 0.8)];
        let evidence = vec![DetectionEvidence {
            evidence_type: EvidenceType::Syntax,
            weight: 0.8,
            description: "Syntax match".to_string(),
        }];

        let result = aggregate_detection_results(candidates, evidence);
        assert_eq!(result.detection_method, DetectionMethod::ContentSignature);
    }

    #[test]
    fn test_apply_project_type_bias_web_frontend() {
        let result = DetectionResult {
            language: Language::JavaScript,
            confidence: 0.5,
            detection_method: DetectionMethod::FileExtension,
            alternatives: vec![],
            evidence: vec![],
        };

        let biased = apply_project_type_bias(result, &ProjectType::WebFrontend);
        assert_eq!(biased.confidence, 0.75); // 0.5 + 0.25
    }

    #[test]
    fn test_apply_project_type_bias_web_backend() {
        let result = DetectionResult {
            language: Language::Python,
            confidence: 0.6,
            detection_method: DetectionMethod::FileExtension,
            alternatives: vec![],
            evidence: vec![],
        };

        let biased = apply_project_type_bias(result, &ProjectType::WebBackend);
        assert_eq!(biased.confidence, 0.85); // 0.6 + 0.25
    }

    #[test]
    fn test_apply_project_type_bias_systems_program() {
        let result = DetectionResult {
            language: Language::Rust,
            confidence: 0.7,
            detection_method: DetectionMethod::FileExtension,
            alternatives: vec![],
            evidence: vec![],
        };

        let biased = apply_project_type_bias(result, &ProjectType::SystemsProgram);
        assert_eq!(biased.confidence, 0.95); // 0.7 + 0.25
    }

    #[test]
    fn test_apply_project_type_bias_data_science() {
        let result = DetectionResult {
            language: Language::Python,
            confidence: 0.7,
            detection_method: DetectionMethod::FileExtension,
            alternatives: vec![],
            evidence: vec![],
        };

        let biased = apply_project_type_bias(result, &ProjectType::DataScience);
        assert_eq!(biased.confidence, 0.95); // 0.7 + 0.25
    }

    #[test]
    fn test_apply_project_type_bias_caps_at_one() {
        let result = DetectionResult {
            language: Language::Rust,
            confidence: 0.9,
            detection_method: DetectionMethod::FileExtension,
            alternatives: vec![],
            evidence: vec![],
        };

        let biased = apply_project_type_bias(result, &ProjectType::SystemsProgram);
        assert_eq!(biased.confidence, 1.0); // Capped at 1.0
    }

    #[test]
    fn test_apply_dominant_language_bias() {
        let result = DetectionResult {
            language: Language::Python,
            confidence: 0.7,
            detection_method: DetectionMethod::FileExtension,
            alternatives: vec![],
            evidence: vec![],
        };

        let biased = apply_dominant_language_bias(result, &[Language::Python, Language::Rust]);
        assert_eq!(biased.confidence, 0.85); // 0.7 + 0.15
    }

    #[test]
    fn test_apply_dominant_language_bias_no_match() {
        let result = DetectionResult {
            language: Language::Python,
            confidence: 0.7,
            detection_method: DetectionMethod::FileExtension,
            alternatives: vec![],
            evidence: vec![],
        };

        let biased = apply_dominant_language_bias(result, &[Language::Rust, Language::Go]);
        assert_eq!(biased.confidence, 0.7); // No change
    }

    #[test]
    fn test_apply_framework_bias_package_json() {
        let result = DetectionResult {
            language: Language::JavaScript,
            confidence: 0.7,
            detection_method: DetectionMethod::FileExtension,
            alternatives: vec![],
            evidence: vec![],
        };

        let biased = apply_framework_bias(result, &["package.json".to_string()]);
        assert_eq!(biased.confidence, 0.8); // 0.7 + 0.1
    }

    #[test]
    fn test_apply_framework_bias_cargo_toml() {
        let result = DetectionResult {
            language: Language::Rust,
            confidence: 0.7,
            detection_method: DetectionMethod::FileExtension,
            alternatives: vec![],
            evidence: vec![],
        };

        let biased = apply_framework_bias(result, &["Cargo.toml".to_string()]);
        assert_eq!(biased.confidence, 0.8); // 0.7 + 0.1
    }

    #[test]
    fn test_apply_framework_bias_python() {
        let result = DetectionResult {
            language: Language::Python,
            confidence: 0.7,
            detection_method: DetectionMethod::FileExtension,
            alternatives: vec![],
            evidence: vec![],
        };

        let biased = apply_framework_bias(result, &["requirements.txt".to_string()]);
        assert_eq!(biased.confidence, 0.8); // 0.7 + 0.1
    }

    #[test]
    fn test_apply_framework_bias_multiple() {
        let result = DetectionResult {
            language: Language::JavaScript,
            confidence: 0.6,
            detection_method: DetectionMethod::FileExtension,
            alternatives: vec![],
            evidence: vec![],
        };

        let biased = apply_framework_bias(
            result,
            &["package.json".to_string(), "node_modules".to_string()],
        );
        assert!((biased.confidence - 0.8).abs() < 0.001); // 0.6 + 0.1 + 0.1 (with floating point tolerance)
    }
}
