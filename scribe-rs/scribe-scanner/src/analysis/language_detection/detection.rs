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
    if content.contains("interface ")
        || content.contains("type ")
        || content.contains(": string")
    {
        likely_languages.push(Language::TypeScript);
    }
    if content.contains("func ") || content.contains("package ") {
        likely_languages.push(Language::Go);
    }

    if likely_languages.is_empty() {
        likely_languages = vec![
            Language::JavaScript,
            Language::Python,
            Language::TypeScript,
            Language::Rust,
            Language::Go,
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
