//! Helper functions for repository analysis.

use scribe_analysis::DocumentAnalysis;
use scribe_core::Language;

/// Analyze document content for structural indicators
pub(crate) fn analyze_document_content(content: &str) -> DocumentAnalysis {
    let mut analysis = DocumentAnalysis::new();
    let mut in_code_block = false;

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("```") {
            if !in_code_block {
                analysis.code_block_count += 1;
            }
            in_code_block = !in_code_block;
            continue;
        }

        if trimmed.starts_with('#') {
            analysis.heading_count += 1;
            if trimmed.to_lowercase().contains("table of contents") {
                analysis.toc_indicators += 1;
            }
        }

        if trimmed.contains("](") {
            analysis.link_count += trimmed.matches("](").count();
        }
    }

    analysis.is_well_structured = analysis.heading_count > 0 && analysis.link_count > 0;
    analysis
}

/// Convert a language identifier string to a Language enum
pub(crate) fn language_from_identifier(language: &str, path: &std::path::Path) -> Language {
    if !language.is_empty() {
        match language.to_lowercase().as_str() {
            "rust" => return Language::Rust,
            "python" => return Language::Python,
            "javascript" => return Language::JavaScript,
            "typescript" => return Language::TypeScript,
            "go" => return Language::Go,
            "java" => return Language::Java,
            "c" => return Language::C,
            "cpp" | "c++" => return Language::Cpp,
            "kotlin" => return Language::Kotlin,
            "swift" => return Language::Swift,
            "php" => return Language::PHP,
            "ruby" => return Language::Ruby,
            _ => {}
        }
    }

    let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
    Language::from_extension(extension)
}
