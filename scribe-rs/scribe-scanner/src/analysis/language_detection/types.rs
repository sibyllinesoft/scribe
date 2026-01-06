//! Type definitions for language detection.

use scribe_core::Language;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Language detection strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetectionStrategy {
    /// Extension-only detection (fastest)
    ExtensionOnly,
    /// Extension + content analysis (default)
    ExtensionWithContent,
    /// Full analysis including statistical detection (most accurate)
    FullAnalysis,
    /// Custom detection with user-defined rules
    Custom(CustomDetectionRules),
}

/// Custom detection rules for specialized cases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomDetectionRules {
    pub extension_overrides: HashMap<String, Language>,
    pub filename_patterns: HashMap<String, Language>,
    pub content_signatures: Vec<ContentSignatureConfig>,
    pub priority_languages: Vec<Language>,
}

/// Content signature for language detection
#[derive(Debug, Clone)]
pub struct ContentSignature {
    pub language: Language,
    pub patterns: Vec<regex::Regex>,
    pub weight: f32,
    pub required_matches: usize,
}

/// Serializable version of ContentSignature for configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentSignatureConfig {
    pub language: Language,
    pub patterns: Vec<String>,
    pub weight: f32,
    pub required_matches: usize,
}

/// Language detection hints for improved accuracy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageHints {
    pub project_type: Option<ProjectType>,
    pub build_files: Vec<String>,
    pub directory_structure: Vec<String>,
    pub dominant_languages: Vec<Language>,
    pub framework_indicators: Vec<String>,
}

/// Project type classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectType {
    WebFrontend,
    WebBackend,
    MobileApp,
    DesktopApp,
    SystemsProgram,
    DataScience,
    GameDevelopment,
    EmbeddedSystem,
    Library,
    Documentation,
    Configuration,
    Unknown,
}

/// Language detection results with confidence scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub language: Language,
    pub confidence: f32,
    pub detection_method: DetectionMethod,
    pub alternatives: Vec<(Language, f32)>,
    pub evidence: Vec<DetectionEvidence>,
}

/// Method used for language detection
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionMethod {
    FileExtension,
    Filename,
    Shebang,
    ContentSignature,
    StatisticalAnalysis,
    Hybrid,
}

/// Evidence supporting language detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionEvidence {
    pub evidence_type: EvidenceType,
    pub description: String,
    pub weight: f32,
}

/// Type of detection evidence
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceType {
    Extension,
    Filename,
    Shebang,
    Keyword,
    Syntax,
    Import,
    Framework,
    BuildSystem,
}

/// AST-based syntax analyzer for content analysis
#[derive(Debug, Clone)]
pub struct SyntaxAnalyzer {
    pub language: Language,
    pub keywords: Vec<String>,
    pub structural_patterns: Vec<String>,
    pub confidence_weights: HashMap<String, f32>,
}

impl Default for DetectionStrategy {
    fn default() -> Self {
        DetectionStrategy::ExtensionWithContent
    }
}

impl Default for LanguageHints {
    fn default() -> Self {
        Self {
            project_type: None,
            build_files: Vec::new(),
            directory_structure: Vec::new(),
            dominant_languages: Vec::new(),
            framework_indicators: Vec::new(),
        }
    }
}
