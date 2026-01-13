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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_strategy_default() {
        let strategy = DetectionStrategy::default();
        assert!(matches!(strategy, DetectionStrategy::ExtensionWithContent));
    }

    #[test]
    fn test_detection_strategy_extension_only() {
        let strategy = DetectionStrategy::ExtensionOnly;
        assert!(matches!(strategy, DetectionStrategy::ExtensionOnly));
    }

    #[test]
    fn test_detection_strategy_full_analysis() {
        let strategy = DetectionStrategy::FullAnalysis;
        assert!(matches!(strategy, DetectionStrategy::FullAnalysis));
    }

    #[test]
    fn test_detection_strategy_custom() {
        let custom_rules = CustomDetectionRules {
            extension_overrides: HashMap::new(),
            filename_patterns: HashMap::new(),
            content_signatures: vec![],
            priority_languages: vec![],
        };
        let strategy = DetectionStrategy::Custom(custom_rules);
        assert!(matches!(strategy, DetectionStrategy::Custom(_)));
    }

    #[test]
    fn test_detection_strategy_clone() {
        let strategy = DetectionStrategy::ExtensionOnly;
        let cloned = strategy.clone();
        assert!(matches!(cloned, DetectionStrategy::ExtensionOnly));
    }

    #[test]
    fn test_detection_strategy_serialize() {
        let strategy = DetectionStrategy::ExtensionOnly;
        let json = serde_json::to_string(&strategy).unwrap();
        let deserialized: DetectionStrategy = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, DetectionStrategy::ExtensionOnly));
    }

    #[test]
    fn test_detection_strategy_debug() {
        let strategy = DetectionStrategy::FullAnalysis;
        let debug = format!("{:?}", strategy);
        assert!(debug.contains("FullAnalysis"));
    }

    #[test]
    fn test_custom_detection_rules_creation() {
        let mut extension_overrides = HashMap::new();
        extension_overrides.insert("xyz".to_string(), Language::Rust);

        let mut filename_patterns = HashMap::new();
        filename_patterns.insert("Makefile".to_string(), Language::Unknown);

        let rules = CustomDetectionRules {
            extension_overrides,
            filename_patterns,
            content_signatures: vec![],
            priority_languages: vec![Language::Rust, Language::Python],
        };

        assert_eq!(rules.extension_overrides.len(), 1);
        assert_eq!(rules.filename_patterns.len(), 1);
        assert_eq!(rules.priority_languages.len(), 2);
    }

    #[test]
    fn test_custom_detection_rules_clone() {
        let rules = CustomDetectionRules {
            extension_overrides: HashMap::new(),
            filename_patterns: HashMap::new(),
            content_signatures: vec![],
            priority_languages: vec![Language::JavaScript],
        };

        let cloned = rules.clone();
        assert_eq!(rules.priority_languages.len(), cloned.priority_languages.len());
    }

    #[test]
    fn test_custom_detection_rules_serialize() {
        let rules = CustomDetectionRules {
            extension_overrides: HashMap::new(),
            filename_patterns: HashMap::new(),
            content_signatures: vec![],
            priority_languages: vec![],
        };

        let json = serde_json::to_string(&rules).unwrap();
        assert!(json.contains("extension_overrides"));
        assert!(json.contains("priority_languages"));

        let deserialized: CustomDetectionRules = serde_json::from_str(&json).unwrap();
        assert!(deserialized.priority_languages.is_empty());
    }

    #[test]
    fn test_custom_detection_rules_debug() {
        let rules = CustomDetectionRules {
            extension_overrides: HashMap::new(),
            filename_patterns: HashMap::new(),
            content_signatures: vec![],
            priority_languages: vec![],
        };

        let debug = format!("{:?}", rules);
        assert!(debug.contains("CustomDetectionRules"));
    }

    #[test]
    fn test_content_signature_config_creation() {
        let config = ContentSignatureConfig {
            language: Language::Rust,
            patterns: vec!["fn ".to_string(), "impl ".to_string()],
            weight: 0.8,
            required_matches: 2,
        };

        assert!(matches!(config.language, Language::Rust));
        assert_eq!(config.patterns.len(), 2);
        assert!((config.weight - 0.8).abs() < 0.001);
        assert_eq!(config.required_matches, 2);
    }

    #[test]
    fn test_content_signature_config_clone() {
        let config = ContentSignatureConfig {
            language: Language::Python,
            patterns: vec!["def ".to_string()],
            weight: 0.5,
            required_matches: 1,
        };

        let cloned = config.clone();
        assert_eq!(config.patterns, cloned.patterns);
        assert_eq!(config.weight, cloned.weight);
    }

    #[test]
    fn test_content_signature_config_serialize() {
        let config = ContentSignatureConfig {
            language: Language::JavaScript,
            patterns: vec!["function".to_string()],
            weight: 0.6,
            required_matches: 1,
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("patterns"));
        assert!(json.contains("weight"));

        let deserialized: ContentSignatureConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.required_matches, deserialized.required_matches);
    }

    #[test]
    fn test_content_signature_config_debug() {
        let config = ContentSignatureConfig {
            language: Language::Go,
            patterns: vec![],
            weight: 0.0,
            required_matches: 0,
        };

        let debug = format!("{:?}", config);
        assert!(debug.contains("ContentSignatureConfig"));
    }

    #[test]
    fn test_language_hints_default() {
        let hints = LanguageHints::default();
        assert!(hints.project_type.is_none());
        assert!(hints.build_files.is_empty());
        assert!(hints.directory_structure.is_empty());
        assert!(hints.dominant_languages.is_empty());
        assert!(hints.framework_indicators.is_empty());
    }

    #[test]
    fn test_language_hints_custom() {
        let hints = LanguageHints {
            project_type: Some(ProjectType::WebBackend),
            build_files: vec!["Cargo.toml".to_string()],
            directory_structure: vec!["src".to_string(), "tests".to_string()],
            dominant_languages: vec![Language::Rust],
            framework_indicators: vec!["actix-web".to_string()],
        };

        assert_eq!(hints.project_type, Some(ProjectType::WebBackend));
        assert_eq!(hints.build_files.len(), 1);
        assert_eq!(hints.dominant_languages.len(), 1);
    }

    #[test]
    fn test_language_hints_clone() {
        let hints = LanguageHints {
            project_type: Some(ProjectType::Library),
            build_files: vec!["package.json".to_string()],
            directory_structure: vec![],
            dominant_languages: vec![],
            framework_indicators: vec![],
        };

        let cloned = hints.clone();
        assert_eq!(hints.project_type, cloned.project_type);
        assert_eq!(hints.build_files, cloned.build_files);
    }

    #[test]
    fn test_language_hints_serialize() {
        let hints = LanguageHints::default();
        let json = serde_json::to_string(&hints).unwrap();
        assert!(json.contains("project_type"));
        assert!(json.contains("build_files"));

        let deserialized: LanguageHints = serde_json::from_str(&json).unwrap();
        assert!(deserialized.project_type.is_none());
    }

    #[test]
    fn test_language_hints_debug() {
        let hints = LanguageHints::default();
        let debug = format!("{:?}", hints);
        assert!(debug.contains("LanguageHints"));
    }

    #[test]
    fn test_project_type_variants() {
        let types = vec![
            ProjectType::WebFrontend,
            ProjectType::WebBackend,
            ProjectType::MobileApp,
            ProjectType::DesktopApp,
            ProjectType::SystemsProgram,
            ProjectType::DataScience,
            ProjectType::GameDevelopment,
            ProjectType::EmbeddedSystem,
            ProjectType::Library,
            ProjectType::Documentation,
            ProjectType::Configuration,
            ProjectType::Unknown,
        ];

        assert_eq!(types.len(), 12);
    }

    #[test]
    fn test_project_type_equality() {
        let type1 = ProjectType::WebBackend;
        let type2 = ProjectType::WebBackend;
        let type3 = ProjectType::WebFrontend;

        assert_eq!(type1, type2);
        assert_ne!(type1, type3);
    }

    #[test]
    fn test_project_type_clone() {
        let pt = ProjectType::DataScience;
        let cloned = pt.clone();
        assert_eq!(pt, cloned);
    }

    #[test]
    fn test_project_type_serialize() {
        let pt = ProjectType::GameDevelopment;
        let json = serde_json::to_string(&pt).unwrap();
        let deserialized: ProjectType = serde_json::from_str(&json).unwrap();
        assert_eq!(pt, deserialized);
    }

    #[test]
    fn test_project_type_debug() {
        let pt = ProjectType::EmbeddedSystem;
        let debug = format!("{:?}", pt);
        assert!(debug.contains("EmbeddedSystem"));
    }

    #[test]
    fn test_detection_result_creation() {
        let result = DetectionResult {
            language: Language::Rust,
            confidence: 0.95,
            detection_method: DetectionMethod::FileExtension,
            alternatives: vec![(Language::Unknown, 0.05)],
            evidence: vec![],
        };

        assert!(matches!(result.language, Language::Rust));
        assert!((result.confidence - 0.95).abs() < 0.001);
        assert_eq!(result.alternatives.len(), 1);
    }

    #[test]
    fn test_detection_result_clone() {
        let result = DetectionResult {
            language: Language::Python,
            confidence: 0.9,
            detection_method: DetectionMethod::Hybrid,
            alternatives: vec![],
            evidence: vec![],
        };

        let cloned = result.clone();
        assert!((result.confidence - cloned.confidence).abs() < 0.001);
    }

    #[test]
    fn test_detection_result_serialize() {
        let result = DetectionResult {
            language: Language::JavaScript,
            confidence: 0.8,
            detection_method: DetectionMethod::ContentSignature,
            alternatives: vec![],
            evidence: vec![],
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("confidence"));
        assert!(json.contains("detection_method"));

        let deserialized: DetectionResult = serde_json::from_str(&json).unwrap();
        assert!((result.confidence - deserialized.confidence).abs() < 0.001);
    }

    #[test]
    fn test_detection_result_debug() {
        let result = DetectionResult {
            language: Language::Go,
            confidence: 0.5,
            detection_method: DetectionMethod::Shebang,
            alternatives: vec![],
            evidence: vec![],
        };

        let debug = format!("{:?}", result);
        assert!(debug.contains("DetectionResult"));
    }

    #[test]
    fn test_detection_method_variants() {
        let methods = vec![
            DetectionMethod::FileExtension,
            DetectionMethod::Filename,
            DetectionMethod::Shebang,
            DetectionMethod::ContentSignature,
            DetectionMethod::StatisticalAnalysis,
            DetectionMethod::Hybrid,
        ];

        assert_eq!(methods.len(), 6);
    }

    #[test]
    fn test_detection_method_equality() {
        let method1 = DetectionMethod::FileExtension;
        let method2 = DetectionMethod::FileExtension;
        let method3 = DetectionMethod::Shebang;

        assert_eq!(method1, method2);
        assert_ne!(method1, method3);
    }

    #[test]
    fn test_detection_method_clone() {
        let method = DetectionMethod::StatisticalAnalysis;
        let cloned = method.clone();
        assert_eq!(method, cloned);
    }

    #[test]
    fn test_detection_method_serialize() {
        let method = DetectionMethod::Hybrid;
        let json = serde_json::to_string(&method).unwrap();
        let deserialized: DetectionMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(method, deserialized);
    }

    #[test]
    fn test_detection_method_debug() {
        let method = DetectionMethod::ContentSignature;
        let debug = format!("{:?}", method);
        assert!(debug.contains("ContentSignature"));
    }

    #[test]
    fn test_detection_evidence_creation() {
        let evidence = DetectionEvidence {
            evidence_type: EvidenceType::Extension,
            description: "File has .rs extension".to_string(),
            weight: 0.9,
        };

        assert_eq!(evidence.evidence_type, EvidenceType::Extension);
        assert!(evidence.description.contains(".rs"));
        assert!((evidence.weight - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_detection_evidence_clone() {
        let evidence = DetectionEvidence {
            evidence_type: EvidenceType::Keyword,
            description: "Contains fn keyword".to_string(),
            weight: 0.7,
        };

        let cloned = evidence.clone();
        assert_eq!(evidence.evidence_type, cloned.evidence_type);
        assert_eq!(evidence.description, cloned.description);
    }

    #[test]
    fn test_detection_evidence_serialize() {
        let evidence = DetectionEvidence {
            evidence_type: EvidenceType::Import,
            description: "Has import statements".to_string(),
            weight: 0.6,
        };

        let json = serde_json::to_string(&evidence).unwrap();
        assert!(json.contains("evidence_type"));
        assert!(json.contains("description"));

        let deserialized: DetectionEvidence = serde_json::from_str(&json).unwrap();
        assert_eq!(evidence.evidence_type, deserialized.evidence_type);
    }

    #[test]
    fn test_detection_evidence_debug() {
        let evidence = DetectionEvidence {
            evidence_type: EvidenceType::Framework,
            description: "Debug test".to_string(),
            weight: 0.5,
        };

        let debug = format!("{:?}", evidence);
        assert!(debug.contains("DetectionEvidence"));
    }

    #[test]
    fn test_evidence_type_variants() {
        let types = vec![
            EvidenceType::Extension,
            EvidenceType::Filename,
            EvidenceType::Shebang,
            EvidenceType::Keyword,
            EvidenceType::Syntax,
            EvidenceType::Import,
            EvidenceType::Framework,
            EvidenceType::BuildSystem,
        ];

        assert_eq!(types.len(), 8);
    }

    #[test]
    fn test_evidence_type_equality() {
        let type1 = EvidenceType::Keyword;
        let type2 = EvidenceType::Keyword;
        let type3 = EvidenceType::Syntax;

        assert_eq!(type1, type2);
        assert_ne!(type1, type3);
    }

    #[test]
    fn test_evidence_type_clone() {
        let et = EvidenceType::BuildSystem;
        let cloned = et.clone();
        assert_eq!(et, cloned);
    }

    #[test]
    fn test_evidence_type_serialize() {
        let et = EvidenceType::Framework;
        let json = serde_json::to_string(&et).unwrap();
        let deserialized: EvidenceType = serde_json::from_str(&json).unwrap();
        assert_eq!(et, deserialized);
    }

    #[test]
    fn test_evidence_type_debug() {
        let et = EvidenceType::Shebang;
        let debug = format!("{:?}", et);
        assert!(debug.contains("Shebang"));
    }

    #[test]
    fn test_syntax_analyzer_creation() {
        let analyzer = SyntaxAnalyzer {
            language: Language::Rust,
            keywords: vec!["fn".to_string(), "let".to_string()],
            structural_patterns: vec!["impl".to_string()],
            confidence_weights: HashMap::new(),
        };

        assert!(matches!(analyzer.language, Language::Rust));
        assert_eq!(analyzer.keywords.len(), 2);
        assert_eq!(analyzer.structural_patterns.len(), 1);
    }

    #[test]
    fn test_syntax_analyzer_clone() {
        let mut weights = HashMap::new();
        weights.insert("fn".to_string(), 0.9_f32);

        let analyzer = SyntaxAnalyzer {
            language: Language::Python,
            keywords: vec!["def".to_string()],
            structural_patterns: vec![],
            confidence_weights: weights,
        };

        let cloned = analyzer.clone();
        assert_eq!(analyzer.keywords, cloned.keywords);
    }

    #[test]
    fn test_syntax_analyzer_debug() {
        let analyzer = SyntaxAnalyzer {
            language: Language::Go,
            keywords: vec![],
            structural_patterns: vec![],
            confidence_weights: HashMap::new(),
        };

        let debug = format!("{:?}", analyzer);
        assert!(debug.contains("SyntaxAnalyzer"));
    }

    #[test]
    fn test_content_signature_creation() {
        let sig = ContentSignature {
            language: Language::Rust,
            patterns: vec![regex::Regex::new("fn ").unwrap()],
            weight: 0.8,
            required_matches: 1,
        };

        assert!(matches!(sig.language, Language::Rust));
        assert_eq!(sig.patterns.len(), 1);
        assert!((sig.weight - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_content_signature_clone() {
        let sig = ContentSignature {
            language: Language::Python,
            patterns: vec![regex::Regex::new("def ").unwrap()],
            weight: 0.7,
            required_matches: 2,
        };

        let cloned = sig.clone();
        assert_eq!(sig.required_matches, cloned.required_matches);
        assert!((sig.weight - cloned.weight).abs() < 0.001);
    }

    #[test]
    fn test_content_signature_debug() {
        let sig = ContentSignature {
            language: Language::JavaScript,
            patterns: vec![],
            weight: 0.5,
            required_matches: 0,
        };

        let debug = format!("{:?}", sig);
        assert!(debug.contains("ContentSignature"));
    }
}
