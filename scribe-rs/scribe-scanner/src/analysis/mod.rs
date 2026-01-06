//! Language detection and code analysis.

pub mod aho_corasick_reference_index;
pub mod language_detection;

pub use aho_corasick_reference_index::{AhoCorasickReferenceIndex, IndexConfig, IndexMetrics};
pub use language_detection::{DetectionStrategy, LanguageDetector, LanguageHints};
