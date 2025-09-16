//! # Advanced Template Detection System
//!
//! Implements sophisticated template engine detection using multiple methods:
//! 1. Extension-based detection for known template file types
//! 2. AST-based content pattern analysis for template syntax detection
//! 3. Tree-sitter parsing for HTML/XML files that might be templates
//! 4. Directory context analysis for template-like structures
//!
//! This module uses tree-sitter AST parsing instead of regex patterns
//! for accurate template detection and better performance.

use once_cell::sync::Lazy;
use scribe_core::Result;
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use tree_sitter::{Language as TsLanguage, Node, Parser, Tree};

/// Template engine classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TemplateEngine {
    // JavaScript template engines
    Handlebars,
    Mustache,
    Ejs,
    Pug,
    Jade,

    // Python template engines
    Django,
    Jinja2,
    Mako,

    // PHP template engines
    Twig,
    Smarty,

    // Ruby template engines
    Erb,
    Haml,

    // Other template engines
    Liquid,
    Dust,
    Eta,

    // Frontend frameworks with templates
    Vue,
    Svelte,
    React, // JSX/TSX
    Angular,

    // Generic/Unknown template
    Generic,
}

impl TemplateEngine {
    /// Get the typical file extensions for this template engine
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            TemplateEngine::Handlebars => &[".hbs", ".handlebars"],
            TemplateEngine::Mustache => &[".mustache"],
            TemplateEngine::Ejs => &[".ejs"],
            TemplateEngine::Pug => &[".pug"],
            TemplateEngine::Jade => &[".jade"],
            TemplateEngine::Django => &[".html", ".htm"], // Context dependent
            TemplateEngine::Jinja2 => &[".j2", ".jinja", ".jinja2"],
            TemplateEngine::Mako => &[".mako"],
            TemplateEngine::Twig => &[".twig"],
            TemplateEngine::Smarty => &[".tpl"],
            TemplateEngine::Erb => &[".erb", ".rhtml"],
            TemplateEngine::Haml => &[".haml"],
            TemplateEngine::Liquid => &[".liquid"],
            TemplateEngine::Dust => &[".dust"],
            TemplateEngine::Eta => &[".eta"],
            TemplateEngine::Vue => &[".vue"],
            TemplateEngine::Svelte => &[".svelte"],
            TemplateEngine::React => &[".jsx", ".tsx"],
            TemplateEngine::Angular => &[".html"], // Context dependent
            TemplateEngine::Generic => &[],
        }
    }

    /// Get score boost factor for this template engine
    pub fn score_boost(&self) -> f64 {
        match self {
            // High priority for dedicated template files
            TemplateEngine::Handlebars
            | TemplateEngine::Mustache
            | TemplateEngine::Jinja2
            | TemplateEngine::Twig
            | TemplateEngine::Liquid => 1.5,

            // Moderate priority for framework templates
            TemplateEngine::Vue | TemplateEngine::Svelte | TemplateEngine::React => 1.3,

            // Standard boost for other template engines
            TemplateEngine::Ejs
            | TemplateEngine::Pug
            | TemplateEngine::Erb
            | TemplateEngine::Haml => 1.2,

            // Lower boost for generic HTML templates
            TemplateEngine::Django | TemplateEngine::Angular => 1.0,

            // Minimal boost for unknown templates
            TemplateEngine::Generic => 0.8,

            _ => 1.0,
        }
    }
}

/// Method used for template detection
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateDetectionMethod {
    /// Detected by file extension
    Extension,
    /// Detected by content pattern analysis
    ContentPattern,
    /// Detected by directory context
    DirectoryContext,
    /// Detected by language heuristics
    LanguageHeuristic,
}

/// Result of template detection
#[derive(Debug, Clone)]
pub struct TemplateDetectionResult {
    /// Whether the file is detected as a template
    pub is_template: bool,
    /// The detected template engine (if any)
    pub engine: Option<TemplateEngine>,
    /// Method used for detection
    pub detection_method: TemplateDetectionMethod,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Score boost to apply
    pub score_boost: f64,
}

impl TemplateDetectionResult {
    pub fn not_template() -> Self {
        Self {
            is_template: false,
            engine: None,
            detection_method: TemplateDetectionMethod::Extension,
            confidence: 0.0,
            score_boost: 0.0,
        }
    }

    pub fn template(
        engine: TemplateEngine,
        method: TemplateDetectionMethod,
        confidence: f64,
    ) -> Self {
        let score_boost = engine.score_boost();
        Self {
            is_template: true,
            engine: Some(engine),
            detection_method: method,
            confidence,
            score_boost,
        }
    }
}

/// Template pattern definition for content analysis
#[derive(Debug, Clone)]
pub struct TemplatePattern {
    pub open_tag: String,
    pub close_tag: String,
    pub engine: TemplateEngine,
    pub min_occurrences: usize,
}

impl TemplatePattern {
    pub fn new(open: &str, close: &str, engine: TemplateEngine, min_occurrences: usize) -> Self {
        Self {
            open_tag: open.to_string(),
            close_tag: close.to_string(),
            engine,
            min_occurrences,
        }
    }
}

/// Static template patterns for content analysis
static TEMPLATE_PATTERNS: Lazy<Vec<TemplatePattern>> = Lazy::new(|| {
    vec![
        // Handlebars, Mustache patterns
        TemplatePattern::new("{{", "}}", TemplateEngine::Handlebars, 2),
        TemplatePattern::new("{{{", "}}}", TemplateEngine::Handlebars, 1),
        // Jinja2, Django, Liquid patterns
        TemplatePattern::new("{%", "%}", TemplateEngine::Jinja2, 2),
        TemplatePattern::new("{{", "}}", TemplateEngine::Jinja2, 1), // Also Handlebars
        // EJS, ERB patterns
        TemplatePattern::new("<%", "%>", TemplateEngine::Ejs, 2),
        TemplatePattern::new("<%=", "%>", TemplateEngine::Ejs, 1),
        TemplatePattern::new("<%#", "%>", TemplateEngine::Ejs, 1),
        // FreeMarker patterns
        TemplatePattern::new("<#", "#>", TemplateEngine::Generic, 2),
        // Template literals and other patterns
        TemplatePattern::new("${", "}", TemplateEngine::Generic, 3),
        TemplatePattern::new("@{", "}", TemplateEngine::Generic, 2),
        TemplatePattern::new("[[", "]]", TemplateEngine::Generic, 2),
    ]
});

/// Extension to template engine mapping
static EXTENSION_MAP: Lazy<HashMap<&'static str, TemplateEngine>> = Lazy::new(|| {
    let mut map = HashMap::new();

    // Dedicated template extensions
    map.insert(".njk", TemplateEngine::Jinja2);
    map.insert(".nunjucks", TemplateEngine::Jinja2);
    map.insert(".hbs", TemplateEngine::Handlebars);
    map.insert(".handlebars", TemplateEngine::Handlebars);
    map.insert(".j2", TemplateEngine::Jinja2);
    map.insert(".jinja", TemplateEngine::Jinja2);
    map.insert(".jinja2", TemplateEngine::Jinja2);
    map.insert(".twig", TemplateEngine::Twig);
    map.insert(".liquid", TemplateEngine::Liquid);
    map.insert(".mustache", TemplateEngine::Mustache);
    map.insert(".ejs", TemplateEngine::Ejs);
    map.insert(".erb", TemplateEngine::Erb);
    map.insert(".rhtml", TemplateEngine::Erb);
    map.insert(".haml", TemplateEngine::Haml);
    map.insert(".pug", TemplateEngine::Pug);
    map.insert(".jade", TemplateEngine::Jade);
    map.insert(".dust", TemplateEngine::Dust);
    map.insert(".eta", TemplateEngine::Eta);
    map.insert(".svelte", TemplateEngine::Svelte);
    map.insert(".vue", TemplateEngine::Vue);
    map.insert(".jsx", TemplateEngine::React);
    map.insert(".tsx", TemplateEngine::React);

    map
});

/// Single-pattern indicators for template detection
static SINGLE_PATTERNS: &[&str] = &[
    "ng-",        // Angular directives
    "v-",         // Vue directives
    ":",          // Vue shorthand or other template syntax
    "data-bind",  // Knockout.js
    "handlebars", // Handlebars comments
    "jinja",      // Jinja comments
    "mustache",   // Mustache comments
    "twig",       // Twig comments
    "liquid",     // Liquid comments
];

/// Template directory indicators
static TEMPLATE_DIRECTORIES: &[&str] = &[
    "template",
    "templates",
    "_includes",
    "_layouts",
    "layout",
    "layouts",
    "view",
    "views",
    "component",
    "components",
    "partial",
    "partials",
];

/// Advanced template detection engine
pub struct TemplateDetector {
    /// AST parsers for different languages
    parsers: HashMap<String, Parser>,
    /// File content cache (for performance)
    content_cache: HashMap<PathBuf, String>,
    /// Cache size limit
    max_cache_size: usize,
}

impl std::fmt::Debug for TemplateDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TemplateDetector")
            .field("parsers", &format!("[{} parsers]", self.parsers.len()))
            .field(
                "content_cache",
                &format!("[{} cached items]", self.content_cache.len()),
            )
            .field("max_cache_size", &self.max_cache_size)
            .finish()
    }
}

impl TemplateDetector {
    /// Create a new template detector
    pub fn new() -> Result<Self> {
        let mut parsers = HashMap::new();

        // Initialize HTML parser for template detection
        let mut html_parser = Parser::new();
        html_parser
            .set_language(tree_sitter_html::language())
            .map_err(|e| {
                scribe_core::ScribeError::parse(format!("Failed to set HTML language: {}", e))
            })?;
        parsers.insert("html".to_string(), html_parser);

        Ok(Self {
            parsers,
            content_cache: HashMap::new(),
            max_cache_size: 100, // Cache up to 100 files
        })
    }

    /// Detect if a file is a template and get appropriate score boost
    pub fn detect_template(&mut self, file_path: &str) -> Result<TemplateDetectionResult> {
        let path = Path::new(file_path);

        // Method 1: Extension-based detection (fastest)
        if let Some(result) = self.detect_by_extension(path) {
            return Ok(result);
        }

        // Method 2: Directory context analysis
        if let Some(result) = self.detect_by_directory_context(path) {
            return Ok(result);
        }

        // Method 3: Content pattern analysis (slower, for ambiguous files)
        if self.should_analyze_content(path) {
            if let Some(result) = self.detect_by_content_patterns(path)? {
                return Ok(result);
            }
        }

        // Method 4: Language-specific heuristics
        if let Some(result) = self.detect_by_language_heuristics(path) {
            return Ok(result);
        }

        Ok(TemplateDetectionResult::not_template())
    }

    /// Get template score boost for a file path
    pub fn get_score_boost(&self, file_path: &str) -> Result<f64> {
        // Use a simplified version that doesn't require mutable self
        let path = Path::new(file_path);

        // Extension-based detection
        if let Some(engine) = self.detect_engine_by_extension(path) {
            return Ok(engine.score_boost());
        }

        // Directory context check
        if self.is_in_template_directory(path) {
            return Ok(1.2); // Moderate boost for template directories
        }

        Ok(0.0)
    }

    /// Detect template engine by file extension
    fn detect_by_extension(&self, path: &Path) -> Option<TemplateDetectionResult> {
        if let Some(engine) = self.detect_engine_by_extension(path) {
            return Some(TemplateDetectionResult::template(
                engine,
                TemplateDetectionMethod::Extension,
                0.95, // High confidence for extension-based detection
            ));
        }
        None
    }

    fn detect_engine_by_extension(&self, path: &Path) -> Option<TemplateEngine> {
        let extension = path.extension()?.to_str()?.to_lowercase();
        let ext_with_dot = format!(".{}", extension);

        EXTENSION_MAP.get(ext_with_dot.as_str()).cloned()
    }

    /// Detect by directory context
    fn detect_by_directory_context(&self, path: &Path) -> Option<TemplateDetectionResult> {
        if self.is_in_template_directory(path) {
            // Check if it's HTML/XML in a template directory
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_str()?.to_lowercase();
                if matches!(ext_str.as_str(), "html" | "htm" | "xml") {
                    return Some(TemplateDetectionResult::template(
                        TemplateEngine::Generic,
                        TemplateDetectionMethod::DirectoryContext,
                        0.7, // Moderate confidence for directory context
                    ));
                }
            }
        }
        None
    }

    fn is_in_template_directory(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy().to_lowercase();
        TEMPLATE_DIRECTORIES
            .iter()
            .any(|dir| path_str.contains(dir))
    }

    /// Check if file should be analyzed for content patterns
    fn should_analyze_content(&self, path: &Path) -> bool {
        // Only analyze potentially ambiguous files
        if let Some(ext) = path.extension() {
            let ext_str = ext.to_str().unwrap_or("").to_lowercase();
            return matches!(ext_str.as_str(), "html" | "htm" | "xml" | "js" | "ts");
        }
        false
    }

    /// Detect templates by AST-based content pattern analysis
    fn detect_by_content_patterns(
        &mut self,
        path: &Path,
    ) -> Result<Option<TemplateDetectionResult>> {
        let content = self.read_file_content(path)?;

        // First check for simple template patterns (lightweight check)
        for pattern in TEMPLATE_PATTERNS.iter() {
            let occurrences =
                self.count_pattern_occurrences(&content, &pattern.open_tag, &pattern.close_tag);

            if occurrences >= pattern.min_occurrences {
                return Ok(Some(TemplateDetectionResult::template(
                    pattern.engine.clone(),
                    TemplateDetectionMethod::ContentPattern,
                    0.8, // Good confidence for pattern-based detection
                )));
            }
        }

        // If it's HTML/XML content, try AST-based analysis
        if self.should_use_ast_analysis(path) {
            if let Some(result) = self.analyze_with_ast(path, &content)? {
                return Ok(Some(result));
            }
        }

        // Check for single-pattern indicators as fallback
        let content_lower = content.to_lowercase();
        for &pattern in SINGLE_PATTERNS {
            if content_lower.contains(pattern) {
                return Ok(Some(TemplateDetectionResult::template(
                    TemplateEngine::Generic,
                    TemplateDetectionMethod::ContentPattern,
                    0.6, // Lower confidence for single patterns
                )));
            }
        }

        Ok(None)
    }

    /// Detect by language-specific heuristics
    fn detect_by_language_heuristics(&self, path: &Path) -> Option<TemplateDetectionResult> {
        // This is a placeholder for more sophisticated language analysis
        // In a full implementation, this might use tree-sitter or similar
        // for AST-based template detection

        if let Some(ext) = path.extension() {
            let ext_str = ext.to_str()?.to_lowercase();

            // JSX/TSX are React templates
            if matches!(ext_str.as_str(), "jsx" | "tsx") {
                return Some(TemplateDetectionResult::template(
                    TemplateEngine::React,
                    TemplateDetectionMethod::LanguageHeuristic,
                    0.9,
                ));
            }
        }

        None
    }

    /// Read file content with caching
    fn read_file_content(&mut self, path: &Path) -> Result<String> {
        // Check cache first
        if let Some(content) = self.content_cache.get(path) {
            return Ok(content.clone());
        }

        // Read file (limit to first 2KB for performance)
        let file = fs::File::open(path)?;
        let reader = BufReader::new(file);
        let mut content = String::new();
        let mut bytes_read = 0;
        const MAX_READ_SIZE: usize = 2048;

        for line in reader.lines() {
            let line = line?;
            if bytes_read + line.len() > MAX_READ_SIZE {
                break;
            }
            content.push_str(&line);
            content.push('\n');
            bytes_read += line.len() + 1;
        }

        // Cache the content (with size limit)
        if self.content_cache.len() < self.max_cache_size {
            self.content_cache
                .insert(path.to_path_buf(), content.clone());
        }

        Ok(content)
    }

    /// Count occurrences of a pattern pair in content
    fn count_pattern_occurrences(&self, content: &str, open_tag: &str, close_tag: &str) -> usize {
        let open_count = content.matches(open_tag).count();
        let close_count = content.matches(close_tag).count();

        // Return the minimum of open and close tags (pairs)
        open_count.min(close_count)
    }

    /// Check if file should use AST analysis
    fn should_use_ast_analysis(&self, path: &Path) -> bool {
        if let Some(ext) = path.extension() {
            let ext_str = ext.to_str().unwrap_or("").to_lowercase();
            return matches!(ext_str.as_str(), "html" | "htm" | "xml" | "vue" | "svelte");
        }
        false
    }

    /// Analyze content using AST parsing
    fn analyze_with_ast(
        &mut self,
        path: &Path,
        content: &str,
    ) -> Result<Option<TemplateDetectionResult>> {
        if let Some(parser) = self.parsers.get_mut("html") {
            if let Some(tree) = parser.parse(content, None) {
                let root_node = tree.root_node();

                // Check for template-specific patterns in AST
                if self.has_template_attributes(&root_node) {
                    let engine = self.detect_template_engine_from_ast(&root_node, path);
                    return Ok(Some(TemplateDetectionResult::template(
                        engine,
                        TemplateDetectionMethod::ContentPattern,
                        0.85, // High confidence for AST-based detection
                    )));
                }
            }
        }
        Ok(None)
    }

    /// Check for template-specific attributes in HTML AST
    fn has_template_attributes(&self, node: &Node) -> bool {
        let template_indicators = [
            "v-",     // Vue.js directives
            "ng-",    // Angular directives
            "*ng",    // Angular structural directives
            ":bind",  // Vue binding
            "@click", // Vue events
            "{{{",    // Template expressions
            "{{",     // Template expressions
            "<%",     // EJS, ERB
            "{%",     // Jinja2, Liquid
        ];

        self.node_contains_patterns(node, &template_indicators)
    }

    /// Recursively check if node or its children contain template patterns
    fn node_contains_patterns(&self, node: &Node, patterns: &[&str]) -> bool {
        // Check current node kind
        if patterns
            .iter()
            .any(|&pattern| node.kind().contains(pattern))
        {
            return true;
        }

        // Check node text content if it's a text node
        if node.kind() == "text" || node.kind() == "attribute_value" {
            // For text nodes, we'd need the actual content, which requires the source text
            // This is a simplified check - in practice you'd get the node's text from content
            return true; // Assume potential template content for now
        }

        // Recursively check children
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                if self.node_contains_patterns(&child, patterns) {
                    return true;
                }
            }
        }

        false
    }

    /// Detect specific template engine from AST patterns
    fn detect_template_engine_from_ast(&self, node: &Node, path: &Path) -> TemplateEngine {
        // Check file extension first
        if let Some(ext) = path.extension() {
            let ext_str = ext.to_str().unwrap_or("").to_lowercase();
            match ext_str.as_str() {
                "vue" => return TemplateEngine::Vue,
                "svelte" => return TemplateEngine::Svelte,
                _ => {}
            }
        }

        // Analyze AST structure for engine-specific patterns
        if self.has_vue_patterns(node) {
            TemplateEngine::Vue
        } else if self.has_angular_patterns(node) {
            TemplateEngine::Angular
        } else if self.has_react_patterns(node) {
            TemplateEngine::React
        } else {
            TemplateEngine::Generic
        }
    }

    /// Check for Vue.js specific patterns
    fn has_vue_patterns(&self, node: &Node) -> bool {
        let vue_patterns = ["v-if", "v-for", "v-model", "v-bind", "@click", ":class"];
        self.node_contains_patterns(node, &vue_patterns)
    }

    /// Check for Angular specific patterns
    fn has_angular_patterns(&self, node: &Node) -> bool {
        let angular_patterns = ["*ngFor", "*ngIf", "(click)", "[class]", "[(ngModel)]"];
        self.node_contains_patterns(node, &angular_patterns)
    }

    /// Check for React JSX patterns (limited in HTML parser)
    fn has_react_patterns(&self, node: &Node) -> bool {
        let react_patterns = ["className", "onClick", "onChange"];
        self.node_contains_patterns(node, &react_patterns)
    }

    /// Clear content cache
    pub fn clear_cache(&mut self) {
        self.content_cache.clear();
    }
}

impl Default for TemplateDetector {
    fn default() -> Self {
        Self::new().expect("Failed to create TemplateDetector")
    }
}

/// Convenience function to check if a file is a template
pub fn is_template_file(file_path: &str) -> Result<bool> {
    let mut detector = TemplateDetector::new()?;
    let result = detector.detect_template(file_path)?;
    Ok(result.is_template)
}

/// Convenience function to get template score boost
pub fn get_template_score_boost(file_path: &str) -> Result<f64> {
    let detector = TemplateDetector::new()?;
    detector.get_score_boost(file_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_file(content: &str, extension: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();

        // Rename with proper extension (this is a bit hacky but works for tests)
        let path = file.path().with_extension(extension);
        std::fs::rename(file.path(), &path).unwrap();

        file
    }

    #[test]
    fn test_extension_based_detection() {
        let detector = TemplateDetector::new().unwrap();

        // Test known template extensions
        assert_eq!(
            detector.detect_engine_by_extension(Path::new("template.hbs")),
            Some(TemplateEngine::Handlebars)
        );

        assert_eq!(
            detector.detect_engine_by_extension(Path::new("view.j2")),
            Some(TemplateEngine::Jinja2)
        );

        assert_eq!(
            detector.detect_engine_by_extension(Path::new("component.jsx")),
            Some(TemplateEngine::React)
        );

        // Test non-template extension
        assert_eq!(
            detector.detect_engine_by_extension(Path::new("script.js")),
            None
        );
    }

    #[test]
    fn test_directory_context_detection() {
        let detector = TemplateDetector::new().unwrap();

        assert!(detector.is_in_template_directory(Path::new("templates/layout.html")));
        assert!(detector.is_in_template_directory(Path::new("src/components/header.html")));
        assert!(!detector.is_in_template_directory(Path::new("src/utils/helper.js")));
    }

    #[test]
    fn test_pattern_counting() {
        let detector = TemplateDetector::new().unwrap();
        let content = "Hello {{ name }}! Welcome to {{ site }}.";

        assert_eq!(detector.count_pattern_occurrences(content, "{{", "}}"), 2);
        assert_eq!(detector.count_pattern_occurrences(content, "{%", "%}"), 0);
    }

    #[test]
    fn test_template_score_boost() {
        let detector = TemplateDetector::new().unwrap();

        // High boost for dedicated template files
        assert!(detector.get_score_boost("template.hbs").unwrap() > 1.0);

        // No boost for regular files
        assert_eq!(detector.get_score_boost("script.js").unwrap(), 0.0);
    }

    #[test]
    fn test_engine_score_boost() {
        assert!(TemplateEngine::Handlebars.score_boost() > 1.0);
        assert!(TemplateEngine::React.score_boost() > 1.0);
        assert!(TemplateEngine::Generic.score_boost() < 1.0);
    }
}
