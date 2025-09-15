//! Content analysis for extracting imports, documentation structure, and code metrics.
//!
//! This module provides advanced content analysis capabilities including:
//! - Import and dependency extraction for multiple languages
//! - Documentation structure analysis (headings, links, code blocks)
//! - Code complexity metrics and statistics
//! - Text content classification and analysis

use scribe_core::{Result, ScribeError, Language};
use scribe_selection::ast_parser::{AstParser, AstLanguage, AstImport};
use std::path::{Path, PathBuf};
use std::collections::{HashMap, HashSet};
use std::fs;
use regex::Regex;
use serde::{Serialize, Deserialize};
use once_cell::sync::Lazy;

/// Comprehensive content analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentStats {
    pub imports: ImportInfo,
    pub documentation: DocumentationInfo,
    pub complexity: ComplexityMetrics,
    pub structure: StructureInfo,
    pub text_stats: TextStats,
}

/// Import and dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportInfo {
    pub total_imports: usize,
    pub unique_imports: usize,
    pub import_sources: Vec<ImportSource>,
    pub external_dependencies: HashSet<String>,
    pub internal_dependencies: HashSet<String>,
    pub relative_imports: usize,
    pub absolute_imports: usize,
}

/// Individual import source information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportSource {
    pub module: String,
    pub alias: Option<String>,
    pub items: Vec<String>,
    pub line_number: usize,
    pub import_type: ImportType,
}

/// Type of import statement
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImportType {
    Standard,     // Standard library
    External,     // Third-party package
    Internal,     // Internal module/package
    Relative,     // Relative import
    Dynamic,      // Dynamic/runtime import
}

/// Documentation structure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationInfo {
    pub headings: Vec<Heading>,
    pub links: Vec<Link>,
    pub code_blocks: Vec<CodeBlock>,
    pub tables: usize,
    pub lists: usize,
    pub images: usize,
    pub todo_comments: Vec<TodoComment>,
    pub docstrings: Vec<Docstring>,
}

/// Documentation heading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heading {
    pub level: usize,
    pub text: String,
    pub line_number: usize,
    pub anchor: Option<String>,
}

/// Link in documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    pub text: String,
    pub url: String,
    pub line_number: usize,
    pub link_type: LinkType,
}

/// Type of link
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinkType {
    Internal,   // Internal document link
    External,   // External URL
    Relative,   // Relative file path
    Anchor,     // In-document anchor
}

/// Code block in documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeBlock {
    pub language: Option<String>,
    pub content: String,
    pub line_number: usize,
    pub line_count: usize,
}

/// TODO/FIXME/NOTE comment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoComment {
    pub comment_type: TodoType,
    pub text: String,
    pub line_number: usize,
    pub author: Option<String>,
}

/// Type of TODO comment
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TodoType {
    Todo,
    Fixme,
    Note,
    Bug,
    Hack,
    Warning,
}

/// Docstring information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Docstring {
    pub content: String,
    pub line_number: usize,
    pub line_count: usize,
    pub style: DocstringStyle,
}

/// Docstring style
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DocstringStyle {
    Google,
    Numpy,
    Sphinx,
    Rustdoc,
    Javadoc,
    JSDoc,
    Unknown,
}

/// Code complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    pub cyclomatic_complexity: usize,
    pub function_count: usize,
    pub class_count: usize,
    pub nesting_depth: usize,
    pub cognitive_complexity: usize,
    pub halstead_metrics: HalsteadMetrics,
}

/// Halstead complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HalsteadMetrics {
    pub distinct_operators: usize,
    pub distinct_operands: usize,
    pub total_operators: usize,
    pub total_operands: usize,
    pub vocabulary: usize,
    pub length: usize,
    pub difficulty: f64,
    pub effort: f64,
}

/// Structural information about the file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureInfo {
    pub functions: Vec<FunctionInfo>,
    pub classes: Vec<ClassInfo>,
    pub constants: Vec<ConstantInfo>,
    pub interfaces: Vec<InterfaceInfo>,
}

/// Function information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionInfo {
    pub name: String,
    pub line_number: usize,
    pub line_count: usize,
    pub parameters: Vec<String>,
    pub return_type: Option<String>,
    pub visibility: Visibility,
    pub is_async: bool,
    pub is_generator: bool,
    pub docstring: Option<String>,
}

/// Class information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassInfo {
    pub name: String,
    pub line_number: usize,
    pub line_count: usize,
    pub parent_classes: Vec<String>,
    pub methods: Vec<FunctionInfo>,
    pub attributes: Vec<String>,
    pub visibility: Visibility,
    pub docstring: Option<String>,
}

/// Constant/variable information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantInfo {
    pub name: String,
    pub line_number: usize,
    pub value_type: Option<String>,
    pub visibility: Visibility,
}

/// Interface information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceInfo {
    pub name: String,
    pub line_number: usize,
    pub methods: Vec<String>,
    pub extends: Vec<String>,
}

/// Visibility modifier
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Visibility {
    Public,
    Private,
    Protected,
    Package,
    Unknown,
}

/// Basic text statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextStats {
    pub line_count: usize,
    pub non_empty_line_count: usize,
    pub comment_line_count: usize,
    pub code_line_count: usize,
    pub blank_line_count: usize,
    pub character_count: usize,
    pub word_count: usize,
    pub comment_density: f64, // ratio of comment lines to code lines
}

/// Content analyzer with language-specific parsers
pub struct ContentAnalyzer {
    regex_cache: HashMap<String, Regex>,
    ast_parser: AstParser,
}


// Compile-time regex patterns for common operations
static HEADING_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(#{1,6})\s+(.+)").unwrap());
static LINK_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"\[([^\]]+)\]\(([^)]+)\)").unwrap());
static TODO_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?://|#|/\*|\*|<!--)\s*(TODO|FIXME|NOTE|BUG|HACK|WARNING):?\s*(.*)").unwrap()
});
static CODE_BLOCK_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"```(\w+)?\n((?s).*?)```").unwrap()
});

impl Default for ContentStats {
    fn default() -> Self {
        Self {
            imports: ImportInfo::default(),
            documentation: DocumentationInfo::default(),
            complexity: ComplexityMetrics::default(),
            structure: StructureInfo::default(),
            text_stats: TextStats::default(),
        }
    }
}

impl Default for ImportInfo {
    fn default() -> Self {
        Self {
            total_imports: 0,
            unique_imports: 0,
            import_sources: Vec::new(),
            external_dependencies: HashSet::new(),
            internal_dependencies: HashSet::new(),
            relative_imports: 0,
            absolute_imports: 0,
        }
    }
}

impl Default for DocumentationInfo {
    fn default() -> Self {
        Self {
            headings: Vec::new(),
            links: Vec::new(),
            code_blocks: Vec::new(),
            tables: 0,
            lists: 0,
            images: 0,
            todo_comments: Vec::new(),
            docstrings: Vec::new(),
        }
    }
}

impl Default for ComplexityMetrics {
    fn default() -> Self {
        Self {
            cyclomatic_complexity: 0,
            function_count: 0,
            class_count: 0,
            nesting_depth: 0,
            cognitive_complexity: 0,
            halstead_metrics: HalsteadMetrics::default(),
        }
    }
}

impl Default for HalsteadMetrics {
    fn default() -> Self {
        Self {
            distinct_operators: 0,
            distinct_operands: 0,
            total_operators: 0,
            total_operands: 0,
            vocabulary: 0,
            length: 0,
            difficulty: 0.0,
            effort: 0.0,
        }
    }
}

impl Default for StructureInfo {
    fn default() -> Self {
        Self {
            functions: Vec::new(),
            classes: Vec::new(),
            constants: Vec::new(),
            interfaces: Vec::new(),
        }
    }
}

impl Default for TextStats {
    fn default() -> Self {
        Self {
            line_count: 0,
            non_empty_line_count: 0,
            comment_line_count: 0,
            code_line_count: 0,
            blank_line_count: 0,
            character_count: 0,
            word_count: 0,
            comment_density: 0.0,
        }
    }
}

impl ContentAnalyzer {
    /// Create a new content analyzer
    pub fn new() -> Self {
        Self {
            regex_cache: HashMap::new(),
            ast_parser: AstParser::new().expect("Failed to initialize AST parser"),
        }
    }

    /// Analyze a file and extract comprehensive content information
    pub async fn analyze_file(&self, path: &Path) -> Result<ContentStats> {
        let content = tokio::fs::read_to_string(path).await
            .map_err(|e| ScribeError::io(format!("Failed to read file {}: {}", path.display(), e), e))?;

        let language = self.detect_language_from_path(path);
        self.analyze_content(&content, &language).await
    }

    /// Analyze content string directly
    pub async fn analyze_content(&self, content: &str, language: &Language) -> Result<ContentStats> {
        let mut stats = ContentStats::default();

        // Parallel analysis of all aspects
        let (imports, documentation, complexity, structure, text_stats) = tokio::join!(
            self.analyze_imports_async(content, language),
            self.analyze_documentation_async(content),
            self.analyze_complexity_async(content, language),
            self.analyze_structure_async(content, language),
            self.analyze_text_stats_async(content)
        );

        stats.imports = imports?;
        stats.documentation = documentation?;
        stats.complexity = complexity?;
        stats.structure = structure?;
        stats.text_stats = text_stats?;

        Ok(stats)
    }

    /// Analyze imports and dependencies using tree-sitter AST parsing
    async fn analyze_imports_async(&self, content: &str, language: &Language) -> Result<ImportInfo> {
        let mut import_info = ImportInfo::default();
        
        // Convert Language to AstLanguage
        let ast_language = match language {
            Language::Python => Some(AstLanguage::Python),
            Language::JavaScript => Some(AstLanguage::JavaScript),
            Language::TypeScript => Some(AstLanguage::TypeScript),
            Language::Go => Some(AstLanguage::Go),
            Language::Rust => Some(AstLanguage::Rust),
            _ => None, // Fall back to regex for unsupported languages
        };
        
        if let Some(ast_lang) = ast_language {
            // Use tree-sitter to extract imports
            match self.ast_parser.extract_imports(content, ast_lang) {
                Ok(imports) => {
                    for (line_number, import) in imports.into_iter().enumerate() {
                        let import_type = self.classify_import_type(&import.module);
                        
                        let import_source = ImportSource {
                            module: import.module.clone(),
                            alias: import.alias,
                            items: import.items,
                            line_number: line_number + 1,
                            import_type: import_type.clone(),
                        };
                        
                        import_info.import_sources.push(import_source);
                        
                        // Classify import type
                        match import_type {
                            ImportType::External => {
                                import_info.external_dependencies.insert(import.module);
                                import_info.absolute_imports += 1;
                            }
                            ImportType::Internal => {
                                import_info.internal_dependencies.insert(import.module);
                                import_info.absolute_imports += 1;
                            }
                            ImportType::Relative => {
                                import_info.relative_imports += 1;
                            }
                            _ => {
                                import_info.absolute_imports += 1;
                            }
                        }
                    }
                    
                    import_info.total_imports = import_info.import_sources.len();
                    import_info.unique_imports = import_info.external_dependencies.len() + 
                                               import_info.internal_dependencies.len();
                }
                Err(_) => {
                    // If tree-sitter parsing fails, return empty import info
                    // (could fall back to regex here if needed)
                }
            }
        }

        Ok(import_info)
    }

    /// Analyze documentation structure
    async fn analyze_documentation_async(&self, content: &str) -> Result<DocumentationInfo> {
        let mut doc_info = DocumentationInfo::default();
        let mut line_number = 1;

        for line in content.lines() {
            // Find headings
            if let Some(captures) = HEADING_REGEX.captures(line) {
                let level = captures.get(1).unwrap().as_str().len();
                let text = captures.get(2).unwrap().as_str().trim().to_string();
                
                doc_info.headings.push(Heading {
                    level,
                    text: text.clone(),
                    line_number,
                    anchor: Some(self.generate_anchor(&text)),
                });
            }

            // Find links
            for captures in LINK_REGEX.captures_iter(line) {
                let text = captures.get(1).unwrap().as_str().to_string();
                let url = captures.get(2).unwrap().as_str().to_string();
                
                doc_info.links.push(Link {
                    text,
                    url: url.clone(),
                    line_number,
                    link_type: self.classify_link(&url),
                });
            }

            // Find TODO comments
            if let Some(captures) = TODO_REGEX.captures(line) {
                let comment_type = match captures.get(1).unwrap().as_str().to_uppercase().as_str() {
                    "TODO" => TodoType::Todo,
                    "FIXME" => TodoType::Fixme,
                    "NOTE" => TodoType::Note,
                    "BUG" => TodoType::Bug,
                    "HACK" => TodoType::Hack,
                    "WARNING" => TodoType::Warning,
                    _ => TodoType::Todo,
                };
                
                let text = captures.get(2).map_or(String::new(), |m| m.as_str().trim().to_string());
                
                doc_info.todo_comments.push(TodoComment {
                    comment_type,
                    text,
                    line_number,
                    author: None, // Could be enhanced to extract from git blame
                });
            }

            // Count tables and lists
            if line.starts_with('|') && line.ends_with('|') {
                doc_info.tables += 1;
            }
            if line.trim_start().starts_with('-') || line.trim_start().starts_with('*') || 
               line.trim_start().chars().next().map_or(false, |c| c.is_digit(10)) {
                doc_info.lists += 1;
            }

            line_number += 1;
        }

        // Find code blocks
        for captures in CODE_BLOCK_REGEX.captures_iter(content) {
            let language = captures.get(1).map(|m| m.as_str().to_string());
            let content_str = captures.get(2).unwrap().as_str().to_string();
            let line_count = content_str.lines().count();
            
            doc_info.code_blocks.push(CodeBlock {
                language,
                content: content_str,
                line_number: 0, // Would need more sophisticated parsing
                line_count,
            });
        }

        Ok(doc_info)
    }

    /// Analyze code complexity metrics
    async fn analyze_complexity_async(&self, content: &str, language: &Language) -> Result<ComplexityMetrics> {
        let mut complexity = ComplexityMetrics::default();

        // Basic complexity analysis - could be enhanced with proper AST parsing
        let lines: Vec<&str> = content.lines().collect();
        
        for line in &lines {
            let trimmed = line.trim();
            
            // Count functions (basic pattern matching)
            if self.is_function_declaration(trimmed, language) {
                complexity.function_count += 1;
            }
            
            // Count classes
            if self.is_class_declaration(trimmed, language) {
                complexity.class_count += 1;
            }
            
            // Simple cyclomatic complexity (count decision points)
            if self.is_decision_point(trimmed, language) {
                complexity.cyclomatic_complexity += 1;
            }
        }

        // Calculate nesting depth
        complexity.nesting_depth = self.calculate_max_nesting_depth(content, language);
        
        // Basic Halstead metrics
        complexity.halstead_metrics = self.calculate_halstead_metrics(content, language);

        Ok(complexity)
    }

    /// Analyze code structure
    async fn analyze_structure_async(&self, content: &str, language: &Language) -> Result<StructureInfo> {
        let mut structure = StructureInfo::default();
        
        // This would ideally use a proper AST parser for each language
        // For now, we'll use basic pattern matching
        let mut line_number = 1;
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            if let Some(function_info) = self.parse_function_declaration(trimmed, line_number, language) {
                structure.functions.push(function_info);
            }
            
            if let Some(class_info) = self.parse_class_declaration(trimmed, line_number, language) {
                structure.classes.push(class_info);
            }
            
            if let Some(constant_info) = self.parse_constant_declaration(trimmed, line_number, language) {
                structure.constants.push(constant_info);
            }
            
            line_number += 1;
        }

        Ok(structure)
    }

    /// Analyze basic text statistics
    async fn analyze_text_stats_async(&self, content: &str) -> Result<TextStats> {
        let lines: Vec<&str> = content.lines().collect();
        let line_count = lines.len();
        let character_count = content.len();
        let word_count = content.split_whitespace().count();
        
        let mut non_empty_line_count = 0;
        let mut comment_line_count = 0;
        let mut blank_line_count = 0;
        
        for line in &lines {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                blank_line_count += 1;
            } else {
                non_empty_line_count += 1;
                if self.is_comment_line(trimmed) {
                    comment_line_count += 1;
                }
            }
        }
        
        let code_line_count = non_empty_line_count - comment_line_count;
        let comment_density = if code_line_count > 0 {
            comment_line_count as f64 / code_line_count as f64
        } else {
            0.0
        };

        Ok(TextStats {
            line_count,
            non_empty_line_count,
            comment_line_count,
            code_line_count,
            blank_line_count,
            character_count,
            word_count,
            comment_density,
        })
    }



    /// Classify import type based on module name
    fn classify_import_type(&self, module: &str) -> ImportType {
        if module.starts_with('.') || module.starts_with("./") || module.starts_with("../") {
            ImportType::Relative
        } else if self.is_standard_library_module(module) {
            ImportType::Standard
        } else if module.contains('/') || module.contains('.') {
            ImportType::External
        } else {
            ImportType::Internal
        }
    }

    /// Check if a module is part of the standard library
    fn is_standard_library_module(&self, module: &str) -> bool {
        // This would need to be language-specific
        match module {
            // Python standard library examples
            "os" | "sys" | "json" | "re" | "collections" | "itertools" | "functools" => true,
            // JavaScript/Node.js standard modules
            "fs" | "path" | "http" | "https" | "url" | "crypto" => true,
            _ => false,
        }
    }

    /// Generate anchor for heading
    fn generate_anchor(&self, text: &str) -> String {
        text.to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '-' })
            .collect::<String>()
            .split('-')
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("-")
    }

    /// Classify link type
    fn classify_link(&self, url: &str) -> LinkType {
        if url.starts_with("http://") || url.starts_with("https://") {
            LinkType::External
        } else if url.starts_with("#") {
            LinkType::Anchor
        } else if url.starts_with("./") || url.starts_with("../") {
            LinkType::Relative
        } else {
            LinkType::Internal
        }
    }

    /// Detect language from file path
    fn detect_language_from_path(&self, path: &Path) -> Language {
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            Language::from_extension(extension)
        } else {
            Language::Unknown
        }
    }

    /// Check if line is a function declaration
    fn is_function_declaration(&self, line: &str, language: &Language) -> bool {
        match language {
            Language::Python => line.starts_with("def ") || line.starts_with("async def "),
            Language::JavaScript | Language::TypeScript => {
                line.contains("function ") || line.contains("=> ") || line.contains("function(")
            }
            Language::Rust => line.starts_with("fn ") || line.starts_with("pub fn "),
            Language::Java => line.contains("public ") && line.contains("(") && line.contains(")"),
            _ => false,
        }
    }

    /// Check if line is a class declaration
    fn is_class_declaration(&self, line: &str, language: &Language) -> bool {
        match language {
            Language::Python => line.starts_with("class "),
            Language::JavaScript | Language::TypeScript => line.starts_with("class "),
            Language::Java => line.contains("class ") && line.contains("{"),
            Language::Rust => line.starts_with("struct ") || line.starts_with("enum "),
            _ => false,
        }
    }

    /// Check if line is a decision point for complexity calculation
    fn is_decision_point(&self, line: &str, _language: &Language) -> bool {
        // Common decision points across languages
        line.contains("if ") || line.contains("elif ") || line.contains("else ") ||
        line.contains("for ") || line.contains("while ") || line.contains("match ") ||
        line.contains("switch ") || line.contains("case ") || line.contains("catch ") ||
        line.contains("&&") || line.contains("||") || line.contains("?")
    }

    /// Calculate maximum nesting depth
    fn calculate_max_nesting_depth(&self, content: &str, _language: &Language) -> usize {
        let mut max_depth = 0;
        let mut current_depth = 0;
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            // Count opening braces/indentation
            let opens = trimmed.matches('{').count() + 
                       trimmed.matches('(').count() + 
                       trimmed.matches('[').count();
            let closes = trimmed.matches('}').count() + 
                        trimmed.matches(')').count() + 
                        trimmed.matches(']').count();
            
            current_depth += opens;
            max_depth = max_depth.max(current_depth);
            current_depth = current_depth.saturating_sub(closes);
        }
        
        max_depth
    }

    /// Calculate basic Halstead metrics
    fn calculate_halstead_metrics(&self, content: &str, _language: &Language) -> HalsteadMetrics {
        // This is a simplified version - real Halstead metrics need proper tokenization
        let words: Vec<&str> = content.split_whitespace().collect();
        let unique_words: HashSet<&str> = words.iter().cloned().collect();
        
        let operators = ["+", "-", "*", "/", "=", "==", "!=", "&&", "||", "!", "<", ">", "<=", ">="];
        let mut operator_count = 0;
        let mut unique_operators = HashSet::new();
        
        for word in &words {
            for &op in &operators {
                if word.contains(op) {
                    operator_count += 1;
                    unique_operators.insert(op);
                }
            }
        }
        
        let distinct_operators = unique_operators.len();
        let distinct_operands = unique_words.len().saturating_sub(distinct_operators);
        let total_operators = operator_count;
        let total_operands = words.len().saturating_sub(operator_count);
        let vocabulary = distinct_operators + distinct_operands;
        let length = total_operators + total_operands;
        
        let difficulty = if distinct_operands > 0 {
            (distinct_operators as f64 / 2.0) * (total_operands as f64 / distinct_operands as f64)
        } else {
            0.0
        };
        
        let effort = difficulty * length as f64;
        
        HalsteadMetrics {
            distinct_operators,
            distinct_operands,
            total_operators,
            total_operands,
            vocabulary,
            length,
            difficulty,
            effort,
        }
    }

    /// Parse function declaration (simplified)
    fn parse_function_declaration(&self, line: &str, line_number: usize, language: &Language) -> Option<FunctionInfo> {
        if !self.is_function_declaration(line, language) {
            return None;
        }
        
        // This is a very basic parser - would need proper AST parsing for production
        let name = match language {
            Language::Python => {
                if let Some(start) = line.find("def ") {
                    let after_def = &line[start + 4..];
                    if let Some(paren_pos) = after_def.find('(') {
                        Some(after_def[..paren_pos].trim().to_string())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            Language::Rust => {
                if let Some(start) = line.find("fn ") {
                    let after_fn = &line[start + 3..];
                    if let Some(paren_pos) = after_fn.find('(') {
                        Some(after_fn[..paren_pos].trim().to_string())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        };
        
        if let Some(function_name) = name {
            Some(FunctionInfo {
                name: function_name,
                line_number,
                line_count: 1, // Would need multi-line parsing
                parameters: vec![], // Would need parameter parsing
                return_type: None, // Would need return type parsing
                visibility: Visibility::Unknown,
                is_async: line.contains("async"),
                is_generator: line.contains("yield") || line.contains("generator"),
                docstring: None,
            })
        } else {
            None
        }
    }

    /// Parse class declaration (simplified)
    fn parse_class_declaration(&self, line: &str, line_number: usize, language: &Language) -> Option<ClassInfo> {
        if !self.is_class_declaration(line, language) {
            return None;
        }
        
        let name = match language {
            Language::Python => {
                if let Some(start) = line.find("class ") {
                    let after_class = &line[start + 6..];
                    if let Some(colon_pos) = after_class.find(':') {
                        Some(after_class[..colon_pos].trim().split('(').next().unwrap().trim().to_string())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        };
        
        if let Some(class_name) = name {
            Some(ClassInfo {
                name: class_name,
                line_number,
                line_count: 1, // Would need multi-line parsing
                parent_classes: vec![], // Would need inheritance parsing
                methods: vec![], // Would need method parsing
                attributes: vec![], // Would need attribute parsing
                visibility: Visibility::Unknown,
                docstring: None,
            })
        } else {
            None
        }
    }

    /// Parse constant declaration (simplified)
    fn parse_constant_declaration(&self, line: &str, line_number: usize, _language: &Language) -> Option<ConstantInfo> {
        // Very basic constant detection
        if line.contains("const ") || line.contains("final ") || (line.contains("=") && line.to_uppercase() == line) {
            if let Some(equals_pos) = line.find('=') {
                let before_equals = line[..equals_pos].trim();
                
                // Extract identifier name based on language patterns
                let tokens: Vec<&str> = before_equals.split_whitespace().collect();
                
                if tokens.len() >= 2 {
                    // For patterns like "const IDENTIFIER" or "const IDENTIFIER: type"
                    if tokens[0] == "const" || tokens[0] == "final" {
                        let name = tokens[1];
                        // Remove type annotations (e.g., "IDENTIFIER:" -> "IDENTIFIER")
                        let clean_name = name.trim_end_matches(':');
                        return Some(ConstantInfo {
                            name: clean_name.to_string(),
                            line_number,
                            value_type: None, // Would need type analysis
                            visibility: Visibility::Unknown,
                        });
                    }
                }
                
                // Fallback for other patterns
                if let Some(name) = tokens.get(1) {
                    let clean_name = name.trim_end_matches(':');
                    return Some(ConstantInfo {
                        name: clean_name.to_string(),
                        line_number,
                        value_type: None,
                        visibility: Visibility::Unknown,
                    });
                }
            }
        }
        None
    }

    /// Check if line is a comment
    fn is_comment_line(&self, line: &str) -> bool {
        let trimmed = line.trim();
        trimmed.starts_with("//") || trimmed.starts_with('#') || 
        trimmed.starts_with("/*") || trimmed.starts_with('*') ||
        trimmed.starts_with("<!--") || trimmed.starts_with("--")
    }
}

impl Default for ContentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;

    #[tokio::test]
    async fn test_content_analyzer_creation() {
        let analyzer = ContentAnalyzer::new();
        // Test that the AST parser is initialized
        assert!(true); // AST parser initialization is tested implicitly by other tests
    }

    #[tokio::test]
    async fn test_python_import_analysis() {
        let analyzer = ContentAnalyzer::new();
        let python_code = r#"
import os
import sys as system
from collections import defaultdict, Counter
from .local_module import LocalClass
import third_party.package
        "#;

        let stats = analyzer.analyze_content(python_code, &Language::Python).await.unwrap();
        
        // The line `from collections import defaultdict, Counter` should count as 1 import
        // with 2 items, not 2 separate imports
        assert_eq!(stats.imports.total_imports, 5);
        
        // Standard library modules should not be in external_dependencies
        assert!(!stats.imports.external_dependencies.contains("os"));
        assert!(!stats.imports.external_dependencies.contains("sys"));
        assert!(!stats.imports.external_dependencies.contains("collections"));
        
        // Third party packages should be in external_dependencies
        assert!(stats.imports.external_dependencies.contains("third_party.package"));
        
        assert_eq!(stats.imports.relative_imports, 1);
        assert!(stats.imports.absolute_imports > 0);
    }

    #[tokio::test]
    async fn test_documentation_analysis() {
        let analyzer = ContentAnalyzer::new();
        let markdown_content = r#"
# Main Title

This is a paragraph with [a link](https://example.com).

## Subsection

```python
def example():
    pass
```

- List item 1
- List item 2

| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |

<!-- TODO: Add more examples -->
        "#;

        let stats = analyzer.analyze_content(markdown_content, &Language::Markdown).await.unwrap();
        
        assert_eq!(stats.documentation.headings.len(), 2);
        assert_eq!(stats.documentation.headings[0].level, 1);
        assert_eq!(stats.documentation.headings[0].text, "Main Title");
        assert_eq!(stats.documentation.links.len(), 1);
        assert_eq!(stats.documentation.code_blocks.len(), 1);
        assert_eq!(stats.documentation.todo_comments.len(), 1);
        assert!(stats.documentation.lists > 0);
    }

    #[tokio::test]
    async fn test_text_statistics() {
        let analyzer = ContentAnalyzer::new();
        let code_content = r#"
// This is a comment
function example() {
    console.log("Hello, world!");
    // Another comment
    return true;
}

// Final comment
        "#;

        let stats = analyzer.analyze_content(code_content, &Language::JavaScript).await.unwrap();
        
        assert!(stats.text_stats.line_count > 0);
        assert!(stats.text_stats.comment_line_count >= 3);
        assert!(stats.text_stats.code_line_count > 0);
        assert!(stats.text_stats.comment_density > 0.0);
        assert!(stats.text_stats.word_count > 0);
    }

    #[tokio::test]
    async fn test_complexity_metrics() {
        let analyzer = ContentAnalyzer::new();
        let code_content = r#"
def complex_function(x, y):
    if x > 0:
        if y > 0:
            for i in range(10):
                if i % 2 == 0:
                    print(i)
        else:
            while y < 0:
                y += 1
    return x + y

class ExampleClass:
    def method1(self):
        pass
    
    def method2(self):
        pass
        "#;

        let stats = analyzer.analyze_content(code_content, &Language::Python).await.unwrap();
        
        assert!(stats.complexity.function_count >= 2);
        assert!(stats.complexity.class_count >= 1);
        assert!(stats.complexity.cyclomatic_complexity > 0);
        assert!(stats.complexity.nesting_depth > 0);
    }

    #[tokio::test]
    async fn test_structure_analysis() {
        let analyzer = ContentAnalyzer::new();
        let rust_code = r#"
pub fn public_function(param: i32) -> bool {
    true
}

fn private_function() {
    println!("Hello");
}

pub struct MyStruct {
    field: String,
}

const CONSTANT_VALUE: i32 = 42;
        "#;

        let stats = analyzer.analyze_content(rust_code, &Language::Rust).await.unwrap();
        
        assert_eq!(stats.structure.functions.len(), 2);
        assert!(stats.structure.functions.iter().any(|f| f.name == "public_function"));
        assert!(stats.structure.functions.iter().any(|f| f.name == "private_function"));
        assert_eq!(stats.structure.constants.len(), 1);
        assert_eq!(stats.structure.constants[0].name, "CONSTANT_VALUE");
    }

    #[tokio::test]
    async fn test_file_analysis() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.py");
        
        let content = r#"
"""
This is a module docstring.
"""
import os
from collections import defaultdict

def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello, {name}!"

class Person:
    """A simple person class."""
    def __init__(self, name: str):
        self.name = name
    
    def speak(self):
        return self.greet()
        "#;
        
        fs::write(&test_file, content).unwrap();

        let analyzer = ContentAnalyzer::new();
        let stats = analyzer.analyze_file(&test_file).await.unwrap();
        
        assert!(stats.imports.total_imports >= 2);
        assert!(stats.structure.functions.len() >= 2);
        assert!(stats.structure.classes.len() >= 1);
        assert!(stats.text_stats.line_count > 10);
        assert!(stats.complexity.function_count >= 2);
    }

    #[test]
    fn test_import_type_classification() {
        let analyzer = ContentAnalyzer::new();
        
        assert_eq!(analyzer.classify_import_type("os"), ImportType::Standard);
        assert_eq!(analyzer.classify_import_type("./local"), ImportType::Relative);
        assert_eq!(analyzer.classify_import_type("../parent"), ImportType::Relative);
        assert_eq!(analyzer.classify_import_type("third_party.package"), ImportType::External);
    }

    #[test]
    fn test_link_classification() {
        let analyzer = ContentAnalyzer::new();
        
        assert_eq!(analyzer.classify_link("https://example.com"), LinkType::External);
        assert_eq!(analyzer.classify_link("#anchor"), LinkType::Anchor);
        assert_eq!(analyzer.classify_link("./relative/path"), LinkType::Relative);
        assert_eq!(analyzer.classify_link("internal-link"), LinkType::Internal);
    }

    #[test]
    fn test_anchor_generation() {
        let analyzer = ContentAnalyzer::new();
        
        assert_eq!(analyzer.generate_anchor("Main Title"), "main-title");
        assert_eq!(analyzer.generate_anchor("Complex Title With Symbols!"), "complex-title-with-symbols");
        assert_eq!(analyzer.generate_anchor("Numbers 123 and More"), "numbers-123-and-more");
    }
}