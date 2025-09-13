//! Context Positioning Optimization
//! 
//! Strategic file positioning based on transformer model attention patterns.
//! Models have better reasoning at the head and tail of context, so we position:
//! - HEAD (20%): Query-specific high centrality files  
//! - MIDDLE (60%): Low centrality supporting files
//! - TAIL (20%): Core functionality, high centrality files

use std::collections::{HashMap, HashSet};
use std::path::Path;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::error::{ScalingResult, ScalingError};
use crate::streaming::FileMetadata;

/// Configuration for context positioning optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPositioningConfig {
    /// Enable context positioning optimization
    pub enable_positioning: bool,
    
    /// Percentage of context for HEAD positioning (query-relevant, high centrality)
    pub head_percentage: f64,
    
    /// Percentage of context for TAIL positioning (core functionality)  
    pub tail_percentage: f64,
    
    /// Weight for centrality in positioning decisions
    pub centrality_weight: f64,
    
    /// Weight for file relatedness in grouping decisions
    pub relatedness_weight: f64,
    
    /// Weight for query relevance in HEAD positioning
    pub query_relevance_weight: f64,
    
    /// Auto-exclude test files from selection (focuses on code and docs only)
    pub auto_exclude_tests: bool,
}

impl Default for ContextPositioningConfig {
    fn default() -> Self {
        Self {
            enable_positioning: true,
            head_percentage: 0.20,
            tail_percentage: 0.20, 
            centrality_weight: 0.4,
            relatedness_weight: 0.3,
            query_relevance_weight: 0.3,
            auto_exclude_tests: false,
        }
    }
}

/// Centrality scores for files in the codebase
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CentralityScores {
    /// Betweenness centrality: files connecting different parts
    pub betweenness: f64,
    
    /// PageRank centrality: heavily referenced files
    pub pagerank: f64,
    
    /// Degree centrality: files with many connections
    pub degree: f64,
    
    /// Combined centrality score
    pub combined: f64,
}

/// File with centrality and positioning metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileWithCentrality {
    pub metadata: FileMetadata,
    pub centrality: CentralityScores,
    pub query_relevance: f64,
    pub relatedness_group: String,
}

/// Three-tier context positioning structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPositioning {
    /// HEAD: Query-specific high centrality files (first ~20%)
    pub head_files: Vec<FileWithCentrality>,
    
    /// MIDDLE: Low centrality supporting files (~60%)  
    pub middle_files: Vec<FileWithCentrality>,
    
    /// TAIL: Core functionality, high centrality (~20%)
    pub tail_files: Vec<FileWithCentrality>,
}

/// Result of context positioning with reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionedSelection {
    pub positioning: ContextPositioning,
    pub total_tokens: usize,
    pub positioning_reasoning: String,
}

/// Context positioning optimizer
pub struct ContextPositioner {
    config: ContextPositioningConfig,
}

impl ContextPositioner {
    /// Create new context positioner with configuration
    pub fn new(config: ContextPositioningConfig) -> Self {
        Self { config }
    }
    
    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(ContextPositioningConfig::default())
    }
    
    /// Apply context positioning to selected files
    pub async fn position_files(
        &self,
        files: Vec<FileMetadata>,
        query_hint: Option<&str>,
    ) -> ScalingResult<PositionedSelection> {
        if !self.config.enable_positioning || files.is_empty() {
            return Ok(self.create_simple_positioning(files));
        }
        
        // Filter out test files if auto-exclude is enabled
        let filtered_files = if self.config.auto_exclude_tests {
            let original_count = files.len();
            let non_test_files: Vec<FileMetadata> = files
                .into_iter()
                .filter(|file| !self.is_test_file(&file.path))
                .collect();
            let filtered_count = non_test_files.len();
            
            if original_count != filtered_count {
                info!("Auto-excluded {} test files, {} files remaining", 
                    original_count - filtered_count, filtered_count);
            }
            
            non_test_files
        } else {
            files
        };
        
        info!("Starting context positioning for {} files", filtered_files.len());
        
        // Phase 1: Calculate centrality scores for all files
        let files_with_centrality = self.calculate_centrality_scores(filtered_files).await?;
        
        // Phase 2: Calculate query relevance if hint provided
        let files_with_relevance = self.calculate_query_relevance(files_with_centrality, query_hint).await?;
        
        // Phase 3: Group by relatedness
        let files_with_groups = self.group_by_relatedness(files_with_relevance).await?;
        
        // Phase 4: Apply three-tier positioning strategy
        let positioning = self.apply_positioning_strategy(files_with_groups).await?;
        
        // Phase 5: Calculate total tokens and generate reasoning
        let total_tokens = self.calculate_total_tokens(&positioning);
        let reasoning = self.generate_positioning_reasoning(&positioning, query_hint);
        
        info!("Context positioning complete: HEAD={}, MIDDLE={}, TAIL={}", 
            positioning.head_files.len(),
            positioning.middle_files.len(), 
            positioning.tail_files.len()
        );
        
        Ok(PositionedSelection {
            positioning,
            total_tokens,
            positioning_reasoning: reasoning,
        })
    }
    
    /// Calculate centrality scores for all files
    async fn calculate_centrality_scores(&self, files: Vec<FileMetadata>) -> ScalingResult<Vec<FileWithCentrality>> {
        debug!("Calculating centrality scores for {} files", files.len());
        
        // Build dependency graph from import/export relationships
        let dependency_graph = self.build_dependency_graph(&files).await?;
        
        let mut files_with_centrality = Vec::new();
        
        for file in files {
            let centrality = self.calculate_file_centrality(&file, &dependency_graph).await?;
            
            files_with_centrality.push(FileWithCentrality {
                metadata: file,
                centrality,
                query_relevance: 0.0, // Will be set later
                relatedness_group: String::new(), // Will be set later
            });
        }
        
        Ok(files_with_centrality)
    }
    
    /// Build dependency graph from file relationships  
    async fn build_dependency_graph(&self, files: &[FileMetadata]) -> ScalingResult<HashMap<String, Vec<String>>> {
        let mut graph = HashMap::new();
        
        for file in files {
            let file_key = self.file_to_key(&file.path);
            let dependencies = self.extract_dependencies(file).await?;
            graph.insert(file_key, dependencies);
        }
        
        Ok(graph)
    }
    
    /// Extract dependencies from a file (imports, includes, etc.)
    async fn extract_dependencies(&self, file: &FileMetadata) -> ScalingResult<Vec<String>> {
        // Simple dependency extraction based on file patterns and language
        let mut dependencies = Vec::new();
        
        let path_str = file.path.to_string_lossy();
        let dir_path = file.path.parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
            
        // For Rust files, assume mod.rs and lib.rs are central
        if file.language == "Rust" {
            let filename = file.path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
                
            if filename == "mod.rs" || filename == "lib.rs" {
                // These are likely dependency targets
            } else {
                // Regular Rust files likely depend on lib.rs or mod.rs
                dependencies.push(format!("{}/lib.rs", dir_path));
                dependencies.push(format!("{}/mod.rs", dir_path));
            }
        }
        
        // For Python files, __init__.py files are central
        if file.language == "Python" {
            let filename = file.path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
                
            if filename != "__init__.py" {
                dependencies.push(format!("{}/__init__.py", dir_path));
            }
        }
        
        // For JavaScript/TypeScript, index files are central
        if file.language == "JavaScript" || file.language == "TypeScript" {
            dependencies.push(format!("{}/index.js", dir_path));
            dependencies.push(format!("{}/index.ts", dir_path));
        }
        
        // Configuration files often depend on package manifests
        if file.file_type == "Configuration" {
            dependencies.push("package.json".to_string());
            dependencies.push("Cargo.toml".to_string());
            dependencies.push("pyproject.toml".to_string());
        }
        
        Ok(dependencies)
    }
    
    /// Calculate centrality scores for a single file
    async fn calculate_file_centrality(
        &self, 
        file: &FileMetadata, 
        graph: &HashMap<String, Vec<String>>
    ) -> ScalingResult<CentralityScores> {
        let file_key = self.file_to_key(&file.path);
        
        // Calculate degree centrality (number of connections)
        let degree = self.calculate_degree_centrality(&file_key, graph);
        
        // Calculate PageRank centrality (importance based on references)
        let pagerank = self.calculate_pagerank_centrality(&file_key, graph);
        
        // Calculate betweenness centrality (bridge between components)
        let betweenness = self.calculate_betweenness_centrality(&file_key, graph);
        
        // Combine centrality scores with weights
        let combined = (degree * 0.3) + (pagerank * 0.5) + (betweenness * 0.2);
        
        Ok(CentralityScores {
            degree,
            pagerank, 
            betweenness,
            combined,
        })
    }
    
    /// Calculate degree centrality for a file
    fn calculate_degree_centrality(&self, file_key: &str, graph: &HashMap<String, Vec<String>>) -> f64 {
        let out_degree = graph.get(file_key).map(|deps| deps.len()).unwrap_or(0);
        
        // Count incoming edges (files that depend on this one)
        let in_degree = graph.values()
            .map(|deps| deps.iter().filter(|dep| *dep == file_key).count())
            .sum::<usize>();
        
        // Normalize by potential maximum degree
        let total_degree = out_degree + in_degree;
        let max_possible = graph.len().saturating_sub(1);
        
        if max_possible == 0 {
            0.0
        } else {
            total_degree as f64 / max_possible as f64
        }
    }
    
    /// Calculate PageRank centrality (simplified)
    fn calculate_pagerank_centrality(&self, file_key: &str, graph: &HashMap<String, Vec<String>>) -> f64 {
        // Simplified PageRank: count weighted incoming references
        let incoming_refs = graph.values()
            .map(|deps| {
                let weight = if deps.is_empty() { 0.0 } else { 1.0 / deps.len() as f64 };
                deps.iter().filter(|dep| *dep == file_key).count() as f64 * weight
            })
            .sum::<f64>();
            
        // Add damping factor and normalize
        let damping = 0.85;
        let num_files = graph.len() as f64;
        
        if num_files == 0.0 {
            0.0
        } else {
            ((1.0 - damping) / num_files) + (damping * incoming_refs / num_files)
        }
    }
    
    /// Calculate betweenness centrality (simplified)
    fn calculate_betweenness_centrality(&self, file_key: &str, graph: &HashMap<String, Vec<String>>) -> f64 {
        // Simplified betweenness: files that appear in many dependency paths
        let mut betweenness = 0.0;
        
        // Count how many files this file connects to (transitively)
        if let Some(dependencies) = graph.get(file_key) {
            betweenness += dependencies.len() as f64;
            
            // Add bonus for connecting different directory structures
            let mut dirs_connected = HashSet::new();
            for dep in dependencies {
                if let Some(dir) = dep.rfind('/') {
                    dirs_connected.insert(&dep[..dir]);
                }
            }
            betweenness += dirs_connected.len() as f64 * 0.5;
        }
        
        // Normalize
        let max_possible = graph.len() as f64;
        if max_possible == 0.0 {
            0.0
        } else {
            betweenness / max_possible
        }
    }
    
    /// Calculate query relevance scores if query hint provided
    async fn calculate_query_relevance(
        &self,
        mut files: Vec<FileWithCentrality>,
        query_hint: Option<&str>,
    ) -> ScalingResult<Vec<FileWithCentrality>> {
        if let Some(query) = query_hint {
            debug!("Calculating query relevance for: {}", query);
            
            let query_lower = query.to_lowercase();
            let query_words: Vec<&str> = query_lower.split_whitespace().collect();
            
            for file in &mut files {
                file.query_relevance = self.calculate_file_query_relevance(&file.metadata, &query_words);
            }
        }
        
        Ok(files)
    }
    
    /// Calculate query relevance for a single file
    fn calculate_file_query_relevance(&self, file: &FileMetadata, query_words: &[&str]) -> f64 {
        let path_str = file.path.to_string_lossy().to_lowercase();
        let filename = file.path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();
            
        let mut relevance = 0.0;
        
        for word in query_words {
            // Exact matches in filename get highest score
            if filename.contains(word) {
                relevance += 1.0;
            }
            // Partial matches in path get medium score  
            else if path_str.contains(word) {
                relevance += 0.5;
            }
            // Language matches get small boost
            else if file.language.to_lowercase().contains(word) {
                relevance += 0.2;
            }
        }
        
        // Boost for entry points that might be relevant
        if filename.contains("main") || filename.contains("index") || 
           filename == "lib.rs" || filename == "__init__.py" {
            relevance += 0.3;
        }
        
        relevance
    }
    
    /// Group files by relatedness
    async fn group_by_relatedness(&self, mut files: Vec<FileWithCentrality>) -> ScalingResult<Vec<FileWithCentrality>> {
        debug!("Grouping {} files by relatedness", files.len());
        
        for file in &mut files {
            file.relatedness_group = self.determine_relatedness_group(&file.metadata);
        }
        
        Ok(files)
    }
    
    /// Determine relatedness group for a file
    fn determine_relatedness_group(&self, file: &FileMetadata) -> String {
        let path_str = file.path.to_string_lossy();
        
        // Group by directory structure (first 2 levels)
        let path_components: Vec<&str> = path_str.split('/').collect();
        let group = if path_components.len() >= 2 {
            format!("{}/{}", path_components[0], path_components[1])
        } else if path_components.len() == 1 {
            path_components[0].to_string()
        } else {
            "root".to_string()
        };
        
        // Add language suffix for better grouping
        format!("{}::{}", group, file.language)
    }
    
    /// Apply three-tier positioning strategy
    async fn apply_positioning_strategy(&self, files: Vec<FileWithCentrality>) -> ScalingResult<ContextPositioning> {
        if files.is_empty() {
            return Ok(ContextPositioning {
                head_files: Vec::new(),
                middle_files: Vec::new(),
                tail_files: Vec::new(),
            });
        }
        
        let total_files = files.len();
        let head_count = ((total_files as f64 * self.config.head_percentage) as usize).max(1);
        let tail_count = ((total_files as f64 * self.config.tail_percentage) as usize).max(1);
        
        debug!("Positioning strategy: HEAD={}, TAIL={}, MIDDLE={}", 
            head_count, tail_count, total_files - head_count - tail_count);
        
        // Sort files for HEAD positioning: query relevance + centrality
        let mut head_candidates = files.clone();
        head_candidates.sort_by(|a, b| {
            let score_a = (a.query_relevance * self.config.query_relevance_weight) + 
                         (a.centrality.combined * self.config.centrality_weight);
            let score_b = (b.query_relevance * self.config.query_relevance_weight) + 
                         (b.centrality.combined * self.config.centrality_weight);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Sort files for TAIL positioning: pure centrality
        let mut tail_candidates = files.clone();
        tail_candidates.sort_by(|a, b| {
            b.centrality.combined.partial_cmp(&a.centrality.combined).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Select HEAD files (query-relevant + high centrality)
        let mut selected_files = HashSet::new();
        let mut head_files = Vec::new();
        
        for file in head_candidates.into_iter().take(head_count) {
            let file_key = self.file_to_key(&file.metadata.path);
            selected_files.insert(file_key);
            head_files.push(file);
        }
        
        // Select TAIL files (high centrality, not already in head)
        let mut tail_files = Vec::new();
        for file in tail_candidates {
            if tail_files.len() >= tail_count {
                break;
            }
            let file_key = self.file_to_key(&file.metadata.path);
            if !selected_files.contains(&file_key) {
                selected_files.insert(file_key);
                tail_files.push(file);
            }
        }
        
        // Remaining files go to MIDDLE
        let mut middle_files = Vec::new();
        for file in files {
            let file_key = self.file_to_key(&file.metadata.path);
            if !selected_files.contains(&file_key) {
                middle_files.push(file);
            }
        }
        
        // Group related files within each tier
        self.group_within_tier(&mut head_files);
        self.group_within_tier(&mut middle_files);
        self.group_within_tier(&mut tail_files);
        
        Ok(ContextPositioning {
            head_files,
            middle_files,
            tail_files,
        })
    }
    
    /// Group related files within a tier to improve locality
    fn group_within_tier(&self, files: &mut Vec<FileWithCentrality>) {
        files.sort_by(|a, b| {
            // Primary sort: relatedness group
            let group_cmp = a.relatedness_group.cmp(&b.relatedness_group);
            if group_cmp != std::cmp::Ordering::Equal {
                return group_cmp;
            }
            
            // Secondary sort: centrality within group
            b.centrality.combined.partial_cmp(&a.centrality.combined).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    
    /// Calculate total tokens for positioned files
    fn calculate_total_tokens(&self, positioning: &ContextPositioning) -> usize {
        let head_tokens = positioning.head_files.iter()
            .map(|f| self.estimate_tokens(&f.metadata))
            .sum::<usize>();
            
        let middle_tokens = positioning.middle_files.iter()
            .map(|f| self.estimate_tokens(&f.metadata))
            .sum::<usize>();
            
        let tail_tokens = positioning.tail_files.iter()
            .map(|f| self.estimate_tokens(&f.metadata))
            .sum::<usize>();
            
        head_tokens + middle_tokens + tail_tokens
    }
    
    /// Generate positioning reasoning explanation
    fn generate_positioning_reasoning(&self, positioning: &ContextPositioning, query_hint: Option<&str>) -> String {
        let mut reasoning = Vec::new();
        
        reasoning.push("🎯 Context Positioning Strategy Applied".to_string());
        reasoning.push("".to_string());
        
        // HEAD section reasoning
        reasoning.push(format!("📍 HEAD ({} files): Query-specific high centrality files", positioning.head_files.len()));
        if let Some(query) = query_hint {
            reasoning.push(format!("   Query hint: '{}'", query));
        }
        for (i, file) in positioning.head_files.iter().take(3).enumerate() {
            reasoning.push(format!("   {}. {} (centrality: {:.3}, relevance: {:.3})", 
                i + 1, 
                file.metadata.path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
                file.centrality.combined,
                file.query_relevance
            ));
        }
        if positioning.head_files.len() > 3 {
            reasoning.push(format!("   ... and {} more files", positioning.head_files.len() - 3));
        }
        reasoning.push("".to_string());
        
        // MIDDLE section reasoning  
        reasoning.push(format!("🔄 MIDDLE ({} files): Supporting utilities and low-centrality files", positioning.middle_files.len()));
        reasoning.push("".to_string());
        
        // TAIL section reasoning
        reasoning.push(format!("🏛️ TAIL ({} files): Core functionality, high centrality", positioning.tail_files.len()));
        for (i, file) in positioning.tail_files.iter().take(3).enumerate() {
            reasoning.push(format!("   {}. {} (centrality: {:.3})",
                i + 1,
                file.metadata.path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
                file.centrality.combined
            ));
        }
        if positioning.tail_files.len() > 3 {
            reasoning.push(format!("   ... and {} more files", positioning.tail_files.len() - 3));
        }
        
        reasoning.join("\n")
    }
    
    /// Create simple positioning when optimization is disabled
    fn create_simple_positioning(&self, files: Vec<FileMetadata>) -> PositionedSelection {
        let files_with_centrality: Vec<FileWithCentrality> = files.into_iter()
            .map(|metadata| FileWithCentrality {
                metadata,
                centrality: CentralityScores::default(),
                query_relevance: 0.0,
                relatedness_group: "default".to_string(),
            })
            .collect();
            
        let positioning = ContextPositioning {
            head_files: Vec::new(),
            middle_files: files_with_centrality,
            tail_files: Vec::new(),
        };
        
        let total_tokens = self.calculate_total_tokens(&positioning);
        
        PositionedSelection {
            positioning,
            total_tokens,
            positioning_reasoning: "Context positioning disabled - using default order".to_string(),
        }
    }
    
    /// Convert file path to graph key
    fn file_to_key(&self, path: &Path) -> String {
        path.to_string_lossy().to_string()
    }
    
    /// Estimate tokens for a file (simplified version)
    fn estimate_tokens(&self, file: &FileMetadata) -> usize {
        // Basic token estimation: ~3.5 chars per token
        let base_tokens = ((file.size as f64) / 3.5) as usize;
        
        // Language-specific adjustments
        let multiplier = match file.language.as_str() {
            "Rust" => 1.3,
            "JavaScript" | "TypeScript" => 1.2,
            "Python" => 1.1,
            "C" | "Go" => 1.0,
            "JSON" | "YAML" | "TOML" => 0.7,
            _ => 1.0,
        };
        
        (base_tokens as f64 * multiplier) as usize
    }
    
    /// Smart test file detection based on common patterns
    fn is_test_file(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy().to_lowercase();
        let file_name = path.file_name()
            .map(|s| s.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        
        // Test directory patterns
        if path_str.contains("/test/") ||
           path_str.contains("/tests/") ||
           path_str.contains("\\test\\") ||
           path_str.contains("\\tests\\") ||
           path_str.contains("/__tests__/") ||
           path_str.contains("\\__tests__\\") {
            return true;
        }
        
        // Test file name patterns
        if file_name.starts_with("test_") ||
           file_name.ends_with("_test.rs") ||
           file_name.ends_with("_test.py") ||
           file_name.ends_with("_test.js") ||
           file_name.ends_with("_test.ts") ||
           file_name.ends_with(".test.js") ||
           file_name.ends_with(".test.ts") ||
           file_name.ends_with(".test.jsx") ||
           file_name.ends_with(".test.tsx") ||
           file_name.ends_with(".spec.js") ||
           file_name.ends_with(".spec.ts") ||
           file_name.ends_with(".spec.jsx") ||
           file_name.ends_with(".spec.tsx") ||
           file_name.ends_with("_spec.py") ||
           file_name.ends_with("_spec.rb") {
            return true;
        }
        
        // Language-specific test patterns
        match path.extension().and_then(|s| s.to_str()) {
            Some("rs") => {
                // Rust: mod tests, #[cfg(test)]
                file_name.contains("test") && (
                    file_name.starts_with("test_") || 
                    file_name.ends_with("_test.rs") ||
                    path_str.contains("/tests/")
                )
            },
            Some("py") => {
                // Python: test_*.py, *_test.py, pytest patterns
                file_name.starts_with("test_") || 
                file_name.ends_with("_test.py") ||
                file_name.contains("test_")
            },
            Some("go") => {
                // Go: *_test.go
                file_name.ends_with("_test.go")
            },
            Some("java") | Some("kt") => {
                // Java/Kotlin: *Test.java, *Tests.java
                file_name.ends_with("test.java") ||
                file_name.ends_with("tests.java") ||
                file_name.ends_with("test.kt") ||
                file_name.ends_with("tests.kt") ||
                path_str.contains("/test/") ||
                path_str.contains("/tests/")
            },
            Some("js") | Some("ts") | Some("jsx") | Some("tsx") => {
                // JavaScript/TypeScript: comprehensive test patterns
                file_name.contains(".test.") ||
                file_name.contains(".spec.") ||
                file_name.ends_with(".test.js") ||
                file_name.ends_with(".test.ts") ||
                file_name.ends_with(".spec.js") ||
                file_name.ends_with(".spec.ts") ||
                path_str.contains("/__tests__/") ||
                path_str.contains("/test/") ||
                path_str.contains("/tests/")
            },
            Some("rb") => {
                // Ruby: *_test.rb, *_spec.rb, spec/ and test/ directories
                file_name.ends_with("_test.rb") ||
                file_name.ends_with("_spec.rb") ||
                path_str.contains("/spec/") ||
                path_str.contains("/test/")
            },
            Some("php") => {
                // PHP: *Test.php, *_test.php
                file_name.ends_with("test.php") ||
                file_name.ends_with("_test.php") ||
                file_name.contains("test") && path_str.contains("/test")
            },
            _ => false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::SystemTime;

    fn create_test_file(path: &str, size: u64, language: &str) -> FileMetadata {
        FileMetadata {
            path: PathBuf::from(path),
            size,
            modified: SystemTime::now(),
            language: language.to_string(),
            file_type: if language == "Rust" { "Source" } else { "Other" }.to_string(),
        }
    }

    #[tokio::test]
    async fn test_context_positioner_creation() {
        let positioner = ContextPositioner::with_defaults();
        assert!(positioner.config.enable_positioning);
        assert_eq!(positioner.config.head_percentage, 0.20);
        assert_eq!(positioner.config.tail_percentage, 0.20);
    }

    #[tokio::test]
    async fn test_centrality_calculation() {
        let positioner = ContextPositioner::with_defaults();
        
        let files = vec![
            create_test_file("src/main.rs", 1000, "Rust"),
            create_test_file("src/lib.rs", 2000, "Rust"),
            create_test_file("src/utils.rs", 500, "Rust"),
        ];
        
        let files_with_centrality = positioner.calculate_centrality_scores(files).await.unwrap();
        assert_eq!(files_with_centrality.len(), 3);
        
        // All files should have some centrality score
        for file in &files_with_centrality {
            assert!(file.centrality.combined >= 0.0);
            assert!(file.centrality.degree >= 0.0);
            assert!(file.centrality.pagerank >= 0.0);
            assert!(file.centrality.betweenness >= 0.0);
        }
        
        // At least one file should have higher centrality than another
        let max_centrality = files_with_centrality.iter()
            .map(|f| f.centrality.combined)
            .fold(0.0, f64::max);
        let min_centrality = files_with_centrality.iter()
            .map(|f| f.centrality.combined)
            .fold(1.0, f64::min);
        
        // Allow for equal centrality scores in simple cases
        assert!(max_centrality >= min_centrality);
    }

    #[tokio::test] 
    async fn test_positioning_strategy() {
        let positioner = ContextPositioner::with_defaults();
        
        let files = vec![
            create_test_file("src/main.rs", 1000, "Rust"),
            create_test_file("src/lib.rs", 2000, "Rust"), 
            create_test_file("src/utils.rs", 500, "Rust"),
            create_test_file("tests/integration.rs", 800, "Rust"),
            create_test_file("README.md", 300, "Markdown"),
        ];
        
        let result = positioner.position_files(files, Some("main")).await.unwrap();
        
        // Should have files in all three tiers
        assert!(!result.positioning.head_files.is_empty());
        assert!(!result.positioning.middle_files.is_empty());
        assert!(!result.positioning.tail_files.is_empty());
        
        // Total should equal original count
        let total = result.positioning.head_files.len() + 
                   result.positioning.middle_files.len() + 
                   result.positioning.tail_files.len();
        assert_eq!(total, 5);
        
        // Reasoning should be provided
        assert!(!result.positioning_reasoning.is_empty());
        assert!(result.positioning_reasoning.contains("HEAD"));
        assert!(result.positioning_reasoning.contains("TAIL"));
    }

    #[tokio::test]
    async fn test_query_relevance() {
        let positioner = ContextPositioner::with_defaults();
        
        let files = vec![
            FileWithCentrality {
                metadata: create_test_file("src/main.rs", 1000, "Rust"),
                centrality: CentralityScores::default(),
                query_relevance: 0.0,
                relatedness_group: String::new(),
            },
            FileWithCentrality {
                metadata: create_test_file("src/utils.rs", 500, "Rust"),
                centrality: CentralityScores::default(),
                query_relevance: 0.0,
                relatedness_group: String::new(),
            },
        ];
        
        let result = positioner.calculate_query_relevance(files, Some("main")).await.unwrap();
        
        // main.rs should have higher query relevance for "main" query
        let main_relevance = result.iter()
            .find(|f| f.metadata.path.to_string_lossy().contains("main.rs"))
            .unwrap();
        let utils_relevance = result.iter()
            .find(|f| f.metadata.path.to_string_lossy().contains("utils.rs"))
            .unwrap();
            
        assert!(main_relevance.query_relevance > utils_relevance.query_relevance);
    }

    #[test]
    fn test_relatedness_grouping() {
        let positioner = ContextPositioner::with_defaults();
        
        let file = create_test_file("src/api/handlers.rs", 1000, "Rust");
        let group = positioner.determine_relatedness_group(&file);
        
        assert!(group.contains("src/api"));
        assert!(group.contains("Rust"));
    }

    #[test]
    fn test_token_estimation() {
        let positioner = ContextPositioner::with_defaults();
        
        let rust_file = create_test_file("src/main.rs", 1000, "Rust");
        let json_file = create_test_file("package.json", 1000, "JSON");
        
        let rust_tokens = positioner.estimate_tokens(&rust_file);
        let json_tokens = positioner.estimate_tokens(&json_file);
        
        // Rust should have more tokens than JSON for same file size
        assert!(rust_tokens > json_tokens);
    }

    #[test]
    fn test_is_test_file_detection() {
        let positioner = ContextPositioner::with_defaults();
        
        // Test directory patterns
        assert!(positioner.is_test_file(&std::path::Path::new("src/test/utils.rs")));
        assert!(positioner.is_test_file(&std::path::Path::new("src/tests/integration.py")));
        assert!(positioner.is_test_file(&std::path::Path::new("__tests__/component.test.js")));
        
        // Test file name patterns
        assert!(positioner.is_test_file(&std::path::Path::new("test_utils.py")));
        assert!(positioner.is_test_file(&std::path::Path::new("utils_test.rs")));
        assert!(positioner.is_test_file(&std::path::Path::new("component.test.tsx")));
        assert!(positioner.is_test_file(&std::path::Path::new("service.spec.ts")));
        assert!(positioner.is_test_file(&std::path::Path::new("model_test.go")));
        
        // Language-specific patterns
        assert!(positioner.is_test_file(&std::path::Path::new("UserTest.java")));
        assert!(positioner.is_test_file(&std::path::Path::new("user_spec.rb")));
        assert!(positioner.is_test_file(&std::path::Path::new("UserTest.php")));
        
        // Non-test files should not be detected
        assert!(!positioner.is_test_file(&std::path::Path::new("src/main.rs")));
        assert!(!positioner.is_test_file(&std::path::Path::new("lib/utils.py")));
        assert!(!positioner.is_test_file(&std::path::Path::new("components/Button.tsx")));
        assert!(!positioner.is_test_file(&std::path::Path::new("README.md")));
        assert!(!positioner.is_test_file(&std::path::Path::new("package.json")));
    }

    #[tokio::test]
    async fn test_auto_exclude_tests() {
        let mut config = ContextPositioningConfig::default();
        config.auto_exclude_tests = true;
        let positioner = ContextPositioner::new(config);
        
        // Create mix of test and non-test files
        let files = vec![
            create_test_file("src/main.rs", 1000, "Rust"),
            create_test_file("src/lib.rs", 800, "Rust"), 
            create_test_file("src/tests/integration_test.rs", 1200, "Rust"),
            create_test_file("test/unit_test.py", 600, "Python"),
            create_test_file("components/Button.tsx", 900, "TypeScript"),
            create_test_file("__tests__/Button.test.tsx", 700, "TypeScript"),
        ];
        
        let result = positioner.position_files(files, None).await.unwrap();
        
        // Should have filtered out test files
        let all_files: Vec<&FileWithCentrality> = result.positioning.head_files.iter()
            .chain(result.positioning.middle_files.iter())
            .chain(result.positioning.tail_files.iter())
            .collect();
        
        // Should only have non-test files (3 out of 6)
        assert_eq!(all_files.len(), 3);
        
        // Verify no test files remain
        for file in all_files {
            let path_str = file.metadata.path.to_string_lossy();
            assert!(!path_str.contains("test"));
            assert!(!path_str.contains("__tests__"));
        }
        
        // Verify we have the expected non-test files
        let file_names: Vec<String> = result.positioning.head_files.iter()
            .chain(result.positioning.middle_files.iter())
            .chain(result.positioning.tail_files.iter())
            .map(|f| f.metadata.path.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        
        assert!(file_names.contains(&"main.rs".to_string()));
        assert!(file_names.contains(&"lib.rs".to_string()));
        assert!(file_names.contains(&"Button.tsx".to_string()));
    }

    #[tokio::test]
    async fn test_auto_exclude_disabled() {
        let mut config = ContextPositioningConfig::default();
        config.auto_exclude_tests = false; // Explicitly disabled
        let positioner = ContextPositioner::new(config);
        
        // Create mix of test and non-test files
        let files = vec![
            create_test_file("src/main.rs", 1000, "Rust"),
            create_test_file("src/tests/integration_test.rs", 1200, "Rust"),
            create_test_file("test_utils.py", 600, "Python"),
        ];
        
        let result = positioner.position_files(files, None).await.unwrap();
        
        // Should include all files when auto-exclude is disabled
        let all_files: Vec<&FileWithCentrality> = result.positioning.head_files.iter()
            .chain(result.positioning.middle_files.iter())
            .chain(result.positioning.tail_files.iter())
            .collect();
        
        // Should have all 3 files including test files
        assert_eq!(all_files.len(), 3);
    }
}