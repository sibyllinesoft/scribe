//! Context Positioning Optimization
//!
//! Strategic file positioning based on transformer model attention patterns.
//! Models have better reasoning at the head and tail of context, so we position:
//! - HEAD (20%): Query-specific high centrality files
//! - MIDDLE (60%): Low centrality supporting files
//! - TAIL (20%): Core functionality, high centrality files

#[cfg(test)]
mod tests;
mod types;

pub use types::{
    CentralityScores, ContextPositioner, ContextPositioning, ContextPositioningConfig,
    FileWithCentrality, PositionedSelection,
};

use petgraph::algo::kosaraju_scc;
use petgraph::visit::EdgeRef;
use petgraph::{graph::NodeIndex, Directed, Graph};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use tracing::{debug, info};

use crate::core::error::ScalingResult;
use crate::io::streaming::FileMetadata;
use scribe_core::file;

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
                info!(
                    "Auto-excluded {} test files, {} files remaining",
                    original_count - filtered_count,
                    filtered_count
                );
            }

            non_test_files
        } else {
            files
        };

        info!(
            "Starting context positioning for {} files",
            filtered_files.len()
        );

        // Phase 1: Calculate centrality scores for all files
        let files_with_centrality = self.calculate_centrality_scores(filtered_files).await?;

        // Phase 2: Calculate query relevance if hint provided
        let files_with_relevance = self
            .calculate_query_relevance(files_with_centrality, query_hint)
            .await?;

        // Phase 3: Group by relatedness
        let files_with_groups = self.group_by_relatedness(files_with_relevance).await?;

        // Phase 4: Apply three-tier positioning strategy
        let positioning = self.apply_positioning_strategy(files_with_groups).await?;

        // Phase 5: Calculate total tokens and generate reasoning
        let total_tokens = self.calculate_total_tokens(&positioning);
        let reasoning = self.generate_positioning_reasoning(&positioning, query_hint);

        info!(
            "Context positioning complete: HEAD={}, MIDDLE={}, TAIL={}",
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

    /// Calculate centrality scores for all files using optimized algorithms
    async fn calculate_centrality_scores(
        &self,
        files: Vec<FileMetadata>,
    ) -> ScalingResult<Vec<FileWithCentrality>> {
        debug!("Calculating centrality scores for {} files", files.len());

        if files.is_empty() {
            return Ok(Vec::new());
        }

        // Build optimized dependency graph
        let (graph, node_map) = self.build_dependency_graph(&files).await?;

        // Calculate all centrality measures efficiently using petgraph
        let centrality_scores = self.calculate_all_centralities(&graph, &node_map).await?;

        // Map centrality scores back to files in parallel
        let files_with_centrality: Vec<FileWithCentrality> = files
            .into_par_iter()
            .map(|file| {
                let file_key = self.file_to_key(&file.path);
                let centrality = centrality_scores
                    .get(&file_key)
                    .cloned()
                    .unwrap_or_default();

                FileWithCentrality {
                    metadata: file,
                    centrality,
                    query_relevance: 0.0,             // Will be set later
                    relatedness_group: String::new(), // Will be set later
                }
            })
            .collect();

        debug!(
            "Calculated centrality for {} files",
            files_with_centrality.len()
        );
        Ok(files_with_centrality)
    }

    /// Build dependency graph from file relationships using petgraph
    async fn build_dependency_graph(
        &self,
        files: &[FileMetadata],
    ) -> ScalingResult<(Graph<String, (), Directed>, HashMap<String, NodeIndex>)> {
        let mut graph = Graph::new();
        let mut node_map = HashMap::new();

        // First pass: create nodes for all files
        for file in files {
            let file_key = self.file_to_key(&file.path);
            let node_idx = graph.add_node(file_key.clone());
            node_map.insert(file_key, node_idx);
        }

        // Second pass: create edges based on dependencies
        for file in files {
            let file_key = self.file_to_key(&file.path);
            let dependencies = self.extract_dependencies(file).await?;

            if let Some(&from_idx) = node_map.get(&file_key) {
                for dep in dependencies {
                    if let Some(&to_idx) = node_map.get(&dep) {
                        graph.add_edge(from_idx, to_idx, ());
                    }
                }
            }
        }

        debug!(
            "Built dependency graph: {} nodes, {} edges",
            graph.node_count(),
            graph.edge_count()
        );

        Ok((graph, node_map))
    }

    /// Extract dependencies from a file (imports, includes, etc.)
    async fn extract_dependencies(&self, file: &FileMetadata) -> ScalingResult<Vec<String>> {
        // Simple dependency extraction based on file patterns and language
        let mut dependencies = Vec::new();

        let dir_path = file
            .path
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();

        // For Rust files, assume mod.rs and lib.rs are central
        if file.language == "Rust" {
            let filename = file.path.file_name().and_then(|n| n.to_str()).unwrap_or("");

            if filename != "mod.rs" && filename != "lib.rs" {
                // Regular Rust files likely depend on lib.rs or mod.rs
                dependencies.push(format!("{}/lib.rs", dir_path));
                dependencies.push(format!("{}/mod.rs", dir_path));
            }
        }

        // For Python files, __init__.py files are central
        if file.language == "Python" {
            let filename = file.path.file_name().and_then(|n| n.to_str()).unwrap_or("");

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

    /// Calculate all centrality measures efficiently using petgraph algorithms
    async fn calculate_all_centralities(
        &self,
        graph: &Graph<String, (), Directed>,
        node_map: &HashMap<String, NodeIndex>,
    ) -> ScalingResult<HashMap<String, CentralityScores>> {
        let mut centrality_scores = HashMap::new();

        if graph.node_count() == 0 {
            return Ok(centrality_scores);
        }

        // Calculate PageRank using simplified approach
        let pagerank_scores = self.calculate_simple_pagerank(graph, node_map)?;

        // Calculate degree centrality in parallel
        let degree_scores: Vec<(NodeIndex, f64)> = node_map
            .par_iter()
            .map(|(_, &node_idx)| {
                let in_degree = graph.edges_directed(node_idx, petgraph::Incoming).count();
                let out_degree = graph.edges_directed(node_idx, petgraph::Outgoing).count();
                let total_degree = in_degree + out_degree;
                let max_possible = graph.node_count().saturating_sub(1);

                let normalized_degree = if max_possible == 0 {
                    0.0
                } else {
                    total_degree as f64 / max_possible as f64
                };

                (node_idx, normalized_degree)
            })
            .collect();

        // Calculate betweenness centrality using strongly connected components
        let betweenness_scores = self.calculate_betweenness_from_scc(graph, node_map)?;

        // Combine all scores
        for (file_key, &node_idx) in node_map {
            let pagerank = pagerank_scores
                .get(node_idx.index())
                .copied()
                .unwrap_or(0.0);
            let degree = degree_scores
                .iter()
                .find(|(idx, _)| *idx == node_idx)
                .map(|(_, score)| *score)
                .unwrap_or(0.0);
            let betweenness = betweenness_scores.get(&node_idx).copied().unwrap_or(0.0);

            // Combine centrality scores with weights
            let combined = (degree * 0.3) + (pagerank * 0.5) + (betweenness * 0.2);

            centrality_scores.insert(
                file_key.clone(),
                CentralityScores {
                    degree,
                    pagerank,
                    betweenness,
                    combined,
                },
            );
        }

        debug!(
            "Calculated centrality scores for {} files",
            centrality_scores.len()
        );
        Ok(centrality_scores)
    }

    /// Calculate betweenness centrality using strongly connected components
    fn calculate_betweenness_from_scc(
        &self,
        graph: &Graph<String, (), Directed>,
        node_map: &HashMap<String, NodeIndex>,
    ) -> ScalingResult<HashMap<NodeIndex, f64>> {
        let mut betweenness_scores = HashMap::new();

        // Use Kosaraju's algorithm to find strongly connected components
        let sccs = kosaraju_scc(graph);

        // Calculate betweenness based on component connectivity
        for &node_idx in node_map.values() {
            let mut betweenness = 0.0;

            // Find which SCC this node belongs to
            let node_scc = sccs.iter().position(|scc| scc.contains(&node_idx));

            if let Some(scc_idx) = node_scc {
                // Count connections to other SCCs
                let out_edges: HashSet<usize> = graph
                    .edges_directed(node_idx, petgraph::Outgoing)
                    .filter_map(|edge| {
                        let target = edge.target();
                        sccs.iter().position(|scc| scc.contains(&target))
                    })
                    .filter(|&target_scc| target_scc != scc_idx)
                    .collect();

                let in_edges: HashSet<usize> = graph
                    .edges_directed(node_idx, petgraph::Incoming)
                    .filter_map(|edge| {
                        let source = edge.source();
                        sccs.iter().position(|scc| scc.contains(&source))
                    })
                    .filter(|&source_scc| source_scc != scc_idx)
                    .collect();

                // Betweenness is based on how many different components this node connects
                betweenness = (out_edges.len() + in_edges.len()) as f64;

                // Normalize by maximum possible connections
                let max_components = sccs.len().saturating_sub(1);
                if max_components > 0 {
                    betweenness /= max_components as f64;
                }
            }

            betweenness_scores.insert(node_idx, betweenness);
        }

        Ok(betweenness_scores)
    }

    /// Calculate simplified PageRank scores
    fn calculate_simple_pagerank(
        &self,
        graph: &Graph<String, (), Directed>,
        node_map: &HashMap<String, NodeIndex>,
    ) -> ScalingResult<Vec<f64>> {
        let node_count = graph.node_count();
        if node_count == 0 {
            return Ok(Vec::new());
        }

        let mut scores = vec![1.0 / node_count as f64; node_count];
        let damping = 0.85;
        let iterations = 10; // Simple approximation

        for _ in 0..iterations {
            let mut new_scores = vec![(1.0 - damping) / node_count as f64; node_count];

            for &node_idx in node_map.values() {
                let out_degree = graph.edges_directed(node_idx, petgraph::Outgoing).count();
                if out_degree > 0 {
                    let contribution = scores[node_idx.index()] * damping / out_degree as f64;

                    for edge in graph.edges_directed(node_idx, petgraph::Outgoing) {
                        let target_idx = edge.target().index();
                        new_scores[target_idx] += contribution;
                    }
                }
            }

            scores = new_scores;
        }

        Ok(scores)
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
                file.query_relevance =
                    self.calculate_file_query_relevance(&file.metadata, &query_words);
            }
        }

        Ok(files)
    }

    /// Calculate query relevance for a single file
    fn calculate_file_query_relevance(&self, file: &FileMetadata, query_words: &[&str]) -> f64 {
        let path_str = file.path.to_string_lossy().to_lowercase();
        let filename = file
            .path
            .file_name()
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
        if filename.contains("main")
            || filename.contains("index")
            || filename == "lib.rs"
            || filename == "__init__.py"
        {
            relevance += 0.3;
        }

        relevance
    }

    /// Group files by relatedness
    async fn group_by_relatedness(
        &self,
        mut files: Vec<FileWithCentrality>,
    ) -> ScalingResult<Vec<FileWithCentrality>> {
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
    async fn apply_positioning_strategy(
        &self,
        files: Vec<FileWithCentrality>,
    ) -> ScalingResult<ContextPositioning> {
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

        debug!(
            "Positioning strategy: HEAD={}, TAIL={}, MIDDLE={}",
            head_count,
            tail_count,
            total_files - head_count - tail_count
        );

        // Sort files for HEAD positioning: query relevance + centrality
        let mut head_candidates = files.clone();
        head_candidates.sort_by(|a, b| {
            let score_a = (a.query_relevance * self.config.query_relevance_weight)
                + (a.centrality.combined * self.config.centrality_weight);
            let score_b = (b.query_relevance * self.config.query_relevance_weight)
                + (b.centrality.combined * self.config.centrality_weight);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Sort files for TAIL positioning: pure centrality
        let mut tail_candidates = files.clone();
        tail_candidates.sort_by(|a, b| {
            b.centrality
                .combined
                .partial_cmp(&a.centrality.combined)
                .unwrap_or(std::cmp::Ordering::Equal)
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
            b.centrality
                .combined
                .partial_cmp(&a.centrality.combined)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Calculate total tokens for positioned files
    fn calculate_total_tokens(&self, positioning: &ContextPositioning) -> usize {
        let head_tokens = positioning
            .head_files
            .iter()
            .map(|f| self.estimate_tokens(&f.metadata))
            .sum::<usize>();

        let middle_tokens = positioning
            .middle_files
            .iter()
            .map(|f| self.estimate_tokens(&f.metadata))
            .sum::<usize>();

        let tail_tokens = positioning
            .tail_files
            .iter()
            .map(|f| self.estimate_tokens(&f.metadata))
            .sum::<usize>();

        head_tokens + middle_tokens + tail_tokens
    }

    /// Generate positioning reasoning explanation
    fn generate_positioning_reasoning(
        &self,
        positioning: &ContextPositioning,
        query_hint: Option<&str>,
    ) -> String {
        let mut reasoning = Vec::new();

        reasoning.push("Context Positioning Strategy Applied".to_string());
        reasoning.push("".to_string());

        // HEAD section reasoning
        reasoning.push(format!(
            "HEAD ({} files): Query-specific high centrality files",
            positioning.head_files.len()
        ));
        if let Some(query) = query_hint {
            reasoning.push(format!("   Query hint: '{}'", query));
        }
        for (i, file) in positioning.head_files.iter().take(3).enumerate() {
            reasoning.push(format!(
                "   {}. {} (centrality: {:.3}, relevance: {:.3})",
                i + 1,
                file.metadata
                    .path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("?"),
                file.centrality.combined,
                file.query_relevance
            ));
        }
        if positioning.head_files.len() > 3 {
            reasoning.push(format!(
                "   ... and {} more files",
                positioning.head_files.len() - 3
            ));
        }
        reasoning.push("".to_string());

        // MIDDLE section reasoning
        reasoning.push(format!(
            "MIDDLE ({} files): Supporting utilities and low-centrality files",
            positioning.middle_files.len()
        ));
        reasoning.push("".to_string());

        // TAIL section reasoning
        reasoning.push(format!(
            "TAIL ({} files): Core functionality, high centrality",
            positioning.tail_files.len()
        ));
        for (i, file) in positioning.tail_files.iter().take(3).enumerate() {
            reasoning.push(format!(
                "   {}. {} (centrality: {:.3})",
                i + 1,
                file.metadata
                    .path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("?"),
                file.centrality.combined
            ));
        }
        if positioning.tail_files.len() > 3 {
            reasoning.push(format!(
                "   ... and {} more files",
                positioning.tail_files.len() - 3
            ));
        }

        reasoning.join("\n")
    }

    /// Create simple positioning when optimization is disabled
    fn create_simple_positioning(&self, files: Vec<FileMetadata>) -> PositionedSelection {
        let files_with_centrality: Vec<FileWithCentrality> = files
            .into_iter()
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
        file::is_test_path(path)
    }
}
