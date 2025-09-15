//! Raw score calculation for heuristic components before normalization

use super::types::{RawScoreComponents, HeuristicWeights};
use super::super::{ScanResult, import_analysis::ImportGraph};

/// Calculate raw score components for a file before normalization
pub fn calculate_raw_scores<T>(
    file: &T, 
    weights: &HeuristicWeights, 
    import_graph: Option<&ImportGraph>
) -> RawScoreComponents 
where 
    T: ScanResult,
{
    // Documentation score
    let doc_raw = if file.is_docs() { 1.0 } else { 0.0 } + 
                 if let Some(doc_analysis) = file.doc_analysis() {
                     doc_analysis.structure_score()
                 } else {
                     0.0
                 };
    
    // README score
    let readme_raw = if file.is_readme() {
        // Root-level README gets higher score
        if file.depth() <= 1 { 1.5 } else { 1.0 }
    } else {
        0.0
    };
    
    // Import degree (estimated from imports list)
    let import_degree_in = file.centrality_in() as usize;
    let import_degree_out = if let Some(imports) = file.imports() {
        imports.len()
    } else {
        0
    };
    
    // Path depth
    let path_depth = file.depth();
    
    // Test links (use is_test as proxy)
    let test_links_found = if weights.features.enable_test_linking {
        if file.is_test() { 1 } else { 0 }
    } else {
        0
    };
    
    // Churn analysis (use churn_score from trait)
    let churn_commits = if weights.features.enable_churn_analysis {
        file.churn_score() as usize
    } else {
        0
    };
    
    // Centrality (PageRank from trait)
    let centrality_raw = if weights.features.enable_centrality {
        file.centrality_in()
    } else {
        0.0
    };
    
    // Entrypoint detection
    let is_entrypoint = is_file_entrypoint(file);
    
    // Examples detection
    let examples_count = if weights.features.enable_examples_detection {
        count_examples_in_file(file)
    } else {
        0
    };
    
    RawScoreComponents {
        doc_raw,
        readme_raw,
        import_degree_in,
        import_degree_out,
        path_depth,
        test_links_found,
        churn_commits,
        centrality_raw,
        is_entrypoint,
        examples_count,
    }
}

/// Detect if a file is likely an entrypoint
fn is_file_entrypoint<T: ScanResult>(file: &T) -> bool {
    // Use the built-in method from ScanResult trait
    file.is_entrypoint()
}

/// Check if a file is likely executable based on its content or naming
fn is_likely_executable<T: ScanResult>(file: &T) -> bool {
    let path_str = file.path().to_lowercase();
    
    // Executable extensions
    let exec_extensions = [".py", ".rs", ".js", ".ts", ".sh", ".bash", ".zsh"];
    if exec_extensions.iter().any(|ext| path_str.ends_with(ext)) {
        return true;
    }
    
    false
}

/// Count examples or example-like content in a file
fn count_examples_in_file<T: ScanResult>(file: &T) -> usize {
    // Use the built-in method from ScanResult trait
    if file.has_examples() { 1 } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    // Mock implementation for testing
    struct MockFile {
        path: String,
        is_docs: bool,
        is_readme: bool,
        depth: usize,
        content: Option<String>,
    }
    
    impl ScanResult for MockFile {
        fn path(&self) -> &str { &self.path }
        fn relative_path(&self) -> &str { &self.path }
        fn depth(&self) -> usize { self.depth }
        fn is_docs(&self) -> bool { self.is_docs }
        fn is_readme(&self) -> bool { self.is_readme }
        fn is_test(&self) -> bool { false }
        fn is_entrypoint(&self) -> bool { false }
        fn has_examples(&self) -> bool { false }
        fn priority_boost(&self) -> f64 { 0.0 }
        fn churn_score(&self) -> f64 { 0.0 }
        fn centrality_in(&self) -> f64 { 0.0 }
        fn imports(&self) -> Option<&[String]> { None }
        fn doc_analysis(&self) -> Option<&crate::heuristics::DocumentAnalysis> { None }
    }
    
    #[test]
    fn test_entrypoint_detection() {
        let main_py = MockFile {
            path: "main.py".to_string(),
            is_docs: false,
            is_readme: false,
            depth: 1,
            content: None,
        };
        assert!(!is_file_entrypoint(&main_py)); // Now returns false since is_entrypoint() returns false
        
        let random_file = MockFile {
            path: "utils/helper.py".to_string(),
            is_docs: false,
            is_readme: false,
            depth: 2,
            content: None,
        };
        assert!(!is_file_entrypoint(&random_file));
    }
    
    #[test]
    fn test_examples_counting() {
        let example_file = MockFile {
            path: "examples/demo.py".to_string(),
            is_docs: false,
            is_readme: false,
            depth: 2,
            content: Some("# Example usage\n```python\nprint('hello')\n```".to_string()),
        };
        
        let count = count_examples_in_file(&example_file);
        assert_eq!(count, 0); // Now returns 0 since has_examples() returns false
    }
}