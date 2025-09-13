//! Basic usage example demonstrating the full Scribe functionality
//! 
//! This example shows how to use Scribe with all features enabled
//! to analyze a repository and get the most important files.

use scribe::prelude::*;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    println!("🔍 Scribe Repository Analysis Example");
    println!("====================================\n");
    
    // Get the repository path from command line or use current directory
    let repo_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| ".".to_string());
    
    println!("📁 Analyzing repository: {}", repo_path);
    
    // Create a configuration with custom settings
    let config = Config::default()
        .with_git_integration(true)
        .with_parallel_processing(true);
    
    println!("⚙️  Configuration: {:?}\n", config.general);
    
    // Perform comprehensive repository analysis
    println!("🚀 Starting analysis...");
    let start_time = std::time::Instant::now();
    
    let analysis = analyze_repository(&repo_path, &config).await?;
    
    let duration = start_time.elapsed();
    println!("✅ Analysis completed in {:.2}s\n", duration.as_secs_f64());
    
    // Display summary
    println!("{}\n", analysis.summary());
    
    // Show top 10 most important files
    println!("🏆 Top 10 Most Important Files:");
    println!("================================");
    for (rank, (file_path, score)) in analysis.top_files(10).iter().enumerate() {
        println!("{:2}. {:<50} {:.3}", rank + 1, file_path, score);
    }
    
    // Show files above threshold
    let threshold = 0.7;
    let high_importance_files = analysis.files_above_threshold(threshold);
    if !high_importance_files.is_empty() {
        println!("\n🎯 High Importance Files (score > {}):", threshold);
        println!("========================================");
        for (file_path, score) in high_importance_files {
            println!("  {:<50} {:.3}", file_path, score);
        }
    }
    
    // Display analysis metadata
    println!("\n📊 Analysis Details:");
    println!("===================");
    println!("Version: {}", scribe::VERSION);
    println!("Files analyzed: {}", analysis.file_count());
    println!("Scribe version: {}", analysis.metadata.scribe_version);
    println!("Features used: {}", analysis.metadata.features_enabled.join(", "));
    
    // Demonstrate feature-specific functionality
    #[cfg(feature = "graph")]
    {
        if let Some(ref centrality_scores) = analysis.centrality_scores {
            println!("\n🕸️  Graph Analysis Results:");
            println!("=========================");
            
            let mut centrality_sorted: Vec<_> = centrality_scores.iter().collect();
            centrality_sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            println!("Top 5 files by centrality:");
            for (rank, (file, score)) in centrality_sorted.iter().take(5).enumerate() {
                println!("{:2}. {:<50} {:.4}", rank + 1, file, score);
            }
        }
    }
    
    // Demonstrate pattern matching
    #[cfg(feature = "patterns")]
    {
        println!("\n🎯 Pattern Matching Demo:");
        println!("========================");
        
        // Create different pattern matchers
        let mut source_matcher = scribe::patterns::presets::source_code()?;
        let mut doc_matcher = scribe::patterns::presets::documentation()?;
        
        let source_files = analysis.files.iter()
            .filter(|f| {
                source_matcher.should_process(&f.path).unwrap_or(false)
            })
            .count();
            
        let doc_files = analysis.files.iter()
            .filter(|f| {
                doc_matcher.should_process(&f.path).unwrap_or(false)
            })
            .count();
        
        println!("Source code files: {}", source_files);
        println!("Documentation files: {}", doc_files);
    }
    
    println!("\n🎉 Analysis complete!");
    
    Ok(())
}

// Helper trait extension for Config (example of how to extend the API)
trait ConfigExt {
    fn with_git_integration(self, enabled: bool) -> Self;
    fn with_parallel_processing(self, enabled: bool) -> Self;
}

impl ConfigExt for Config {
    fn with_git_integration(mut self, enabled: bool) -> Self {
        self.git.enabled = enabled;
        self
    }
    
    fn with_parallel_processing(mut self, enabled: bool) -> Self {
        // Performance config doesn't have parallel_processing field
        // This is a placeholder implementation
        self
    }
}