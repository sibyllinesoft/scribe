//! Selective features example demonstrating minimal Scribe usage
//! 
//! This example shows how to use Scribe with only core and scanner features
//! enabled, which is useful for applications that only need file discovery
//! without complex analysis.

use scribe::core::{Config, FileInfo, Result, Language, FileType, bytes_to_human};
use scribe::core::{normalize_path, truncate, mean, median, generate_hash, duration_to_human};
use scribe::scanner::{Scanner, ScanOptions};

#[tokio::main]
async fn main() -> Result<()> {
    println!("📂 Scribe Selective Features Example");
    println!("====================================\n");
    
    // Get the directory path from command line or use current directory
    let scan_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| ".".to_string());
    
    println!("📁 Scanning directory: {}", scan_path);
    
    // Create a scanner with minimal configuration
    let scanner = Scanner::new();
    
    // Configure scan options for fast discovery
    let options = ScanOptions::default()
        .with_git_integration(true)
        .with_metadata_extraction(true)
        .with_parallel_processing(true);
    
    println!("⚙️  Using features: core + scanner only");
    println!("🚀 Starting fast scan...\n");
    
    let start_time = std::time::Instant::now();
    
    // Perform the scan
    let files = scanner.scan(&scan_path, options).await?;
    
    let duration = start_time.elapsed();
    println!("✅ Scan completed in {:.2}s", duration.as_secs_f64());
    println!("📊 Found {} files\n", files.len());
    
    // Analyze the results
    analyze_file_types(&files);
    show_largest_files(&files, 5);
    show_file_distribution(&files);
    
    // Demonstrate core utilities
    demonstrate_core_utilities();
    
    println!("\n🎉 Selective features demo complete!");
    println!("💡 To use full analysis features, add 'analysis' and 'graph' features");
    
    Ok(())
}

/// Analyze and display file type distribution
fn analyze_file_types(files: &[FileInfo]) {
    use std::collections::HashMap;
    use scribe::core::{Language, FileType};
    
    println!("📋 File Type Analysis:");
    println!("=====================");
    
    let mut language_counts = HashMap::new();
    let mut type_counts = HashMap::new();
    
    for file in files {
        // Count by language
        *language_counts.entry(file.language.clone()).or_insert(0) += 1;
        
        // Count by file type
        *type_counts.entry(file.file_type.clone()).or_insert(0) += 1;
    }
    
    // Show top languages
    let mut lang_sorted: Vec<_> = language_counts.iter().collect();
    lang_sorted.sort_by(|a, b| b.1.cmp(a.1));
    
    println!("Top languages:");
    for (lang, count) in lang_sorted.iter().take(5) {
        println!("  {:?}: {} files", lang, count);
    }
    
    // Show file types
    println!("\nFile types:");
    for (file_type, count) in type_counts {
        println!("  {:?}: {} files", file_type, count);
    }
    println!();
}

/// Show the largest files
fn show_largest_files(files: &[FileInfo], limit: usize) {
    let mut files_with_size: Vec<_> = files.iter()
        .map(|f| (f, f.size))
        .collect();
    
    files_with_size.sort_by(|a, b| b.1.cmp(&a.1));
    
    println!("📏 Largest Files:");
    println!("================");
    
    for (rank, (file, size)) in files_with_size.iter().take(limit).enumerate() {
        let size_human = bytes_to_human(*size as u64);
        let path_display = file.path.to_string_lossy();
        println!("{:2}. {:<50} {}", rank + 1, path_display, size_human);
    }
    println!();
}

/// Show distribution of files by directory depth
fn show_file_distribution(files: &[FileInfo]) {
    use std::collections::HashMap;
    
    println!("📊 Directory Depth Distribution:");
    println!("===============================");
    
    let mut depth_counts = HashMap::new();
    
    for file in files {
        let depth = file.path.components().count();
        *depth_counts.entry(depth).or_insert(0) += 1;
    }
    
    let mut depth_sorted: Vec<_> = depth_counts.iter().collect();
    depth_sorted.sort_by_key(|&(depth, _)| depth);
    
    for (depth, count) in depth_sorted {
        let bar = "█".repeat((*count as f64 / 10.0).ceil() as usize);
        println!("  Depth {}: {:3} files {}", depth, count, bar);
    }
    println!();
}

/// Demonstrate core utility functions
fn demonstrate_core_utilities() {
    use scribe::core::utils::*;
    
    println!("🛠️  Core Utilities Demo:");
    println!("=======================");
    
    // Path utilities
    let normalized = normalize_path("./src/../src/main.rs");
    println!("Path normalization: './src/../src/main.rs' -> '{}'", normalized);
    
    // String utilities
    let long_text = "This is a very long string that needs to be truncated for display";
    let truncated = truncate(long_text, 30);
    println!("String truncation: '{}' -> '{}'", long_text, truncated);
    
    // Math utilities
    let numbers = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let avg = mean(&numbers);
    let med = median(&mut numbers.clone());
    println!("Math utilities: mean={:.1}, median={:.1}", avg, med);
    
    // Hash utilities
    let content = "Hello, Scribe!";
    let hash = generate_hash(&content);
    println!("Hash generation: '{}' -> '{}'", content, &hash[..8]);
    
    // Time utilities
    let duration = std::time::Duration::from_millis(1234567);
    let human_time = duration_to_human(duration);
    println!("Time formatting: {:?} -> '{}'", duration, human_time);
    
    println!();
}

/// Helper trait to extend ScanOptions with a builder pattern
trait ScanOptionsExt {
    fn with_git_integration(self, enabled: bool) -> Self;
    fn with_metadata_extraction(self, enabled: bool) -> Self;
    fn with_parallel_processing(self, enabled: bool) -> Self;
}

impl ScanOptionsExt for ScanOptions {
    fn with_git_integration(mut self, enabled: bool) -> Self {
        self.git_integration = enabled;
        self
    }
    
    fn with_metadata_extraction(mut self, enabled: bool) -> Self {
        self.metadata_extraction = enabled;
        self
    }
    
    fn with_parallel_processing(mut self, enabled: bool) -> Self {
        self.parallel_processing = enabled;
        self
    }
}