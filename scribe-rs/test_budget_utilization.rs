use scribe::prelude::*;
use std::path::Path;

#[tokio::main]
async fn main() -> scribe::Result<()> {
    // Enable debug output
    std::env::set_var("SCRIBE_DEBUG", "1");

    println!("Testing token budget utilization improvements...");

    // Create config with a 250k token budget
    let mut config = Config::default();
    config.analysis.token_budget = Some(250_000);

    // Test on the current repository
    let repo_path = Path::new(".");

    println!("Analyzing repository with 250k token budget...");
    let analysis = scribe::analyze_repository(repo_path, &config).await?;

    println!("\n=== Analysis Results ===");
    println!("Files analyzed: {}", analysis.file_count());
    println!("Summary: {}", analysis.summary());

    // Get token usage from the files if available
    let mut total_tokens = 0;
    let mut files_with_tokens = 0;

    for file in &analysis.files {
        if let Some(tokens) = file.token_estimate {
            total_tokens += tokens;
            files_with_tokens += 1;
        }
    }

    if files_with_tokens > 0 {
        let utilization = (total_tokens as f64 / 250_000.0) * 100.0;
        println!(
            "Token utilization: {} / 250,000 ({:.1}%)",
            total_tokens, utilization
        );

        if utilization > 90.0 {
            println!("✅ Excellent budget utilization (>90%)");
        } else if utilization > 80.0 {
            println!("🟡 Good budget utilization (>80%)");
        } else {
            println!("❌ Poor budget utilization (<80%)");
        }
    } else {
        println!("⚠️  No token estimates available in results");
    }

    Ok(())
}
