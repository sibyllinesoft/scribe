// Temporary debug script
fn main() {
    use scribe_patterns::gitignore::GitignoreMatcher;
    
    let mut matcher = GitignoreMatcher::with_defaults();
    println!("Stats: {:?}", matcher.stats());
    
    // Debug what patterns are loaded
    println!("Testing node_modules/package.json");
    let result = matcher.match_path("node_modules/package.json").unwrap();
    println!("Match result: {:?}", result);
    
    println!("Testing target/debug/main");
    let result = matcher.match_path("target/debug/main").unwrap();
    println!("Match result: {:?}", result);
}
