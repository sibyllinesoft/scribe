//! File scoring helper functions for intelligent selection.

/// Calculate entry point bonus for a file path
pub fn entry_point_score(path_str: &str) -> f64 {
    let mut score = 0.0;
    if path_str.contains("main") || path_str.contains("index") {
        score += 2.0;
    }
    if path_str.contains("lib.rs") || path_str.contains("mod.rs") {
        score += 1.5;
    }
    if path_str.contains("__init__.py") {
        score += 1.3;
    }
    score
}

/// Calculate root-level file bonus
pub fn root_level_score(path_str: &str, path_components: usize) -> f64 {
    if path_components > 2 {
        return 0.0;
    }
    let mut score = 1.0;
    const ROOT_FILE_PATTERNS: &[&str] = &[
        "readme", "license", "cargo.toml", "package.json", "pyproject.toml", "setup.py",
    ];
    if ROOT_FILE_PATTERNS.iter().any(|p| path_str.contains(p)) {
        score += 1.5;
    }
    score
}

/// Calculate language importance score
pub fn language_score(language: &str) -> f64 {
    match language {
        "Rust" | "Python" | "JavaScript" | "TypeScript" => 0.8,
        "C" | "C++" | "Go" | "Java" => 0.6,
        "Shell" | "Makefile" => 0.4,
        _ => 0.0,
    }
}

/// Calculate file type importance score
pub fn file_type_score(file_type: &str) -> f64 {
    match file_type {
        "Source" => 0.6,
        "Configuration" => 0.5,
        "Documentation" => 0.3,
        _ => 0.0,
    }
}

/// Calculate size-related score adjustments
pub fn size_score(size: u64, path_str: &str) -> f64 {
    let mut score = 0.0;
    if size > 50_000 {
        score -= 0.5;
    }
    if size > 100_000 {
        score -= 1.0;
    }
    if size < 10_000 && (path_str.contains("config") || path_str.contains("env")) {
        score += 0.4;
    }
    score
}

/// Calculate nesting depth penalty
pub fn nesting_score(path_components: usize) -> f64 {
    if path_components > 4 {
        -0.3 * (path_components - 4) as f64
    } else {
        0.0
    }
}

/// Calculate test pattern bonus
pub fn test_pattern_score(path_str: &str) -> f64 {
    if path_str.contains("test") && !path_str.contains("tests/") {
        0.2
    } else {
        0.0
    }
}

/// Calculate combined file score from all factors
pub fn calculate_combined_score(
    path_str: &str,
    path_components: usize,
    language: &str,
    file_type: &str,
    size: u64,
) -> f64 {
    let score = 0.1 // Base score
        + entry_point_score(path_str)
        + root_level_score(path_str, path_components)
        + language_score(language)
        + file_type_score(file_type)
        + size_score(size, path_str)
        + nesting_score(path_components)
        + test_pattern_score(path_str);

    score.clamp(0.0, 5.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_point_score_python_init() {
        // Tests line 13: __init__.py detection
        let score = entry_point_score("mypackage/__init__.py");
        assert!((score - 1.3).abs() < 0.001);
    }

    #[test]
    fn test_file_type_score_unknown() {
        // Tests line 49: default case for unknown file type
        let score = file_type_score("Unknown");
        assert_eq!(score, 0.0);

        let score = file_type_score("Binary");
        assert_eq!(score, 0.0);

        let score = file_type_score("Web");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_size_score_large_files() {
        // Tests lines 57, 60: large file penalties
        let score_50k = size_score(60_000, "main.rs");
        assert!(score_50k < 0.0); // Should have -0.5 penalty

        let score_100k = size_score(150_000, "large_file.rs");
        assert!(score_100k < -1.0); // Should have both -0.5 and -1.0 penalties
    }

    #[test]
    fn test_test_pattern_score_in_file() {
        // Tests line 80: test pattern within a file (not in tests/ directory)
        let score = test_pattern_score("src/test_utils.rs");
        assert!((score - 0.2).abs() < 0.001);

        // Files in tests/ directory should not get bonus
        let score_tests_dir = test_pattern_score("tests/unit_test.rs");
        assert_eq!(score_tests_dir, 0.0);
    }

    #[test]
    fn test_entry_point_score_combinations() {
        // main and index
        assert!(entry_point_score("src/main.rs") > 0.0);
        assert!(entry_point_score("src/index.js") > 0.0);

        // lib.rs and mod.rs
        assert!(entry_point_score("src/lib.rs") > 0.0);
        assert!(entry_point_score("src/config/mod.rs") > 0.0);

        // No entry point
        assert_eq!(entry_point_score("src/utils.rs"), 0.0);
    }

    #[test]
    fn test_root_level_score_patterns() {
        // Root level files with patterns
        let score = root_level_score("readme.md", 1);
        assert!(score > 1.0);

        let score = root_level_score("cargo.toml", 1);
        assert!(score > 1.0);

        let score = root_level_score("package.json", 1);
        assert!(score > 1.0);

        // Too deep
        let score = root_level_score("src/deep/readme.md", 4);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_language_score_all_variants() {
        // High priority languages
        assert_eq!(language_score("Rust"), 0.8);
        assert_eq!(language_score("Python"), 0.8);
        assert_eq!(language_score("JavaScript"), 0.8);
        assert_eq!(language_score("TypeScript"), 0.8);

        // Medium priority
        assert_eq!(language_score("C"), 0.6);
        assert_eq!(language_score("C++"), 0.6);
        assert_eq!(language_score("Go"), 0.6);
        assert_eq!(language_score("Java"), 0.6);

        // Low priority
        assert_eq!(language_score("Shell"), 0.4);
        assert_eq!(language_score("Makefile"), 0.4);

        // Unknown
        assert_eq!(language_score("Unknown"), 0.0);
    }

    #[test]
    fn test_file_type_score_all_variants() {
        assert_eq!(file_type_score("Source"), 0.6);
        assert_eq!(file_type_score("Configuration"), 0.5);
        assert_eq!(file_type_score("Documentation"), 0.3);
    }

    #[test]
    fn test_size_score_config_bonus() {
        // Small config files get a bonus
        let score = size_score(5000, "config.json");
        assert!(score > 0.0);

        let score = size_score(5000, ".env");
        assert!(score > 0.0);

        // Regular files don't get bonus
        let score = size_score(5000, "main.rs");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_nesting_score() {
        // Shallow paths
        assert_eq!(nesting_score(1), 0.0);
        assert_eq!(nesting_score(4), 0.0);

        // Deep paths get penalties
        assert!(nesting_score(5) < 0.0);
        assert!(nesting_score(6) < nesting_score(5)); // More penalty for deeper
    }

    #[test]
    fn test_calculate_combined_score_clamp() {
        // Score should be clamped between 0 and 5
        let score = calculate_combined_score("src/main.rs", 2, "Rust", "Source", 1000);
        assert!(score >= 0.0 && score <= 5.0);

        // Even with negative factors
        let score = calculate_combined_score("very/deep/path/to/file.unknown", 10, "Unknown", "Unknown", 200_000);
        assert!(score >= 0.0);
    }
}
