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
