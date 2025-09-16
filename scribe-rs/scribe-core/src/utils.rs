//! Utility functions and helpers for the Scribe library.
//!
//! Provides common functionality used across the Scribe ecosystem,
//! including path manipulation, string processing, and validation.

use crate::error::{Result, ScribeError};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Path utility functions
pub mod path {
    use super::*;

    /// Normalize a path to use forward slashes (for cross-platform consistency)
    pub fn normalize_path<P: AsRef<Path>>(path: P) -> String {
        path.as_ref().to_string_lossy().replace('\\', "/")
    }

    /// Get relative path from base to target
    pub fn relative_path<P1: AsRef<Path>, P2: AsRef<Path>>(
        base: P1,
        target: P2,
    ) -> Result<PathBuf> {
        let base_ref = base.as_ref();
        let target_ref = target.as_ref();
        let base = base_ref.canonicalize().map_err(|e| {
            ScribeError::path_with_source("Failed to canonicalize base path", base_ref, e)
        })?;
        let target = target_ref.canonicalize().map_err(|e| {
            ScribeError::path_with_source("Failed to canonicalize target path", target_ref, e)
        })?;

        target
            .strip_prefix(&base)
            .map(|p| p.to_path_buf())
            .map_err(|_| ScribeError::path("Target path is not under base path", &target))
    }

    /// Check if path is under a given directory
    pub fn is_under_directory<P1: AsRef<Path>, P2: AsRef<Path>>(path: P1, directory: P2) -> bool {
        match relative_path(directory, path) {
            Ok(rel_path) => !rel_path.to_string_lossy().starts_with(".."),
            Err(_) => false,
        }
    }

    /// Get the depth of a path (number of components)
    pub fn path_depth<P: AsRef<Path>>(path: P) -> usize {
        path.as_ref().components().count()
    }

    /// Check if path represents a hidden file or directory (starts with .)
    pub fn is_hidden<P: AsRef<Path>>(path: P) -> bool {
        path.as_ref()
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.starts_with('.'))
            .unwrap_or(false)
    }

    /// Ensure directory exists, creating if necessary
    pub fn ensure_dir_exists<P: AsRef<Path>>(path: P) -> Result<()> {
        let path = path.as_ref();
        if !path.exists() {
            std::fs::create_dir_all(path).map_err(|e| {
                ScribeError::path_with_source("Failed to create directory", path, e)
            })?;
        } else if !path.is_dir() {
            return Err(ScribeError::path(
                "Path exists but is not a directory",
                path,
            ));
        }
        Ok(())
    }

    /// Find the repository root by looking for common markers (.git, Cargo.toml, etc.)
    pub fn find_repo_root<P: AsRef<Path>>(start_path: P) -> Option<PathBuf> {
        const REPO_MARKERS: &[&str] = &[
            ".git",
            "Cargo.toml",
            "package.json",
            "pyproject.toml",
            "setup.py",
            "go.mod",
            "pom.xml",
            "build.gradle",
            "Makefile",
        ];

        let mut current = start_path.as_ref();

        loop {
            for marker in REPO_MARKERS {
                if current.join(marker).exists() {
                    return Some(current.to_path_buf());
                }
            }

            match current.parent() {
                Some(parent) => current = parent,
                None => return None,
            }
        }
    }
}

/// String processing utilities
pub mod string {
    use super::*;

    /// Truncate a string to a maximum length, adding ellipsis if needed
    pub fn truncate(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else if max_len <= 3 {
            "...".to_string()
        } else {
            format!("{}...", &s[..max_len - 3])
        }
    }

    /// Remove common leading whitespace from all lines
    pub fn dedent(s: &str) -> String {
        let lines: Vec<&str> = s.lines().collect();
        if lines.is_empty() {
            return String::new();
        }

        // Find minimum indentation (ignoring empty lines)
        let min_indent = lines
            .iter()
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.len() - line.trim_start().len())
            .min()
            .unwrap_or(0);

        lines
            .iter()
            .map(|line| {
                if line.trim().is_empty() {
                    String::new()
                } else {
                    line.chars().skip(min_indent).collect()
                }
            })
            .collect::<Vec<String>>()
            .join("\n")
    }

    /// Count lines in a string
    pub fn count_lines(s: &str) -> usize {
        if s.is_empty() {
            0
        } else {
            s.matches('\n').count() + 1
        }
    }

    /// Check if string is likely binary content
    pub fn is_likely_binary(s: &str) -> bool {
        // Check for null bytes or high proportion of non-printable characters
        let null_bytes = s.bytes().filter(|&b| b == 0).count();
        if null_bytes > 0 {
            return true;
        }

        let total_chars = s.chars().count();
        if total_chars == 0 {
            return false;
        }

        let non_printable = s
            .chars()
            .filter(|&c| {
                !c.is_ascii_graphic()
                    && !c.is_ascii_whitespace()
                    && c != '\n'
                    && c != '\r'
                    && c != '\t'
            })
            .count();

        // If more than 30% non-printable characters, likely binary
        (non_printable as f64 / total_chars as f64) > 0.3
    }

    /// Extract identifier/name from a file path
    pub fn extract_identifier<P: AsRef<Path>>(path: P) -> String {
        path.as_ref()
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("unknown")
            .to_string()
    }

    /// Convert snake_case to camelCase
    pub fn snake_to_camel(s: &str) -> String {
        let mut result = String::new();
        let mut capitalize_next = false;

        for c in s.chars() {
            if c == '_' {
                capitalize_next = true;
            } else if capitalize_next {
                result.push(c.to_uppercase().next().unwrap_or(c));
                capitalize_next = false;
            } else {
                result.push(c);
            }
        }

        result
    }

    /// Convert camelCase to snake_case
    pub fn camel_to_snake(s: &str) -> String {
        let mut result = String::new();

        for (i, c) in s.char_indices() {
            if c.is_uppercase() && i > 0 {
                result.push('_');
            }
            result.push(c.to_lowercase().next().unwrap_or(c));
        }

        result
    }
}

/// Time and duration utilities
pub mod time {
    use super::*;

    /// Convert Duration to human-readable string
    pub fn duration_to_human(duration: Duration) -> String {
        let total_secs = duration.as_secs();
        let millis = duration.subsec_millis();

        if total_secs >= 3600 {
            let hours = total_secs / 3600;
            let mins = (total_secs % 3600) / 60;
            let secs = total_secs % 60;
            format!("{}h {}m {}s", hours, mins, secs)
        } else if total_secs >= 60 {
            let mins = total_secs / 60;
            let secs = total_secs % 60;
            format!("{}m {}s", mins, secs)
        } else if total_secs > 0 {
            format!("{}.{:03}s", total_secs, millis)
        } else {
            format!("{}ms", millis)
        }
    }

    /// Get current timestamp as seconds since Unix epoch
    pub fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Convert SystemTime to timestamp
    pub fn system_time_to_timestamp(time: SystemTime) -> u64 {
        time.duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Convert timestamp to SystemTime
    pub fn timestamp_to_system_time(timestamp: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(timestamp)
    }
}

/// Collection utilities
pub mod collections {
    use std::collections::HashMap;
    use std::hash::Hash;

    /// Count occurrences of items in an iterator
    pub fn count_occurrences<T, I>(iter: I) -> HashMap<T, usize>
    where
        T: Eq + Hash,
        I: Iterator<Item = T>,
    {
        let mut counts = HashMap::new();
        for item in iter {
            *counts.entry(item).or_insert(0) += 1;
        }
        counts
    }

    /// Find the most common item in an iterator
    pub fn most_common<T, I>(iter: I) -> Option<T>
    where
        T: Eq + Hash + Clone,
        I: Iterator<Item = T>,
    {
        let counts = count_occurrences(iter);
        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(item, _)| item)
    }

    /// Group items by a key function
    pub fn group_by<T, K, F>(items: Vec<T>, key_fn: F) -> HashMap<K, Vec<T>>
    where
        K: Eq + Hash,
        F: Fn(&T) -> K,
    {
        let mut groups = HashMap::new();
        for item in items {
            let key = key_fn(&item);
            groups.entry(key).or_insert_with(Vec::new).push(item);
        }
        groups
    }
}

/// Math and statistics utilities
pub mod math {
    /// Calculate mean of a slice of numbers
    pub fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }

    /// Calculate median of a slice of numbers
    pub fn median(values: &mut [f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = values.len() / 2;

        if values.len() % 2 == 0 {
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[mid]
        }
    }

    /// Calculate standard deviation
    pub fn std_deviation(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean_val = mean(values);
        let variance = values
            .iter()
            .map(|x| {
                let diff = x - mean_val;
                diff * diff
            })
            .sum::<f64>()
            / values.len() as f64;

        variance.sqrt()
    }

    /// Normalize values to 0-1 range
    pub fn normalize(values: &mut [f64]) {
        if values.is_empty() {
            return;
        }

        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        if range == 0.0 {
            // All values are the same
            values.iter_mut().for_each(|x| *x = 0.0);
        } else {
            values.iter_mut().for_each(|x| *x = (*x - min_val) / range);
        }
    }

    /// Clamp a value between min and max
    pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
        value.max(min).min(max)
    }
}

/// Validation utilities
pub mod validation {
    use super::*;

    /// Validate that a path exists and is readable
    pub fn validate_readable_path<P: AsRef<Path>>(path: P) -> Result<()> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(ScribeError::path("Path does not exist", path));
        }

        // Try to read metadata to check if readable
        std::fs::metadata(path)
            .map_err(|e| ScribeError::path_with_source("Path is not readable", path, e))?;

        Ok(())
    }

    /// Validate that a path is a directory
    pub fn validate_directory<P: AsRef<Path>>(path: P) -> Result<()> {
        let path = path.as_ref();
        validate_readable_path(path)?;

        if !path.is_dir() {
            return Err(ScribeError::path("Path is not a directory", path));
        }

        Ok(())
    }

    /// Validate that a path is a file
    pub fn validate_file<P: AsRef<Path>>(path: P) -> Result<()> {
        let path = path.as_ref();
        validate_readable_path(path)?;

        if !path.is_file() {
            return Err(ScribeError::path("Path is not a file", path));
        }

        Ok(())
    }

    /// Validate configuration values
    pub fn validate_config_value<T>(value: T, min: T, max: T, field_name: &str) -> Result<T>
    where
        T: PartialOrd + std::fmt::Display + Copy,
    {
        if value < min || value > max {
            return Err(ScribeError::config_field(
                format!("{} must be between {} and {}", field_name, min, max),
                field_name,
            ));
        }
        Ok(value)
    }
}

/// Hash generation utilities  
pub mod hash {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    /// Generate a hash string from a hashable value
    pub fn generate_hash<T: Hash>(value: &T) -> String {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Generate a hash from file contents
    pub fn hash_file_content(content: &str) -> String {
        generate_hash(&content)
    }

    /// Generate a hash from multiple values
    pub fn hash_multiple<T: Hash>(values: &[T]) -> String {
        let mut hasher = DefaultHasher::new();
        for value in values {
            value.hash(&mut hasher);
        }
        format!("{:x}", hasher.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_normalize() {
        let windows_path = r"src\lib\mod.rs";
        let normalized = path::normalize_path(windows_path);
        assert_eq!(normalized, "src/lib/mod.rs");
    }

    #[test]
    fn test_path_depth() {
        assert_eq!(path::path_depth("file.txt"), 1);
        assert_eq!(path::path_depth("src/lib.rs"), 2);
        assert_eq!(path::path_depth("src/nested/deep/file.rs"), 4);
    }

    #[test]
    fn test_is_hidden() {
        assert!(path::is_hidden(".gitignore"));
        assert!(path::is_hidden(".cargo")); // Directory itself is hidden
        assert!(!path::is_hidden("src/lib.rs"));
        assert!(!path::is_hidden("README.md"));
        // Test nested path - only checks the final component
        assert!(!path::is_hidden(".cargo/config"));
    }

    #[test]
    fn test_string_truncate() {
        assert_eq!(string::truncate("hello", 10), "hello");
        assert_eq!(string::truncate("hello world", 8), "hello...");
        assert_eq!(string::truncate("hi", 2), "hi");
        assert_eq!(string::truncate("hello", 3), "...");
    }

    #[test]
    fn test_string_dedent() {
        let indented = "    line 1\n    line 2\n        line 3";
        let expected = "line 1\nline 2\n    line 3";
        assert_eq!(string::dedent(indented), expected);
    }

    #[test]
    fn test_count_lines() {
        assert_eq!(string::count_lines(""), 0);
        assert_eq!(string::count_lines("single line"), 1);
        assert_eq!(string::count_lines("line 1\nline 2"), 2);
        assert_eq!(string::count_lines("line 1\nline 2\n"), 3);
    }

    #[test]
    fn test_case_conversion() {
        assert_eq!(string::snake_to_camel("hello_world"), "helloWorld");
        assert_eq!(string::snake_to_camel("test_case_name"), "testCaseName");

        assert_eq!(string::camel_to_snake("helloWorld"), "hello_world");
        assert_eq!(string::camel_to_snake("TestCaseName"), "test_case_name");
    }

    #[test]
    fn test_binary_detection() {
        assert!(!string::is_likely_binary("Hello world"));
        assert!(!string::is_likely_binary("let x = 42;\nfn main() {}"));
        assert!(string::is_likely_binary("Hello\x00world"));

        // Test high proportion of non-printable characters
        let mostly_non_printable = (0..100)
            .map(|i| if i % 3 == 0 { 'a' } else { '\x01' })
            .collect::<String>();
        assert!(string::is_likely_binary(&mostly_non_printable));
    }

    #[test]
    fn test_duration_formatting() {
        assert_eq!(time::duration_to_human(Duration::from_millis(500)), "500ms");
        assert_eq!(time::duration_to_human(Duration::from_secs(5)), "5.000s");
        assert_eq!(time::duration_to_human(Duration::from_secs(65)), "1m 5s");
        assert_eq!(
            time::duration_to_human(Duration::from_secs(3661)),
            "1h 1m 1s"
        );
    }

    #[test]
    fn test_math_functions() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(math::mean(&values), 3.0);

        let mut values_for_median = values.clone();
        assert_eq!(math::median(&mut values_for_median), 3.0);

        let std_dev = math::std_deviation(&values);
        // Standard deviation of [1,2,3,4,5] is sqrt(2) ≈ 1.4142
        assert!((std_dev - 1.4142135623730951).abs() < 1e-10);

        assert_eq!(math::clamp(5.0, 2.0, 4.0), 4.0);
        assert_eq!(math::clamp(1.0, 2.0, 4.0), 2.0);
        assert_eq!(math::clamp(3.0, 2.0, 4.0), 3.0);
    }

    #[test]
    fn test_collections_utilities() {
        let items = vec!['a', 'b', 'a', 'c', 'a'];
        let counts = collections::count_occurrences(items.iter().cloned());
        assert_eq!(counts[&'a'], 3);
        assert_eq!(counts[&'b'], 1);
        assert_eq!(counts[&'c'], 1);

        let most_common = collections::most_common(items.iter().cloned());
        assert_eq!(most_common, Some('a'));
    }

    #[test]
    fn test_hash_generation() {
        let hash1 = hash::generate_hash(&"test string");
        let hash2 = hash::generate_hash(&"test string");
        let hash3 = hash::generate_hash(&"different string");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
}
