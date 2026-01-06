//! Tests for the covering set computation module.

use super::*;
use crate::ast::ast_parser::EntityQuery;

#[test]
fn test_covering_set_options() {
    let opts = CoveringSetOptions::default();
    assert!(opts.include_dependencies);
    assert!(!opts.include_dependents);

    let minimal = CoveringSetOptions::minimal();
    assert_eq!(minimal.max_depth, Some(1));
    assert_eq!(minimal.max_files, Some(20));
}

#[test]
fn test_covering_set_computer_creation() {
    let computer = CoveringSetComputer::new();
    assert!(computer.is_ok());
}

#[test]
fn test_inclusion_reason_formatting() {
    let computer = CoveringSetComputer::new().unwrap();

    let reason = computer.format_inclusion_reason(&InclusionReason::TargetFile, 0);
    assert_eq!(reason, "Contains the target entity");

    let reason = computer.format_inclusion_reason(&InclusionReason::ChangedFile, 0);
    assert_eq!(reason, "Changed in diff");

    let reason = computer.format_inclusion_reason(&InclusionReason::DirectDependency, 1);
    assert_eq!(reason, "Direct dependency of target");

    let reason = computer.format_inclusion_reason(&InclusionReason::TransitiveDependency, 3);
    assert!(reason.contains("distance: 3"));
}

#[test]
fn test_covering_set_for_changed_files() {
    let computer = CoveringSetComputer::new().unwrap();
    let graph = DependencyGraph::new();

    let changed = vec!["src/lib.rs".to_string(), "src/main.rs".to_string()];
    let result = computer
        .compute_covering_set_for_files(
            &changed,
            &graph,
            None,
            &CoveringSetOptions::default(),
        )
        .unwrap();

    assert!(result.target_entity.is_none());
    assert_eq!(result.files.len(), 2);
    assert!(result
        .files
        .iter()
        .all(|f| f.reason == InclusionReason::ChangedFile));
}

#[test]
fn test_entity_level_options() {
    let opts = CoveringSetOptions::entity_level();
    assert_eq!(opts.granularity, CoveringSetGranularity::Entity);
    assert!(opts.include_dependencies);
    assert_eq!(opts.max_depth, Some(3));

    let focused = CoveringSetOptions::entity_focused();
    assert_eq!(focused.granularity, CoveringSetGranularity::Entity);
    assert_eq!(focused.max_depth, Some(2));
}

#[test]
fn test_granularity_default() {
    assert_eq!(CoveringSetGranularity::default(), CoveringSetGranularity::File);

    let opts = CoveringSetOptions::default();
    assert_eq!(opts.granularity, CoveringSetGranularity::File);
}

#[test]
fn test_entity_level_covering_set() {
    let mut computer = CoveringSetComputer::new().unwrap();
    let graph = DependencyGraph::new();

    // Create test file contents with a function that calls another
    let mut file_contents = HashMap::new();
    file_contents.insert(
        "src/lib.py".to_string(),
        r#"
def main():
    result = helper_func(42)
    return result

def helper_func(x):
    return x * 2
"#.to_string(),
    );

    // Use file:entity format
    let query = EntityQuery::for_file_entity("src/lib.py", "main");

    let result = computer
        .compute_covering_set(
            &query,
            &file_contents,
            &graph,
            &CoveringSetOptions::entity_level(),
        )
        .unwrap();

    // Should find the target entity
    assert!(result.target_entity.is_some());
    let target = result.target_entity.as_ref().unwrap();
    assert_eq!(target.entity_name, "main");

    // In entity-level mode, files should be empty
    assert!(result.files.is_empty());

    // Should have at least the target entity
    assert!(!result.entities.is_empty());
    assert!(result.entities.iter().any(|e| e.name == "main"));

    // Statistics should reflect entity-level results
    assert!(result.statistics.entities_selected > 0);
}

#[test]
fn test_file_only_covering_set() {
    let mut computer = CoveringSetComputer::new().unwrap();
    let graph = DependencyGraph::new();

    let mut file_contents = HashMap::new();
    file_contents.insert(
        "src/lib.py".to_string(),
        "def main(): pass".to_string(),
    );

    // Use file-only format (no entity)
    let query = EntityQuery::for_file("src/lib.py");

    let result = computer
        .compute_covering_set(
            &query,
            &file_contents,
            &graph,
            &CoveringSetOptions::default(),
        )
        .unwrap();

    // Should find the file but no specific entity
    assert!(result.target_entity.is_none());
    assert!(!result.files.is_empty());
    assert_eq!(result.files[0].path, "src/lib.py");
    assert_eq!(result.files[0].reason, InclusionReason::TargetFile);
}

#[test]
fn test_file_pattern_requires_file() {
    let mut computer = CoveringSetComputer::new().unwrap();
    let graph = DependencyGraph::new();
    let file_contents = HashMap::new();

    // Query with no file pattern should error
    let query = EntityQuery::by_name("main");

    let result = computer.compute_covering_set(
        &query,
        &file_contents,
        &graph,
        &CoveringSetOptions::default(),
    );

    assert!(result.is_err());
}

#[test]
fn test_is_common_keyword() {
    assert!(super::is_common_keyword("if"));
    assert!(super::is_common_keyword("def"));
    assert!(super::is_common_keyword("print"));
    assert!(super::is_common_keyword("self"));
    assert!(!super::is_common_keyword("my_function"));
    assert!(!super::is_common_keyword("CustomClass"));
}

#[test]
fn test_is_common_type() {
    assert!(super::is_common_type("String"));
    assert!(super::is_common_type("Option"));
    assert!(super::is_common_type("Vec"));
    assert!(!super::is_common_type("MyCustomType"));
    assert!(!super::is_common_type("UserService"));
}
