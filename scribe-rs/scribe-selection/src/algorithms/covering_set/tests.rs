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

#[test]
fn test_covering_set_statistics_default() {
    let stats = CoveringSetStatistics {
        files_examined: 100,
        files_selected: 50,
        files_excluded: 10,
        entities_selected: 0,
        max_depth_reached: 3,
        limits_reached: false,
    };

    assert_eq!(stats.files_examined, 100);
    assert_eq!(stats.files_selected, 50);
    assert_eq!(stats.max_depth_reached, 3);
}

#[test]
fn test_covering_set_file_fields() {
    let file = CoveringSetFile {
        path: "src/lib.rs".to_string(),
        reason: InclusionReason::DirectDependency,
        distance: 1,
        importance: Some(0.8),
        line_ranges: vec![LineRange { start_line: 10, end_line: 50 }],
    };

    assert_eq!(file.path, "src/lib.rs");
    assert_eq!(file.reason, InclusionReason::DirectDependency);
    assert_eq!(file.distance, 1);
    assert_eq!(file.importance, Some(0.8));
    assert_eq!(file.line_ranges.len(), 1);
}

#[test]
fn test_covering_set_entity_fields() {
    let entity = CoveringSetEntity {
        file_path: "src/utils.py".to_string(),
        name: "helper_function".to_string(),
        entity_type: "function".to_string(),
        content: "def helper_function(): pass".to_string(),
        start_line: 1,
        end_line: 1,
        reason: InclusionReason::TransitiveDependency,
        distance: 2,
        references: vec!["other_func".to_string()],
    };

    assert_eq!(entity.file_path, "src/utils.py");
    assert_eq!(entity.name, "helper_function");
    assert_eq!(entity.entity_type, "function");
    assert_eq!(entity.distance, 2);
    assert!(!entity.references.is_empty());
}

#[test]
fn test_line_range() {
    let range = LineRange {
        start_line: 10,
        end_line: 25,
    };

    assert_eq!(range.start_line, 10);
    assert_eq!(range.end_line, 25);
}

#[test]
fn test_inclusion_reason_equality() {
    assert_eq!(InclusionReason::TargetFile, InclusionReason::TargetFile);
    assert_eq!(InclusionReason::ChangedFile, InclusionReason::ChangedFile);
    assert_ne!(InclusionReason::TargetFile, InclusionReason::ChangedFile);
    assert_ne!(InclusionReason::DirectDependency, InclusionReason::TransitiveDependency);
}

#[test]
fn test_covering_set_options_dependents() {
    let mut opts = CoveringSetOptions::default();
    opts.include_dependents = true;

    assert!(opts.include_dependents);
    assert!(opts.include_dependencies); // Still true
}

#[test]
fn test_covering_set_result_empty() {
    let result = CoveringSetResult {
        target_entity: None,
        files: Vec::new(),
        entities: Vec::new(),
        statistics: CoveringSetStatistics {
            files_examined: 0,
            files_selected: 0,
            files_excluded: 0,
            entities_selected: 0,
            max_depth_reached: 0,
            limits_reached: false,
        },
        inclusion_reasons: HashMap::new(),
    };

    assert!(result.files.is_empty());
    assert!(result.entities.is_empty());
    assert!(result.target_entity.is_none());
}

#[test]
fn test_covering_set_for_empty_changed_files() {
    let computer = CoveringSetComputer::new().unwrap();
    let graph = DependencyGraph::new();

    let result = computer
        .compute_covering_set_for_files(
            &[],
            &graph,
            None,
            &CoveringSetOptions::default(),
        )
        .unwrap();

    assert!(result.files.is_empty());
    assert!(result.statistics.files_examined == 0);
}

#[test]
fn test_covering_set_with_line_map() {
    let computer = CoveringSetComputer::new().unwrap();
    let graph = DependencyGraph::new();

    let mut line_map = HashMap::new();
    line_map.insert(
        "src/main.rs".to_string(),
        vec![LineRange { start_line: 10, end_line: 20 }],
    );

    let changed = vec!["src/main.rs".to_string()];
    let result = computer
        .compute_covering_set_for_files(
            &changed,
            &graph,
            Some(&line_map),
            &CoveringSetOptions::default(),
        )
        .unwrap();

    assert_eq!(result.files.len(), 1);
    assert!(!result.files[0].line_ranges.is_empty());
    assert_eq!(result.files[0].line_ranges[0].start_line, 10);
}

#[test]
fn test_covering_set_file_not_found() {
    let mut computer = CoveringSetComputer::new().unwrap();
    let graph = DependencyGraph::new();
    let file_contents = HashMap::new(); // Empty - file not found

    let query = EntityQuery::for_file("nonexistent.py");

    let result = computer
        .compute_covering_set(
            &query,
            &file_contents,
            &graph,
            &CoveringSetOptions::default(),
        )
        .unwrap();

    // Should return empty result when file not found
    assert!(result.files.is_empty());
}

#[test]
fn test_covering_set_options_with_max_files() {
    let opts = CoveringSetOptions {
        max_files: Some(10),
        max_depth: Some(2),
        ..Default::default()
    };

    assert_eq!(opts.max_files, Some(10));
    assert_eq!(opts.max_depth, Some(2));
}

#[test]
fn test_extract_regex_matches() {
    let content = "def foo(): pass\ndef bar(): pass";
    let mut references = Vec::new();

    super::CoveringSetComputer::extract_regex_matches(
        content,
        r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
        super::is_common_keyword,
        &mut references,
    ).unwrap();

    // Should extract function names
    assert!(references.contains(&"foo".to_string()) || references.contains(&"bar".to_string()));
}

#[test]
fn test_format_inclusion_reason_direct_dependent() {
    let computer = CoveringSetComputer::new().unwrap();

    let reason = computer.format_inclusion_reason(&InclusionReason::DirectDependent, 1);
    assert!(reason.contains("Direct") || reason.contains("dependent"));
}

#[test]
fn test_format_inclusion_reason_transitive_dependent() {
    let computer = CoveringSetComputer::new().unwrap();

    let reason = computer.format_inclusion_reason(&InclusionReason::TransitiveDependent, 4);
    assert!(reason.contains("distance: 4") || reason.contains("Transitive"));
}

#[test]
fn test_covering_set_granularity_clone() {
    let granularity = CoveringSetGranularity::Entity;
    let cloned = granularity.clone();
    assert_eq!(granularity, cloned);
}

#[test]
fn test_covering_set_options_clone() {
    let opts = CoveringSetOptions::entity_level();
    let cloned = opts.clone();
    assert_eq!(opts.granularity, cloned.granularity);
    assert_eq!(opts.max_depth, cloned.max_depth);
}

#[test]
fn test_default_implementation() {
    let computer = CoveringSetComputer::default();
    // Should be able to create using default
    let _ = computer;
}

#[test]
fn test_is_path_suffix_match() {
    // Exact suffix match at path boundary
    assert!(CoveringSetComputer::is_path_suffix_match("src/lib.rs", "lib.rs"));
    assert!(CoveringSetComputer::is_path_suffix_match("src/utils/lib.rs", "utils/lib.rs"));
    assert!(CoveringSetComputer::is_path_suffix_match("lib.rs", "lib.rs"));

    // Non-match cases
    assert!(!CoveringSetComputer::is_path_suffix_match("src/mylib.rs", "lib.rs"));
    assert!(!CoveringSetComputer::is_path_suffix_match("src/lib.rs", "main.rs"));
}

#[test]
fn test_find_exact_or_suffix_match() {
    let mut file_contents = HashMap::new();
    file_contents.insert("src/lib.rs".to_string(), "".to_string());
    file_contents.insert("src/main.rs".to_string(), "".to_string());

    // Exact match
    let result = CoveringSetComputer::find_exact_or_suffix_match("src/lib.rs", &file_contents);
    assert_eq!(result, Some("src/lib.rs".to_string()));

    // Suffix match
    let result = CoveringSetComputer::find_exact_or_suffix_match("lib.rs", &file_contents);
    assert_eq!(result, Some("src/lib.rs".to_string()));

    // No match
    let result = CoveringSetComputer::find_exact_or_suffix_match("nonexistent.rs", &file_contents);
    assert!(result.is_none());
}

#[test]
fn test_find_substring_match() {
    let mut file_contents = HashMap::new();
    file_contents.insert("src/utils/helpers.rs".to_string(), "".to_string());

    let result = CoveringSetComputer::find_substring_match("helpers", &file_contents);
    assert!(result.is_some());

    let result = CoveringSetComputer::find_substring_match("nonexistent", &file_contents);
    assert!(result.is_none());
}

#[test]
fn test_get_traversal_direction() {
    let computer = CoveringSetComputer::new().unwrap();

    // Both directions
    let mut opts = CoveringSetOptions::default();
    opts.include_dependencies = true;
    opts.include_dependents = true;
    let direction = computer.get_traversal_direction(&opts);
    assert_eq!(direction, TraversalDirection::Both);

    // Dependencies only
    opts.include_dependencies = true;
    opts.include_dependents = false;
    let direction = computer.get_traversal_direction(&opts);
    assert_eq!(direction, TraversalDirection::Dependencies);

    // Dependents only
    opts.include_dependencies = false;
    opts.include_dependents = true;
    let direction = computer.get_traversal_direction(&opts);
    assert_eq!(direction, TraversalDirection::Dependents);

    // Neither (defaults to dependencies)
    opts.include_dependencies = false;
    opts.include_dependents = false;
    let direction = computer.get_traversal_direction(&opts);
    assert_eq!(direction, TraversalDirection::Dependencies);
}

#[test]
fn test_compute_distance() {
    let computer = CoveringSetComputer::new().unwrap();
    let mut graph = DependencyGraph::new();

    graph.add_edge("A".to_string(), "B".to_string()).unwrap();
    graph.add_edge("B".to_string(), "C".to_string()).unwrap();

    let distance = computer.compute_distance("A", "C", &graph);
    assert_eq!(distance, 2);

    let distance = computer.compute_distance("A", "B", &graph);
    assert_eq!(distance, 1);

    let distance = computer.compute_distance("A", "A", &graph);
    assert_eq!(distance, 0);

    // Unreachable
    let distance = computer.compute_distance("C", "A", &graph);
    assert_eq!(distance, 999); // Not reachable in forward direction
}

#[test]
fn test_compute_inclusion_info_direct_dependency() {
    let computer = CoveringSetComputer::new().unwrap();
    let mut graph = DependencyGraph::new();

    graph.add_edge("target.rs".to_string(), "dep.rs".to_string()).unwrap();

    let (reason, distance) = computer.compute_inclusion_info(
        "dep.rs",
        "target.rs",
        &graph,
        &CoveringSetOptions::default(),
    );

    assert_eq!(reason, InclusionReason::DirectDependency);
    assert_eq!(distance, 1);
}

#[test]
fn test_compute_inclusion_info_direct_dependent() {
    let computer = CoveringSetComputer::new().unwrap();
    let mut graph = DependencyGraph::new();

    graph.add_edge("dep.rs".to_string(), "target.rs".to_string()).unwrap();

    let (reason, distance) = computer.compute_inclusion_info(
        "dep.rs",
        "target.rs",
        &graph,
        &CoveringSetOptions::default(),
    );

    assert_eq!(reason, InclusionReason::DirectDependent);
    assert_eq!(distance, 1);
}

#[test]
fn test_apply_limits() {
    let computer = CoveringSetComputer::new().unwrap();
    let mut stats = CoveringSetStatistics {
        files_examined: 10,
        files_selected: 0,
        files_excluded: 0,
        entities_selected: 0,
        max_depth_reached: 0,
        limits_reached: false,
    };

    let mut covering_set = vec![
        CoveringSetFile {
            path: "a.rs".to_string(),
            reason: InclusionReason::TargetFile,
            distance: 0,
            importance: None,
            line_ranges: vec![],
        },
        CoveringSetFile {
            path: "b.rs".to_string(),
            reason: InclusionReason::DirectDependency,
            distance: 1,
            importance: None,
            line_ranges: vec![],
        },
        CoveringSetFile {
            path: "c.rs".to_string(),
            reason: InclusionReason::TransitiveDependency,
            distance: 2,
            importance: None,
            line_ranges: vec![],
        },
    ];

    let opts = CoveringSetOptions {
        max_files: Some(2),
        ..Default::default()
    };

    computer.apply_limits(&mut covering_set, &opts, &mut stats);

    assert_eq!(covering_set.len(), 2);
    assert!(stats.limits_reached);
}

#[test]
fn test_apply_limits_min_importance() {
    let computer = CoveringSetComputer::new().unwrap();
    let mut stats = CoveringSetStatistics {
        files_examined: 10,
        files_selected: 0,
        files_excluded: 0,
        entities_selected: 0,
        max_depth_reached: 0,
        limits_reached: false,
    };

    let mut covering_set = vec![
        CoveringSetFile {
            path: "target.rs".to_string(),
            reason: InclusionReason::TargetFile,
            distance: 0,
            importance: Some(0.3),  // Below threshold but target
            line_ranges: vec![],
        },
        CoveringSetFile {
            path: "high.rs".to_string(),
            reason: InclusionReason::DirectDependency,
            distance: 1,
            importance: Some(0.8),  // Above threshold
            line_ranges: vec![],
        },
        CoveringSetFile {
            path: "low.rs".to_string(),
            reason: InclusionReason::TransitiveDependency,
            distance: 2,
            importance: Some(0.2),  // Below threshold
            line_ranges: vec![],
        },
    ];

    let opts = CoveringSetOptions {
        min_importance: Some(0.5),
        ..Default::default()
    };

    computer.apply_limits(&mut covering_set, &opts, &mut stats);

    // Target is always kept, high is above threshold, low is filtered
    assert_eq!(covering_set.len(), 2);
    assert!(covering_set.iter().any(|f| f.path == "target.rs"));
    assert!(covering_set.iter().any(|f| f.path == "high.rs"));
    assert!(!covering_set.iter().any(|f| f.path == "low.rs"));
}

#[test]
fn test_is_common_keyword_exhaustive() {
    // Control flow
    assert!(is_common_keyword("if"));
    assert!(is_common_keyword("else"));
    assert!(is_common_keyword("for"));
    assert!(is_common_keyword("while"));
    assert!(is_common_keyword("return"));
    assert!(is_common_keyword("break"));
    assert!(is_common_keyword("continue"));

    // Declarations
    assert!(is_common_keyword("def"));
    assert!(is_common_keyword("fn"));
    assert!(is_common_keyword("class"));
    assert!(is_common_keyword("struct"));

    // Other keywords
    assert!(is_common_keyword("import"));
    assert!(is_common_keyword("from"));
    assert!(is_common_keyword("async"));
    assert!(is_common_keyword("await"));

    // Builtins
    assert!(is_common_keyword("print"));
    assert!(is_common_keyword("len"));
    assert!(is_common_keyword("self"));
    assert!(is_common_keyword("this"));

    // Literals
    assert!(is_common_keyword("true"));
    assert!(is_common_keyword("false"));
    assert!(is_common_keyword("None"));
    assert!(is_common_keyword("null"));
}

#[test]
fn test_is_common_type_exhaustive() {
    // Basic types
    assert!(is_common_type("String"));
    assert!(is_common_type("Int"));
    assert!(is_common_type("Float"));
    assert!(is_common_type("Bool"));

    // Collections
    assert!(is_common_type("List"));
    assert!(is_common_type("Vec"));
    assert!(is_common_type("Dict"));
    assert!(is_common_type("HashMap"));
    assert!(is_common_type("Set"));

    // Rust types
    assert!(is_common_type("Option"));
    assert!(is_common_type("Result"));
    assert!(is_common_type("Some"));
    assert!(is_common_type("None"));
    assert!(is_common_type("Ok"));
    assert!(is_common_type("Err"));

    // Other
    assert!(is_common_type("Any"));
    assert!(is_common_type("Promise"));
    assert!(is_common_type("Future"));
}

#[test]
fn test_extract_symbol_references_unknown_language() {
    let computer = CoveringSetComputer::new().unwrap();

    // Unknown extension should return empty
    let refs = computer.extract_symbol_references("some code", "file.unknown").unwrap();
    assert!(refs.is_empty());
}

#[test]
fn test_extract_symbol_references_python() {
    let computer = CoveringSetComputer::new().unwrap();

    let content = r#"
def main():
    helper_func()
    result = MyClass()
    value: CustomType = None
"#;

    let refs = computer.extract_symbol_references(content, "test.py").unwrap();

    // Should extract function calls and type annotations
    assert!(refs.contains(&"main".to_string()) || refs.contains(&"helper_func".to_string()));
}

#[test]
fn test_covering_set_with_graph_dependencies() {
    let mut computer = CoveringSetComputer::new().unwrap();
    let mut graph = DependencyGraph::new();

    // Create dependency: main.py -> utils.py -> helpers.py
    graph.add_edge("main.py".to_string(), "utils.py".to_string()).unwrap();
    graph.add_edge("utils.py".to_string(), "helpers.py".to_string()).unwrap();

    let mut file_contents = HashMap::new();
    file_contents.insert("main.py".to_string(), "def main(): pass".to_string());
    file_contents.insert("utils.py".to_string(), "def util(): pass".to_string());
    file_contents.insert("helpers.py".to_string(), "def helper(): pass".to_string());

    let query = EntityQuery::for_file("main.py");
    let result = computer.compute_covering_set(
        &query,
        &file_contents,
        &graph,
        &CoveringSetOptions::default(),
    ).unwrap();

    // Should include main.py and its dependencies
    assert!(!result.files.is_empty());
    assert!(result.files.iter().any(|f| f.path == "main.py"));
    // With default options (include_dependencies=true), should include utils.py
    assert!(result.files.iter().any(|f| f.path == "utils.py"));
}

#[test]
fn test_covering_set_file_debug() {
    let file = CoveringSetFile {
        path: "test.rs".to_string(),
        reason: InclusionReason::TargetFile,
        distance: 0,
        importance: None,
        line_ranges: vec![],
    };

    let debug_str = format!("{:?}", file);
    assert!(debug_str.contains("test.rs"));
}

#[test]
fn test_covering_set_entity_debug() {
    let entity = CoveringSetEntity {
        file_path: "test.py".to_string(),
        name: "my_func".to_string(),
        entity_type: "function".to_string(),
        content: "def my_func(): pass".to_string(),
        start_line: 1,
        end_line: 1,
        reason: InclusionReason::TargetFile,
        distance: 0,
        references: vec![],
    };

    let debug_str = format!("{:?}", entity);
    assert!(debug_str.contains("my_func"));
}

#[test]
fn test_covering_set_result_debug() {
    let result = CoveringSetResult {
        target_entity: None,
        files: vec![],
        entities: vec![],
        statistics: CoveringSetStatistics {
            files_examined: 0,
            files_selected: 0,
            files_excluded: 0,
            entities_selected: 0,
            max_depth_reached: 0,
            limits_reached: false,
        },
        inclusion_reasons: HashMap::new(),
    };

    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("CoveringSetResult"));
}

#[test]
fn test_line_range_clone() {
    let range = LineRange {
        start_line: 10,
        end_line: 20,
    };
    let cloned = range.clone();
    assert_eq!(range.start_line, cloned.start_line);
    assert_eq!(range.end_line, cloned.end_line);
}

#[test]
fn test_inclusion_reason_clone() {
    let reason = InclusionReason::DirectDependency;
    let cloned = reason.clone();
    assert_eq!(reason, cloned);
}
