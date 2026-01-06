//! Tests for the dependency graph module.

use super::*;

#[test]
fn test_graph_creation() {
    let graph = DependencyGraph::new();
    assert_eq!(graph.node_count(), 0);
    assert_eq!(graph.edge_count(), 0);
}

#[test]
fn test_node_operations() {
    let mut graph = DependencyGraph::new();

    // Add nodes
    graph.add_node("main.py".to_string()).unwrap();
    graph.add_node("utils.py".to_string()).unwrap();

    assert_eq!(graph.node_count(), 2);
    assert!(graph.contains_node(&"main.py".to_string()));
    assert!(graph.contains_node(&"utils.py".to_string()));

    // Remove node
    let removed = graph.remove_node(&"utils.py".to_string()).unwrap();
    assert!(removed);
    assert_eq!(graph.node_count(), 1);
    assert!(!graph.contains_node(&"utils.py".to_string()));
}

#[test]
fn test_edge_operations() {
    let mut graph = DependencyGraph::new();

    // Add edge (automatically creates nodes)
    graph
        .add_edge("main.py".to_string(), "utils.py".to_string())
        .unwrap();

    assert_eq!(graph.node_count(), 2);
    assert_eq!(graph.edge_count(), 1);
    assert!(graph.contains_edge(&"main.py".to_string(), &"utils.py".to_string()));

    // Check degrees
    assert_eq!(graph.out_degree(&"main.py".to_string()), 1);
    assert_eq!(graph.in_degree(&"utils.py".to_string()), 1);
    assert_eq!(graph.in_degree(&"main.py".to_string()), 0);
    assert_eq!(graph.out_degree(&"utils.py".to_string()), 0);
}

#[test]
fn test_multiple_edges() {
    let mut graph = DependencyGraph::new();

    let edges = vec![
        ("main.py".to_string(), "utils.py".to_string()),
        ("main.py".to_string(), "config.py".to_string()),
        ("utils.py".to_string(), "config.py".to_string()),
    ];

    graph.add_edges(&edges).unwrap();

    assert_eq!(graph.node_count(), 3);
    assert_eq!(graph.edge_count(), 3);

    // main.py should have out-degree 2
    assert_eq!(graph.out_degree(&"main.py".to_string()), 2);

    // config.py should have in-degree 2
    assert_eq!(graph.in_degree(&"config.py".to_string()), 2);
}

#[test]
fn test_node_metadata() {
    let mut graph = DependencyGraph::new();

    let metadata = NodeMetadata::new("main.py".to_string()).with_size(1024);
    graph
        .add_node_with_metadata("main.py".to_string(), metadata)
        .unwrap();

    let retrieved = graph.node_metadata(&"main.py".to_string()).unwrap();
    assert_eq!(retrieved.file_path, "main.py");
    assert_eq!(retrieved.language, Some("python".to_string()));
    assert!(retrieved.is_entrypoint);
    assert!(!retrieved.is_test);
    assert_eq!(retrieved.size_bytes, 1024);
}

#[test]
fn test_pagerank_iterator() {
    let mut graph = DependencyGraph::new();

    // Build a small graph: A -> B -> C, C -> A (creates cycle)
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();
    graph.add_edge("B".to_string(), "C".to_string()).unwrap();
    graph.add_edge("C".to_string(), "A".to_string()).unwrap();

    let pagerank_data: Vec<_> = graph.pagerank_iterator().collect();
    assert_eq!(pagerank_data.len(), 3);

    // Each node should have incoming edges (reverse edges)
    for (_node, reverse_edges) in pagerank_data {
        assert!(reverse_edges.is_some());
        assert!(!reverse_edges.unwrap().is_empty());
    }
}

#[test]
fn test_dangling_nodes() {
    let mut graph = DependencyGraph::new();

    // A -> B, C is isolated, D has no outgoing edges
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();
    graph.add_node("C".to_string()).unwrap();
    graph.add_edge("B".to_string(), "D".to_string()).unwrap();

    let dangling = graph.dangling_nodes();

    // C and D should be dangling (no outgoing edges)
    assert_eq!(dangling.len(), 2);
    assert!(dangling.contains(&&"C".to_string()));
    assert!(dangling.contains(&&"D".to_string()));
}

#[test]
fn test_concurrent_graph() {
    let mut graph = DependencyGraph::new();
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();
    graph.add_edge("B".to_string(), "C".to_string()).unwrap();

    let concurrent = graph.into_concurrent();

    // Test concurrent operations
    assert_eq!(concurrent.in_degree(&"B".to_string()), 1);
    assert_eq!(concurrent.out_degree(&"B".to_string()), 1);

    // Add node concurrently
    concurrent.add_node("D".to_string()).unwrap();

    // Convert back to sequential
    let sequential = concurrent.into_sequential();
    assert_eq!(sequential.node_count(), 4); // A, B, C, D
}

#[test]
fn test_scc_estimation() {
    let mut graph = DependencyGraph::new();

    // Create a graph with potential cycles: A <-> B, C -> D
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();
    graph.add_edge("B".to_string(), "A".to_string()).unwrap();
    graph.add_edge("C".to_string(), "D".to_string()).unwrap();
    graph.add_node("E".to_string()).unwrap(); // Isolated

    let scc_count = graph.estimate_scc_count();

    // Should estimate: 1 SCC (A,B have both in/out), plus isolated/chain nodes (C,D,E)
    assert!(scc_count >= 3); // At least C, D, E as separate components
}

#[test]
fn test_nodes_by_language() {
    let mut graph = DependencyGraph::new();

    graph.add_node("main.py".to_string()).unwrap();
    graph.add_node("utils.py".to_string()).unwrap();
    graph.add_node("app.js".to_string()).unwrap();
    graph.add_node("lib.rs".to_string()).unwrap();

    let python_nodes = graph.nodes_by_language("python");
    let js_nodes = graph.nodes_by_language("javascript");
    let rust_nodes = graph.nodes_by_language("rust");

    assert_eq!(python_nodes.len(), 2);
    assert_eq!(js_nodes.len(), 1);
    assert_eq!(rust_nodes.len(), 1);

    assert!(python_nodes.contains(&&"main.py".to_string()));
    assert!(python_nodes.contains(&&"utils.py".to_string()));
}

#[test]
fn test_transitive_dependencies() {
    let mut graph = DependencyGraph::new();

    // Create dependency chain: A -> B -> C -> D
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();
    graph.add_edge("B".to_string(), "C".to_string()).unwrap();
    graph.add_edge("C".to_string(), "D".to_string()).unwrap();

    // A should transitively depend on B, C, D
    let deps = graph.transitive_dependencies(&"A".to_string(), None);
    assert_eq!(deps.len(), 3);
    assert!(deps.contains(&"B".to_string()));
    assert!(deps.contains(&"C".to_string()));
    assert!(deps.contains(&"D".to_string()));

    // B should transitively depend on C, D
    let deps = graph.transitive_dependencies(&"B".to_string(), None);
    assert_eq!(deps.len(), 2);
    assert!(deps.contains(&"C".to_string()));
    assert!(deps.contains(&"D".to_string()));

    // D has no dependencies
    let deps = graph.transitive_dependencies(&"D".to_string(), None);
    assert_eq!(deps.len(), 0);
}

#[test]
fn test_transitive_dependencies_with_depth_limit() {
    let mut graph = DependencyGraph::new();

    // Create dependency chain: A -> B -> C -> D
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();
    graph.add_edge("B".to_string(), "C".to_string()).unwrap();
    graph.add_edge("C".to_string(), "D".to_string()).unwrap();

    // Limit to depth 1: only direct dependencies
    let deps = graph.transitive_dependencies(&"A".to_string(), Some(1));
    assert_eq!(deps.len(), 1);
    assert!(deps.contains(&"B".to_string()));

    // Limit to depth 2
    let deps = graph.transitive_dependencies(&"A".to_string(), Some(2));
    assert_eq!(deps.len(), 2);
    assert!(deps.contains(&"B".to_string()));
    assert!(deps.contains(&"C".to_string()));
}

#[test]
fn test_transitive_dependents() {
    let mut graph = DependencyGraph::new();

    // Create dependency chain: A -> B -> C -> D
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();
    graph.add_edge("B".to_string(), "C".to_string()).unwrap();
    graph.add_edge("C".to_string(), "D".to_string()).unwrap();

    // D should have transitive dependents: C, B, A
    let dependents = graph.transitive_dependents(&"D".to_string(), None);
    assert_eq!(dependents.len(), 3);
    assert!(dependents.contains(&"C".to_string()));
    assert!(dependents.contains(&"B".to_string()));
    assert!(dependents.contains(&"A".to_string()));

    // C should have transitive dependents: B, A
    let dependents = graph.transitive_dependents(&"C".to_string(), None);
    assert_eq!(dependents.len(), 2);
    assert!(dependents.contains(&"B".to_string()));
    assert!(dependents.contains(&"A".to_string()));

    // A has no dependents
    let dependents = graph.transitive_dependents(&"A".to_string(), None);
    assert_eq!(dependents.len(), 0);
}

#[test]
fn test_compute_closure_dependencies() {
    let mut graph = DependencyGraph::new();

    // Create a diamond dependency: A -> B, A -> C, B -> D, C -> D
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();
    graph.add_edge("A".to_string(), "C".to_string()).unwrap();
    graph.add_edge("B".to_string(), "D".to_string()).unwrap();
    graph.add_edge("C".to_string(), "D".to_string()).unwrap();

    // Closure of A should include A, B, C, D
    let closure = graph.compute_closure(
        &["A".to_string()],
        TraversalDirection::Dependencies,
        None,
    );
    assert_eq!(closure.len(), 4);
    assert!(closure.contains(&"A".to_string()));
    assert!(closure.contains(&"B".to_string()));
    assert!(closure.contains(&"C".to_string()));
    assert!(closure.contains(&"D".to_string()));
}

#[test]
fn test_compute_closure_both_directions() {
    let mut graph = DependencyGraph::new();

    // Create: A -> B -> C
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();
    graph.add_edge("B".to_string(), "C".to_string()).unwrap();

    // Closure of B in both directions should include A, B, C
    let closure =
        graph.compute_closure(&["B".to_string()], TraversalDirection::Both, None);
    assert_eq!(closure.len(), 3);
    assert!(closure.contains(&"A".to_string()));
    assert!(closure.contains(&"B".to_string()));
    assert!(closure.contains(&"C".to_string()));
}

#[test]
fn test_compute_closure_multiple_seeds() {
    let mut graph = DependencyGraph::new();

    // Create two separate chains: A -> B and C -> D
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();
    graph.add_edge("C".to_string(), "D".to_string()).unwrap();

    // Closure of [A, C] should include all four nodes
    let closure = graph.compute_closure(
        &["A".to_string(), "C".to_string()],
        TraversalDirection::Dependencies,
        None,
    );
    assert_eq!(closure.len(), 4);
    assert!(closure.contains(&"A".to_string()));
    assert!(closure.contains(&"B".to_string()));
    assert!(closure.contains(&"C".to_string()));
    assert!(closure.contains(&"D".to_string()));
}
