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

#[test]
fn test_graph_with_capacity() {
    let graph = DependencyGraph::with_capacity(100);
    assert_eq!(graph.node_count(), 0);
    assert_eq!(graph.edge_count(), 0);
}

#[test]
fn test_add_existing_node() {
    let mut graph = DependencyGraph::new();

    let id1 = graph.add_node("main.py".to_string()).unwrap();
    let id2 = graph.add_node("main.py".to_string()).unwrap();

    // Same internal ID should be returned
    assert_eq!(id1, id2);
    assert_eq!(graph.node_count(), 1);
}

#[test]
fn test_remove_nonexistent_node() {
    let mut graph = DependencyGraph::new();
    let removed = graph.remove_node(&"nonexistent".to_string()).unwrap();
    assert!(!removed);
}

#[test]
fn test_remove_edge() {
    let mut graph = DependencyGraph::new();
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();

    assert!(graph.contains_edge(&"A".to_string(), &"B".to_string()));

    let removed = graph.remove_edge(&"A".to_string(), &"B".to_string()).unwrap();
    assert!(removed);
    assert!(!graph.contains_edge(&"A".to_string(), &"B".to_string()));

    // Nodes should still exist
    assert!(graph.contains_node(&"A".to_string()));
    assert!(graph.contains_node(&"B".to_string()));
}

#[test]
fn test_remove_edge_nonexistent() {
    let mut graph = DependencyGraph::new();
    graph.add_node("A".to_string()).unwrap();
    graph.add_node("B".to_string()).unwrap();

    // No edge between A and B
    let removed = graph.remove_edge(&"A".to_string(), &"B".to_string()).unwrap();
    assert!(!removed);

    // Nonexistent nodes
    let removed = graph.remove_edge(&"X".to_string(), &"Y".to_string()).unwrap();
    assert!(!removed);
}

#[test]
fn test_contains_edge_nonexistent_nodes() {
    let graph = DependencyGraph::new();
    assert!(!graph.contains_edge(&"A".to_string(), &"B".to_string()));
}

#[test]
fn test_degree_nonexistent_node() {
    let graph = DependencyGraph::new();
    assert_eq!(graph.in_degree(&"nonexistent".to_string()), 0);
    assert_eq!(graph.out_degree(&"nonexistent".to_string()), 0);
    assert_eq!(graph.degree(&"nonexistent".to_string()), 0);
}

#[test]
fn test_outgoing_neighbors() {
    let mut graph = DependencyGraph::new();
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();
    graph.add_edge("A".to_string(), "C".to_string()).unwrap();

    let neighbors = graph.outgoing_neighbors(&"A".to_string()).unwrap();
    assert_eq!(neighbors.len(), 2);
    assert!(neighbors.contains(&&"B".to_string()));
    assert!(neighbors.contains(&&"C".to_string()));

    // Nonexistent node
    assert!(graph.outgoing_neighbors(&"X".to_string()).is_none());
}

#[test]
fn test_incoming_neighbors() {
    let mut graph = DependencyGraph::new();
    graph.add_edge("A".to_string(), "C".to_string()).unwrap();
    graph.add_edge("B".to_string(), "C".to_string()).unwrap();

    let neighbors = graph.incoming_neighbors(&"C".to_string()).unwrap();
    assert_eq!(neighbors.len(), 2);
    assert!(neighbors.contains(&&"A".to_string()));
    assert!(neighbors.contains(&&"B".to_string()));

    // Nonexistent node
    assert!(graph.incoming_neighbors(&"X".to_string()).is_none());
}

#[test]
fn test_all_neighbors() {
    let mut graph = DependencyGraph::new();
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();
    graph.add_edge("B".to_string(), "C".to_string()).unwrap();

    // B has both incoming (A) and outgoing (C) neighbors
    let neighbors = graph.all_neighbors(&"B".to_string());
    assert_eq!(neighbors.len(), 2);
    assert!(neighbors.contains(&"A".to_string()));
    assert!(neighbors.contains(&"C".to_string()));

    // Nonexistent node returns empty set
    let neighbors = graph.all_neighbors(&"X".to_string());
    assert!(neighbors.is_empty());
}

#[test]
fn test_transitive_dependencies_nonexistent() {
    let graph = DependencyGraph::new();
    let deps = graph.transitive_dependencies(&"nonexistent".to_string(), None);
    assert!(deps.is_empty());
}

#[test]
fn test_transitive_dependents_nonexistent() {
    let graph = DependencyGraph::new();
    let deps = graph.transitive_dependents(&"nonexistent".to_string(), None);
    assert!(deps.is_empty());
}

#[test]
fn test_get_degree_info() {
    let mut graph = DependencyGraph::new();
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();
    graph.add_edge("B".to_string(), "C".to_string()).unwrap();

    let info = graph.get_degree_info(&"B".to_string()).unwrap();
    assert_eq!(info.node_id, "B".to_string());
    assert_eq!(info.in_degree, 1);
    assert_eq!(info.out_degree, 1);
    assert_eq!(info.total_degree, 2);

    // Nonexistent node
    assert!(graph.get_degree_info(&"X".to_string()).is_none());
}

#[test]
fn test_set_node_metadata() {
    let mut graph = DependencyGraph::new();
    graph.add_node("main.py".to_string()).unwrap();

    let mut new_meta = NodeMetadata::new("main.py".to_string()).with_size(2048);
    new_meta.is_test = true;

    graph.set_node_metadata("main.py".to_string(), new_meta).unwrap();

    let retrieved = graph.node_metadata(&"main.py".to_string()).unwrap();
    assert_eq!(retrieved.size_bytes, 2048);
    assert!(retrieved.is_test);
}

#[test]
fn test_set_metadata_nonexistent_node() {
    let mut graph = DependencyGraph::new();
    let meta = NodeMetadata::new("test.py".to_string());

    let result = graph.set_node_metadata("nonexistent.py".to_string(), meta);
    assert!(result.is_err());
}

#[test]
fn test_entrypoint_nodes() {
    let mut graph = DependencyGraph::new();
    graph.add_node("main.py".to_string()).unwrap(); // Auto-detected as entrypoint
    graph.add_node("utils.py".to_string()).unwrap();
    graph.add_node("index.js".to_string()).unwrap(); // Auto-detected as entrypoint

    let entrypoints = graph.entrypoint_nodes();
    assert!(entrypoints.len() >= 2);
    assert!(entrypoints.contains(&&"main.py".to_string()));
    assert!(entrypoints.contains(&&"index.js".to_string()));
}

#[test]
fn test_test_nodes() {
    let mut graph = DependencyGraph::new();

    // Add a test file - test_main.py auto-detects as test file due to "test_" prefix
    let mut meta = NodeMetadata::new("test_main.py".to_string());
    meta.is_test = true;
    graph.add_node_with_metadata("test_main.py".to_string(), meta).unwrap();

    // Add a normal file
    graph.add_node("main.py".to_string()).unwrap();

    let test_nodes = graph.test_nodes();
    assert_eq!(test_nodes.len(), 1);
    assert!(test_nodes.contains(&&"test_main.py".to_string()));
}

#[test]
fn test_nodes_iterator() {
    let mut graph = DependencyGraph::new();
    graph.add_node("A".to_string()).unwrap();
    graph.add_node("B".to_string()).unwrap();
    graph.add_node("C".to_string()).unwrap();

    let nodes: Vec<_> = graph.nodes().collect();
    assert_eq!(nodes.len(), 3);
}

#[test]
fn test_edges_iterator() {
    let mut graph = DependencyGraph::new();
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();
    graph.add_edge("B".to_string(), "C".to_string()).unwrap();

    let edges: Vec<_> = graph.edges().collect();
    assert_eq!(edges.len(), 2);

    let edge_set: std::collections::HashSet<_> = edges.iter().collect();
    assert!(edge_set.contains(&("A".to_string(), "B".to_string())));
    assert!(edge_set.contains(&("B".to_string(), "C".to_string())));
}

#[test]
fn test_is_strongly_connected_empty() {
    let graph = DependencyGraph::new();
    assert!(graph.is_strongly_connected()); // Empty graph is trivially strongly connected
}

#[test]
fn test_is_strongly_connected_yes() {
    let mut graph = DependencyGraph::new();
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();
    graph.add_edge("B".to_string(), "A".to_string()).unwrap();

    // Both nodes have in and out edges
    assert!(graph.is_strongly_connected());
}

#[test]
fn test_is_strongly_connected_no() {
    let mut graph = DependencyGraph::new();
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();

    // A has only out edges, B has only in edges
    assert!(!graph.is_strongly_connected());
}

#[test]
fn test_estimate_scc_empty() {
    let graph = DependencyGraph::new();
    assert_eq!(graph.estimate_scc_count(), 0);
}

#[test]
fn test_default_implementation() {
    let graph = DependencyGraph::default();
    assert_eq!(graph.node_count(), 0);
    assert_eq!(graph.edge_count(), 0);
}

#[test]
fn test_internal_apis() {
    let mut graph = DependencyGraph::new();
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();

    // Test internal ID operations
    let id_a = graph.get_internal_id(&"A".to_string()).unwrap();
    let id_b = graph.get_internal_id(&"B".to_string()).unwrap();
    assert_ne!(id_a, id_b);

    // Test path retrieval
    let path_a = graph.get_path(id_a).unwrap();
    assert_eq!(path_a, &"A".to_string());

    // Test incoming neighbors by ID
    let incoming = graph.incoming_neighbors_by_id(id_b).unwrap();
    assert!(incoming.contains(&id_a));

    // Test out degree by ID
    assert_eq!(graph.out_degree_by_id(id_a), 1);
    assert_eq!(graph.out_degree_by_id(id_b), 0);

    // Test internal node count
    assert_eq!(graph.internal_node_count(), 2);

    // Test internal nodes iterator
    let internal_nodes: Vec<_> = graph.internal_nodes().collect();
    assert_eq!(internal_nodes.len(), 2);
}

#[test]
fn test_internal_id_nonexistent() {
    let graph = DependencyGraph::new();
    assert!(graph.get_internal_id(&"nonexistent".to_string()).is_none());
}

#[test]
fn test_compute_closure_dependents() {
    let mut graph = DependencyGraph::new();
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();
    graph.add_edge("B".to_string(), "C".to_string()).unwrap();

    // Closure of C in dependents direction should include B and A
    let closure = graph.compute_closure(
        &["C".to_string()],
        TraversalDirection::Dependents,
        None,
    );
    assert_eq!(closure.len(), 3);
    assert!(closure.contains(&"A".to_string()));
    assert!(closure.contains(&"B".to_string()));
    assert!(closure.contains(&"C".to_string()));
}

#[test]
fn test_remove_node_with_edges() {
    let mut graph = DependencyGraph::new();
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();
    graph.add_edge("B".to_string(), "C".to_string()).unwrap();

    // Remove B (has both incoming and outgoing edges)
    let removed = graph.remove_node(&"B".to_string()).unwrap();
    assert!(removed);

    // B should be gone
    assert!(!graph.contains_node(&"B".to_string()));

    // Edges involving B should be gone
    assert!(!graph.contains_edge(&"A".to_string(), &"B".to_string()));
    assert!(!graph.contains_edge(&"B".to_string(), &"C".to_string()));

    // A and C should still exist
    assert!(graph.contains_node(&"A".to_string()));
    assert!(graph.contains_node(&"C".to_string()));
}

#[test]
fn test_clone_graph() {
    let mut graph = DependencyGraph::new();
    graph.add_edge("A".to_string(), "B".to_string()).unwrap();

    let cloned = graph.clone();
    assert_eq!(cloned.node_count(), 2);
    assert_eq!(cloned.edge_count(), 1);
    assert!(cloned.contains_edge(&"A".to_string(), &"B".to_string()));
}

#[test]
fn test_debug_implementation() {
    let graph = DependencyGraph::new();
    let debug_str = format!("{:?}", graph);
    assert!(debug_str.contains("DependencyGraph"));
}

#[test]
fn test_remove_edge_to_nonexistent() {
    let mut graph = DependencyGraph::new();
    // Add only node A
    graph.add_node("A".to_string()).unwrap();

    // Try to remove edge where to_node doesn't exist - exercises line 191
    let removed = graph.remove_edge(&"A".to_string(), &"nonexistent".to_string()).unwrap();
    assert!(!removed);
}

#[test]
fn test_node_metadata_nonexistent() {
    let graph = DependencyGraph::new();
    // Exercises line 461 - None branch for node_metadata
    assert!(graph.node_metadata(&"nonexistent".to_string()).is_none());
}

#[test]
fn test_pagerank_iterator_no_incoming() {
    let mut graph = DependencyGraph::new();
    // Create a node with no incoming edges
    graph.add_node("isolated".to_string()).unwrap();

    // Exercises line 541 - Some(Vec::new()) for nodes with no reverse edges
    let pagerank_data: Vec<_> = graph.pagerank_iterator().collect();
    assert_eq!(pagerank_data.len(), 1);

    let (node, incoming) = &pagerank_data[0];
    assert_eq!(*node, &"isolated".to_string());
    assert!(incoming.as_ref().unwrap().is_empty());
}
