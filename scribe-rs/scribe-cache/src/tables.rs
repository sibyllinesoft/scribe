//! Database table definitions for redb

use redb::TableDefinition;

/// Table for file data: content_hash (u64) -> serialized CachedFileData
pub const FILE_DATA: TableDefinition<u64, &[u8]> = TableDefinition::new("file_data");

/// Table for path to hash mapping: path_bytes -> content_hash (u64)
pub const PATH_HASHES: TableDefinition<&[u8], u64> = TableDefinition::new("path_hashes");

/// Table for path to mtime mapping: path_bytes -> mtime_nanos (u64)
pub const PATH_MTIMES: TableDefinition<&[u8], u64> = TableDefinition::new("path_mtimes");

/// Table for graph data: key_bytes -> serialized data
pub const GRAPH_DATA: TableDefinition<&[u8], &[u8]> = TableDefinition::new("graph_data");

/// Table for metadata: key_str -> value_bytes
pub const METADATA: TableDefinition<&str, &[u8]> = TableDefinition::new("metadata");

/// Keys used in the metadata table
pub mod meta_keys {
    pub const VERSION: &str = "version";
    pub const REPO_ID: &str = "repo_id";
    pub const CREATED_AT: &str = "created_at";
    pub const UPDATED_AT: &str = "updated_at";
}

/// Keys used in the graph data table
pub mod graph_keys {
    pub const PAGERANK: &[u8] = b"pagerank";
    pub const CENTRALITY: &[u8] = b"centrality";
    pub const EDGES_HASH: &[u8] = b"edges_hash";
}
