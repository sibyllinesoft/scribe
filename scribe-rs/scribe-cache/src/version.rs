//! Cache version management
//!
//! The cache version is independent of the scribe version.
//! Increment CACHE_VERSION when:
//! - Changing the storage format
//! - Changing hash algorithms
//! - Changing what data is cached
//! - Any breaking change to cache structure

/// Current cache format version
///
/// Increment this when making breaking changes to the cache format.
/// The cache will be invalidated when version mismatches.
pub const CACHE_VERSION: u32 = 1;

/// Version history for documentation
/// - v1: Initial version with redb storage, xxh3 hashing
#[allow(dead_code)]
const VERSION_HISTORY: &[&str] =
    &["v1: Initial version - redb storage, xxh3 hashing, bincode serialization"];
