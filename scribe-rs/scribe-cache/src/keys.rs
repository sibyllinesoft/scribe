//! Cache key generation and content hashing

use std::path::Path;
use xxhash_rust::xxh3::xxh3_64;

/// A content hash representing file contents
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash(pub u64);

impl ContentHash {
    /// Create a hash from file contents
    #[inline]
    pub fn from_content(content: &[u8]) -> Self {
        Self(xxh3_64(content))
    }

    /// Create a hash from a string
    #[inline]
    pub fn from_str(s: &str) -> Self {
        Self(xxh3_64(s.as_bytes()))
    }

    /// Get the raw u64 value
    #[inline]
    pub fn as_u64(&self) -> u64 {
        self.0
    }

    /// Convert to bytes for storage
    #[inline]
    pub fn to_bytes(&self) -> [u8; 8] {
        self.0.to_le_bytes()
    }

    /// Create from bytes
    #[inline]
    pub fn from_bytes(bytes: [u8; 8]) -> Self {
        Self(u64::from_le_bytes(bytes))
    }
}

impl From<u64> for ContentHash {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

impl From<ContentHash> for u64 {
    fn from(h: ContentHash) -> Self {
        h.0
    }
}

/// Generate a stable repository identifier
/// Uses git remote URL if available, otherwise canonical path
pub fn repo_identifier(repo_path: &Path) -> String {
    // Try git remote URL first
    #[cfg(feature = "git")]
    if let Ok(id) = git_based_identifier(repo_path) {
        return id;
    }

    // Fall back to path-based identifier
    path_based_identifier(repo_path)
}

#[cfg(feature = "git")]
fn git_based_identifier(repo_path: &Path) -> Result<String, git2::Error> {
    let repo = git2::Repository::discover(repo_path)?;

    // Try to get origin remote URL
    if let Ok(remote) = repo.find_remote("origin") {
        if let Some(url) = remote.url() {
            return Ok(format!("{:016x}", xxh3_64(url.as_bytes())));
        }
    }

    // Try any remote
    let remotes = repo.remotes()?;
    for remote_name in remotes.iter().flatten() {
        if let Ok(remote) = repo.find_remote(remote_name) {
            if let Some(url) = remote.url() {
                return Ok(format!("{:016x}", xxh3_64(url.as_bytes())));
            }
        }
    }

    Err(git2::Error::from_str("No remote URL found"))
}

fn path_based_identifier(repo_path: &Path) -> String {
    let canonical = repo_path
        .canonicalize()
        .unwrap_or_else(|_| repo_path.to_path_buf());
    format!("{:016x}", xxh3_64(canonical.to_string_lossy().as_bytes()))
}

/// Hash a collection of edges for graph cache invalidation
pub fn hash_edges(edges: &[(String, String)]) -> u64 {
    let mut hasher_input = Vec::new();
    for (from, to) in edges {
        hasher_input.extend_from_slice(from.as_bytes());
        hasher_input.push(0);
        hasher_input.extend_from_slice(to.as_bytes());
        hasher_input.push(0);
    }
    xxh3_64(&hasher_input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_hash() {
        let content = b"hello world";
        let hash = ContentHash::from_content(content);

        // Same content should produce same hash
        let hash2 = ContentHash::from_content(content);
        assert_eq!(hash, hash2);

        // Different content should produce different hash
        let hash3 = ContentHash::from_content(b"different content");
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_hash_roundtrip() {
        let hash = ContentHash::from_content(b"test");
        let bytes = hash.to_bytes();
        let recovered = ContentHash::from_bytes(bytes);
        assert_eq!(hash, recovered);
    }
}
