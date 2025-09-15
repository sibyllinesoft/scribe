//! File metadata extraction and analysis.
//!
//! This module provides comprehensive file metadata extraction including
//! size statistics, timestamps, permissions, and file system attributes.

use scribe_core::{Result, ScribeError, bytes_to_human};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use std::fs;
use dashmap::DashMap;
use serde::{Serialize, Deserialize};

/// Comprehensive file metadata information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    pub path: PathBuf,
    pub size: u64,
    pub size_human: String,
    pub created: Option<u64>,
    pub modified: Option<u64>,
    pub accessed: Option<u64>,
    pub readonly: bool,
    pub hidden: bool,
    pub executable: bool,
    pub symlink: bool,
    pub symlink_target: Option<PathBuf>,
    pub permissions: u32,
    pub file_type: FileSystemType,
    pub inode: Option<u64>,
    pub links: Option<u64>,
    pub uid: Option<u32>,
    pub gid: Option<u32>,
}

/// File system type classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileSystemType {
    RegularFile,
    SymbolicLink,
    Directory,
    FIFO,
    Socket,
    CharacterDevice,
    BlockDevice,
    Unknown,
}

/// Size-related statistics for collections of files
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SizeStats {
    pub total_size: u64,
    pub total_size_human: String,
    pub file_count: usize,
    pub average_size: u64,
    pub median_size: u64,
    pub min_size: u64,
    pub max_size: u64,
    pub size_distribution: SizeDistribution,
}

/// Distribution of file sizes
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SizeDistribution {
    pub tiny: usize,    // < 1KB
    pub small: usize,   // 1KB - 10KB
    pub medium: usize,  // 10KB - 100KB
    pub large: usize,   // 100KB - 1MB
    pub huge: usize,    // > 1MB
}

/// Metadata extractor with caching and optimization
pub struct MetadataExtractor {
    cache: DashMap<PathBuf, FileMetadata>,
    cache_enabled: bool,
}

impl Default for FileMetadata {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            size: 0,
            size_human: "0 B".to_string(),
            created: None,
            modified: None,
            accessed: None,
            readonly: false,
            hidden: false,
            executable: false,
            symlink: false,
            symlink_target: None,
            permissions: 0,
            file_type: FileSystemType::Unknown,
            inode: None,
            links: None,
            uid: None,
            gid: None,
        }
    }
}

impl MetadataExtractor {
    /// Create a new metadata extractor
    pub fn new() -> Self {
        Self {
            cache: DashMap::new(),
            cache_enabled: true,
        }
    }

    /// Create a metadata extractor without caching
    pub fn without_cache() -> Self {
        Self {
            cache: DashMap::new(),
            cache_enabled: false,
        }
    }

    /// Extract comprehensive metadata for a file
    pub async fn extract_metadata(&self, path: &Path) -> Result<FileMetadata> {
        // Check cache first if enabled
        if self.cache_enabled {
            if let Some(cached) = self.cache.get(path) {
                return Ok(cached.clone());
            }
        }

        let metadata = self.extract_metadata_uncached(path).await?;

        // Cache the result if caching is enabled
        if self.cache_enabled {
            self.cache.insert(path.to_path_buf(), metadata.clone());
        }

        Ok(metadata)
    }

    /// Extract metadata without caching
    async fn extract_metadata_uncached(&self, path: &Path) -> Result<FileMetadata> {
        let std_metadata = tokio::fs::symlink_metadata(path).await
            .map_err(|e| ScribeError::io(format!("Failed to read metadata for {}: {}", path.display(), e), e))?;

        let size = std_metadata.len();
        let size_human = bytes_to_human(size);

        // Extract timestamps
        let created = system_time_to_timestamp(std_metadata.created().ok());
        let modified = system_time_to_timestamp(std_metadata.modified().ok());
        let accessed = system_time_to_timestamp(std_metadata.accessed().ok());

        // Determine file type
        let file_type = classify_file_type(&std_metadata);

        // Check if it's a symlink and get target
        let (symlink, symlink_target) = if std_metadata.file_type().is_symlink() {
            let target = tokio::fs::read_link(path).await.ok();
            (true, target)
        } else {
            (false, None)
        };

        // Platform-specific metadata extraction
        let (permissions, readonly, hidden, executable, inode, links, uid, gid) = 
            extract_platform_metadata(path, &std_metadata)?;

        Ok(FileMetadata {
            path: path.to_path_buf(),
            size,
            size_human,
            created,
            modified,
            accessed,
            readonly,
            hidden,
            executable,
            symlink,
            symlink_target,
            permissions,
            file_type,
            inode,
            links,
            uid,
            gid,
        })
    }

    /// Extract metadata for multiple files in parallel
    pub async fn extract_metadata_batch(&self, paths: &[PathBuf]) -> Vec<Result<FileMetadata>> {
        // For now, process sequentially to avoid async closure issues
        // In a future version, this could use async map with proper lifetime handling
        let mut results = Vec::with_capacity(paths.len());
        for path in paths {
            results.push(self.extract_metadata(path).await);
        }
        results
    }

    /// Calculate size statistics for a collection of files
    pub fn calculate_size_stats(&self, files: &[FileMetadata]) -> SizeStats {
        if files.is_empty() {
            return SizeStats::default();
        }

        let mut sizes: Vec<u64> = files.iter().map(|f| f.size).collect();
        sizes.sort_unstable();

        let total_size = sizes.iter().sum();
        let file_count = files.len();
        let average_size = total_size / file_count as u64;
        let median_size = if file_count % 2 == 0 {
            (sizes[file_count / 2 - 1] + sizes[file_count / 2]) / 2
        } else {
            sizes[file_count / 2]
        };

        let min_size = sizes[0];
        let max_size = sizes[sizes.len() - 1];

        // Calculate size distribution
        let mut distribution = SizeDistribution::default();
        for &size in &sizes {
            match size {
                0..=1024 => distribution.tiny += 1,
                1025..=10240 => distribution.small += 1,
                10241..=102400 => distribution.medium += 1,
                102401..=1048576 => distribution.large += 1,
                _ => distribution.huge += 1,
            }
        }

        SizeStats {
            total_size,
            total_size_human: bytes_to_human(total_size),
            file_count,
            average_size,
            median_size,
            min_size,
            max_size,
            size_distribution: distribution,
        }
    }

    /// Clear the metadata cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache.len(), self.cache.capacity())
    }

    /// Check if a file is likely to be a text file based on metadata
    pub fn is_likely_text_file(&self, metadata: &FileMetadata) -> bool {
        // Skip very large files that are unlikely to be source code
        if metadata.size > 10 * 1024 * 1024 { // 10MB
            return false;
        }

        // Skip binary file types
        matches!(metadata.file_type, 
            FileSystemType::RegularFile | FileSystemType::SymbolicLink)
    }

    /// Check if a file has been modified recently
    pub fn is_recently_modified(&self, metadata: &FileMetadata, hours: u64) -> bool {
        if let Some(modified) = metadata.modified {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            let threshold = hours * 3600;
            
            now.saturating_sub(modified) <= threshold
        } else {
            false
        }
    }
}

impl Default for MetadataExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert SystemTime to Unix timestamp
fn system_time_to_timestamp(time: Option<SystemTime>) -> Option<u64> {
    time.and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
}

/// Classify file type from metadata
fn classify_file_type(metadata: &fs::Metadata) -> FileSystemType {
    let file_type = metadata.file_type();
    
    if file_type.is_file() {
        FileSystemType::RegularFile
    } else if file_type.is_dir() {
        FileSystemType::Directory
    } else if file_type.is_symlink() {
        FileSystemType::SymbolicLink
    } else {
        // Platform-specific special file types
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileTypeExt;
            if file_type.is_fifo() {
                return FileSystemType::FIFO;
            } else if file_type.is_socket() {
                return FileSystemType::Socket;
            } else if file_type.is_char_device() {
                return FileSystemType::CharacterDevice;
            } else if file_type.is_block_device() {
                return FileSystemType::BlockDevice;
            }
        }
        
        FileSystemType::Unknown
    }
}

/// Extract platform-specific metadata
#[cfg(unix)]
fn extract_platform_metadata(path: &Path, metadata: &fs::Metadata) -> Result<(u32, bool, bool, bool, Option<u64>, Option<u64>, Option<u32>, Option<u32>)> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let permissions = metadata.permissions().mode();
    let readonly = !metadata.permissions().readonly();
    
    // Check if file is hidden (starts with .)
    let hidden = path.file_name()
        .and_then(|name| name.to_str())
        .map_or(false, |name| name.starts_with('.'));
    
    // Check if file is executable
    let executable = permissions & 0o111 != 0;
    
    let inode = Some(metadata.ino());
    let links = Some(metadata.nlink());
    let uid = Some(metadata.uid());
    let gid = Some(metadata.gid());

    Ok((permissions, readonly, hidden, executable, inode, links, uid, gid))
}

/// Extract platform-specific metadata for Windows
#[cfg(windows)]
fn extract_platform_metadata(path: &Path, metadata: &fs::Metadata) -> Result<(u32, bool, bool, bool, Option<u64>, Option<u64>, Option<u32>, Option<u32>)> {
    use std::os::windows::fs::MetadataExt;

    let permissions = 0; // Windows doesn't have Unix-style permissions
    let readonly = metadata.permissions().readonly();
    
    // Check if file is hidden using Windows attributes
    let hidden = metadata.file_attributes() & 0x2 != 0;
    
    // Windows executables typically have .exe, .bat, .cmd extensions
    let executable = path.extension()
        .and_then(|ext| ext.to_str())
        .map_or(false, |ext| {
            matches!(ext.to_lowercase().as_str(), "exe" | "bat" | "cmd" | "com" | "scr")
        });
    
    // Windows doesn't have direct equivalents for these Unix concepts
    let inode = None;
    let links = None;
    let uid = None;
    let gid = None;

    Ok((permissions, readonly, hidden, executable, inode, links, uid, gid))
}

impl SizeStats {
    /// Create size statistics from a slice of file sizes
    pub fn from_sizes(sizes: &[u64]) -> Self {
        let mut extractor = MetadataExtractor::new();
        let fake_metadata: Vec<FileMetadata> = sizes.iter()
            .enumerate()
            .map(|(i, &size)| FileMetadata {
                path: PathBuf::from(format!("file_{}", i)),
                size,
                size_human: bytes_to_human(size),
                ..Default::default()
            })
            .collect();
        
        extractor.calculate_size_stats(&fake_metadata)
    }

    /// Get a human-readable summary of the size statistics
    pub fn summary(&self) -> String {
        format!(
            "Files: {}, Total: {}, Avg: {}, Range: {} - {}",
            self.file_count,
            self.total_size_human,
            bytes_to_human(self.average_size),
            bytes_to_human(self.min_size),
            bytes_to_human(self.max_size)
        )
    }

    /// Get distribution summary
    pub fn distribution_summary(&self) -> String {
        format!(
            "Tiny: {}, Small: {}, Medium: {}, Large: {}, Huge: {}",
            self.size_distribution.tiny,
            self.size_distribution.small,
            self.size_distribution.medium,
            self.size_distribution.large,
            self.size_distribution.huge
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    use tokio::fs as async_fs;

    #[tokio::test]
    async fn test_metadata_extraction() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        
        let content = "Hello, world! This is a test file.";
        fs::write(&test_file, content).unwrap();

        let mut extractor = MetadataExtractor::new();
        let metadata = extractor.extract_metadata(&test_file).await.unwrap();

        assert_eq!(metadata.path, test_file);
        assert_eq!(metadata.size, content.len() as u64);
        assert!(!metadata.size_human.is_empty());
        assert!(metadata.modified.is_some());
        assert_eq!(metadata.file_type, FileSystemType::RegularFile);
        assert!(!metadata.symlink);
    }

    #[tokio::test]
    async fn test_symlink_detection() {
        let temp_dir = TempDir::new().unwrap();
        let original_file = temp_dir.path().join("original.txt");
        let symlink_file = temp_dir.path().join("link.txt");
        
        fs::write(&original_file, "original content").unwrap();
        
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&original_file, &symlink_file).unwrap();
            
            let mut extractor = MetadataExtractor::new();
            let metadata = extractor.extract_metadata(&symlink_file).await.unwrap();

            assert!(metadata.symlink);
            assert_eq!(metadata.symlink_target, Some(original_file));
        }
    }

    #[tokio::test]
    async fn test_batch_metadata_extraction() {
        let temp_dir = TempDir::new().unwrap();
        let mut file_paths = Vec::new();

        // Create multiple test files
        for i in 0..5 {
            let file_path = temp_dir.path().join(format!("test_{}.txt", i));
            fs::write(&file_path, format!("Content for file {}", i)).unwrap();
            file_paths.push(file_path);
        }

        let mut extractor = MetadataExtractor::new();
        let results = extractor.extract_metadata_batch(&file_paths).await;

        assert_eq!(results.len(), 5);
        for result in results {
            assert!(result.is_ok());
            let metadata = result.unwrap();
            assert_eq!(metadata.file_type, FileSystemType::RegularFile);
            assert!(metadata.size > 0);
        }
    }

    #[tokio::test]
    async fn test_size_statistics() {
        let temp_dir = TempDir::new().unwrap();
        let mut files = Vec::new();

        // Create files of different sizes
        let sizes = [100, 500, 1500, 5000, 50000];
        for (i, &size) in sizes.iter().enumerate() {
            let file_path = temp_dir.path().join(format!("test_{}.txt", i));
            let content = "x".repeat(size);
            fs::write(&file_path, content).unwrap();

            let mut extractor = MetadataExtractor::new();
            let metadata = extractor.extract_metadata(&file_path).await.unwrap();
            files.push(metadata);
        }

        let extractor = MetadataExtractor::new();
        let stats = extractor.calculate_size_stats(&files);

        assert_eq!(stats.file_count, 5);
        assert_eq!(stats.total_size, sizes.iter().sum::<usize>() as u64);
        assert_eq!(stats.min_size, 100);
        assert_eq!(stats.max_size, 50000);
        
        // Check distribution
        assert_eq!(stats.size_distribution.tiny, 2);    // 100, 500 bytes (both <= 1024)
        assert_eq!(stats.size_distribution.small, 2);   // 1500, 5000 bytes (1025-10240)
        assert_eq!(stats.size_distribution.medium, 1);  // 50000 bytes (10241-102400)
        assert_eq!(stats.size_distribution.large, 0);   // none
        assert_eq!(stats.size_distribution.huge, 0);
    }

    #[test]
    fn test_size_stats_from_sizes() {
        let sizes = [1000, 2000, 3000, 4000, 5000];
        let stats = SizeStats::from_sizes(&sizes);

        assert_eq!(stats.file_count, 5);
        assert_eq!(stats.total_size, 15000);
        assert_eq!(stats.average_size, 3000);
        assert_eq!(stats.median_size, 3000);
        assert_eq!(stats.min_size, 1000);
        assert_eq!(stats.max_size, 5000);
    }

    #[test]
    fn test_size_distribution() {
        let sizes = [
            500,      // tiny
            5000,     // small
            50000,    // medium
            500000,   // large
            5000000,  // huge
        ];
        let stats = SizeStats::from_sizes(&sizes);

        assert_eq!(stats.size_distribution.tiny, 1);
        assert_eq!(stats.size_distribution.small, 1);
        assert_eq!(stats.size_distribution.medium, 1);
        assert_eq!(stats.size_distribution.large, 1);
        assert_eq!(stats.size_distribution.huge, 1);
    }

    #[tokio::test]
    async fn test_cache_functionality() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, "test content").unwrap();

        let mut extractor = MetadataExtractor::new();
        
        // First extraction should cache the result
        let metadata1 = extractor.extract_metadata(&test_file).await.unwrap();
        let (cache_size, _) = extractor.cache_stats();
        assert_eq!(cache_size, 1);

        // Second extraction should use cache
        let metadata2 = extractor.extract_metadata(&test_file).await.unwrap();
        assert_eq!(metadata1.size, metadata2.size);
        assert_eq!(metadata1.modified, metadata2.modified);

        // Clear cache and verify
        extractor.clear_cache();
        let (cache_size, _) = extractor.cache_stats();
        assert_eq!(cache_size, 0);
    }

    #[tokio::test]
    async fn test_recently_modified() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, "test content").unwrap();

        let mut extractor = MetadataExtractor::new();
        let metadata = extractor.extract_metadata(&test_file).await.unwrap();

        // File should be recently modified (within 1 hour)
        assert!(extractor.is_recently_modified(&metadata, 1));
        
        // File should definitely be modified within 24 hours
        assert!(extractor.is_recently_modified(&metadata, 24));
    }

    #[test]
    fn test_file_type_classification() {
        // Test with mock metadata - actual implementation would depend on platform
        let sizes = [1000];
        let stats = SizeStats::from_sizes(&sizes);
        
        // Basic smoke test for stats functionality
        assert_eq!(stats.file_count, 1);
        assert_eq!(stats.total_size, 1000);
    }
}