//! Progressive loading and streaming for memory-efficient file processing.

use std::path::PathBuf;
use std::time::SystemTime;
use serde::{Deserialize, Serialize};

/// File metadata for streaming operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    /// File path
    pub path: PathBuf,
    
    /// File size in bytes
    pub size: u64,
    
    /// Last modified time
    pub modified: SystemTime,
    
    /// Detected programming language
    pub language: String,
    
    /// File type classification
    pub file_type: String,
}

/// Configuration for streaming operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Whether to enable streaming (vs loading all at once)
    pub enable_streaming: bool,
    
    /// Number of files to process in each chunk
    pub chunk_size: usize,
    
    /// Memory limit for streaming operations (bytes)
    pub memory_limit: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enable_streaming: true,
            chunk_size: 1000,
            memory_limit: 100 * 1024 * 1024, // 100MB
        }
    }
}

/// Simple file chunk for processing
#[derive(Debug, Clone)]
pub struct FileChunk {
    /// Files in this chunk
    pub files: Vec<FileMetadata>,
    
    /// Chunk index
    pub index: usize,
    
    /// Total number of chunks
    pub total_chunks: usize,
}

impl FileChunk {
    /// Create a new file chunk
    pub fn new(files: Vec<FileMetadata>, index: usize, total_chunks: usize) -> Self {
        Self {
            files,
            index,
            total_chunks,
        }
    }
    
    /// Get the number of files in this chunk
    pub fn len(&self) -> usize {
        self.files.len()
    }
    
    /// Check if the chunk is empty
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }
    
    /// Get total size of all files in this chunk
    pub fn total_size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}