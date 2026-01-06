//! Core scanning types and operations.

pub mod filtering;
pub mod metadata;
pub mod scanner;

pub use filtering::{DirectoryFilter, FileFilter, FilterReason, FilterResult};
pub use metadata::{FileMetadata, MetadataExtractor, SizeStats};
pub use scanner::{ScanOptions, ScanProgress, ScanResult, Scanner};
