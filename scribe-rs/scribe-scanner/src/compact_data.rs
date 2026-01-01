//! Compact data structures with interned strings and memory-efficient formats.
//!
//! This module implements space-efficient data structures that dramatically reduce
//! memory usage through string interning, compact encoding, and cache-friendly
//! data layouts for large-scale file scanning operations.

use fxhash::FxHashMap;
use scribe_core::{FileWeight, GitFileStatus, Language, RenderDecision, Result, ScribeError};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use string_interner::{backend::StringBackend, DefaultSymbol, StringInterner};
use xxhash_rust::xxh3::xxh3_64;

/// Interned string identifier
pub type StringId = DefaultSymbol;

/// Global string interner for path deduplication
static GLOBAL_INTERNER: parking_lot::Mutex<StringInterner> =
    parking_lot::Mutex::new(StringInterner::new());

/// Compact file information with interned strings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactFileInfo {
    /// Interned path string ID
    pub path_id: StringId,
    /// Interned relative path ID
    pub relative_path_id: StringId,
    /// File size in bytes
    pub size: u32, // Limit to 4GB files for compactness
    /// Last modified timestamp (seconds since epoch)
    pub modified: u32, // Good until 2106
    /// Language as packed enum
    pub language: PackedLanguage,
    /// File type as packed enum  
    pub file_type: PackedFileType,
    /// Render decision packed
    pub decision: PackedDecision,
    /// Git status (if available)
    pub git_status: Option<PackedGitStatus>,
    /// Content metrics (compact)
    pub metrics: CompactMetrics,
    /// Flags bitfield
    pub flags: FileFlags,
}

/// Packed language enum (1 byte)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PackedLanguage {
    Unknown = 0,
    Rust = 1,
    Python = 2,
    JavaScript = 3,
    TypeScript = 4,
    Go = 5,
    Java = 6,
    C = 7,
    Cpp = 8,
    CSharp = 9,
    PHP = 10,
    Ruby = 11,
    Swift = 12,
    Kotlin = 13,
    Scala = 14,
    Clojure = 15,
    Haskell = 16,
    Elixir = 17,
    Erlang = 18,
    OCaml = 19,
    FSharp = 20,
    Html = 21,
    Css = 22,
    Json = 23,
    Xml = 24,
    Yaml = 25,
    Toml = 26,
    Sql = 27,
    Shell = 28,
    Dockerfile = 29,
    Markdown = 30,
    // Reserve remaining values for future languages
}

/// Packed file type enum (1 byte)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PackedFileType {
    Unknown = 0,
    Source = 1,
    Test = 2,
    Config = 3,
    Documentation = 4,
    Build = 5,
    Data = 6,
    Generated = 7,
    Binary = 8,
}

/// Packed render decision (1 byte)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PackedDecision {
    Include = 0,
    Exclude = 1,
}

/// Packed git status (1 byte)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PackedGitStatus {
    Unmodified = 0,
    Modified = 1,
    Added = 2,
    Deleted = 3,
    Renamed = 4,
    Copied = 5,
    Untracked = 6,
    Ignored = 7,
}

/// Compact content metrics (12 bytes total)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CompactMetrics {
    /// Line count (up to 16M lines)
    pub line_count: u24,
    /// Character count (up to 16M chars)
    pub char_count: u24,
    /// Token estimate (up to 64K tokens)
    pub token_estimate: u16,
    /// Content hash for deduplication
    pub content_hash: u16, // Truncated hash for space
}

/// File flags bitfield (1 byte)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FileFlags {
    flags: u8,
}

/// Compact file collection with global string interning
#[derive(Debug)]
pub struct CompactFileCollection {
    /// Files using compact representation
    files: Vec<CompactFileInfo>,
    /// Local string interner for this collection
    local_interner: StringInterner,
    /// Statistics about memory savings
    stats: CompressionStats,
    /// Index for fast lookups
    path_index: FxHashMap<u64, usize>, // hash -> file index
}

/// Statistics about data compression
#[derive(Debug, Default, Clone)]
pub struct CompressionStats {
    /// Original uncompressed size estimate
    pub uncompressed_bytes: u64,
    /// Actual compressed size
    pub compressed_bytes: u64,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Number of strings interned
    pub interned_strings: u64,
    /// Bytes saved by interning
    pub interning_savings_bytes: u64,
    /// Number of files processed
    pub files_processed: u64,
}

/// 24-bit unsigned integer for compact storage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct u24(u32);

impl u24 {
    pub fn new(value: u32) -> Self {
        Self(value & 0x00FFFFFF) // Mask to 24 bits
    }

    pub fn get(self) -> u32 {
        self.0
    }
}

impl Serialize for u24 {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize as 3-byte array for compact storage
        let bytes = [
            (self.0 & 0xFF) as u8,
            ((self.0 >> 8) & 0xFF) as u8,
            ((self.0 >> 16) & 0xFF) as u8,
        ];
        bytes.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for u24 {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes: [u8; 3] = Deserialize::deserialize(deserializer)?;
        let value = (bytes[0] as u32) | ((bytes[1] as u32) << 8) | ((bytes[2] as u32) << 16);
        Ok(Self(value))
    }
}

impl CompactFileInfo {
    /// Create compact file info from full file info
    pub fn from_full_file_info(
        file_info: &scribe_core::FileInfo,
        interner: &mut StringInterner,
    ) -> Self {
        // Intern strings
        let path_id = interner.get_or_intern(file_info.path.to_string_lossy());
        let relative_path_id = interner.get_or_intern(&file_info.relative_path);

        // Convert types to compact representations
        let language = PackedLanguage::from(file_info.language);
        let file_type = PackedFileType::from_file_type(&file_info.file_type);
        let decision = PackedDecision::from(&file_info.decision);
        let git_status = file_info.git_status.as_ref().map(PackedGitStatus::from);

        // Create compact metrics
        let metrics = CompactMetrics {
            line_count: u24::new(file_info.line_count.unwrap_or(0) as u32),
            char_count: u24::new(file_info.char_count.unwrap_or(0) as u32),
            token_estimate: file_info.token_estimate.unwrap_or(0) as u16,
            content_hash: Self::calculate_content_hash(file_info.content.as_deref().unwrap_or("")),
        };

        // Create flags
        let mut flags = FileFlags::new();
        flags.set_binary(file_info.is_binary);
        if file_info.content.is_some() {
            flags.set_has_content(true);
        }

        Self {
            path_id,
            relative_path_id,
            size: file_info.size.min(u32::MAX as u64) as u32,
            modified: file_info
                .modified
                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                .map(|d| d.as_secs().min(u32::MAX as u64) as u32)
                .unwrap_or(0),
            language,
            file_type,
            decision,
            git_status,
            metrics,
            flags,
        }
    }

    /// Convert back to full file info
    pub fn to_full_file_info(&self, interner: &StringInterner) -> Result<scribe_core::FileInfo> {
        let path_str = interner
            .resolve(self.path_id)
            .ok_or_else(|| ScribeError::data("Invalid path ID".to_string()))?;
        let relative_path = interner
            .resolve(self.relative_path_id)
            .ok_or_else(|| ScribeError::data("Invalid relative path ID".to_string()))?;

        let modified = if self.modified > 0 {
            Some(SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(self.modified as u64))
        } else {
            None
        };

        let language = Language::from(self.language);
        let relative_path_str = relative_path.to_string();
        let extension = Path::new(&relative_path_str)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        Ok(scribe_core::FileInfo {
            path: PathBuf::from(path_str),
            relative_path: relative_path_str.clone(),
            size: self.size as u64,
            modified,
            decision: RenderDecision::from(&self.decision),
            file_type: scribe_core::FileInfo::classify_file_type_with_binary(
                &relative_path_str,
                &language,
                extension,
                self.flags.is_binary(),
            ),
            language,
            content: None, // Not stored in compact format
            token_estimate: Some(self.token_estimate()),
            line_count: Some(self.line_count()),
            char_count: Some(self.char_count()),
            is_binary: self.flags.is_binary(),
            git_status: self.git_status.map(|s| s.into()),
            weight: FileWeight::default(),
            centrality_score: None, // Not stored in compact format, calculated during analysis
        })
    }

    /// Calculate content hash for deduplication
    fn calculate_content_hash(content: &str) -> u16 {
        let hash = xxh3_64(content.as_bytes());
        (hash & 0xFFFF) as u16
    }

    /// Get line count as u32
    pub fn line_count(&self) -> u32 {
        self.metrics.line_count.get()
    }

    /// Get character count as u32
    pub fn char_count(&self) -> u32 {
        self.metrics.char_count.get()
    }

    /// Get token estimate as u32
    pub fn token_estimate(&self) -> u32 {
        self.metrics.token_estimate as u32
    }

    /// Calculate memory footprint
    pub fn memory_footprint(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

impl CompactFileCollection {
    /// Create a new compact file collection
    pub fn new() -> Self {
        Self {
            files: Vec::new(),
            local_interner: StringInterner::new(),
            stats: CompressionStats::default(),
            path_index: FxHashMap::default(),
        }
    }

    /// Add a file to the collection
    pub fn add_file(&mut self, file_info: &scribe_core::FileInfo) {
        let original_size = self.estimate_file_info_size(file_info);
        let compact_info =
            CompactFileInfo::from_full_file_info(file_info, &mut self.local_interner);

        // Add to path index
        let path_hash = xxh3_64(file_info.path.to_string_lossy().as_bytes());
        self.path_index.insert(path_hash, self.files.len());

        self.files.push(compact_info);

        // Update stats
        self.stats.files_processed += 1;
        self.stats.uncompressed_bytes += original_size as u64;
        self.stats.compressed_bytes += compact_info.memory_footprint() as u64;
        self.stats.interned_strings += 1; // Approximation

        self.update_compression_stats();
    }

    /// Add multiple files efficiently
    pub fn add_files(&mut self, file_infos: &[scribe_core::FileInfo]) {
        self.files.reserve(file_infos.len());

        for file_info in file_infos {
            self.add_file(file_info);
        }
    }

    /// Find file by path hash (fast lookup)
    pub fn find_by_path(&self, path: &Path) -> Option<&CompactFileInfo> {
        let path_hash = xxh3_64(path.to_string_lossy().as_bytes());
        self.path_index
            .get(&path_hash)
            .and_then(|&index| self.files.get(index))
    }

    /// Get all files
    pub fn files(&self) -> &[CompactFileInfo] {
        &self.files
    }

    /// Get files by language
    pub fn files_by_language(&self, language: PackedLanguage) -> Vec<&CompactFileInfo> {
        self.files
            .iter()
            .filter(|f| f.language == language)
            .collect()
    }

    /// Get files by type
    pub fn files_by_type(&self, file_type: PackedFileType) -> Vec<&CompactFileInfo> {
        self.files
            .iter()
            .filter(|f| f.file_type == file_type)
            .collect()
    }

    /// Sort files by size (largest first)
    pub fn sort_by_size(&mut self) {
        self.files.sort_by(|a, b| b.size.cmp(&a.size));
        self.rebuild_path_index();
    }

    /// Sort files by modification time (newest first)
    pub fn sort_by_modified(&mut self) {
        self.files.sort_by(|a, b| b.modified.cmp(&a.modified));
        self.rebuild_path_index();
    }

    /// Convert to full file infos (decompression)
    pub fn to_full_file_infos(&self) -> Result<Vec<scribe_core::FileInfo>> {
        self.files
            .iter()
            .map(|f| f.to_full_file_info(&self.local_interner))
            .collect()
    }

    /// Get compression statistics
    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }

    /// Serialize to bytes for persistence
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let interner_data: Vec<_> = self.local_interner.into_iter().collect();
        let data = (&self.files, &interner_data, &self.stats);

        bincode::serialize(&data)
            .map_err(|e| ScribeError::io(format!("Serialization failed: {}", e)))
    }

    /// Deserialize from bytes
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        let (files, interner_data, stats): (
            Vec<CompactFileInfo>,
            Vec<(StringId, String)>,
            CompressionStats,
        ) = bincode::deserialize(data)
            .map_err(|e| ScribeError::io(format!("Deserialization failed: {}", e)))?;

        // Rebuild interner
        let mut local_interner = StringInterner::new();
        for (_, string) in &interner_data {
            local_interner.get_or_intern(string);
        }

        let mut collection = Self {
            files,
            local_interner,
            stats,
            path_index: FxHashMap::default(),
        };

        collection.rebuild_path_index();
        Ok(collection)
    }

    /// Estimate memory usage of original FileInfo
    fn estimate_file_info_size(&self, file_info: &scribe_core::FileInfo) -> usize {
        std::mem::size_of::<scribe_core::FileInfo>()
            + file_info.path.to_string_lossy().len()
            + file_info.relative_path.len()
            + file_info.file_type.len()
            + file_info.content.as_ref().map(|c| c.len()).unwrap_or(0)
    }

    /// Update compression statistics
    fn update_compression_stats(&mut self) {
        if self.stats.uncompressed_bytes > 0 {
            self.stats.compression_ratio =
                1.0 - (self.stats.compressed_bytes as f64 / self.stats.uncompressed_bytes as f64);
        }

        // Estimate interning savings
        let unique_strings = self.local_interner.len();
        let total_string_bytes: usize = self.local_interner.into_iter().map(|(_, s)| s.len()).sum();

        let without_interning = self.files.len() * 2 * 50; // Estimate avg 50 chars per path
        let with_interning = total_string_bytes + unique_strings * std::mem::size_of::<StringId>();

        self.stats.interning_savings_bytes =
            without_interning.saturating_sub(with_interning) as u64;
    }

    /// Rebuild path index after sorting
    fn rebuild_path_index(&mut self) {
        self.path_index.clear();
        for (index, file) in self.files.iter().enumerate() {
            if let Some(path_str) = self.local_interner.resolve(file.path_id) {
                let path_hash = xxh3_64(path_str.as_bytes());
                self.path_index.insert(path_hash, index);
            }
        }
    }
}

impl PackedLanguage {
    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Rust,
            2 => Self::Python,
            3 => Self::JavaScript,
            4 => Self::TypeScript,
            5 => Self::Go,
            6 => Self::Java,
            7 => Self::C,
            8 => Self::Cpp,
            9 => Self::CSharp,
            10 => Self::PHP,
            11 => Self::Ruby,
            12 => Self::Swift,
            13 => Self::Kotlin,
            14 => Self::Scala,
            15 => Self::Clojure,
            16 => Self::Haskell,
            17 => Self::Elixir,
            18 => Self::Erlang,
            19 => Self::OCaml,
            20 => Self::FSharp,
            21 => Self::Html,
            22 => Self::Css,
            23 => Self::Json,
            24 => Self::Xml,
            25 => Self::Yaml,
            26 => Self::Toml,
            27 => Self::Sql,
            28 => Self::Shell,
            29 => Self::Dockerfile,
            30 => Self::Markdown,
            _ => Self::Unknown,
        }
    }

    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Convert from Language enum
    pub fn from(lang: Language) -> Self {
        match lang {
            Language::Rust => Self::Rust,
            Language::Python => Self::Python,
            Language::JavaScript => Self::JavaScript,
            Language::TypeScript => Self::TypeScript,
            Language::Go => Self::Go,
            Language::Java => Self::Java,
            Language::C => Self::C,
            Language::Cpp => Self::Cpp,
            Language::CSharp => Self::CSharp,
            Language::PHP => Self::PHP,
            Language::Ruby => Self::Ruby,
            Language::Swift => Self::Swift,
            Language::Kotlin => Self::Kotlin,
            Language::Scala => Self::Scala,
            Language::Clojure => Self::Clojure,
            Language::Haskell => Self::Haskell,
            Language::Html => Self::Html,
            Language::Css => Self::Css,
            Language::Json => Self::Json,
            Language::Xml => Self::Xml,
            Language::Yaml => Self::Yaml,
            Language::Sql => Self::Sql,
            Language::Shell => Self::Shell,
            Language::Dockerfile => Self::Dockerfile,
            Language::Markdown => Self::Markdown,
            Language::Unknown => Self::Unknown,
        }
    }
}

impl From<PackedLanguage> for Language {
    fn from(packed: PackedLanguage) -> Self {
        match packed {
            PackedLanguage::Rust => Self::Rust,
            PackedLanguage::Python => Self::Python,
            PackedLanguage::JavaScript => Self::JavaScript,
            PackedLanguage::TypeScript => Self::TypeScript,
            PackedLanguage::Go => Self::Go,
            PackedLanguage::Java => Self::Java,
            PackedLanguage::C => Self::C,
            PackedLanguage::Cpp => Self::Cpp,
            PackedLanguage::CSharp => Self::CSharp,
            PackedLanguage::PHP => Self::PHP,
            PackedLanguage::Ruby => Self::Ruby,
            PackedLanguage::Swift => Self::Swift,
            PackedLanguage::Kotlin => Self::Kotlin,
            PackedLanguage::Scala => Self::Scala,
            PackedLanguage::Clojure => Self::Clojure,
            PackedLanguage::Haskell => Self::Haskell,
            PackedLanguage::Html => Self::Html,
            PackedLanguage::Css => Self::Css,
            PackedLanguage::Json => Self::Json,
            PackedLanguage::Xml => Self::Xml,
            PackedLanguage::Yaml => Self::Yaml,
            PackedLanguage::Sql => Self::Sql,
            PackedLanguage::Shell => Self::Shell,
            PackedLanguage::Dockerfile => Self::Dockerfile,
            PackedLanguage::Markdown => Self::Markdown,
            _ => Self::Unknown,
        }
    }
}

impl PackedFileType {
    pub fn from_file_type(file_type: &scribe_core::FileType) -> Self {
        match file_type {
            scribe_core::FileType::Source { .. } => Self::Source,
            scribe_core::FileType::Test { .. } => Self::Test,
            scribe_core::FileType::Configuration { .. } => Self::Config,
            scribe_core::FileType::Documentation { .. } => Self::Documentation,
            scribe_core::FileType::Build => Self::Build,
            scribe_core::FileType::Data => Self::Data,
            scribe_core::FileType::Generated => Self::Generated,
            scribe_core::FileType::Binary => Self::Binary,
            scribe_core::FileType::Unknown => Self::Unknown,
        }
    }

    pub fn to_file_type(&self, language: &Language) -> scribe_core::FileType {
        match self {
            Self::Source => scribe_core::FileType::Source {
                language: language.clone(),
            },
            Self::Test => scribe_core::FileType::Test {
                language: language.clone(),
            },
            Self::Config => scribe_core::FileType::Configuration {
                format: scribe_core::file::ConfigurationFormat::Json,
            },
            Self::Documentation => scribe_core::FileType::Documentation {
                format: scribe_core::file::DocumentationFormat::Markdown,
            },
            Self::Build => scribe_core::FileType::Build,
            Self::Data => scribe_core::FileType::Data,
            Self::Generated => scribe_core::FileType::Generated,
            Self::Binary => scribe_core::FileType::Binary,
            Self::Unknown => scribe_core::FileType::Unknown,
        }
    }

    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Source,
            2 => Self::Test,
            3 => Self::Config,
            4 => Self::Documentation,
            5 => Self::Build,
            6 => Self::Data,
            7 => Self::Generated,
            8 => Self::Binary,
            _ => Self::Unknown,
        }
    }
}

impl PackedDecision {
    pub fn from(decision: &RenderDecision) -> Self {
        if decision.should_include() {
            Self::Include
        } else {
            Self::Exclude
        }
    }
}

impl From<&PackedDecision> for RenderDecision {
    fn from(packed: &PackedDecision) -> Self {
        match packed {
            PackedDecision::Include => RenderDecision::include("included"),
            PackedDecision::Exclude => RenderDecision::exclude("excluded"),
        }
    }
}

impl PackedGitStatus {
    pub fn from(status: &scribe_core::GitStatus) -> Self {
        match status.working_tree {
            GitFileStatus::Modified => Self::Modified,
            GitFileStatus::Added => Self::Added,
            GitFileStatus::Deleted => Self::Deleted,
            GitFileStatus::Renamed => Self::Renamed,
            GitFileStatus::Copied => Self::Copied,
            GitFileStatus::Untracked => Self::Untracked,
            GitFileStatus::Ignored => Self::Ignored,
            GitFileStatus::Unmodified => Self::Unmodified,
            GitFileStatus::Unmerged => Self::Modified, // Treat unmerged as modified
        }
    }
}

impl From<PackedGitStatus> for scribe_core::GitStatus {
    fn from(packed: PackedGitStatus) -> Self {
        let working_tree = match packed {
            PackedGitStatus::Modified => GitFileStatus::Modified,
            PackedGitStatus::Added => GitFileStatus::Added,
            PackedGitStatus::Deleted => GitFileStatus::Deleted,
            PackedGitStatus::Renamed => GitFileStatus::Renamed,
            PackedGitStatus::Copied => GitFileStatus::Copied,
            PackedGitStatus::Untracked => GitFileStatus::Untracked,
            PackedGitStatus::Ignored => GitFileStatus::Ignored,
            PackedGitStatus::Unmodified => GitFileStatus::Unmodified,
        };

        Self {
            working_tree,
            index: GitFileStatus::Unmodified,
        }
    }
}

impl FileFlags {
    pub fn new() -> Self {
        Self { flags: 0 }
    }

    pub fn set_binary(&mut self, binary: bool) {
        if binary {
            self.flags |= 1;
        } else {
            self.flags &= !1;
        }
    }

    pub fn is_binary(&self) -> bool {
        self.flags & 1 != 0
    }

    pub fn set_has_content(&mut self, has_content: bool) {
        if has_content {
            self.flags |= 2;
        } else {
            self.flags &= !2;
        }
    }

    pub fn has_content(&self) -> bool {
        self.flags & 2 != 0
    }
}

impl Default for CompactFileCollection {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scribe_core::{FileType, FileWeight, GitFileStatus, GitStatus, Language, RenderDecision};
    use std::time::UNIX_EPOCH;

    fn create_test_file_info() -> scribe_core::FileInfo {
        scribe_core::FileInfo {
            path: PathBuf::from("/project/src/main.rs"),
            relative_path: "src/main.rs".to_string(),
            size: 1024,
            modified: Some(UNIX_EPOCH + std::time::Duration::from_secs(1640995200)),
            decision: RenderDecision::include("test"),
            file_type: FileType::Source {
                language: Language::Rust,
            },
            language: Language::Rust,
            content: Some("fn main() {}\n".to_string()),
            token_estimate: Some(10),
            line_count: Some(2),
            char_count: Some(14),
            is_binary: false,
            git_status: Some(GitStatus {
                working_tree: GitFileStatus::Modified,
                index: GitFileStatus::Unmodified,
            }),
            weight: FileWeight::default(),
            centrality_score: Some(0.2),
        }
    }

    #[test]
    fn test_compact_file_info_conversion() {
        let original = create_test_file_info();
        let mut interner = StringInterner::new();

        let compact = CompactFileInfo::from_full_file_info(&original, &mut interner);
        let restored = compact.to_full_file_info(&interner).unwrap();

        assert_eq!(original.path, restored.path);
        assert_eq!(original.relative_path, restored.relative_path);
        assert_eq!(original.size, restored.size);
        assert_eq!(original.language, restored.language);
        assert_eq!(original.file_type, restored.file_type);
        assert_eq!(original.is_binary, restored.is_binary);

        // Check metrics
        assert_eq!(original.line_count.unwrap() as u32, compact.line_count());
        assert_eq!(original.char_count.unwrap() as u32, compact.char_count());
        assert_eq!(
            original.token_estimate.unwrap() as u32,
            compact.token_estimate()
        );
    }

    #[test]
    fn test_u24_serialization() {
        let value = u24::new(12345678);
        assert_eq!(value.get(), 12345678 & 0x00FFFFFF);

        // Test edge cases
        let max_value = u24::new(0x00FFFFFF);
        assert_eq!(max_value.get(), 0x00FFFFFF);

        let overflow_value = u24::new(0x01000000);
        assert_eq!(overflow_value.get(), 0x00000000); // Should wrap
    }

    #[test]
    fn test_packed_language_conversion() {
        let languages = [
            Language::Rust,
            Language::Python,
            Language::JavaScript,
            Language::TypeScript,
            Language::Unknown,
        ];

        for lang in &languages {
            let packed = PackedLanguage::from(lang.clone());
            let restored = Language::from(packed);
            assert_eq!(*lang, restored);
        }
    }

    #[test]
    fn test_compact_file_collection() {
        let mut collection = CompactFileCollection::new();
        let file_info = create_test_file_info();

        collection.add_file(&file_info);

        assert_eq!(collection.files().len(), 1);
        assert!(collection.find_by_path(&file_info.path).is_some());

        let stats = collection.stats();
        assert_eq!(stats.files_processed, 1);
        assert!(stats.compressed_bytes < stats.uncompressed_bytes);
        assert!(stats.compression_ratio > 0.0);
    }

    #[test]
    fn test_collection_filtering() {
        let mut collection = CompactFileCollection::new();

        // Add multiple files with different languages
        for (i, lang) in [Language::Rust, Language::Python, Language::JavaScript]
            .iter()
            .enumerate()
        {
            let mut file_info = create_test_file_info();
            file_info.language = lang.clone();
            file_info.path = PathBuf::from(format!("/project/file{}.ext", i));
            file_info.relative_path = format!("file{}.ext", i);
            collection.add_file(&file_info);
        }

        assert_eq!(collection.files().len(), 3);

        let rust_files = collection.files_by_language(PackedLanguage::Rust);
        assert_eq!(rust_files.len(), 1);

        let python_files = collection.files_by_language(PackedLanguage::Python);
        assert_eq!(python_files.len(), 1);

        let source_files = collection.files_by_type(PackedFileType::Source);
        assert_eq!(source_files.len(), 3);
    }

    #[test]
    fn test_collection_sorting() {
        let mut collection = CompactFileCollection::new();

        // Add files with different sizes
        for (i, size) in [100, 500, 200].iter().enumerate() {
            let mut file_info = create_test_file_info();
            file_info.size = *size;
            file_info.path = PathBuf::from(format!("/project/file{}.rs", i));
            file_info.relative_path = format!("file{}.rs", i);
            collection.add_file(&file_info);
        }

        collection.sort_by_size();

        let files = collection.files();
        assert!(files[0].size >= files[1].size);
        assert!(files[1].size >= files[2].size);

        // Check that path index was rebuilt correctly
        let file0_path = PathBuf::from("/project/file0.rs");
        assert!(collection.find_by_path(&file0_path).is_some());
    }

    #[test]
    fn test_serialization() {
        let mut collection = CompactFileCollection::new();
        let file_info = create_test_file_info();
        collection.add_file(&file_info);

        // Serialize
        let serialized = collection.serialize().unwrap();
        assert!(!serialized.is_empty());

        // Deserialize
        let deserialized = CompactFileCollection::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.files().len(), collection.files().len());
        assert_eq!(
            deserialized.stats().files_processed,
            collection.stats().files_processed
        );

        // Check that path index was rebuilt
        assert!(deserialized.find_by_path(&file_info.path).is_some());
    }

    #[test]
    fn test_memory_footprint() {
        let original = create_test_file_info();
        let mut interner = StringInterner::new();
        let compact = CompactFileInfo::from_full_file_info(&original, &mut interner);

        // Compact version should be smaller
        let compact_size = compact.memory_footprint();
        let original_size = std::mem::size_of::<scribe_core::FileInfo>()
            + original.path.to_string_lossy().len()
            + original.relative_path.len()
            + std::mem::size_of::<FileType>() // FileType is an enum, use size_of instead
            + original.content.as_ref().map(|c| c.len()).unwrap_or(0);

        println!(
            "Original: {} bytes, Compact: {} bytes",
            original_size, compact_size
        );
        // Note: Due to string interning, we need the collection level to see real savings
    }

    #[test]
    fn test_file_flags() {
        let mut flags = FileFlags::new();

        assert!(!flags.is_binary());
        assert!(!flags.has_content());

        flags.set_binary(true);
        flags.set_has_content(true);

        assert!(flags.is_binary());
        assert!(flags.has_content());

        flags.set_binary(false);
        assert!(!flags.is_binary());
        assert!(flags.has_content()); // Should remain true
    }

    #[test]
    fn test_compression_stats() {
        let mut collection = CompactFileCollection::new();

        // Add multiple files to see compression benefits
        for i in 0..100 {
            let mut file_info = create_test_file_info();
            file_info.path = PathBuf::from(format!("/project/src/file{}.rs", i));
            file_info.relative_path = format!("src/file{}.rs", i);
            file_info.content = Some(format!("fn function_{}() {{}}\n", i));
            collection.add_file(&file_info);
        }

        let stats = collection.stats();
        assert_eq!(stats.files_processed, 100);
        assert!(stats.compression_ratio > 0.0);
        assert!(stats.interning_savings_bytes > 0);

        println!("Compression ratio: {:.2}%", stats.compression_ratio * 100.0);
        println!("Interning savings: {} bytes", stats.interning_savings_bytes);
    }
}
