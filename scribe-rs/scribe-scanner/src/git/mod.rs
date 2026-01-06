//! Git integration for file discovery and history analysis.

// Note: git_batch.rs is not compiled - it has unresolved dependencies
pub mod git_integration;

pub use git_integration::{GitCommitInfo, GitFileInfo, GitIntegrator};
