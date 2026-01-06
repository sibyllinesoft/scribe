//! Git integration for file discovery and history analysis.

pub mod diff;
pub mod diff_analysis;
pub mod git_integration;
pub mod types;

pub use diff::{DiffAnalysisConfig, DiffAnalysisResult, DiffChangeType, DiffSource, GitDiffEntry};
pub use git_integration::GitIntegrator;
pub use types::{
    ActivityPeriod, AgeDistribution, BranchHealth, ContributorStats, GitBlameInfo, GitBlameLine,
    GitCommitInfo, GitFileInfo, GitRepositoryStats, RepositoryHealth,
};
