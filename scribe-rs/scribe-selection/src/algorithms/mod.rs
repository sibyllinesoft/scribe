//! Selection algorithms and strategies.

pub mod covering_set;
pub mod demotion;
pub mod quota;
pub mod simple_router;
pub mod two_pass;

// Note: demotion_test.rs is not compiled - it was not declared in original lib.rs
// and has test failures due to API changes

pub use covering_set::{
    CoveringSetComputer, CoveringSetEntity, CoveringSetFile, CoveringSetGranularity,
    CoveringSetOptions, CoveringSetResult, CoveringSetStatistics, InclusionReason, LineRange,
};
pub use demotion::{
    ChunkInfo, CodeChunker, DemotionEngine, DemotionResult, FidelityMode, SignatureExtractor,
};
pub use quota::{
    create_quota_manager, CategoryDetector, CategoryQuota, FileCategory, QuotaAllocation,
    QuotaManager, QuotaScanResult,
};
pub use simple_router::{
    ProjectSize, RoutingDecision, SelectionStrategy, SimpleRouter, TimeConstraint,
};
pub use two_pass::{
    CoverageGap, FileInfo, SelectionContext, SelectionMetrics, SelectionRule, TwoPassConfig,
    TwoPassResult, TwoPassSelector,
};
