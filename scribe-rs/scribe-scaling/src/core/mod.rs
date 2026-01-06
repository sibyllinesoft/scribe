//! Core scaling types and algorithms.

pub mod error;
pub mod positioning;
pub mod selector;
pub mod signatures;
pub mod utils;

pub use error::{ScalingError, ScalingResult};
pub use positioning::{
    ContextPositioner, ContextPositioning, ContextPositioningConfig, PositionedSelection,
};
pub use selector::{
    ScalingSelectionConfig, ScalingSelectionResult, ScalingSelector, SelectionAlgorithm,
};
pub use signatures::{SignatureConfig, SignatureLevel};
pub use utils::classify_file_type_string;
