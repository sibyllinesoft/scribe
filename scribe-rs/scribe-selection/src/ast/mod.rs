//! AST parsing and code structure analysis.

pub mod ast_parser;

pub use ast_parser::{
    AstChunk, AstLanguage, AstParser, AstSignature, EntityLocation, EntityQuery, EntityType,
};
