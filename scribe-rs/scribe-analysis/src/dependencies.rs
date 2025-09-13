//! # Dependency Analysis
//! 
//! Placeholder module for dependency extraction and analysis.

use std::collections::HashSet;

#[derive(Debug, Clone, Default)]
pub struct Dependencies {
    pub imports: HashSet<String>,
    pub exports: HashSet<String>,
}

impl Dependencies {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn add_import(&mut self, import: String) {
        self.imports.insert(import);
    }
    
    pub fn add_export(&mut self, export: String) {
        self.exports.insert(export);
    }
}