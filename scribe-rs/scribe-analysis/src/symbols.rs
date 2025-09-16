//! # Symbol Table Management
//!
//! Placeholder module for symbol resolution and scoping.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub symbol_type: String,
    pub scope: String,
}

impl Symbol {
    pub fn new(name: String, symbol_type: String, scope: String) -> Self {
        Self {
            name,
            symbol_type,
            scope,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    symbols: HashMap<String, Symbol>,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, symbol: Symbol) {
        self.symbols.insert(symbol.name.clone(), symbol);
    }

    pub fn lookup(&self, name: &str) -> Option<&Symbol> {
        self.symbols.get(name)
    }
}
