//! Entity types and queries for AST-based code search

use serde::{Deserialize, Serialize};

/// Entity type for search queries
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityType {
    Function,
    Class,
    Module,
    Interface,
    Constant,
    Any,
}

/// Query for finding entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityQuery {
    /// Type of entity to search for (None means any type)
    pub entity_type: Option<EntityType>,
    /// Name pattern to match (None means any name)
    pub name_pattern: Option<String>,
    /// Whether to match name exactly (vs substring)
    pub exact_match: bool,
    /// Only return public/exported entities
    pub public_only: Option<bool>,
    /// File path pattern to filter by (None means any file)
    pub file_pattern: Option<String>,
}

impl EntityQuery {
    /// Parse a query string in the format "file" or "file:entity"
    ///
    /// The rightmost colon separates file from entity, with special handling
    /// for Windows drive letters (single char before lone colon).
    ///
    /// Examples:
    /// - "src/auth.rs" -> file only, no entity
    /// - "src/auth.rs:login" -> file with entity
    /// - "C:\path\file.rs" -> Windows path, file only (single colon after drive letter)
    /// - "C:\path\file.rs:login" -> Windows path with entity (rightmost colon)
    pub fn parse(query: &str) -> Self {
        let colon_count = query.matches(':').count();

        if colon_count == 0 {
            // No colon: entire string is a file path
            Self::for_file(query)
        } else if colon_count == 1 {
            // Single colon: check for Windows drive letter
            if let Some((before, after)) = query.split_once(':') {
                if before.len() == 1 && before.chars().next().unwrap().is_ascii_alphabetic() {
                    // Windows drive letter (e.g., "C:\path") - whole thing is file
                    Self::for_file(query)
                } else {
                    // file:entity format
                    Self::for_file_entity(before, after)
                }
            } else {
                Self::for_file(query)
            }
        } else {
            // Multiple colons: rightmost colon is the separator
            if let Some((file_part, entity_part)) = query.rsplit_once(':') {
                Self::for_file_entity(file_part, entity_part)
            } else {
                Self::for_file(query)
            }
        }
    }

    /// Create a query for a file only (no specific entity)
    pub fn for_file(file: &str) -> Self {
        Self {
            entity_type: None,
            name_pattern: None,
            exact_match: false,
            public_only: None,
            file_pattern: Some(file.to_string()),
        }
    }

    /// Create a query for a specific entity within a file
    pub fn for_file_entity(file: &str, entity: &str) -> Self {
        Self {
            entity_type: None,
            name_pattern: Some(entity.to_string()),
            exact_match: false,
            public_only: None,
            file_pattern: Some(file.to_string()),
        }
    }

    /// Create a query for any entity with a specific name (searches all files)
    pub fn by_name(name: &str) -> Self {
        Self {
            entity_type: None,
            name_pattern: Some(name.to_string()),
            exact_match: false,
            public_only: None,
            file_pattern: None,
        }
    }

    /// Create a query for a specific entity type
    pub fn by_type(entity_type: EntityType) -> Self {
        Self {
            entity_type: Some(entity_type),
            name_pattern: None,
            exact_match: false,
            public_only: None,
            file_pattern: None,
        }
    }

    /// Create a query for a specific function by name
    pub fn function(name: &str) -> Self {
        Self {
            entity_type: Some(EntityType::Function),
            name_pattern: Some(name.to_string()),
            exact_match: false,
            public_only: None,
            file_pattern: None,
        }
    }

    /// Create a query for a specific class/struct by name
    pub fn class(name: &str) -> Self {
        Self {
            entity_type: Some(EntityType::Class),
            name_pattern: Some(name.to_string()),
            exact_match: false,
            public_only: None,
            file_pattern: None,
        }
    }

    /// Create a query for a specific module by path
    pub fn module(path: &str) -> Self {
        Self {
            entity_type: Some(EntityType::Module),
            name_pattern: Some(path.to_string()),
            exact_match: false,
            public_only: None,
            file_pattern: None,
        }
    }

    /// Filter to a specific file or file pattern
    pub fn in_file(mut self, file_pattern: &str) -> Self {
        self.file_pattern = Some(file_pattern.to_string());
        self
    }

    /// Set whether to match exactly
    pub fn exact(mut self) -> Self {
        self.exact_match = true;
        self
    }

    /// Only match public/exported entities
    pub fn public(mut self) -> Self {
        self.public_only = Some(true);
        self
    }

    /// Check if a file path matches the file pattern (if any)
    pub fn matches_file(&self, file_path: &str) -> bool {
        match &self.file_pattern {
            None => true, // No pattern means match all files
            Some(pattern) => {
                let pattern_lower = pattern.to_lowercase();
                let path_lower = file_path.to_lowercase();
                // Match if the file path contains the pattern
                path_lower.contains(&pattern_lower)
            }
        }
    }
}

/// Location of an entity in the codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityLocation {
    /// File path containing the entity
    pub file_path: String,
    /// Type of entity (function, class, etc.)
    pub entity_type: String,
    /// Name of the entity
    pub entity_name: String,
    /// Start line number (1-indexed)
    pub start_line: usize,
    /// End line number (1-indexed)
    pub end_line: usize,
    /// Whether this entity is public/exported
    pub is_public: bool,
    /// Full content of the entity
    pub content: String,
}

impl EntityLocation {
    /// Get a unique identifier for this entity
    pub fn identifier(&self) -> String {
        format!("{}::{}", self.file_path, self.entity_name)
    }
}
