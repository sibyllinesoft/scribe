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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_type_variants() {
        let types = vec![
            EntityType::Function,
            EntityType::Class,
            EntityType::Module,
            EntityType::Interface,
            EntityType::Constant,
            EntityType::Any,
        ];

        // Test clone and eq
        for t in &types {
            let cloned = t.clone();
            assert_eq!(t, &cloned);
        }
    }

    #[test]
    fn test_entity_type_serialize() {
        let func = EntityType::Function;
        let json = serde_json::to_string(&func).unwrap();
        let deserialized: EntityType = serde_json::from_str(&json).unwrap();
        assert_eq!(func, deserialized);
    }

    #[test]
    fn test_entity_query_parse_file_only() {
        let query = EntityQuery::parse("src/auth.rs");
        assert_eq!(query.file_pattern, Some("src/auth.rs".to_string()));
        assert!(query.name_pattern.is_none());
    }

    #[test]
    fn test_entity_query_parse_file_entity() {
        let query = EntityQuery::parse("src/auth.rs:login");
        assert_eq!(query.file_pattern, Some("src/auth.rs".to_string()));
        assert_eq!(query.name_pattern, Some("login".to_string()));
    }

    #[test]
    fn test_entity_query_parse_windows_drive_only() {
        // Windows path without entity
        let query = EntityQuery::parse(r"C:\path\file.rs");
        assert_eq!(query.file_pattern, Some(r"C:\path\file.rs".to_string()));
        assert!(query.name_pattern.is_none());
    }

    #[test]
    fn test_entity_query_parse_windows_drive_with_entity() {
        // Windows path with entity (two colons)
        let query = EntityQuery::parse(r"C:\path\file.rs:login");
        assert_eq!(query.file_pattern, Some(r"C:\path\file.rs".to_string()));
        assert_eq!(query.name_pattern, Some("login".to_string()));
    }

    #[test]
    fn test_entity_query_parse_multiple_colons() {
        // Multiple colons - rightmost is the separator
        let query = EntityQuery::parse("src:module:entity");
        assert_eq!(query.file_pattern, Some("src:module".to_string()));
        assert_eq!(query.name_pattern, Some("entity".to_string()));
    }

    #[test]
    fn test_entity_query_for_file() {
        let query = EntityQuery::for_file("src/lib.rs");
        assert_eq!(query.file_pattern, Some("src/lib.rs".to_string()));
        assert!(query.name_pattern.is_none());
        assert!(query.entity_type.is_none());
        assert!(!query.exact_match);
        assert!(query.public_only.is_none());
    }

    #[test]
    fn test_entity_query_for_file_entity() {
        let query = EntityQuery::for_file_entity("src/lib.rs", "main");
        assert_eq!(query.file_pattern, Some("src/lib.rs".to_string()));
        assert_eq!(query.name_pattern, Some("main".to_string()));
    }

    #[test]
    fn test_entity_query_by_name() {
        let query = EntityQuery::by_name("login_user");
        assert_eq!(query.name_pattern, Some("login_user".to_string()));
        assert!(query.file_pattern.is_none());
    }

    #[test]
    fn test_entity_query_by_type() {
        let query = EntityQuery::by_type(EntityType::Function);
        assert_eq!(query.entity_type, Some(EntityType::Function));
        assert!(query.name_pattern.is_none());
        assert!(query.file_pattern.is_none());
    }

    #[test]
    fn test_entity_query_function() {
        let query = EntityQuery::function("main");
        assert_eq!(query.entity_type, Some(EntityType::Function));
        assert_eq!(query.name_pattern, Some("main".to_string()));
    }

    #[test]
    fn test_entity_query_class() {
        let query = EntityQuery::class("UserService");
        assert_eq!(query.entity_type, Some(EntityType::Class));
        assert_eq!(query.name_pattern, Some("UserService".to_string()));
    }

    #[test]
    fn test_entity_query_module() {
        let query = EntityQuery::module("auth::login");
        assert_eq!(query.entity_type, Some(EntityType::Module));
        assert_eq!(query.name_pattern, Some("auth::login".to_string()));
    }

    #[test]
    fn test_entity_query_builder_in_file() {
        let query = EntityQuery::function("main").in_file("src/lib.rs");
        assert_eq!(query.file_pattern, Some("src/lib.rs".to_string()));
        assert_eq!(query.name_pattern, Some("main".to_string()));
        assert_eq!(query.entity_type, Some(EntityType::Function));
    }

    #[test]
    fn test_entity_query_builder_exact() {
        let query = EntityQuery::function("main").exact();
        assert!(query.exact_match);
    }

    #[test]
    fn test_entity_query_builder_public() {
        let query = EntityQuery::function("main").public();
        assert_eq!(query.public_only, Some(true));
    }

    #[test]
    fn test_entity_query_builder_chain() {
        let query = EntityQuery::function("main")
            .in_file("src/lib.rs")
            .exact()
            .public();

        assert_eq!(query.entity_type, Some(EntityType::Function));
        assert_eq!(query.name_pattern, Some("main".to_string()));
        assert_eq!(query.file_pattern, Some("src/lib.rs".to_string()));
        assert!(query.exact_match);
        assert_eq!(query.public_only, Some(true));
    }

    #[test]
    fn test_entity_query_matches_file_no_pattern() {
        let query = EntityQuery::by_name("test");
        assert!(query.matches_file("any/path/file.rs"));
        assert!(query.matches_file("another.py"));
    }

    #[test]
    fn test_entity_query_matches_file_with_pattern() {
        let query = EntityQuery::for_file("auth");
        assert!(query.matches_file("src/auth/login.rs"));
        assert!(query.matches_file("auth.rs"));
        assert!(!query.matches_file("src/user/profile.rs"));
    }

    #[test]
    fn test_entity_query_matches_file_case_insensitive() {
        let query = EntityQuery::for_file("AUTH");
        assert!(query.matches_file("src/auth/login.rs"));
        assert!(query.matches_file("src/Auth/Login.rs"));
    }

    #[test]
    fn test_entity_query_serialize() {
        let query = EntityQuery::function("main").in_file("src/lib.rs");
        let json = serde_json::to_string(&query).unwrap();
        let deserialized: EntityQuery = serde_json::from_str(&json).unwrap();

        assert_eq!(query.file_pattern, deserialized.file_pattern);
        assert_eq!(query.name_pattern, deserialized.name_pattern);
        assert_eq!(query.entity_type, deserialized.entity_type);
    }

    #[test]
    fn test_entity_query_debug() {
        let query = EntityQuery::function("main");
        let debug_str = format!("{:?}", query);
        assert!(debug_str.contains("main"));
    }

    #[test]
    fn test_entity_location_identifier() {
        let location = EntityLocation {
            file_path: "src/auth.rs".to_string(),
            entity_type: "function".to_string(),
            entity_name: "login".to_string(),
            start_line: 10,
            end_line: 25,
            is_public: true,
            content: "pub fn login() {}".to_string(),
        };

        assert_eq!(location.identifier(), "src/auth.rs::login");
    }

    #[test]
    fn test_entity_location_clone() {
        let location = EntityLocation {
            file_path: "src/main.rs".to_string(),
            entity_type: "function".to_string(),
            entity_name: "main".to_string(),
            start_line: 1,
            end_line: 5,
            is_public: false,
            content: "fn main() {}".to_string(),
        };

        let cloned = location.clone();
        assert_eq!(location.file_path, cloned.file_path);
        assert_eq!(location.entity_name, cloned.entity_name);
    }

    #[test]
    fn test_entity_location_serialize() {
        let location = EntityLocation {
            file_path: "src/lib.rs".to_string(),
            entity_type: "class".to_string(),
            entity_name: "MyClass".to_string(),
            start_line: 10,
            end_line: 50,
            is_public: true,
            content: "struct MyClass {}".to_string(),
        };

        let json = serde_json::to_string(&location).unwrap();
        let deserialized: EntityLocation = serde_json::from_str(&json).unwrap();

        assert_eq!(location.file_path, deserialized.file_path);
        assert_eq!(location.entity_name, deserialized.entity_name);
        assert_eq!(location.start_line, deserialized.start_line);
    }

    #[test]
    fn test_entity_location_debug() {
        let location = EntityLocation {
            file_path: "test.rs".to_string(),
            entity_type: "function".to_string(),
            entity_name: "test_func".to_string(),
            start_line: 1,
            end_line: 1,
            is_public: false,
            content: "fn test_func() {}".to_string(),
        };

        let debug_str = format!("{:?}", location);
        assert!(debug_str.contains("test_func"));
        assert!(debug_str.contains("test.rs"));
    }
}
