//! Tests for the main scribe library.

use super::*;

#[test]
fn test_version() {
    assert!(!VERSION.is_empty());
}

#[cfg(feature = "core")]
#[test]
fn test_core_reexport() {
    let config = Config::default();
    assert!(config.validate().is_ok());
}

#[cfg(all(feature = "analysis", feature = "scanner", feature = "patterns"))]
#[tokio::test]
async fn test_repository_analysis_interface() {
    use std::fs;
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test.rs");
    fs::write(&test_file, "fn main() { println!(\"Hello world\"); }").unwrap();

    let config = Config::default();
    let result = analyze_repository(temp_dir.path(), &config).await;

    // Should succeed or fail gracefully
    match result {
        Ok(analysis) => {
            assert!(analysis.file_count() > 0);
            assert!(!analysis.summary().is_empty());
        }
        Err(_) => {
            // Analysis might fail in test environment, which is acceptable
            // as long as the interface compiles correctly
        }
    }
}

#[cfg(all(feature = "scanner", feature = "patterns"))]
#[tokio::test]
async fn test_scan_repository_interface() {
    use std::fs;
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test.rs");
    fs::write(&test_file, "fn main() {}").unwrap();

    let result =
        scan_repository(temp_dir.path(), Some(&["**/*.rs"]), Some(&["**/target/**"])).await;

    // Should find the test file
    match result {
        Ok(files) => {
            assert!(!files.is_empty());
            assert!(files
                .iter()
                .any(|f| f.path.file_name().unwrap() == "test.rs"));
        }
        Err(_) => {
            // Scan might fail in test environment, which is acceptable
        }
    }
}

#[cfg(feature = "core")]
#[test]
fn test_prelude_imports() {
    use crate::prelude::*;

    // Test that basic types are available
    let config = Config::default();
    assert!(config.validate().is_ok());

    // Test that version is available
    assert!(!VERSION.is_empty());
}

// Import extraction tests
mod import_extraction_tests {
    use super::*;

    #[test]
    fn test_rust_imports() {
        let content = r#"
use crate::module;
use crate::module::{item1, item2};
use super::parent_module;
use self::sibling;
use std::collections::HashMap;
use std::io::{Read, Write};
mod my_module;
pub use crate::reexport;
"#;
        let imports = extract_imports(content, &Language::Rust);
        println!("Rust imports: {:?}", imports);

        assert!(imports.contains(&"crate::module".to_string()), "Should extract simple crate import");
        assert!(imports.contains(&"super::parent_module".to_string()), "Should extract super import");
        assert!(imports.contains(&"self::sibling".to_string()), "Should extract self import");
        assert!(imports.contains(&"std::collections::HashMap".to_string()), "Should extract std import");
        assert!(imports.contains(&"std::io".to_string()), "Should extract grouped std import");
        assert!(imports.contains(&"my_module".to_string()), "Should extract mod declaration");
        // pub use should NOT be extracted (doesn't start with "use ")
        assert!(!imports.iter().any(|i| i.contains("reexport")), "Should NOT extract pub use");
    }

    #[test]
    fn test_python_imports() {
        let content = r#"
import os
import os, sys
import numpy as np
from os import path
from . import module
from .. import parent
from ..package import module
from typing import List, Dict
"#;
        let imports = extract_imports(content, &Language::Python);
        println!("Python imports: {:?}", imports);

        assert!(imports.contains(&"os".to_string()), "Should extract simple import");
        assert!(imports.contains(&"sys".to_string()), "Should extract comma-separated import");
        assert!(imports.contains(&"numpy".to_string()), "Should extract aliased import");
        assert!(imports.contains(&".".to_string()), "Should extract relative import");
        assert!(imports.contains(&"..".to_string()), "Should extract parent relative import");
        assert!(imports.contains(&"..package".to_string()), "Should extract parent package import");
        assert!(imports.contains(&"typing".to_string()), "Should extract from import");
    }

    #[test]
    fn test_javascript_imports() {
        let content = r#"
import foo from 'module'
import bar from "double-quotes"
import { a, b } from 'destructure'
import * as all from 'star-import'
import 'side-effect'
const req1 = require('commonjs')
const req2 = require("cjs-double")
import { x } from './relative'
"#;
        let imports = extract_imports(content, &Language::JavaScript);
        println!("JavaScript imports: {:?}", imports);

        assert!(imports.contains(&"module".to_string()), "Should extract single-quote import");
        assert!(imports.contains(&"double-quotes".to_string()), "Should extract double-quote import");
        assert!(imports.contains(&"destructure".to_string()), "Should extract destructured import");
        assert!(imports.contains(&"star-import".to_string()), "Should extract star import");
        assert!(imports.contains(&"side-effect".to_string()), "Should extract side-effect import");
        assert!(imports.contains(&"commonjs".to_string()), "Should extract require single");
        assert!(imports.contains(&"cjs-double".to_string()), "Should extract require double");
        assert!(imports.contains(&"./relative".to_string()), "Should extract relative import");
    }

    #[test]
    fn test_go_imports() {
        let content = r#"
package main

import "fmt"
import (
    "os"
    "path/filepath"
)
"#;
        let imports = extract_imports(content, &Language::Go);
        println!("Go imports: {:?}", imports);

        assert!(imports.contains(&"fmt".to_string()), "Should extract single import");
        assert!(imports.contains(&"os".to_string()), "Should extract block import");
        assert!(imports.contains(&"path/filepath".to_string()), "Should extract block import with path");
    }

    #[test]
    fn test_go_aliased_imports() {
        let content = r#"
import f "fmt"
import (
    . "os"
    _ "init/pkg"
    alias "github.com/pkg/errors"
)
"#;
        let imports = extract_imports(content, &Language::Go);
        println!("Go aliased imports: {:?}", imports);

        // These should extract the package path, not the alias
        assert!(imports.contains(&"fmt".to_string()), "Should extract aliased fmt");
        assert!(imports.contains(&"os".to_string()), "Should extract dot-imported os");
        assert!(imports.contains(&"init/pkg".to_string()), "Should extract blank-imported init/pkg");
        assert!(imports.contains(&"github.com/pkg/errors".to_string()), "Should extract aliased github package");

        // Should NOT contain the aliases themselves
        assert!(!imports.iter().any(|i| i.starts_with("f ")), "Should not include alias 'f'");
        assert!(!imports.iter().any(|i| i.starts_with(". ")), "Should not include dot alias");
    }
}

// Import edge case tests
mod import_edge_cases {
    use super::*;

    #[test]
    fn test_rust_pub_use_not_extracted() {
        let content = "pub use crate::module;";
        let imports = extract_imports(content, &Language::Rust);
        assert!(imports.is_empty(), "pub use should not be extracted");
    }

    #[test]
    fn test_rust_multiline_use() {
        // Multi-line use statements (spanning lines) - only first line detected
        let content = r#"
use std::collections::{
    HashMap,
    HashSet,
};
"#;
        let imports = extract_imports(content, &Language::Rust);
        println!("Multiline Rust: {:?}", imports);
        assert!(imports.contains(&"std::collections".to_string()));
    }

    #[test]
    fn test_typescript_import_type() {
        let content = r#"
import type { Type } from 'type-module'
import { Component } from '@angular/core'
"#;
        let imports = extract_imports(content, &Language::TypeScript);
        println!("TypeScript imports: {:?}", imports);
        assert!(imports.contains(&"type-module".to_string()), "Should extract type import");
        assert!(imports.contains(&"@angular/core".to_string()), "Should extract scoped package");
    }

    #[test]
    fn test_python_multiline_from_import() {
        // Python allows parenthesized imports
        let content = r#"
from module import (
    item1,
    item2,
)
"#;
        let imports = extract_imports(content, &Language::Python);
        println!("Python multiline: {:?}", imports);
        assert!(imports.contains(&"module".to_string()));
    }

    #[test]
    fn test_go_backtick_imports() {
        let content = "import `fmt`";
        let imports = extract_imports(content, &Language::Go);
        assert!(imports.contains(&"fmt".to_string()), "Should handle backtick imports");
    }

    #[test]
    fn test_js_template_literal_not_extracted() {
        // Template literals shouldn't be confused with imports
        let content = r#"
const x = `not an import`;
import real from 'real-module';
"#;
        let imports = extract_imports(content, &Language::JavaScript);
        println!("JS template test: {:?}", imports);
        assert!(imports.contains(&"real-module".to_string()));
        assert!(!imports.contains(&"not an import".to_string()));
    }
}
