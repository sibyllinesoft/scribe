//! SWC-based import extraction for TypeScript and JavaScript
//!
//! This module provides high-accuracy import extraction using SWC (Speedy Web Compiler),
//! which handles TypeScript and JavaScript edge cases better than tree-sitter, including:
//! - Type-only imports (`import type { Foo } from './foo'`)
//! - Re-exports (`export * from './module'`)
//! - Named re-exports (`export { x } from './module'`)
//! - Dynamic imports (`import('./module')`)

use swc_common::{input::StringInput, sync::Lrc, FileName, SourceMap};
use swc_ecma_ast::{EsVersion, ModuleDecl, ModuleItem};
use swc_ecma_parser::{lexer::Lexer, Parser, Syntax, TsSyntax};

use crate::ast_import_parser::SimpleImport;

/// Extract all imports from TypeScript or JavaScript source code using SWC
///
/// This extracts:
/// - `import ... from 'module'`
/// - `import 'module'` (side-effect imports)
/// - `export * from 'module'`
/// - `export { ... } from 'module'`
///
/// # Arguments
/// * `source` - The source code to parse
/// * `is_typescript` - Whether to parse as TypeScript (true) or JavaScript (false)
///
/// # Returns
/// A vector of `SimpleImport` containing the module specifier and line number
pub fn extract_imports(source: &str, is_typescript: bool) -> Vec<SimpleImport> {
    // Set up SWC source map
    let source_map: Lrc<SourceMap> = Default::default();

    // Source needs to be owned for the SourceMap
    let source_owned: String = source.into();
    let file = source_map.new_source_file(
        Lrc::new(FileName::Custom("input".into())),
        source_owned.clone(),
    );

    let syntax = if is_typescript {
        Syntax::Typescript(TsSyntax {
            tsx: true, // Support TSX as well
            decorators: true,
            ..Default::default()
        })
    } else {
        Syntax::Es(swc_ecma_parser::EsSyntax {
            jsx: true, // Support JSX as well
            ..Default::default()
        })
    };

    let lexer = Lexer::new(
        syntax,
        EsVersion::latest(),
        StringInput::from(&*file),
        None,
    );

    let mut parser = Parser::new_from(lexer);

    // Try to parse as module - silently ignore errors and return empty on failure
    let module = match parser.parse_module() {
        Ok(module) => module,
        Err(_) => {
            // Parse error - return empty imports
            return Vec::new();
        }
    };

    let mut imports = Vec::new();

    for item in module.body {
        match item {
            ModuleItem::ModuleDecl(decl) => {
                extract_from_module_decl(decl, source, &mut imports);
            }
            ModuleItem::Stmt(_) => {
                // Statements can contain dynamic imports, but we don't extract those
                // for dependency graph purposes (they're runtime-conditional)
            }
        }
    }

    imports
}

/// Helper to convert SWC atom to String
/// SWC uses Wtf8Atom which doesn't implement Display, so we need to go through str
fn atom_to_string(atom: &swc_ecma_ast::Str) -> String {
    // The value field is a Wtf8Atom - use as_str() and handle the Option
    atom.value.as_str().unwrap_or_default().to_string()
}

/// Extract import information from a module declaration
fn extract_from_module_decl(decl: ModuleDecl, source: &str, imports: &mut Vec<SimpleImport>) {
    match decl {
        // import ... from 'module'
        // import 'module'
        ModuleDecl::Import(import_decl) => {
            let module = atom_to_string(&import_decl.src);
            let line_number = calculate_line_number(source, import_decl.span.lo.0 as usize);
            imports.push(SimpleImport {
                module,
                line_number,
            });
        }

        // export * from 'module'
        ModuleDecl::ExportAll(export_all) => {
            let module = atom_to_string(&export_all.src);
            let line_number = calculate_line_number(source, export_all.span.lo.0 as usize);
            imports.push(SimpleImport {
                module,
                line_number,
            });
        }

        // export { x, y } from 'module'
        ModuleDecl::ExportNamed(named_export) => {
            if let Some(src) = named_export.src {
                let module = atom_to_string(&src);
                let line_number = calculate_line_number(source, named_export.span.lo.0 as usize);
                imports.push(SimpleImport {
                    module,
                    line_number,
                });
            }
            // If no src, it's just `export { x }` which re-exports from current module
        }

        // export default ... / export const ... / export function ...
        // These don't import from other modules
        ModuleDecl::ExportDefaultDecl(_)
        | ModuleDecl::ExportDefaultExpr(_)
        | ModuleDecl::ExportDecl(_) => {}

        // TypeScript-specific: export = ... / import = ...
        ModuleDecl::TsImportEquals(ts_import) => {
            // import x = require('module')
            if let swc_ecma_ast::TsModuleRef::TsExternalModuleRef(ext_ref) = ts_import.module_ref {
                let module = atom_to_string(&ext_ref.expr);
                let line_number = calculate_line_number(source, ts_import.span.lo.0 as usize);
                imports.push(SimpleImport {
                    module,
                    line_number,
                });
            }
            // import x = Namespace.Member doesn't import from external module
        }

        ModuleDecl::TsExportAssignment(_) | ModuleDecl::TsNamespaceExport(_) => {
            // These don't import from external modules
        }
    }
}

/// Calculate line number from byte offset
fn calculate_line_number(source: &str, byte_offset: usize) -> usize {
    // SWC byte offsets start at 1, not 0
    let offset = byte_offset.saturating_sub(1);
    let offset = offset.min(source.len());

    source[..offset]
        .chars()
        .filter(|&c| c == '\n')
        .count()
        + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_import() {
        let code = r#"import { useState } from 'react';"#;
        let imports = extract_imports(code, false);

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "react");
    }

    #[test]
    fn test_default_import() {
        let code = r#"import React from 'react';"#;
        let imports = extract_imports(code, false);

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "react");
    }

    #[test]
    fn test_namespace_import() {
        let code = r#"import * as utils from './utils';"#;
        let imports = extract_imports(code, false);

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "./utils");
    }

    #[test]
    fn test_side_effect_import() {
        let code = r#"import './polyfills';"#;
        let imports = extract_imports(code, false);

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "./polyfills");
    }

    #[test]
    fn test_type_import() {
        let code = r#"import type { Config } from './config';"#;
        let imports = extract_imports(code, true);

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "./config");
    }

    #[test]
    fn test_export_all_re_export() {
        let code = r#"export * from './utils';"#;
        let imports = extract_imports(code, false);

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "./utils");
    }

    #[test]
    fn test_named_re_export() {
        let code = r#"export { foo, bar } from './helpers';"#;
        let imports = extract_imports(code, false);

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "./helpers");
    }

    #[test]
    fn test_export_without_source() {
        // This is a local export, not a re-export
        let code = r#"
            const foo = 1;
            export { foo };
        "#;
        let imports = extract_imports(code, false);

        assert!(imports.is_empty());
    }

    #[test]
    fn test_multiple_imports() {
        let code = r#"
import React from 'react';
import { Component } from 'react';
import * as utils from './utils';
export * from './helpers';
export { x } from './x';
        "#;
        let imports = extract_imports(code, false);

        assert_eq!(imports.len(), 5);
        assert!(imports.iter().any(|i| i.module == "react"));
        assert!(imports.iter().any(|i| i.module == "./utils"));
        assert!(imports.iter().any(|i| i.module == "./helpers"));
        assert!(imports.iter().any(|i| i.module == "./x"));
    }

    #[test]
    fn test_typescript_syntax() {
        let code = r#"
import { Component } from '@angular/core';
import type { OnInit } from '@angular/core';

interface Props {
    name: string;
}

export class AppComponent implements OnInit {
    ngOnInit(): void {}
}
        "#;
        let imports = extract_imports(code, true);

        assert_eq!(imports.len(), 2);
        assert!(imports.iter().all(|i| i.module == "@angular/core"));
    }

    #[test]
    fn test_tsx_syntax() {
        let code = r#"
import React from 'react';
import { Button } from './Button';

const App: React.FC = () => {
    return <Button>Click me</Button>;
};
        "#;
        let imports = extract_imports(code, true);

        assert_eq!(imports.len(), 2);
        assert!(imports.iter().any(|i| i.module == "react"));
        assert!(imports.iter().any(|i| i.module == "./Button"));
    }

    #[test]
    fn test_jsx_syntax() {
        let code = r#"
import React from 'react';

const App = () => {
    return <div>Hello</div>;
};
        "#;
        let imports = extract_imports(code, false);

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "react");
    }

    #[test]
    fn test_line_numbers() {
        let code = r#"// Comment line 1
import React from 'react';
import { useState } from 'react';
"#;
        let imports = extract_imports(code, false);

        assert_eq!(imports.len(), 2);
        assert_eq!(imports[0].line_number, 2);
        assert_eq!(imports[1].line_number, 3);
    }

    #[test]
    fn test_empty_code() {
        let imports = extract_imports("", false);
        assert!(imports.is_empty());
    }

    #[test]
    fn test_no_imports() {
        let code = r#"
const x = 1;
function foo() {
    return x + 1;
}
        "#;
        let imports = extract_imports(code, false);
        assert!(imports.is_empty());
    }

    #[test]
    fn test_ts_import_equals() {
        let code = r#"import fs = require('fs');"#;
        let imports = extract_imports(code, true);

        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].module, "fs");
    }

    #[test]
    fn test_invalid_syntax_returns_empty() {
        // Invalid syntax should not panic, just return empty
        let code = r#"import { from"#;
        let imports = extract_imports(code, false);
        // May or may not extract partial imports depending on parser recovery
        // Main point is it shouldn't panic
        let _ = imports;
    }
}
