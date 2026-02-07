//! Rule initialization for language detection.

use super::types::{ContentSignature, SyntaxAnalyzer};
use scribe_core::{Language, Result, ScribeError};
use std::collections::HashMap;
use tree_sitter::Parser;

/// Initialize file extension to language mapping
pub fn initialize_extension_map() -> HashMap<String, Vec<(Language, f32)>> {
    let extensions = vec![
        // Rust
        ("rs", vec![(Language::Rust, 1.0)]),
        // Python
        ("py", vec![(Language::Python, 0.95)]),
        ("pyw", vec![(Language::Python, 1.0)]),
        ("pyi", vec![(Language::Python, 1.0)]),
        // JavaScript/TypeScript
        ("js", vec![(Language::JavaScript, 0.9)]),
        ("jsx", vec![(Language::JavaScript, 1.0)]),
        ("mjs", vec![(Language::JavaScript, 1.0)]),
        ("ts", vec![(Language::TypeScript, 1.0)]),
        ("tsx", vec![(Language::TypeScript, 1.0)]),
        // Java/Kotlin/Scala
        ("java", vec![(Language::Java, 1.0)]),
        ("kt", vec![(Language::Kotlin, 1.0)]),
        ("kts", vec![(Language::Kotlin, 1.0)]),
        ("scala", vec![(Language::Scala, 1.0)]),
        ("sc", vec![(Language::Scala, 0.8)]),
        // C/C++
        ("c", vec![(Language::C, 0.9)]),
        ("h", vec![(Language::C, 0.7), (Language::Cpp, 0.3)]),
        ("cpp", vec![(Language::Cpp, 1.0)]),
        ("cxx", vec![(Language::Cpp, 1.0)]),
        ("cc", vec![(Language::Cpp, 1.0)]),
        ("hpp", vec![(Language::Cpp, 1.0)]),
        ("hxx", vec![(Language::Cpp, 1.0)]),
        // C#
        ("cs", vec![(Language::CSharp, 1.0)]),
        // Go
        ("go", vec![(Language::Go, 1.0)]),
        // Ruby
        ("rb", vec![(Language::Ruby, 1.0)]),
        ("rbw", vec![(Language::Ruby, 1.0)]),
        // PHP
        ("php", vec![(Language::PHP, 1.0)]),
        ("phtml", vec![(Language::PHP, 1.0)]),
        // Swift
        ("swift", vec![(Language::Swift, 1.0)]),
        // Dart
        ("dart", vec![(Language::Dart, 1.0)]),
        // Shell scripts
        ("sh", vec![(Language::Bash, 1.0)]),
        ("bash", vec![(Language::Bash, 1.0)]),
        ("zsh", vec![(Language::Bash, 1.0)]),
        ("fish", vec![(Language::Bash, 1.0)]),
        // Web technologies
        ("html", vec![(Language::HTML, 1.0)]),
        ("htm", vec![(Language::HTML, 1.0)]),
        ("css", vec![(Language::CSS, 1.0)]),
        ("scss", vec![(Language::SCSS, 1.0)]),
        ("sass", vec![(Language::SASS, 1.0)]),
        // Markup and data formats
        ("md", vec![(Language::Markdown, 1.0)]),
        ("markdown", vec![(Language::Markdown, 1.0)]),
        ("xml", vec![(Language::XML, 1.0)]),
        ("json", vec![(Language::JSON, 1.0)]),
        ("yaml", vec![(Language::YAML, 1.0)]),
        ("yml", vec![(Language::YAML, 1.0)]),
        ("toml", vec![(Language::TOML, 1.0)]),
        // Configuration
        ("ini", vec![(Language::Unknown, 1.0)]),
        ("cfg", vec![(Language::Unknown, 0.8)]),
        ("conf", vec![(Language::Unknown, 0.7)]),
        // SQL
        ("sql", vec![(Language::SQL, 1.0)]),
        // Documentation
        ("rst", vec![(Language::Unknown, 1.0)]),
        ("tex", vec![(Language::Unknown, 1.0)]),
        // Other languages
        ("r", vec![(Language::R, 1.0)]),
        ("R", vec![(Language::R, 1.0)]),
        (
            "m",
            vec![(Language::ObjectiveC, 0.6), (Language::Matlab, 0.4)],
        ),
        ("mm", vec![(Language::ObjectiveC, 1.0)]),
        ("pl", vec![(Language::Unknown, 0.8)]),
        ("pm", vec![(Language::Unknown, 1.0)]),
        ("lua", vec![(Language::Unknown, 1.0)]),
        ("vim", vec![(Language::Unknown, 1.0)]),
        ("hs", vec![(Language::Haskell, 1.0)]),
        ("lhs", vec![(Language::Haskell, 1.0)]),
        ("ex", vec![(Language::Elixir, 1.0)]),
        ("exs", vec![(Language::Elixir, 1.0)]),
    ];

    let mut map = HashMap::new();
    for (ext, languages) in extensions {
        map.insert(ext.to_string(), languages);
    }
    map
}

/// Initialize filename patterns for special files
pub fn initialize_filename_patterns() -> HashMap<String, Language> {
    let patterns = vec![
        ("Makefile", Language::Unknown),
        ("makefile", Language::Unknown),
        ("Dockerfile", Language::Unknown),
        ("dockerfile", Language::Unknown),
        ("Cargo.toml", Language::TOML),
        ("Cargo.lock", Language::TOML),
        ("package.json", Language::JSON),
        ("tsconfig.json", Language::JSON),
        ("pyproject.toml", Language::TOML),
        ("setup.py", Language::Python),
        ("requirements.txt", Language::Unknown),
        ("README", Language::Unknown),
        ("LICENSE", Language::Unknown),
        ("CHANGELOG", Language::Unknown),
        ("CMakeLists.txt", Language::Unknown),
        (".gitignore", Language::Unknown),
        (".dockerignore", Language::Unknown),
        ("Jenkinsfile", Language::Unknown),
        ("build.gradle", Language::Unknown),
        ("pom.xml", Language::XML),
    ];

    let mut map = HashMap::new();
    for (filename, language) in patterns {
        map.insert(filename.to_string(), language);
    }
    map
}

/// Initialize shebang patterns
pub fn initialize_shebang_patterns() -> HashMap<String, Language> {
    let patterns = vec![
        ("python", Language::Python),
        ("python3", Language::Python),
        ("python2", Language::Python),
        ("node", Language::JavaScript),
        ("bash", Language::Bash),
        ("sh", Language::Bash),
        ("zsh", Language::Bash),
        ("fish", Language::Bash),
        ("ruby", Language::Ruby),
        ("php", Language::PHP),
        ("elixir", Language::Elixir),
        ("env python", Language::Python),
        ("env node", Language::JavaScript),
        ("env bash", Language::Bash),
        ("env ruby", Language::Ruby),
        ("env elixir", Language::Elixir),
    ];

    let mut map = HashMap::new();
    for (pattern, language) in patterns {
        map.insert(pattern.to_string(), language);
    }
    map
}

/// Compile regex patterns, logging errors but not failing
pub fn compile_patterns(patterns: Vec<&str>) -> Result<Vec<regex::Regex>> {
    let mut compiled = Vec::new();
    for pattern in patterns {
        match regex::Regex::new(pattern) {
            Ok(regex) => compiled.push(regex),
            Err(e) => {
                log::warn!("Failed to compile regex pattern '{}': {}", pattern, e);
                return Err(ScribeError::pattern(
                    format!("Failed to compile regex pattern: {}", e),
                    pattern.to_string(),
                ));
            }
        }
    }
    Ok(compiled)
}

/// Initialize content signatures for language detection with pre-compiled regexes
pub fn initialize_content_signatures() -> HashMap<Language, Vec<ContentSignature>> {
    let mut signatures = HashMap::new();

    // Python signatures
    let python_patterns = vec![
        r"def\s+\w+\s*\(",
        r"import\s+\w+",
        r"from\s+\w+\s+import",
        r"class\s+\w+\s*\(",
        r"__\w+__",
    ];
    if let Ok(compiled_patterns) = compile_patterns(python_patterns) {
        let python_sigs = vec![ContentSignature {
            language: Language::Python,
            patterns: compiled_patterns,
            weight: 0.9,
            required_matches: 2,
        }];
        signatures.insert(Language::Python, python_sigs);
    }

    // JavaScript signatures
    let js_patterns = vec![
        r"function\s+\w+\s*\(",
        r"const\s+\w+\s*=",
        r"let\s+\w+\s*=",
        r"=>\s*\{",
        r"require\s*\(",
        r"console\.log\s*\(",
    ];
    if let Ok(compiled_patterns) = compile_patterns(js_patterns) {
        let js_sigs = vec![ContentSignature {
            language: Language::JavaScript,
            patterns: compiled_patterns,
            weight: 0.8,
            required_matches: 2,
        }];
        signatures.insert(Language::JavaScript, js_sigs);
    }

    // Rust signatures
    let rust_patterns = vec![
        r"fn\s+\w+\s*\(",
        r"use\s+[\w:]+",
        r"struct\s+\w+",
        r"impl\s+[\w<>]+",
        r"let\s+mut\s+\w+",
        r"match\s+\w+\s*\{",
    ];
    if let Ok(compiled_patterns) = compile_patterns(rust_patterns) {
        let rust_sigs = vec![ContentSignature {
            language: Language::Rust,
            patterns: compiled_patterns,
            weight: 0.95,
            required_matches: 2,
        }];
        signatures.insert(Language::Rust, rust_sigs);
    }

    // Elixir signatures
    let elixir_patterns = vec![
        r"defmodule\s+[A-Z][\w\.]*\s+do",
        r"def\s+\w+\s*\(",
        r"alias\s+[A-Z][\w\.]*",
        r"use\s+[A-Z][\w\.]*",
        r"@moduledoc",
    ];
    if let Ok(compiled_patterns) = compile_patterns(elixir_patterns) {
        let elixir_sigs = vec![ContentSignature {
            language: Language::Elixir,
            patterns: compiled_patterns,
            weight: 0.9,
            required_matches: 2,
        }];
        signatures.insert(Language::Elixir, elixir_sigs);
    }

    signatures
}

/// Initialize AST parsers for content analysis
pub fn initialize_ast_parsers(
    ts_languages: &HashMap<Language, fn() -> tree_sitter::Language>,
) -> HashMap<Language, Parser> {
    let mut parsers = HashMap::new();
    for (language, ts_lang_fn) in ts_languages.iter() {
        let mut parser = Parser::new();
        if parser.set_language(ts_lang_fn()).is_ok() {
            parsers.insert(language.clone(), parser);
        }
    }
    parsers
}

/// Initialize syntax analyzers for AST-based content analysis
pub fn initialize_syntax_analyzers() -> HashMap<Language, SyntaxAnalyzer> {
    let mut analyzers = HashMap::new();

    // Python syntax analyzer
    let python_analyzer = SyntaxAnalyzer {
        language: Language::Python,
        keywords: vec![
            "def".to_string(),
            "class".to_string(),
            "import".to_string(),
            "from".to_string(),
            "if".to_string(),
            "elif".to_string(),
        ],
        structural_patterns: vec![
            "function_definition".to_string(),
            "class_definition".to_string(),
            "import_statement".to_string(),
            "import_from_statement".to_string(),
        ],
        confidence_weights: HashMap::from([
            ("function_definition".to_string(), 0.9),
            ("class_definition".to_string(), 0.9),
            ("import_statement".to_string(), 0.8),
        ]),
    };
    analyzers.insert(Language::Python, python_analyzer);

    // JavaScript/TypeScript syntax analyzer
    let js_analyzer = SyntaxAnalyzer {
        language: Language::JavaScript,
        keywords: vec![
            "function".to_string(),
            "class".to_string(),
            "import".to_string(),
            "const".to_string(),
            "let".to_string(),
            "var".to_string(),
        ],
        structural_patterns: vec![
            "function_declaration".to_string(),
            "class_declaration".to_string(),
            "import_statement".to_string(),
            "variable_declaration".to_string(),
        ],
        confidence_weights: HashMap::from([
            ("function_declaration".to_string(), 0.9),
            ("class_declaration".to_string(), 0.9),
            ("import_statement".to_string(), 0.8),
        ]),
    };
    analyzers.insert(Language::JavaScript, js_analyzer);

    // Rust syntax analyzer
    let rust_analyzer = SyntaxAnalyzer {
        language: Language::Rust,
        keywords: vec![
            "fn".to_string(),
            "struct".to_string(),
            "enum".to_string(),
            "impl".to_string(),
            "use".to_string(),
            "mod".to_string(),
        ],
        structural_patterns: vec![
            "function_item".to_string(),
            "struct_item".to_string(),
            "enum_item".to_string(),
            "use_declaration".to_string(),
        ],
        confidence_weights: HashMap::from([
            ("function_item".to_string(), 0.9),
            ("struct_item".to_string(), 0.9),
            ("use_declaration".to_string(), 0.8),
        ]),
    };
    analyzers.insert(Language::Rust, rust_analyzer);

    // Elixir syntax analyzer
    let elixir_analyzer = SyntaxAnalyzer {
        language: Language::Elixir,
        keywords: vec![
            "defmodule".to_string(),
            "def".to_string(),
            "defp".to_string(),
            "alias".to_string(),
            "import".to_string(),
            "use".to_string(),
        ],
        structural_patterns: vec![
            "call".to_string(),
            "identifier".to_string(),
            "alias".to_string(),
        ],
        confidence_weights: HashMap::from([
            ("call".to_string(), 0.8),
            ("alias".to_string(), 0.8),
        ]),
    };
    analyzers.insert(Language::Elixir, elixir_analyzer);

    analyzers
}
