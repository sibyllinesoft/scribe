"""Language-specific parsing support using tree-sitter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, List, Set

try:
    import tree_sitter
    import tree_sitter_python as tspython
    import tree_sitter_typescript as tsjs
    import tree_sitter_go as tsgo
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    tree_sitter = None
    tspython = None
    tsjs = None
    tsgo = None

from .base import ChunkKind


class LanguageSupport:
    """
    Language-specific parsing and chunk extraction.
    
    Uses tree-sitter grammars to identify semantic code structures
    and extract meaningful chunks with dependency information.
    """
    
    # Language configurations
    LANGUAGES = {
        'python': {
            'extensions': {'.py'},
            'parser_lib': 'tspython',
            'function_query': '(function_def name: (identifier) @name)',
            'class_query': '(class_def name: (identifier) @name)',
            'import_query': '(import_statement) @import',
        },
        'typescript': {
            'extensions': {'.ts', '.tsx'},
            'parser_lib': 'tsjs', 
            'function_query': '(function_declaration name: (identifier) @name)',
            'class_query': '(class_declaration name: (identifier) @name)',
            'import_query': '(import_statement) @import',
        },
        'javascript': {
            'extensions': {'.js', '.jsx'},
            'parser_lib': 'tsjs',
            'function_query': '(function_declaration name: (identifier) @name)',
            'class_query': '(class_declaration name: (identifier) @name)',
            'import_query': '(import_statement) @import',
        },
        'go': {
            'extensions': {'.go'},
            'parser_lib': 'tsgo',
            'function_query': '(function_declaration name: (identifier) @name)',
            'class_query': '(type_declaration (type_spec name: (type_identifier) @name))',
            'import_query': '(import_declaration) @import',
        }
    }
    
    def __init__(self):
        """Initialize language support with available parsers."""
        self._parsers: Dict[str, Optional[tree_sitter.Parser]] = {}
        self._load_parsers()
    
    def _load_parsers(self):
        """Load available tree-sitter parsers."""
        if not TREE_SITTER_AVAILABLE:
            return
            
        try:
            # Python parser
            if tspython:
                python_parser = tree_sitter.Parser()
                python_parser.set_language(tspython.language())
                self._parsers['python'] = python_parser
            
            # TypeScript/JavaScript parser
            if tsjs:
                ts_parser = tree_sitter.Parser() 
                ts_parser.set_language(tsjs.language())
                self._parsers['typescript'] = ts_parser
                self._parsers['javascript'] = ts_parser
            
            # Go parser
            if tsgo:
                go_parser = tree_sitter.Parser()
                go_parser.set_language(tsgo.language())
                self._parsers['go'] = go_parser
                
        except Exception as e:
            # Log error but continue with degraded functionality
            print(f"Warning: Failed to initialize tree-sitter parsers: {e}")
    
    def detect_language(self, file_path: Path) -> Optional[str]:
        """
        Detect programming language from file extension.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            Language identifier or None if not supported
        """
        extension = file_path.suffix.lower()
        
        for lang, config in self.LANGUAGES.items():
            if extension in config['extensions']:
                return lang
                
        return None
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages with available parsers."""
        return [lang for lang in self.LANGUAGES.keys() if self._parsers.get(lang)]
    
    def is_supported(self, language: str) -> bool:
        """Check if a language is supported."""
        return language in self._parsers and self._parsers[language] is not None
    
    def parse_file(self, file_path: Path, content: str) -> Optional[tree_sitter.Tree]:
        """
        Parse a source file using appropriate tree-sitter parser.
        
        Args:
            file_path: Path to the source file
            content: File content as string
            
        Returns:
            Parsed syntax tree or None if parsing failed
        """
        language = self.detect_language(file_path)
        if not language or not self.is_supported(language):
            return None
            
        parser = self._parsers[language]
        try:
            return parser.parse(content.encode('utf-8'))
        except Exception as e:
            print(f"Warning: Failed to parse {file_path}: {e}")
            return None
    
    def extract_function_nodes(self, tree: tree_sitter.Tree, language: str) -> List[tree_sitter.Node]:
        """Extract function definition nodes from syntax tree."""
        if not tree or language not in self.LANGUAGES:
            return []
            
        query_str = self.LANGUAGES[language].get('function_query', '')
        if not query_str:
            return []
            
        try:
            query = self._parsers[language].language.query(query_str)
            captures = query.captures(tree.root_node)
            return [node for node, _ in captures]
        except Exception:
            return []
    
    def extract_class_nodes(self, tree: tree_sitter.Tree, language: str) -> List[tree_sitter.Node]:
        """Extract class definition nodes from syntax tree.""" 
        if not tree or language not in self.LANGUAGES:
            return []
            
        query_str = self.LANGUAGES[language].get('class_query', '')
        if not query_str:
            return []
            
        try:
            query = self._parsers[language].language.query(query_str)
            captures = query.captures(tree.root_node)
            return [node for node, _ in captures]
        except Exception:
            return []
    
    def extract_import_nodes(self, tree: tree_sitter.Tree, language: str) -> List[tree_sitter.Node]:
        """Extract import statement nodes from syntax tree."""
        if not tree or language not in self.LANGUAGES:
            return []
            
        query_str = self.LANGUAGES[language].get('import_query', '')
        if not query_str:
            return []
            
        try:
            query = self._parsers[language].language.query(query_str)
            captures = query.captures(tree.root_node)  
            return [node for node, _ in captures]
        except Exception:
            return []
    
    def get_node_text(self, node: tree_sitter.Node, content: bytes) -> str:
        """Extract text content from a syntax tree node."""
        return content[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
    
    def get_node_name(self, node: tree_sitter.Node, content: bytes) -> Optional[str]:
        """Extract the name/identifier from a node (function, class, etc.)."""
        try:
            # Look for identifier child nodes
            for child in node.children:
                if child.type == 'identifier':
                    return self.get_node_text(child, content)
            return None
        except Exception:
            return None
    
    def calculate_complexity(self, node: tree_sitter.Node) -> float:
        """
        Calculate code complexity score for a node.
        
        Simple heuristic based on nesting depth and node count.
        """
        def count_nodes(n: tree_sitter.Node, depth: int = 0) -> tuple[int, int]:
            node_count = 1
            max_depth = depth
            
            for child in n.children:
                child_count, child_depth = count_nodes(child, depth + 1)
                node_count += child_count
                max_depth = max(max_depth, child_depth)
            
            return node_count, max_depth
        
        try:
            node_count, max_depth = count_nodes(node)
            # Complexity score combining node count and nesting
            return min(10.0, (node_count / 10.0) + (max_depth / 5.0))
        except Exception:
            return 1.0  # Default complexity
    
    def generate_chunk_id(self, file_path: Path, node: tree_sitter.Node, name: str) -> str:
        """
        Generate deterministic chunk ID.
        
        Uses file path, node position, and name for reproducible IDs.
        """
        identifier = f"{file_path}:{node.start_point[0]}:{node.start_point[1]}:{name}"
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]