"""Main code chunker implementation using tree-sitter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple

from .base import Chunk, ChunkKind, ChunkDependency
from .languages import LanguageSupport
from ..tokenizer.base import Tokenizer


class CodeChunker:
    """
    Tree-sitter based semantic code chunker.
    
    Analyzes source code files to extract meaningful semantic chunks
    (functions, classes, imports, etc.) with dependency relationships
    and characteristics for selection algorithms.
    """
    
    def __init__(self, tokenizer: Tokenizer):
        """
        Initialize the chunker.
        
        Args:
            tokenizer: Tokenizer for calculating token costs
        """
        self.tokenizer = tokenizer
        self.language_support = LanguageSupport()
        self._chunk_cache: Dict[str, List[Chunk]] = {}
    
    def chunk_file(self, file_path: Path, content: str, repo_root: Path) -> List[Chunk]:
        """
        Extract semantic chunks from a single file.
        
        Args:
            file_path: Absolute path to the source file
            content: File content as string
            repo_root: Repository root path for relative path calculation
            
        Returns:
            List of extracted chunks
        """
        # Calculate relative path
        rel_path = str(file_path.relative_to(repo_root)).replace('\\', '/')
        
        # Check cache
        cache_key = f"{file_path}:{hash(content)}"
        if cache_key in self._chunk_cache:
            return self._chunk_cache[cache_key]
        
        chunks = []
        
        # Detect language
        language = self.language_support.detect_language(file_path)
        if not language or not self.language_support.is_supported(language):
            # Fallback to simple text chunking
            chunks = self._fallback_chunk_file(file_path, content, repo_root)
        else:
            # Use tree-sitter parsing
            chunks = self._tree_sitter_chunk_file(file_path, content, repo_root, language)
        
        # Cache results
        self._chunk_cache[cache_key] = chunks
        return chunks
    
    def _tree_sitter_chunk_file(
        self, 
        file_path: Path, 
        content: str, 
        repo_root: Path,
        language: str
    ) -> List[Chunk]:
        """Extract chunks using tree-sitter parsing."""
        rel_path = str(file_path.relative_to(repo_root)).replace('\\', '/')
        content_bytes = content.encode('utf-8')
        
        # Parse the file
        tree = self.language_support.parse_file(file_path, content)
        if not tree:
            return self._fallback_chunk_file(file_path, content, repo_root)
        
        chunks = []
        content_lines = content.splitlines(keepends=False)
        
        # Extract imports first
        import_nodes = self.language_support.extract_import_nodes(tree, language)
        for node in import_nodes:
            chunk = self._create_chunk_from_node(
                node, ChunkKind.IMPORT, file_path, rel_path, language,
                content_bytes, content_lines, "import"
            )
            if chunk:
                chunks.append(chunk)
        
        # Extract classes
        class_nodes = self.language_support.extract_class_nodes(tree, language)
        for node in class_nodes:
            name = self.language_support.get_node_name(node, content_bytes) or "unknown_class"
            chunk = self._create_chunk_from_node(
                node, ChunkKind.CLASS, file_path, rel_path, language,
                content_bytes, content_lines, name
            )
            if chunk:
                chunks.append(chunk)
        
        # Extract functions
        function_nodes = self.language_support.extract_function_nodes(tree, language)
        for node in function_nodes:
            name = self.language_support.get_node_name(node, content_bytes) or "unknown_function"
            # Determine if it's a test function
            chunk_kind = ChunkKind.TEST if self._is_test_function(name, file_path) else ChunkKind.FUNCTION
            chunk = self._create_chunk_from_node(
                node, chunk_kind, file_path, rel_path, language,
                content_bytes, content_lines, name
            )
            if chunk:
                chunks.append(chunk)
        
        # Add module-level docstring if present
        module_docstring = self._extract_module_docstring(content, language)
        if module_docstring:
            chunk = Chunk(
                id=self.language_support.generate_chunk_id(file_path, None, "module_docstring"),
                path=file_path,
                rel_path=rel_path,
                start_line=1,
                end_line=len(module_docstring.splitlines()),
                kind=ChunkKind.DOCSTRING,
                name="module_docstring",
                language=language,
                content=module_docstring,
                docstring=module_docstring,
                doc_density=1.0,  # Pure documentation
            )
            self._calculate_tokens(chunk)
            chunks.append(chunk)
        
        return chunks
    
    def _fallback_chunk_file(self, file_path: Path, content: str, repo_root: Path) -> List[Chunk]:
        """
        Fallback chunking for unsupported languages or parse failures.
        
        Uses simple heuristics to identify functions and classes.
        """
        rel_path = str(file_path.relative_to(repo_root)).replace('\\', '/')
        chunks = []
        lines = content.splitlines(keepends=False)
        
        # Simple pattern-based extraction
        current_chunk = None
        chunk_start = 1
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Function patterns
            if re.match(r'^\s*(def|function|func)\s+\w+', line):
                if current_chunk:
                    # Finalize previous chunk
                    current_chunk['end_line'] = i - 1
                    chunks.append(self._create_fallback_chunk(current_chunk, file_path, rel_path, lines))
                
                # Start new function chunk
                name_match = re.search(r'^\s*(?:def|function|func)\s+(\w+)', line)
                name = name_match.group(1) if name_match else "unknown_function"
                current_chunk = {
                    'start_line': i,
                    'kind': ChunkKind.TEST if self._is_test_function(name, file_path) else ChunkKind.FUNCTION,
                    'name': name
                }
            
            # Class patterns
            elif re.match(r'^\s*class\s+\w+', line):
                if current_chunk:
                    current_chunk['end_line'] = i - 1
                    chunks.append(self._create_fallback_chunk(current_chunk, file_path, rel_path, lines))
                
                name_match = re.search(r'^\s*class\s+(\w+)', line)
                name = name_match.group(1) if name_match else "unknown_class"
                current_chunk = {
                    'start_line': i,
                    'kind': ChunkKind.CLASS,
                    'name': name
                }
        
        # Finalize last chunk
        if current_chunk:
            current_chunk['end_line'] = len(lines)
            chunks.append(self._create_fallback_chunk(current_chunk, file_path, rel_path, lines))
        
        # If no structured chunks found, create one chunk for the entire file
        if not chunks:
            chunk = Chunk(
                id=f"file_{hash(str(file_path))}_{len(content)}",
                path=file_path,
                rel_path=rel_path,
                start_line=1,
                end_line=len(lines),
                kind=ChunkKind.UNKNOWN,
                name=file_path.stem,
                language="unknown",
                content=content,
            )
            self._calculate_tokens(chunk)
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_from_node(
        self,
        node,  # tree_sitter.Node
        kind: ChunkKind,
        file_path: Path,
        rel_path: str,
        language: str,
        content_bytes: bytes,
        content_lines: List[str],
        name: str
    ) -> Optional[Chunk]:
        """Create a Chunk from a tree-sitter node."""
        try:
            # Get node boundaries
            start_line = node.start_point[0] + 1  # Convert to 1-based
            end_line = node.end_point[0] + 1
            
            # Extract content
            node_content = self.language_support.get_node_text(node, content_bytes)
            
            # Extract docstring if present
            docstring = self._extract_docstring(node_content, language)
            
            # Calculate metrics
            complexity = self.language_support.calculate_complexity(node)
            doc_density = self._calculate_doc_density(node_content, docstring)
            
            # Generate deterministic ID
            chunk_id = self.language_support.generate_chunk_id(file_path, node, name)
            
            # Create signature
            signature = self._extract_signature(node_content, language, kind)
            
            chunk = Chunk(
                id=chunk_id,
                path=file_path,
                rel_path=rel_path,
                start_line=start_line,
                end_line=end_line,
                kind=kind,
                name=name,
                language=language,
                content=node_content,
                signature=signature,
                docstring=docstring,
                doc_density=doc_density,
                complexity_score=complexity,
            )
            
            # Calculate token costs
            self._calculate_tokens(chunk)
            
            return chunk
            
        except Exception as e:
            print(f"Warning: Failed to create chunk from node in {file_path}: {e}")
            return None
    
    def _create_fallback_chunk(
        self, 
        chunk_info: Dict, 
        file_path: Path, 
        rel_path: str, 
        lines: List[str]
    ) -> Chunk:
        """Create chunk from fallback parsing info."""
        start_line = chunk_info['start_line']
        end_line = chunk_info['end_line']
        content = '\n'.join(lines[start_line-1:end_line])
        
        chunk = Chunk(
            id=f"fallback_{hash(str(file_path))}_{start_line}_{chunk_info['name']}",
            path=file_path,
            rel_path=rel_path,
            start_line=start_line,
            end_line=end_line,
            kind=chunk_info['kind'],
            name=chunk_info['name'],
            language="unknown",
            content=content,
        )
        
        self._calculate_tokens(chunk)
        return chunk
    
    def _calculate_tokens(self, chunk: Chunk):
        """Calculate token costs for different fidelity modes."""
        # Full content tokens
        chunk.full_tokens = self.tokenizer.count_tokens(chunk.content)
        
        # Signature tokens (if available)
        if chunk.signature:
            chunk.signature_tokens = self.tokenizer.count_tokens(chunk.signature)
        else:
            # Estimate signature tokens as ~10% of full content
            chunk.signature_tokens = max(1, chunk.full_tokens // 10)
    
    def _extract_docstring(self, content: str, language: str) -> Optional[str]:
        """Extract docstring from content."""
        if language == 'python':
            # Look for triple-quoted strings at the start
            match = re.match(r'^\s*"""(.*?)"""|^\s*\'\'\'(.*?)\'\'\'', content, re.DOTALL)
            if match:
                return match.group(1) or match.group(2)
        
        # For other languages, look for leading comments
        lines = content.splitlines()
        docstring_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('#') or stripped.startswith('/*'):
                docstring_lines.append(stripped)
            elif stripped and not stripped.startswith('*'):
                break
        
        if docstring_lines:
            return '\n'.join(docstring_lines)
        
        return None
    
    def _extract_signature(self, content: str, language: str, kind: ChunkKind) -> Optional[str]:
        """Extract type signature or interface from content."""
        lines = content.splitlines()
        if not lines:
            return None
        
        # For functions and methods, extract the signature line
        if kind in [ChunkKind.FUNCTION, ChunkKind.METHOD]:
            for line in lines:
                stripped = line.strip()
                if any(keyword in stripped for keyword in ['def ', 'function ', 'func ']):
                    # Extract just the signature line
                    if ':' in stripped and language == 'python':
                        return stripped.split(':')[0] + ':'
                    elif '{' in stripped:
                        return stripped.split('{')[0]
                    else:
                        return stripped
        
        # For classes, extract class definition
        elif kind == ChunkKind.CLASS:
            for line in lines:
                stripped = line.strip()
                if 'class ' in stripped:
                    if ':' in stripped:
                        return stripped.split(':')[0] + ':'
                    elif '{' in stripped:
                        return stripped.split('{')[0] 
                    else:
                        return stripped
        
        return None
    
    def _extract_module_docstring(self, content: str, language: str) -> Optional[str]:
        """Extract module-level docstring."""
        if language == 'python':
            lines = content.splitlines()
            # Skip imports and comments to find module docstring
            in_docstring = False
            docstring_lines = []
            quote_type = None
            
            for line in lines:
                stripped = line.strip()
                
                if not stripped or stripped.startswith('#'):
                    continue
                elif stripped.startswith('"""') or stripped.startswith("'''"):
                    if not in_docstring:
                        quote_type = '"""' if stripped.startswith('"""') else "'''"
                        in_docstring = True
                        docstring_lines.append(line)
                        if stripped.count(quote_type) >= 2:  # Single line docstring
                            break
                    else:
                        docstring_lines.append(line)
                        break
                elif in_docstring:
                    docstring_lines.append(line)
                else:
                    # Hit non-docstring code
                    break
            
            if docstring_lines:
                return '\n'.join(docstring_lines)
        
        return None
    
    def _calculate_doc_density(self, content: str, docstring: Optional[str]) -> float:
        """Calculate documentation density (0.0-1.0)."""
        if not content:
            return 0.0
        
        doc_chars = len(docstring) if docstring else 0
        
        # Count comment lines
        comment_chars = 0
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('*'):
                comment_chars += len(stripped)
        
        total_doc_chars = doc_chars + comment_chars
        total_chars = len(content)
        
        return min(1.0, total_doc_chars / total_chars) if total_chars > 0 else 0.0
    
    def _is_test_function(self, name: str, file_path: Path) -> bool:
        """Determine if a function is a test function."""
        # Check function name patterns
        if name.startswith('test_') or name.endswith('_test') or 'test' in name.lower():
            return True
        
        # Check file path patterns
        path_str = str(file_path).lower()
        return 'test' in path_str or 'spec' in path_str
    
    def chunk_repository(self, repo_path: Path, file_filter: Optional[callable] = None) -> List[Chunk]:
        """
        Extract chunks from an entire repository.
        
        Args:
            repo_path: Path to repository root
            file_filter: Optional filter function for files to process
            
        Returns:
            List of all extracted chunks from the repository
        """
        all_chunks = []
        
        # Find supported source files
        source_files = []
        supported_extensions = set()
        for lang_config in self.language_support.LANGUAGES.values():
            supported_extensions.update(lang_config['extensions'])
        
        for file_path in repo_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                if file_filter is None or file_filter(file_path):
                    source_files.append(file_path)
        
        # Process each file
        for file_path in source_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                file_chunks = self.chunk_file(file_path, content, repo_path)
                all_chunks.extend(file_chunks)
                
            except Exception as e:
                print(f"Warning: Failed to process {file_path}: {e}")
                continue
        
        return all_chunks
    
    def get_statistics(self) -> Dict[str, int]:
        """Get chunker statistics."""
        total_chunks = sum(len(chunks) for chunks in self._chunk_cache.values())
        
        return {
            "cached_files": len(self._chunk_cache),
            "total_chunks": total_chunks,
            "supported_languages": len(self.language_support.get_supported_languages()),
        }