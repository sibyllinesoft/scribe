"""
Repository structure fuzzing tests for PackRepo.

Tests PackRepo's handling of various repository structures:
- Unusual directory hierarchies
- Circular symlinks
- Permission issues
- Mixed file types
- Large repositories with many files
- Nested repositories
"""

from __future__ import annotations

import pytest
import tempfile
import os
import shutil
import random
import string
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .fuzzer_engine import FuzzerEngine, PackRepoSelectorTarget


@dataclass
class RepositoryStructure:
    """Represents a test repository structure."""
    root_path: str
    files: Dict[str, str]  # path -> content
    directories: List[str]
    symlinks: Dict[str, str]  # link_path -> target_path
    special_files: List[str]  # Fifos, sockets, etc.
    gitignore_patterns: List[str] = None
    
    def __post_init__(self):
        if self.gitignore_patterns is None:
            self.gitignore_patterns = []


class RepositoryGenerator:
    """Generate various repository structures for testing."""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def generate_deep_nested_repo(self, base_path: str, max_depth: int = 10) -> RepositoryStructure:
        """Generate repository with deeply nested directory structure."""
        files = {}
        directories = []
        
        current_depth = 0
        current_path = ""
        
        # Create nested directories
        for depth in range(max_depth):
            dir_name = f"level_{depth}_{random.choice(['src', 'lib', 'tests', 'docs'])}"
            current_path = os.path.join(current_path, dir_name)
            directories.append(current_path)
            
            # Add some files at each level
            for i in range(random.randint(1, 3)):
                filename = f"file_{depth}_{i}.py"
                file_path = os.path.join(current_path, filename)
                files[file_path] = self._generate_file_content(filename)
        
        return RepositoryStructure(
            root_path=base_path,
            files=files,
            directories=directories,
            symlinks={},
            special_files=[]
        )
    
    def generate_wide_flat_repo(self, base_path: str, num_files: int = 1000) -> RepositoryStructure:
        """Generate repository with many files in flat structure."""
        files = {}
        directories = []
        
        # Create multiple top-level directories
        top_dirs = ['src', 'lib', 'tests', 'docs', 'scripts', 'data', 'config', 'tools']
        directories.extend(top_dirs)
        
        # Distribute files across directories
        for i in range(num_files):
            dir_choice = random.choice(top_dirs)
            file_ext = random.choice(['.py', '.js', '.md', '.txt', '.json', '.yaml'])
            filename = f"file_{i:04d}{file_ext}"
            file_path = os.path.join(dir_choice, filename)
            
            files[file_path] = self._generate_file_content(filename)
        
        return RepositoryStructure(
            root_path=base_path,
            files=files,
            directories=directories,
            symlinks={},
            special_files=[]
        )
    
    def generate_symlink_heavy_repo(self, base_path: str) -> RepositoryStructure:
        """Generate repository with many symbolic links."""
        files = {}
        directories = ['src', 'lib', 'links']
        symlinks = {}
        
        # Create some real files
        real_files = [
            'src/main.py',
            'src/utils.py', 
            'lib/helper.py',
            'lib/config.py'
        ]
        
        for file_path in real_files:
            files[file_path] = self._generate_file_content(file_path)
        
        # Create symlinks to real files
        for i, real_file in enumerate(real_files):
            link_path = f"links/link_{i}.py"
            symlinks[link_path] = os.path.join('..', real_file)
        
        # Create circular symlinks
        symlinks['links/circular_a'] = 'circular_b'
        symlinks['links/circular_b'] = 'circular_a'
        
        # Create broken symlinks
        symlinks['links/broken_link'] = 'nonexistent_file.py'
        
        # Create directory symlinks
        symlinks['links/src_link'] = '../src'
        
        return RepositoryStructure(
            root_path=base_path,
            files=files,
            directories=directories,
            symlinks=symlinks,
            special_files=[]
        )
    
    def generate_mixed_file_types_repo(self, base_path: str) -> RepositoryStructure:
        """Generate repository with diverse file types."""
        files = {}
        directories = ['code', 'data', 'media', 'docs']
        
        # Code files
        code_files = {
            'code/main.py': 'def main():\n    print("Hello World")',
            'code/utils.js': 'function utils() {\n  return "utility";\n}',
            'code/app.rs': 'fn main() {\n    println!("Hello Rust");\n}',
            'code/service.go': 'package main\n\nfunc main() {\n    fmt.Println("Go service")\n}',
            'code/Component.tsx': 'export const Component = () => <div>React</div>;'
        }
        files.update(code_files)
        
        # Data files
        data_files = {
            'data/config.json': '{"setting": "value", "debug": true}',
            'data/data.csv': 'name,age,city\nJohn,30,NYC\nJane,25,LA',
            'data/config.yaml': 'database:\n  host: localhost\n  port: 5432',
            'data/sample.xml': '<?xml version="1.0"?><root><item>data</item></root>',
            'data/log.txt': 'INFO: Application started\nERROR: Something failed'
        }
        files.update(data_files)
        
        # Documentation
        doc_files = {
            'docs/README.md': '# Project\n\nThis is a test project.',
            'docs/api.md': '## API\n\n### GET /api/data\n\nReturns data.',
            'docs/manual.rst': 'Manual\n======\n\nThis is documentation.'
        }
        files.update(doc_files)
        
        # Binary-like content (as text for testing)
        binary_content = ''.join(chr(i % 256) for i in range(1000))
        files['media/binary.dat'] = binary_content
        
        # Large text file
        large_content = 'This is line {}\n' * 10000
        files['data/large_file.txt'] = large_content.format(*range(10000))
        
        return RepositoryStructure(
            root_path=base_path,
            files=files,
            directories=directories,
            symlinks={},
            special_files=[]
        )
    
    def generate_gitignore_heavy_repo(self, base_path: str) -> RepositoryStructure:
        """Generate repository with complex .gitignore patterns."""
        files = {}
        directories = ['src', 'build', 'dist', 'node_modules', 'target', '__pycache__', 'vendor']
        
        # Regular source files
        source_files = {
            'src/main.py': 'def main(): pass',
            'src/utils.py': 'def util(): pass', 
            'src/test.py': 'def test(): pass'
        }
        files.update(source_files)
        
        # Files that should be ignored
        ignored_files = {
            'build/output.o': 'binary content',
            'dist/bundle.js': 'minified js',
            'node_modules/package/index.js': 'dependency',
            'target/debug/binary': 'rust binary',
            '__pycache__/module.pyc': 'python bytecode',
            'vendor/lib.so': 'vendor library',
            '.env': 'SECRET_KEY=secret',
            'config.local.json': '{"secret": "value"}',
            'debug.log': 'DEBUG: messages',
            'temp_file.tmp': 'temporary content'
        }
        files.update(ignored_files)
        
        # Complex .gitignore patterns
        gitignore_patterns = [
            # Standard patterns
            '*.pyc',
            '__pycache__/',
            '*.tmp',
            '.env',
            'node_modules/',
            'dist/',
            'build/',
            'target/',
            'vendor/',
            
            # Complex patterns
            '*.log',
            '!important.log',  # Negation
            'config.*.json',
            'temp_*',
            '[Dd]ebug/',
            '*.{o,so,dll}',
            
            # Directory-specific ignores
            'src/*.bak',
            '**/cache/',
            'docs/*.pdf',
            
            # Comments and blank lines
            '# IDE files',
            '.vscode/',
            '.idea/',
            '*.swp',
            '*.swo',
            '',  # Blank line
            '# OS files',
            '.DS_Store',
            'Thumbs.db'
        ]
        
        files['.gitignore'] = '\n'.join(gitignore_patterns)
        
        return RepositoryStructure(
            root_path=base_path,
            files=files,
            directories=directories,
            symlinks={},
            special_files=[],
            gitignore_patterns=gitignore_patterns
        )
    
    def generate_nested_repos_structure(self, base_path: str) -> RepositoryStructure:
        """Generate repository with nested git repositories."""
        files = {}
        directories = ['main', 'submodules', 'submodules/repo1', 'submodules/repo2']
        
        # Main repo files
        main_files = {
            'main/app.py': 'from submodules.repo1 import lib1\n\ndef main():\n    lib1.run()',
            'main/config.json': '{"submodules": ["repo1", "repo2"]}',
            '.gitmodules': '[submodule "repo1"]\n\tpath = submodules/repo1\n\turl = https://github.com/example/repo1.git'
        }
        files.update(main_files)
        
        # Nested repo 1
        repo1_files = {
            'submodules/repo1/lib1.py': 'def run():\n    print("Running lib1")',
            'submodules/repo1/.git/config': '[core]\n\trepositoryformatversion = 0',
            'submodules/repo1/README.md': '# Repo 1\n\nSubmodule repository'
        }
        files.update(repo1_files)
        
        # Nested repo 2
        repo2_files = {
            'submodules/repo2/lib2.py': 'def helper():\n    return "help"',
            'submodules/repo2/.git/config': '[core]\n\trepositoryformatversion = 0',
            'submodules/repo2/package.json': '{"name": "repo2", "version": "1.0.0"}'
        }
        files.update(repo2_files)
        
        # Add .git directory entries for main repo
        directories.extend(['submodules/repo1/.git', 'submodules/repo2/.git', '.git'])
        files['.git/config'] = '[core]\n\trepositoryformatversion = 0'
        
        return RepositoryStructure(
            root_path=base_path,
            files=files,
            directories=directories,
            symlinks={},
            special_files=[]
        )
    
    def generate_unicode_paths_repo(self, base_path: str) -> RepositoryStructure:
        """Generate repository with Unicode file and directory names."""
        files = {}
        directories = []
        
        # Unicode directory names
        unicode_dirs = [
            '中文目录',
            'русский',
            '日本語',
            'العربية',
            'français',
            'español',
            'emoji_🚀_dir'
        ]
        directories.extend(unicode_dirs)
        
        # Files with Unicode names
        unicode_files = {
            '中文目录/测试.py': '# 中文注释\nprint("测试")',
            'русский/файл.py': '# Русский комментарий\nprint("тест")',
            '日本語/テスト.py': '# 日本語のコメント\nprint("テスト")',
            'العربية/ملف.py': '# تعليق عربي\nprint("اختبار")',
            'emoji_🚀_dir/rocket_🚀.py': '# Emoji in filename\nrocket = "🚀"',
            'mixed_scripts_混合_файл_テスト.py': '# Mixed scripts in filename\ntest = "mixed"'
        }
        files.update(unicode_files)
        
        # Regular files for contrast
        files['regular/normal_file.py'] = 'def normal(): pass'
        directories.append('regular')
        
        return RepositoryStructure(
            root_path=base_path,
            files=files,
            directories=directories,
            symlinks={},
            special_files=[]
        )
    
    def _generate_file_content(self, filename: str) -> str:
        """Generate appropriate content based on file extension."""
        ext = os.path.splitext(filename)[1].lower()
        
        if ext == '.py':
            return self._generate_python_content()
        elif ext in ['.js', '.ts']:
            return self._generate_javascript_content()
        elif ext == '.md':
            return self._generate_markdown_content()
        elif ext == '.json':
            return self._generate_json_content()
        elif ext in ['.txt', '.log']:
            return self._generate_text_content()
        else:
            return f"# Generated content for {filename}\ndata = '{filename}'\n"
    
    def _generate_python_content(self) -> str:
        """Generate Python code content."""
        templates = [
            'def function_{n}():\n    """Function {n}."""\n    return {n}\n\n',
            'class Class{n}:\n    """Class {n}."""\n    def __init__(self):\n        self.value = {n}\n\n',
            'import os\nimport sys\n\nif __name__ == "__main__":\n    print("Module {n}")\n\n',
            '# Configuration module {n}\nCONFIG = {{\n    "value": {n},\n    "enabled": True\n}}\n\n'
        ]
        
        template = random.choice(templates)
        return template.format(n=random.randint(1, 100))
    
    def _generate_javascript_content(self) -> str:
        """Generate JavaScript code content."""
        templates = [
            'function func{n}() {{\n  // Function {n}\n  return {n};\n}}\n\n',
            'const obj{n} = {{\n  value: {n},\n  method() {{\n    return this.value;\n  }}\n}};\n\n',
            'import {{ utils }} from "./utils";\n\nconst result{n} = utils.process({n});\n\n',
            '// Module {n}\nexport const config{n} = {{\n  value: {n},\n  enabled: true\n}};\n\n'
        ]
        
        template = random.choice(templates)
        return template.format(n=random.randint(1, 100))
    
    def _generate_markdown_content(self) -> str:
        """Generate Markdown content."""
        templates = [
            '# Document {n}\n\nThis is document number {n}.\n\n## Section\n\nContent here.\n\n',
            '# API Reference {n}\n\n## Function `func{n}`\n\nReturns value {n}.\n\n',
            '# README {n}\n\nProject description.\n\n### Installation\n\n```bash\nnpm install\n```\n\n'
        ]
        
        template = random.choice(templates)
        return template.format(n=random.randint(1, 100))
    
    def _generate_json_content(self) -> str:
        """Generate JSON content."""
        data = {
            "id": random.randint(1, 1000),
            "name": f"item_{random.randint(1, 100)}",
            "config": {
                "enabled": random.choice([True, False]),
                "value": random.randint(1, 100)
            },
            "items": [random.randint(1, 10) for _ in range(random.randint(1, 5))]
        }
        
        import json
        return json.dumps(data, indent=2)
    
    def _generate_text_content(self) -> str:
        """Generate text content."""
        lines = []
        for i in range(random.randint(5, 20)):
            lines.append(f"Line {i}: {random.choice(['info', 'debug', 'warning', 'error'])} message")
        
        return '\n'.join(lines)


class TestRepositoryStructureFuzzing:
    """Test repository structure fuzzing."""
    
    @pytest.fixture
    def repo_generator(self):
        """Create repository generator for tests."""
        return RepositoryGenerator(seed=42)
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp(prefix="packrepo_test_")
        yield temp_dir
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to cleanup temp dir {temp_dir}: {e}")
    
    def _create_repository_on_disk(self, repo_struct: RepositoryStructure, base_path: str):
        """Create the repository structure on disk."""
        # Create directories
        for directory in repo_struct.directories:
            dir_path = os.path.join(base_path, directory)
            os.makedirs(dir_path, exist_ok=True)
        
        # Create files
        for file_path, content in repo_struct.files.items():
            full_path = os.path.join(base_path, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            try:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except UnicodeEncodeError:
                # Handle binary-like content
                with open(full_path, 'wb') as f:
                    f.write(content.encode('utf-8', errors='ignore'))
        
        # Create symlinks (if supported by OS)
        for link_path, target in repo_struct.symlinks.items():
            full_link_path = os.path.join(base_path, link_path)
            os.makedirs(os.path.dirname(full_link_path), exist_ok=True)
            
            try:
                os.symlink(target, full_link_path)
            except (OSError, NotImplementedError):
                # Symlinks not supported on this system
                print(f"Warning: Could not create symlink {link_path} -> {target}")
    
    def test_deep_nested_directory_handling(self, repo_generator, temp_dir):
        """Test handling of deeply nested directory structures."""
        from packrepo.packer.selector.selector import Selector
        
        # Generate deep nested repository
        repo_struct = repo_generator.generate_deep_nested_repo(temp_dir, max_depth=15)
        self._create_repository_on_disk(repo_struct, temp_dir)
        
        # Test PackRepo selector with this structure
        try:
            selector = Selector()
            
            # Mock chunk data from repository
            chunks = []
            for file_path, content in repo_struct.files.items():
                chunks.append({
                    'id': f'chunk_{len(chunks)}',
                    'rel_path': file_path,
                    'start_line': 1,
                    'end_line': len(content.split('\n')),
                    'content': content,
                    'cost': len(content) // 4,  # Approximate token cost
                    'score': random.uniform(0.1, 1.0)
                })
            
            # Test selection with various budgets
            budgets = [1000, 5000, 20000]
            results = []
            
            for budget in budgets:
                try:
                    selected = selector.select(chunks, budget)
                    results.append({
                        'budget': budget,
                        'selected_count': len(selected),
                        'success': True
                    })
                except Exception as e:
                    results.append({
                        'budget': budget,
                        'selected_count': 0,
                        'success': False,
                        'error': str(e)
                    })
            
            # Should handle nested structure gracefully
            successful_results = [r for r in results if r['success']]
            success_rate = len(successful_results) / len(results)
            
            print(f"Deep nested directory test results:")
            print(f"  Repository depth: 15 levels")
            print(f"  Total files: {len(repo_struct.files)}")
            print(f"  Success rate: {success_rate:.2%}")
            
            assert success_rate > 0.8, f"Deep nested directory handling insufficient: {success_rate:.2%}"
            
        except ImportError:
            pytest.skip("Selector not available for testing")
    
    def test_wide_flat_repository_handling(self, repo_generator, temp_dir):
        """Test handling of repositories with many files."""
        # Generate repository with many files
        repo_struct = repo_generator.generate_wide_flat_repo(temp_dir, num_files=500)
        self._create_repository_on_disk(repo_struct, temp_dir)
        
        # Test with fuzzer engine
        fuzzer = FuzzerEngine(seed=42)
        selector_target = PackRepoSelectorTarget()
        
        # Create chunks from repository files
        chunks = []
        for i, (file_path, content) in enumerate(repo_struct.files.items()):
            chunks.append({
                'id': f'chunk_{i}',
                'rel_path': file_path,
                'content': content,
                'cost': max(1, len(content) // 4),
                'score': random.uniform(0.1, 1.0)
            })
        
        # Test selection with various budgets
        test_cases = [
            {'chunks': chunks[:100], 'budget': 1000},   # Small subset
            {'chunks': chunks[:250], 'budget': 5000},   # Medium subset
            {'chunks': chunks, 'budget': 10000},        # Full set
        ]
        
        results = []
        for test_case in test_cases:
            try:
                result = selector_target.execute(test_case)
                results.append({
                    'chunk_count': len(test_case['chunks']),
                    'budget': test_case['budget'],
                    'success': True,
                    'selected_count': len(result) if isinstance(result, (list, set)) else 0
                })
            except Exception as e:
                results.append({
                    'chunk_count': len(test_case['chunks']),
                    'budget': test_case['budget'],
                    'success': False,
                    'error': str(e)
                })
        
        # Analyze results
        successful_results = [r for r in results if r['success']]
        success_rate = len(successful_results) / len(results)
        
        print(f"Wide flat repository test results:")
        print(f"  Total files generated: {len(repo_struct.files)}")
        print(f"  Success rate: {success_rate:.2%}")
        
        for result in results:
            if result['success']:
                print(f"  {result['chunk_count']} chunks, budget {result['budget']}: {result['selected_count']} selected")
            else:
                print(f"  {result['chunk_count']} chunks, budget {result['budget']}: FAILED - {result['error']}")
        
        assert success_rate > 0.8, f"Wide flat repository handling insufficient: {success_rate:.2%}"
    
    def test_symlink_handling(self, repo_generator, temp_dir):
        """Test handling of symbolic links."""
        # Generate repository with symlinks
        repo_struct = repo_generator.generate_symlink_heavy_repo(temp_dir)
        self._create_repository_on_disk(repo_struct, temp_dir)
        
        # Test chunker with symlinked files
        from .fuzzer_engine import PackRepoChunkerTarget
        
        chunker_target = PackRepoChunkerTarget()
        
        # Test both real files and symlinked files
        test_files = []
        
        # Add real files
        for file_path, content in repo_struct.files.items():
            test_files.append({
                'content': content,
                'file_path': file_path,
                'file_type': 'real'
            })
        
        # Add symlinked file content (resolve manually for testing)
        for link_path, target_path in repo_struct.symlinks.items():
            if target_path in repo_struct.files:  # Symlink to real file
                test_files.append({
                    'content': repo_struct.files[target_path],
                    'file_path': link_path,
                    'file_type': 'symlink'
                })
        
        results = []
        for test_file in test_files:
            try:
                input_data = {
                    'content': test_file['content'],
                    'file_path': test_file['file_path'],
                    'config': {}
                }
                
                chunks = chunker_target.execute(input_data)
                results.append({
                    'file_path': test_file['file_path'],
                    'file_type': test_file['file_type'],
                    'success': True,
                    'num_chunks': len(chunks)
                })
                
            except Exception as e:
                results.append({
                    'file_path': test_file['file_path'],
                    'file_type': test_file['file_type'],
                    'success': False,
                    'error': str(e)
                })
        
        # Analyze results
        real_file_results = [r for r in results if r['file_type'] == 'real']
        symlink_results = [r for r in results if r['file_type'] == 'symlink']
        
        real_success_rate = sum(1 for r in real_file_results if r['success']) / len(real_file_results)
        symlink_success_rate = sum(1 for r in symlink_results if r['success']) / len(symlink_results) if symlink_results else 1.0
        
        print(f"Symlink handling test results:")
        print(f"  Real files success rate: {real_success_rate:.2%}")
        print(f"  Symlink files success rate: {symlink_success_rate:.2%}")
        
        # Should handle both real and symlinked files gracefully
        assert real_success_rate > 0.9, f"Real file handling insufficient: {real_success_rate:.2%}"
        assert symlink_success_rate > 0.7, f"Symlink handling insufficient: {symlink_success_rate:.2%}"
    
    def test_mixed_file_types_handling(self, repo_generator, temp_dir):
        """Test handling of diverse file types."""
        repo_struct = repo_generator.generate_mixed_file_types_repo(temp_dir)
        self._create_repository_on_disk(repo_struct, temp_dir)
        
        from .fuzzer_engine import PackRepoChunkerTarget, PackRepoTokenizerTarget
        
        chunker_target = PackRepoChunkerTarget()
        tokenizer_target = PackRepoTokenizerTarget()
        
        # Categorize files by type
        file_categories = {
            'code': [f for f in repo_struct.files.keys() if f.startswith('code/')],
            'data': [f for f in repo_struct.files.keys() if f.startswith('data/')],
            'docs': [f for f in repo_struct.files.keys() if f.startswith('docs/')],
            'media': [f for f in repo_struct.files.keys() if f.startswith('media/')]
        }
        
        results_by_category = {}
        
        for category, file_paths in file_categories.items():
            category_results = []
            
            for file_path in file_paths:
                content = repo_struct.files[file_path]
                
                # Test chunker
                try:
                    chunker_input = {
                        'content': content,
                        'file_path': file_path,
                        'config': {}
                    }
                    chunks = chunker_target.execute(chunker_input)
                    chunker_success = True
                    num_chunks = len(chunks)
                except Exception as e:
                    chunker_success = False
                    num_chunks = 0
                
                # Test tokenizer (skip binary-like content)
                tokenizer_success = True
                token_count = 0
                if not file_path.endswith('.dat') and len(content) < 100000:
                    try:
                        tokenizer_input = {
                            'text': content,
                            'tokenizer': 'cl100k'
                        }
                        token_count = tokenizer_target.execute(tokenizer_input)
                    except Exception as e:
                        tokenizer_success = False
                
                category_results.append({
                    'file_path': file_path,
                    'chunker_success': chunker_success,
                    'tokenizer_success': tokenizer_success,
                    'num_chunks': num_chunks,
                    'token_count': token_count
                })
            
            results_by_category[category] = category_results
        
        # Analyze results by category
        print(f"Mixed file types handling results:")
        
        for category, results in results_by_category.items():
            if not results:
                continue
                
            chunker_success_rate = sum(1 for r in results if r['chunker_success']) / len(results)
            tokenizer_success_rate = sum(1 for r in results if r['tokenizer_success']) / len(results)
            
            print(f"  {category}:")
            print(f"    Chunker success: {chunker_success_rate:.2%}")
            print(f"    Tokenizer success: {tokenizer_success_rate:.2%}")
            
            # Different expectations for different file types
            if category == 'code':
                assert chunker_success_rate > 0.9, f"Code file chunking insufficient: {chunker_success_rate:.2%}"
                assert tokenizer_success_rate > 0.9, f"Code file tokenization insufficient: {tokenizer_success_rate:.2%}"
            elif category in ['data', 'docs']:
                assert chunker_success_rate > 0.8, f"{category} file chunking insufficient: {chunker_success_rate:.2%}"
                assert tokenizer_success_rate > 0.8, f"{category} file tokenization insufficient: {tokenizer_success_rate:.2%}"
            elif category == 'media':
                # Binary/media files might have lower success rates
                assert chunker_success_rate > 0.5, f"Media file handling too poor: {chunker_success_rate:.2%}"
    
    def test_gitignore_pattern_handling(self, repo_generator, temp_dir):
        """Test handling of .gitignore patterns."""
        repo_struct = repo_generator.generate_gitignore_heavy_repo(temp_dir)
        self._create_repository_on_disk(repo_struct, temp_dir)
        
        # Test that file discovery respects .gitignore
        # This would normally be tested with actual PackRepo file discovery
        # For now, we'll simulate the logic
        
        gitignore_patterns = repo_struct.gitignore_patterns
        all_files = list(repo_struct.files.keys())
        
        # Simple gitignore pattern matching (simplified)
        def should_ignore(file_path: str, patterns: List[str]) -> bool:
            import fnmatch
            
            for pattern in patterns:
                pattern = pattern.strip()
                if not pattern or pattern.startswith('#'):
                    continue
                
                # Handle negation
                if pattern.startswith('!'):
                    continue  # Skip negation for this simple test
                
                # Simple pattern matching
                if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(os.path.basename(file_path), pattern):
                    return True
                
                # Directory patterns
                if pattern.endswith('/') and pattern.rstrip('/') in file_path:
                    return True
            
            return False
        
        # Categorize files
        source_files = []
        ignored_files = []
        
        for file_path in all_files:
            if file_path == '.gitignore':
                continue
                
            if should_ignore(file_path, gitignore_patterns):
                ignored_files.append(file_path)
            else:
                source_files.append(file_path)
        
        print(f"Gitignore pattern handling results:")
        print(f"  Total files: {len(all_files)}")
        print(f"  Source files: {len(source_files)}")
        print(f"  Ignored files: {len(ignored_files)}")
        print(f"  Gitignore patterns: {len(gitignore_patterns)}")
        
        # Should have identified some files to ignore
        assert len(ignored_files) > 0, "Should identify files to ignore based on patterns"
        assert len(source_files) > 0, "Should identify source files not matching ignore patterns"
        
        # Test with PackRepo chunker on source files only
        from .fuzzer_engine import PackRepoChunkerTarget
        chunker_target = PackRepoChunkerTarget()
        
        source_file_results = []
        for file_path in source_files[:10]:  # Test subset for performance
            try:
                content = repo_struct.files[file_path]
                input_data = {
                    'content': content,
                    'file_path': file_path,
                    'config': {}
                }
                
                chunks = chunker_target.execute(input_data)
                source_file_results.append(True)
                
            except Exception as e:
                source_file_results.append(False)
                print(f"Source file chunking failed for {file_path}: {e}")
        
        if source_file_results:
            source_success_rate = sum(source_file_results) / len(source_file_results)
            print(f"  Source file chunking success rate: {source_success_rate:.2%}")
            assert source_success_rate > 0.8, f"Source file chunking insufficient: {source_success_rate:.2%}"
    
    def test_unicode_paths_handling(self, repo_generator, temp_dir):
        """Test handling of Unicode file and directory names."""
        repo_struct = repo_generator.generate_unicode_paths_repo(temp_dir)
        
        try:
            self._create_repository_on_disk(repo_struct, temp_dir)
        except (UnicodeError, OSError) as e:
            pytest.skip(f"Unicode paths not supported on this system: {e}")
        
        from .fuzzer_engine import PackRepoChunkerTarget
        chunker_target = PackRepoChunkerTarget()
        
        # Test chunking of files with Unicode paths
        unicode_results = []
        regular_results = []
        
        for file_path, content in repo_struct.files.items():
            try:
                input_data = {
                    'content': content,
                    'file_path': file_path,
                    'config': {}
                }
                
                chunks = chunker_target.execute(input_data)
                
                is_unicode = any(ord(c) > 127 for c in file_path)
                if is_unicode:
                    unicode_results.append(True)
                else:
                    regular_results.append(True)
                    
            except Exception as e:
                is_unicode = any(ord(c) > 127 for c in file_path)
                if is_unicode:
                    unicode_results.append(False)
                    print(f"Unicode path chunking failed for {file_path}: {e}")
                else:
                    regular_results.append(False)
        
        # Analyze results
        unicode_success_rate = sum(unicode_results) / len(unicode_results) if unicode_results else 1.0
        regular_success_rate = sum(regular_results) / len(regular_results) if regular_results else 1.0
        
        print(f"Unicode paths handling results:")
        print(f"  Unicode path files success rate: {unicode_success_rate:.2%}")
        print(f"  Regular path files success rate: {regular_success_rate:.2%}")
        
        # Should handle Unicode paths reasonably well
        assert regular_success_rate > 0.9, f"Regular path handling insufficient: {regular_success_rate:.2%}"
        assert unicode_success_rate > 0.7, f"Unicode path handling insufficient: {unicode_success_rate:.2%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])