"""
File content fuzzing tests for PackRepo.

Tests PackRepo's robustness against various file content patterns:
- Malformed syntax
- Unicode edge cases
- Binary content mixed with text
- Extremely large files
- Parser-breaking constructs
"""

from __future__ import annotations

import pytest
import tempfile
import os
from typing import Dict, Any, List
from pathlib import Path

from .tree_sitter_fuzzer import TreeSitterFuzzer, LanguageGrammar, generate_tree_sitter_fuzzing_samples
from .fuzzer_engine import FuzzerEngine, PackRepoChunkerTarget, PackRepoTokenizerTarget


class TestFileContentFuzzing:
    """Test fuzzing with various file content patterns."""
    
    @pytest.fixture
    def fuzzer_engine(self):
        """Create fuzzing engine for tests."""
        return FuzzerEngine(seed=42)
    
    @pytest.fixture 
    def tree_sitter_fuzzer(self):
        """Create Tree-sitter fuzzer for tests."""
        return TreeSitterFuzzer(seed=42)
    
    def test_python_syntax_fuzzing(self, fuzzer_engine):
        """Test Python syntax variations against chunker."""
        chunker_target = PackRepoChunkerTarget()
        
        # Generate various Python code samples
        samples = generate_tree_sitter_fuzzing_samples(
            'python', 
            num_samples=20,
            corruption_probability=0.2
        )
        
        crashes = []
        successful_executions = 0
        
        for sample in samples:
            try:
                input_data = {
                    'content': sample['content'],
                    'file_path': sample['file_path'],
                    'config': {}
                }
                
                result = chunker_target.execute(input_data)
                successful_executions += 1
                
                # Validate result structure
                assert isinstance(result, list), "Chunker should return list"
                
                # Check that chunks cover the content reasonably
                if len(sample['content'].strip()) > 0:
                    assert len(result) > 0, "Non-empty content should produce chunks"
                
            except Exception as e:
                crash_info = {
                    'sample_id': sample.get('sample_id', 'unknown'),
                    'exception': str(e),
                    'exception_type': type(e).__name__,
                    'content_preview': sample['content'][:100]
                }
                crashes.append(crash_info)
        
        # Analyze results
        total_samples = len(samples)
        success_rate = successful_executions / total_samples
        
        print(f"Python syntax fuzzing results:")
        print(f"  Total samples: {total_samples}")
        print(f"  Successful: {successful_executions}")
        print(f"  Crashes: {len(crashes)}")
        print(f"  Success rate: {success_rate:.2%}")
        
        # Should handle most samples gracefully
        assert success_rate > 0.7, f"Success rate too low: {success_rate:.2%}"
        
        # Log crash details for analysis
        if crashes:
            print("\nCrash details:")
            for crash in crashes[:5]:  # Show first 5 crashes
                print(f"  {crash['exception_type']}: {crash['exception']}")
    
    def test_javascript_syntax_fuzzing(self, fuzzer_engine):
        """Test JavaScript syntax variations."""
        chunker_target = PackRepoChunkerTarget()
        
        samples = generate_tree_sitter_fuzzing_samples(
            'javascript',
            num_samples=15, 
            corruption_probability=0.15
        )
        
        crashes = []
        successful_executions = 0
        
        for sample in samples:
            try:
                input_data = {
                    'content': sample['content'],
                    'file_path': sample['file_path'], 
                    'config': {'language': 'javascript'}
                }
                
                result = chunker_target.execute(input_data)
                successful_executions += 1
                
                # Basic validation
                assert isinstance(result, list)
                
            except Exception as e:
                crashes.append({
                    'sample_id': sample.get('sample_id', 'unknown'),
                    'exception': str(e),
                    'exception_type': type(e).__name__
                })
        
        success_rate = successful_executions / len(samples)
        print(f"JavaScript syntax fuzzing - Success rate: {success_rate:.2%}")
        
        # Should handle most JavaScript samples
        assert success_rate > 0.6, f"JavaScript success rate too low: {success_rate:.2%}"
    
    def test_unicode_content_fuzzing(self, fuzzer_engine):
        """Test Unicode and international character handling."""
        chunker_target = PackRepoChunkerTarget()
        tokenizer_target = PackRepoTokenizerTarget()
        
        # Generate Unicode test cases
        unicode_test_cases = [
            # Chinese characters
            {
                'content': '# 中文注释\n变量 = "测试字符串"\nprint(变量)',
                'file_path': 'chinese_test.py',
                'description': 'Chinese characters'
            },
            # Japanese
            {
                'content': '// 日本語のコメント\nconst テスト = "こんにちは";\nconsole.log(テスト);',
                'file_path': 'japanese_test.js',
                'description': 'Japanese characters'
            },
            # Arabic (RTL script)
            {
                'content': '# تعليق بالعربية\nمتغير = "نص عربي"\nprint(متغير)',
                'file_path': 'arabic_test.py',
                'description': 'Arabic text'
            },
            # Emojis and symbols
            {
                'content': '# 🚀 Test with emojis\nrocket = "🚀🎯🔥"\nprint(f"Status: {rocket}")',
                'file_path': 'emoji_test.py',
                'description': 'Emoji characters'
            },
            # Mixed scripts
            {
                'content': 'def test_混合_функция_テスト():\n    """Mixed العربية scripts."""\n    return "🌍"',
                'file_path': 'mixed_scripts.py',
                'description': 'Mixed scripts'
            },
            # Unicode escape sequences
            {
                'content': 'text = "\\u4e2d\\u6587\\u6d4b\\u8bd5"\nprint(text)',
                'file_path': 'unicode_escapes.py',
                'description': 'Unicode escape sequences'
            },
            # Zero-width characters
            {
                'content': 'def​test():\n    return​"zero​width​chars"',  # Contains zero-width spaces
                'file_path': 'zero_width.py', 
                'description': 'Zero-width characters'
            }
        ]
        
        chunker_crashes = 0
        tokenizer_crashes = 0
        total_cases = len(unicode_test_cases)
        
        for test_case in unicode_test_cases:
            # Test chunker
            try:
                chunker_input = {
                    'content': test_case['content'],
                    'file_path': test_case['file_path'],
                    'config': {}
                }
                chunks = chunker_target.execute(chunker_input)
                assert isinstance(chunks, list)
            except Exception as e:
                chunker_crashes += 1
                print(f"Chunker crashed on {test_case['description']}: {e}")
            
            # Test tokenizer
            try:
                tokenizer_input = {
                    'text': test_case['content'],
                    'tokenizer': 'cl100k'
                }
                token_count = tokenizer_target.execute(tokenizer_input)
                assert isinstance(token_count, int) and token_count >= 0
            except Exception as e:
                tokenizer_crashes += 1
                print(f"Tokenizer crashed on {test_case['description']}: {e}")
        
        # Results analysis
        chunker_success_rate = (total_cases - chunker_crashes) / total_cases
        tokenizer_success_rate = (total_cases - tokenizer_crashes) / total_cases
        
        print(f"Unicode content fuzzing results:")
        print(f"  Chunker success rate: {chunker_success_rate:.2%}")
        print(f"  Tokenizer success rate: {tokenizer_success_rate:.2%}")
        
        # Both should handle Unicode gracefully
        assert chunker_success_rate > 0.8, f"Chunker Unicode handling insufficient: {chunker_success_rate:.2%}"
        assert tokenizer_success_rate > 0.9, f"Tokenizer Unicode handling insufficient: {tokenizer_success_rate:.2%}"
    
    def test_binary_content_fuzzing(self, fuzzer_engine):
        """Test handling of binary/non-text content."""
        chunker_target = PackRepoChunkerTarget()
        
        # Generate binary-like content mixed with text
        binary_test_cases = [
            # Null bytes in code
            {
                'content': 'def test():\n\x00\x01\x02\n    return "after binary"',
                'file_path': 'null_bytes.py'
            },
            # Control characters
            {
                'content': 'print("test")\x07\x08\x09\x0a\x0b\x0c\x0d\nprint("control chars")',
                'file_path': 'control_chars.py'
            },
            # High byte values
            {
                'content': 'code = "test"\n' + ''.join(chr(i) for i in range(128, 256)) + '\nprint(code)',
                'file_path': 'high_bytes.py'
            },
            # Mixed binary and UTF-8
            {
                'content': b'def test():\n    data = b"\xff\xfe\xfd"\n    return data'.decode('utf-8', errors='replace'),
                'file_path': 'mixed_binary.py'
            },
            # Very long binary sequences
            {
                'content': 'binary_data = "' + '\\x' + '\\x'.join(f'{i:02x}' for i in range(256)) * 10 + '"',
                'file_path': 'long_binary.py'
            }
        ]
        
        crashes = 0
        total_cases = len(binary_test_cases)
        
        for test_case in binary_test_cases:
            try:
                input_data = {
                    'content': test_case['content'],
                    'file_path': test_case['file_path'],
                    'config': {}
                }
                
                result = chunker_target.execute(input_data)
                
                # Should handle gracefully - either produce chunks or handle as binary
                assert isinstance(result, list)
                
            except Exception as e:
                crashes += 1
                print(f"Binary content crash: {type(e).__name__}: {e}")
        
        success_rate = (total_cases - crashes) / total_cases
        print(f"Binary content handling success rate: {success_rate:.2%}")
        
        # Should handle most binary content gracefully
        assert success_rate > 0.6, f"Binary content handling insufficient: {success_rate:.2%}"
    
    def test_extreme_file_sizes(self, fuzzer_engine):
        """Test handling of extremely large and small files."""
        chunker_target = PackRepoChunkerTarget()
        tokenizer_target = PackRepoTokenizerTarget()
        
        extreme_size_cases = [
            # Empty file
            {
                'content': '',
                'description': 'empty file'
            },
            # Single character
            {
                'content': 'x',
                'description': 'single character'
            },
            # Very long single line
            {
                'content': 'x = "' + 'a' * 100000 + '"',
                'description': 'very long single line'
            },
            # Many empty lines
            {
                'content': '\n' * 10000,
                'description': 'many empty lines'
            },
            # Many very short lines
            {
                'content': '\n'.join([f'x{i} = {i}' for i in range(10000)]),
                'description': 'many short lines'
            },
            # Deeply nested structure
            {
                'content': 'def f():\n' + '    if True:\n' * 1000 + '        return 42',
                'description': 'deeply nested structure'
            }
        ]
        
        chunker_results = []
        tokenizer_results = []
        
        for test_case in extreme_size_cases:
            content = test_case['content']
            description = test_case['description']
            
            # Test chunker with size limits
            try:
                chunker_input = {
                    'content': content,
                    'file_path': f'{description.replace(" ", "_")}.py',
                    'config': {'max_chunk_size': 1000}  # Limit chunk size
                }
                
                chunks = chunker_target.execute(chunker_input)
                chunker_results.append({
                    'description': description,
                    'success': True,
                    'num_chunks': len(chunks),
                    'content_size': len(content)
                })
                
            except Exception as e:
                chunker_results.append({
                    'description': description,
                    'success': False,
                    'error': str(e),
                    'content_size': len(content)
                })
            
            # Test tokenizer with reasonable size limits
            if len(content) < 1000000:  # Skip extremely large for tokenizer
                try:
                    tokenizer_input = {
                        'text': content,
                        'tokenizer': 'cl100k'
                    }
                    
                    token_count = tokenizer_target.execute(tokenizer_input)
                    tokenizer_results.append({
                        'description': description,
                        'success': True,
                        'token_count': token_count,
                        'content_size': len(content)
                    })
                    
                except Exception as e:
                    tokenizer_results.append({
                        'description': description,
                        'success': False,
                        'error': str(e),
                        'content_size': len(content)
                    })
        
        # Analyze results
        chunker_success_count = sum(1 for r in chunker_results if r['success'])
        tokenizer_success_count = sum(1 for r in tokenizer_results if r['success'])
        
        chunker_success_rate = chunker_success_count / len(chunker_results)
        tokenizer_success_rate = tokenizer_success_count / len(tokenizer_results)
        
        print(f"Extreme size handling results:")
        print(f"  Chunker success rate: {chunker_success_rate:.2%}")
        print(f"  Tokenizer success rate: {tokenizer_success_rate:.2%}")
        
        # Print failure details
        for result in chunker_results:
            if not result['success']:
                print(f"  Chunker failed on {result['description']}: {result['error']}")
        
        for result in tokenizer_results:
            if not result['success']:
                print(f"  Tokenizer failed on {result['description']}: {result['error']}")
        
        # Should handle most extreme cases gracefully
        assert chunker_success_rate > 0.7, f"Chunker extreme size handling insufficient"
        assert tokenizer_success_rate > 0.8, f"Tokenizer extreme size handling insufficient"
    
    def test_malformed_syntax_patterns(self, tree_sitter_fuzzer):
        """Test various malformed syntax patterns."""
        chunker_target = PackRepoChunkerTarget()
        
        # Generate corrupted samples
        languages = [LanguageGrammar.PYTHON, LanguageGrammar.JAVASCRIPT]
        malformed_samples = []
        
        for language in languages:
            # Generate with high corruption probability
            for sample in tree_sitter_fuzzer.generate_fuzzing_samples(
                language, 
                num_samples=10, 
                corruption_probability=0.8
            ):
                malformed_samples.append(sample)
        
        crashes = 0
        parse_failures = 0  # Samples that couldn't be parsed but didn't crash
        successful_parses = 0
        
        for sample in malformed_samples:
            try:
                input_data = {
                    'content': sample['content'],
                    'file_path': sample['file_path'],
                    'config': {}
                }
                
                result = chunker_target.execute(input_data)
                
                if isinstance(result, list) and len(result) > 0:
                    successful_parses += 1
                else:
                    parse_failures += 1
                    
            except Exception as e:
                crashes += 1
                # Log severe crashes for analysis
                if 'memory' in str(e).lower() or 'recursion' in str(e).lower():
                    print(f"Severe crash on malformed syntax: {type(e).__name__}: {e}")
        
        total_samples = len(malformed_samples)
        
        print(f"Malformed syntax handling results:")
        print(f"  Total samples: {total_samples}")
        print(f"  Successful parses: {successful_parses}")
        print(f"  Parse failures (graceful): {parse_failures}")
        print(f"  Crashes: {crashes}")
        
        # Should handle malformed syntax gracefully (not necessarily parse successfully)
        graceful_handling_rate = (successful_parses + parse_failures) / total_samples
        assert graceful_handling_rate > 0.8, f"Malformed syntax handling insufficient: {graceful_handling_rate:.2%}"
    
    @pytest.mark.performance
    def test_performance_with_large_files(self, fuzzer_engine):
        """Test performance characteristics with large files."""
        import time
        
        chunker_target = PackRepoChunkerTarget()
        
        # Generate files of increasing sizes
        sizes = [1000, 10000, 50000, 100000]  # Characters
        performance_results = []
        
        for size in sizes:
            # Generate content of target size
            content = 'def function():\n    return "test"\n\n' * (size // 30)
            content = content[:size]  # Trim to exact size
            
            input_data = {
                'content': content,
                'file_path': f'large_file_{size}.py',
                'config': {}
            }
            
            start_time = time.time()
            try:
                result = chunker_target.execute(input_data)
                end_time = time.time()
                
                processing_time = end_time - start_time
                performance_results.append({
                    'size': size,
                    'processing_time': processing_time,
                    'chunks_produced': len(result),
                    'success': True
                })
                
            except Exception as e:
                end_time = time.time()
                performance_results.append({
                    'size': size,
                    'processing_time': end_time - start_time,
                    'success': False,
                    'error': str(e)
                })
        
        # Analyze performance characteristics
        print(f"Performance test results:")
        for result in performance_results:
            if result['success']:
                rate = result['size'] / result['processing_time']
                print(f"  Size {result['size']:,}: {result['processing_time']:.3f}s, {rate:,.0f} chars/sec")
            else:
                print(f"  Size {result['size']:,}: FAILED - {result['error']}")
        
        # Verify reasonable performance characteristics
        successful_results = [r for r in performance_results if r['success']]
        if len(successful_results) > 1:
            # Check that processing time scales reasonably
            largest = successful_results[-1]
            smallest = successful_results[0]
            
            size_ratio = largest['size'] / smallest['size']
            time_ratio = largest['processing_time'] / smallest['processing_time']
            
            # Time should not scale worse than O(n^2)
            assert time_ratio < size_ratio ** 2, f"Performance scaling too poor: {time_ratio:.2f}x time for {size_ratio:.2f}x size"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])