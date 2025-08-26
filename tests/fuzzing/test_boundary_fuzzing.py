"""
Boundary condition fuzzing tests for PackRepo.

Tests edge cases and boundary conditions:
- Empty inputs
- Maximum/minimum values
- Buffer overflows and underflows
- Memory limits
- Numerical edge cases
- Encoding boundary conditions
"""

from __future__ import annotations

import pytest
import sys
import random
import string
import math
from typing import Dict, Any, List, Optional, Union

from .fuzzer_engine import FuzzerEngine, PackRepoSelectorTarget, PackRepoChunkerTarget, PackRepoTokenizerTarget


class TestBoundaryConditionFuzzing:
    """Test boundary conditions across PackRepo components."""
    
    @pytest.fixture
    def fuzzer_engine(self):
        """Create fuzzer engine for tests."""
        return FuzzerEngine(seed=12345)
    
    def test_empty_input_boundaries(self, fuzzer_engine):
        """Test various empty input conditions."""
        
        # Test cases for different components
        empty_test_cases = [
            # Selector with empty inputs
            {
                'target': 'selector',
                'inputs': [
                    {'chunks': [], 'budget': 0},
                    {'chunks': [], 'budget': 1000},
                    {'chunks': [{'id': 'empty', 'cost': 0, 'score': 0}], 'budget': 0},
                ]
            },
            # Chunker with empty inputs
            {
                'target': 'chunker',
                'inputs': [
                    {'content': '', 'file_path': 'empty.py'},
                    {'content': '', 'file_path': ''},
                    {'content': '\n\n\n', 'file_path': 'whitespace.py'},
                    {'content': '   \t   ', 'file_path': 'tabs_spaces.py'},
                ]
            },
            # Tokenizer with empty inputs
            {
                'target': 'tokenizer',
                'inputs': [
                    {'text': '', 'tokenizer': 'cl100k'},
                    {'text': '', 'tokenizer': 'o200k'},
                    {'text': ' ', 'tokenizer': 'cl100k'},
                    {'text': '\n', 'tokenizer': 'cl100k'},
                    {'text': '\t', 'tokenizer': 'cl100k'},
                ]
            }
        ]
        
        targets = {
            'selector': PackRepoSelectorTarget(),
            'chunker': PackRepoChunkerTarget(),
            'tokenizer': PackRepoTokenizerTarget()
        }
        
        results = []
        
        for test_case in empty_test_cases:
            target_name = test_case['target']
            target = targets[target_name]
            
            for input_data in test_case['inputs']:
                try:
                    # Add config if missing
                    if target_name == 'chunker' and 'config' not in input_data:
                        input_data['config'] = {}
                    
                    result = target.execute(input_data)
                    
                    # Validate empty input handling
                    if target_name == 'selector':
                        assert isinstance(result, (list, set)), f"Selector should return list/set for empty input"
                    elif target_name == 'chunker':
                        assert isinstance(result, list), f"Chunker should return list for empty input"
                    elif target_name == 'tokenizer':
                        assert isinstance(result, int) and result >= 0, f"Tokenizer should return non-negative int for empty input"
                    
                    results.append({
                        'target': target_name,
                        'input': str(input_data)[:100],
                        'success': True,
                        'result_type': type(result).__name__
                    })
                    
                except Exception as e:
                    results.append({
                        'target': target_name,
                        'input': str(input_data)[:100],
                        'success': False,
                        'error': f"{type(e).__name__}: {e}"
                    })
        
        # Analyze results
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r['success'])
        success_rate = successful_tests / total_tests
        
        print(f"Empty input boundary test results:")
        print(f"  Total tests: {total_tests}")
        print(f"  Successful: {successful_tests}")
        print(f"  Success rate: {success_rate:.2%}")
        
        # Show failures for debugging
        failures = [r for r in results if not r['success']]
        if failures:
            print(f"  Failures:")
            for failure in failures[:5]:
                print(f"    {failure['target']}: {failure['error']}")
        
        # Should handle empty inputs gracefully
        assert success_rate > 0.8, f"Empty input handling insufficient: {success_rate:.2%}"
    
    def test_maximum_value_boundaries(self, fuzzer_engine):
        """Test maximum value boundaries."""
        
        # Define maximum value test cases
        max_value_cases = [
            # Selector with maximum values
            {
                'target': 'selector',
                'inputs': [
                    # Maximum budget
                    {
                        'chunks': [{'id': 'chunk1', 'cost': 1, 'score': 1.0}],
                        'budget': sys.maxsize
                    },
                    # Maximum number of chunks (limited for performance)
                    {
                        'chunks': [{'id': f'chunk_{i}', 'cost': 1, 'score': random.uniform(0, 1)} 
                                 for i in range(10000)],
                        'budget': 5000
                    },
                    # Maximum cost per chunk
                    {
                        'chunks': [{'id': 'expensive', 'cost': sys.maxsize // 2, 'score': 1.0}],
                        'budget': sys.maxsize
                    }
                ]
            },
            # Chunker with maximum values
            {
                'target': 'chunker',
                'inputs': [
                    # Very long single line
                    {
                        'content': 'x = "' + 'a' * 1000000 + '"',
                        'file_path': 'very_long_line.py',
                        'config': {'max_chunk_size': 100000}
                    },
                    # Many lines
                    {
                        'content': '\n'.join([f'line_{i} = {i}' for i in range(100000)]),
                        'file_path': 'many_lines.py',
                        'config': {'max_chunk_size': 50000}
                    },
                ]
            },
            # Tokenizer with maximum values
            {
                'target': 'tokenizer',
                'inputs': [
                    # Very long text (but reasonable for tokenizer)
                    {
                        'text': 'This is a test sentence. ' * 100000,
                        'tokenizer': 'cl100k'
                    },
                    # Maximum Unicode codepoints
                    {
                        'text': ''.join([chr(0x10FFFF)] * 1000),  # Maximum Unicode
                        'tokenizer': 'cl100k'
                    }
                ]
            }
        ]
        
        targets = {
            'selector': PackRepoSelectorTarget(),
            'chunker': PackRepoChunkerTarget(),
            'tokenizer': PackRepoTokenizerTarget()
        }
        
        results = []
        timeouts = 0
        
        for test_case in max_value_cases:
            target_name = test_case['target']
            target = targets[target_name]
            
            for input_data in test_case['inputs']:
                try:
                    import signal
                    
                    # Set timeout for maximum value tests
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Test exceeded time limit")
                    
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(30)  # 30 second timeout
                    
                    try:
                        result = target.execute(input_data)
                        
                        # Basic validation
                        if target_name == 'selector':
                            assert isinstance(result, (list, set)), "Selector result type"
                        elif target_name == 'chunker':
                            assert isinstance(result, list), "Chunker result type"
                        elif target_name == 'tokenizer':
                            assert isinstance(result, int) and result >= 0, "Tokenizer result type"
                        
                        results.append({
                            'target': target_name,
                            'input_size': self._estimate_input_size(input_data),
                            'success': True,
                            'result_size': len(result) if hasattr(result, '__len__') else str(result)
                        })
                        
                    finally:
                        signal.alarm(0)  # Cancel timeout
                    
                except TimeoutError:
                    timeouts += 1
                    results.append({
                        'target': target_name,
                        'input_size': self._estimate_input_size(input_data),
                        'success': False,
                        'error': 'Timeout'
                    })
                    
                except Exception as e:
                    results.append({
                        'target': target_name,
                        'input_size': self._estimate_input_size(input_data),
                        'success': False,
                        'error': f"{type(e).__name__}: {str(e)[:100]}"
                    })
        
        # Analyze results
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r['success'])
        success_rate = successful_tests / total_tests
        
        print(f"Maximum value boundary test results:")
        print(f"  Total tests: {total_tests}")
        print(f"  Successful: {successful_tests}")
        print(f"  Timeouts: {timeouts}")
        print(f"  Success rate: {success_rate:.2%}")
        
        # Show sample results
        successful_results = [r for r in results if r['success']][:3]
        for result in successful_results:
            print(f"    {result['target']}: input_size={result['input_size']}, result_size={result['result_size']}")
        
        # Should handle reasonably large inputs
        assert success_rate > 0.5, f"Maximum value handling insufficient: {success_rate:.2%}"
    
    def test_minimum_value_boundaries(self, fuzzer_engine):
        """Test minimum value boundaries."""
        
        min_value_cases = [
            # Selector with minimum values
            {
                'target': 'selector',
                'inputs': [
                    # Minimum budget
                    {
                        'chunks': [{'id': 'chunk1', 'cost': 1, 'score': 0.0}],
                        'budget': 1
                    },
                    # Zero scores
                    {
                        'chunks': [{'id': 'chunk1', 'cost': 1, 'score': 0.0},
                                 {'id': 'chunk2', 'cost': 1, 'score': 0.0}],
                        'budget': 100
                    },
                    # Zero costs
                    {
                        'chunks': [{'id': 'free1', 'cost': 0, 'score': 1.0},
                                 {'id': 'free2', 'cost': 0, 'score': 0.5}],
                        'budget': 100
                    },
                    # Negative values (edge case)
                    {
                        'chunks': [{'id': 'negative', 'cost': -1, 'score': -0.5}],
                        'budget': 100
                    }
                ]
            },
            # Chunker with minimum values
            {
                'target': 'chunker',
                'inputs': [
                    # Single character
                    {
                        'content': 'x',
                        'file_path': 'single.py',
                        'config': {'max_chunk_size': 1}
                    },
                    # Minimum chunk size
                    {
                        'content': 'def f(): pass',
                        'file_path': 'minimal.py',
                        'config': {'max_chunk_size': 1}
                    }
                ]
            },
            # Tokenizer with minimum values
            {
                'target': 'tokenizer',
                'inputs': [
                    # Single character
                    {'text': 'a', 'tokenizer': 'cl100k'},
                    {'text': ' ', 'tokenizer': 'cl100k'},
                    {'text': '\n', 'tokenizer': 'cl100k'},
                    # Minimum Unicode
                    {'text': chr(1), 'tokenizer': 'cl100k'},
                ]
            }
        ]
        
        targets = {
            'selector': PackRepoSelectorTarget(),
            'chunker': PackRepoChunkerTarget(),
            'tokenizer': PackRepoTokenizerTarget()
        }
        
        results = []
        
        for test_case in min_value_cases:
            target_name = test_case['target']
            target = targets[target_name]
            
            for input_data in test_case['inputs']:
                try:
                    result = target.execute(input_data)
                    
                    # Validate minimum input handling
                    if target_name == 'tokenizer':
                        # Single characters should produce at least 1 token
                        if len(input_data['text']) > 0:
                            assert result > 0, f"Non-empty text should produce tokens: got {result}"
                    
                    results.append({
                        'target': target_name,
                        'input': str(input_data)[:100],
                        'success': True,
                        'result': str(result)[:50]
                    })
                    
                except Exception as e:
                    results.append({
                        'target': target_name,
                        'input': str(input_data)[:100],
                        'success': False,
                        'error': f"{type(e).__name__}: {e}"
                    })
        
        # Analyze results
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r['success'])
        success_rate = successful_tests / total_tests
        
        print(f"Minimum value boundary test results:")
        print(f"  Total tests: {total_tests}")
        print(f"  Successful: {successful_tests}")
        print(f"  Success rate: {success_rate:.2%}")
        
        # Show failures
        failures = [r for r in results if not r['success']]
        if failures:
            print(f"  Sample failures:")
            for failure in failures[:3]:
                print(f"    {failure['target']}: {failure['error']}")
        
        assert success_rate > 0.8, f"Minimum value handling insufficient: {success_rate:.2%}"
    
    def test_numerical_edge_cases(self, fuzzer_engine):
        """Test numerical edge cases and special float values."""
        
        # Special numerical values
        special_numbers = [
            float('inf'),      # Positive infinity
            float('-inf'),     # Negative infinity
            float('nan'),      # Not a number
            sys.float_info.max,    # Maximum float
            sys.float_info.min,    # Minimum positive float
            sys.float_info.epsilon, # Machine epsilon
            0.0,              # Zero
            -0.0,             # Negative zero
            1e-308,           # Very small number
            1e308,            # Very large number
        ]
        
        # Test selector with special numerical values
        selector_target = PackRepoSelectorTarget()
        selector_results = []
        
        for special_num in special_numbers:
            try:
                # Test as score
                input_data = {
                    'chunks': [{'id': 'test', 'cost': 1, 'score': special_num}],
                    'budget': 100
                }
                result = selector_target.execute(input_data)
                selector_results.append({
                    'value': special_num,
                    'context': 'score',
                    'success': True
                })
            except Exception as e:
                selector_results.append({
                    'value': special_num,
                    'context': 'score',
                    'success': False,
                    'error': str(e)
                })
            
            # Test as cost (only for finite positive values)
            if math.isfinite(special_num) and special_num >= 0:
                try:
                    input_data = {
                        'chunks': [{'id': 'test', 'cost': special_num, 'score': 1.0}],
                        'budget': special_num + 1 if special_num < 1e100 else 1e100
                    }
                    result = selector_target.execute(input_data)
                    selector_results.append({
                        'value': special_num,
                        'context': 'cost',
                        'success': True
                    })
                except Exception as e:
                    selector_results.append({
                        'value': special_num,
                        'context': 'cost',
                        'success': False,
                        'error': str(e)
                    })
            
            # Test as budget (only for finite positive values)
            if math.isfinite(special_num) and special_num > 0:
                try:
                    input_data = {
                        'chunks': [{'id': 'test', 'cost': 1, 'score': 1.0}],
                        'budget': special_num
                    }
                    result = selector_target.execute(input_data)
                    selector_results.append({
                        'value': special_num,
                        'context': 'budget',
                        'success': True
                    })
                except Exception as e:
                    selector_results.append({
                        'value': special_num,
                        'context': 'budget',
                        'success': False,
                        'error': str(e)
                    })
        
        # Analyze selector results
        total_selector_tests = len(selector_results)
        successful_selector = sum(1 for r in selector_results if r['success'])
        selector_success_rate = successful_selector / total_selector_tests if total_selector_tests > 0 else 1.0
        
        print(f"Numerical edge cases test results:")
        print(f"  Selector tests: {total_selector_tests}")
        print(f"  Selector success rate: {selector_success_rate:.2%}")
        
        # Show problematic special numbers
        failures = [r for r in selector_results if not r['success']]
        if failures:
            print(f"  Problematic special numbers:")
            for failure in failures[:5]:
                print(f"    {failure['value']} as {failure['context']}: {failure['error']}")
        
        # Should handle most numerical edge cases
        assert selector_success_rate > 0.6, f"Numerical edge case handling insufficient: {selector_success_rate:.2%}"
    
    def test_encoding_boundary_conditions(self, fuzzer_engine):
        """Test encoding and Unicode boundary conditions."""
        
        # Various encoding boundary cases
        encoding_test_cases = [
            # UTF-8 boundary conditions
            {
                'name': 'utf8_boundaries',
                'texts': [
                    '\u0000',           # Null character
                    '\u007F',           # Last 1-byte UTF-8 character
                    '\u0080',           # First 2-byte UTF-8 character
                    '\u07FF',           # Last 2-byte UTF-8 character
                    '\u0800',           # First 3-byte UTF-8 character
                    '\uFFFF',           # Last 3-byte UTF-8 character (BMP boundary)
                    '\U00010000',       # First 4-byte UTF-8 character
                    '\U0010FFFF',       # Last valid Unicode character
                ]
            },
            # Surrogate pairs and invalid sequences
            {
                'name': 'surrogate_pairs',
                'texts': [
                    '\uD800',           # High surrogate (invalid alone)
                    '\uDFFF',           # Low surrogate (invalid alone)
                    '\uD800\uDC00',     # Valid surrogate pair
                    '\uDBFF\uDFFF',     # Last valid surrogate pair
                ]
            },
            # Control characters
            {
                'name': 'control_chars',
                'texts': [
                    '\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F',
                    '\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1A\x1B\x1C\x1D\x1E\x1F',
                    '\x7F\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8A\x8B\x8C\x8D\x8E\x8F',
                ]
            },
            # Mixed valid and invalid
            {
                'name': 'mixed_validity',
                'texts': [
                    'Valid text ' + '\uFFFF' + ' more text',
                    'Hello' + '\x00' + 'World',
                    '测试' + '\uD800' + '中文',  # Chinese with invalid surrogate
                ]
            }
        ]
        
        # Test with tokenizer and chunker
        tokenizer_target = PackRepoTokenizerTarget()
        chunker_target = PackRepoChunkerTarget()
        
        encoding_results = []
        
        for test_case in encoding_test_cases:
            case_name = test_case['name']
            
            for i, text in enumerate(test_case['texts']):
                # Test tokenizer
                try:
                    tokenizer_input = {
                        'text': text,
                        'tokenizer': 'cl100k'
                    }
                    token_count = tokenizer_target.execute(tokenizer_input)
                    
                    encoding_results.append({
                        'case': case_name,
                        'text_id': i,
                        'target': 'tokenizer',
                        'success': True,
                        'result': token_count
                    })
                    
                except Exception as e:
                    encoding_results.append({
                        'case': case_name,
                        'text_id': i,
                        'target': 'tokenizer',
                        'success': False,
                        'error': str(e)[:100]
                    })
                
                # Test chunker
                try:
                    chunker_input = {
                        'content': text,
                        'file_path': f'{case_name}_{i}.txt',
                        'config': {}
                    }
                    chunks = chunker_target.execute(chunker_input)
                    
                    encoding_results.append({
                        'case': case_name,
                        'text_id': i,
                        'target': 'chunker',
                        'success': True,
                        'result': len(chunks)
                    })
                    
                except Exception as e:
                    encoding_results.append({
                        'case': case_name,
                        'text_id': i,
                        'target': 'chunker',
                        'success': False,
                        'error': str(e)[:100]
                    })
        
        # Analyze encoding results by case and target
        print(f"Encoding boundary conditions test results:")
        
        for test_case in encoding_test_cases:
            case_name = test_case['name']
            case_results = [r for r in encoding_results if r['case'] == case_name]
            
            tokenizer_results = [r for r in case_results if r['target'] == 'tokenizer']
            chunker_results = [r for r in case_results if r['target'] == 'chunker']
            
            tokenizer_success = sum(1 for r in tokenizer_results if r['success']) / len(tokenizer_results) if tokenizer_results else 1.0
            chunker_success = sum(1 for r in chunker_results if r['success']) / len(chunker_results) if chunker_results else 1.0
            
            print(f"  {case_name}:")
            print(f"    Tokenizer success: {tokenizer_success:.2%}")
            print(f"    Chunker success: {chunker_success:.2%}")
            
            # Different expectations for different cases
            if case_name == 'utf8_boundaries':
                # Should handle valid UTF-8 boundaries well
                assert tokenizer_success > 0.8, f"UTF-8 boundary tokenizer handling insufficient: {tokenizer_success:.2%}"
                assert chunker_success > 0.8, f"UTF-8 boundary chunker handling insufficient: {chunker_success:.2%}"
            elif case_name in ['control_chars', 'mixed_validity']:
                # Control chars and mixed validity might have lower success rates
                assert tokenizer_success > 0.5, f"{case_name} tokenizer handling too poor: {tokenizer_success:.2%}"
                assert chunker_success > 0.5, f"{case_name} chunker handling too poor: {chunker_success:.2%}"
    
    def test_memory_boundary_conditions(self, fuzzer_engine):
        """Test memory-related boundary conditions."""
        import gc
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Test progressively larger inputs
        memory_test_cases = [
            # Small inputs
            {'size': 1000, 'description': '1KB'},
            {'size': 10000, 'description': '10KB'},
            {'size': 100000, 'description': '100KB'},
            {'size': 1000000, 'description': '1MB'},
            {'size': 5000000, 'description': '5MB'},
        ]
        
        chunker_target = PackRepoChunkerTarget()
        tokenizer_target = PackRepoTokenizerTarget()
        
        memory_results = []
        
        for test_case in memory_test_cases:
            size = test_case['size']
            description = test_case['description']
            
            # Generate content of target size
            content = 'def function():\n    return "test"\n\n' * (size // 30)
            content = content[:size]
            
            # Force garbage collection
            gc.collect()
            
            # Test chunker
            try:
                before_memory = process.memory_info().rss / 1024 / 1024
                
                chunker_input = {
                    'content': content,
                    'file_path': f'large_{description}.py',
                    'config': {'max_chunk_size': min(10000, size // 10)}
                }
                
                chunks = chunker_target.execute(chunker_input)
                
                after_memory = process.memory_info().rss / 1024 / 1024
                memory_delta = after_memory - before_memory
                
                memory_results.append({
                    'target': 'chunker',
                    'input_size': size,
                    'description': description,
                    'success': True,
                    'memory_delta': memory_delta,
                    'chunks_produced': len(chunks)
                })
                
            except Exception as e:
                memory_results.append({
                    'target': 'chunker',
                    'input_size': size,
                    'description': description,
                    'success': False,
                    'error': str(e)[:100]
                })
            
            # Test tokenizer with smaller inputs (tokenizers can be memory intensive)
            if size <= 1000000:  # Limit tokenizer to 1MB
                try:
                    before_memory = process.memory_info().rss / 1024 / 1024
                    
                    tokenizer_input = {
                        'text': content,
                        'tokenizer': 'cl100k'
                    }
                    
                    token_count = tokenizer_target.execute(tokenizer_input)
                    
                    after_memory = process.memory_info().rss / 1024 / 1024
                    memory_delta = after_memory - before_memory
                    
                    memory_results.append({
                        'target': 'tokenizer',
                        'input_size': size,
                        'description': description,
                        'success': True,
                        'memory_delta': memory_delta,
                        'token_count': token_count
                    })
                    
                except Exception as e:
                    memory_results.append({
                        'target': 'tokenizer',
                        'input_size': size,
                        'description': description,
                        'success': False,
                        'error': str(e)[:100]
                    })
        
        # Analyze memory results
        print(f"Memory boundary conditions test results:")
        
        chunker_results = [r for r in memory_results if r['target'] == 'chunker']
        tokenizer_results = [r for r in memory_results if r['target'] == 'tokenizer']
        
        chunker_success_rate = sum(1 for r in chunker_results if r['success']) / len(chunker_results)
        tokenizer_success_rate = sum(1 for r in tokenizer_results if r['success']) / len(tokenizer_results)
        
        print(f"  Chunker success rate: {chunker_success_rate:.2%}")
        print(f"  Tokenizer success rate: {tokenizer_success_rate:.2%}")
        
        # Show memory usage patterns
        successful_chunker = [r for r in chunker_results if r['success']]
        successful_tokenizer = [r for r in tokenizer_results if r['success']]
        
        if successful_chunker:
            print(f"  Chunker memory usage:")
            for result in successful_chunker:
                print(f"    {result['description']}: +{result['memory_delta']:.1f}MB, {result['chunks_produced']} chunks")
        
        if successful_tokenizer:
            print(f"  Tokenizer memory usage:")
            for result in successful_tokenizer:
                print(f"    {result['description']}: +{result['memory_delta']:.1f}MB, {result['token_count']} tokens")
        
        # Should handle reasonable memory loads
        assert chunker_success_rate > 0.6, f"Memory handling chunker insufficient: {chunker_success_rate:.2%}"
        assert tokenizer_success_rate > 0.7, f"Memory handling tokenizer insufficient: {tokenizer_success_rate:.2%}"
        
        # Check for reasonable memory usage (not growing exponentially)
        if len(successful_chunker) > 1:
            largest = successful_chunker[-1]
            smallest = successful_chunker[0]
            size_ratio = largest['input_size'] / smallest['input_size']
            memory_ratio = max(largest['memory_delta'], 0.1) / max(smallest['memory_delta'], 0.1)
            
            # Memory usage should not grow much faster than input size
            if memory_ratio > size_ratio * 10:
                print(f"Warning: Memory usage growing too fast - {memory_ratio:.1f}x for {size_ratio:.1f}x input")
    
    def _estimate_input_size(self, input_data: Dict[str, Any]) -> int:
        """Estimate size of input data."""
        total_size = 0
        
        if 'content' in input_data:
            total_size += len(str(input_data['content']))
        
        if 'text' in input_data:
            total_size += len(str(input_data['text']))
        
        if 'chunks' in input_data:
            total_size += len(input_data['chunks']) * 100  # Rough estimate
        
        return total_size if total_size > 0 else len(str(input_data))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])