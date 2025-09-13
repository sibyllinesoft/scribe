"""
Tree-sitter based fuzzing for PackRepo code parsing.

Implements grammar-aware fuzzing for testing PackRepo's code chunking logic
with syntactically valid and invalid code samples.
"""

from __future__ import annotations

import random
import string
from typing import Dict, Any, List, Optional, Set, Iterator
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LanguageGrammar(Enum):
    """Supported language grammars."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    CPP = "cpp"
    HTML = "html"
    CSS = "css"
    JSON = "json"
    MARKDOWN = "markdown"


@dataclass 
class GrammarRule:
    """Represents a grammar rule for generating code."""
    name: str
    patterns: List[str]
    weight: float = 1.0  # Probability weight for selection
    requires: List[str] = None  # Dependencies on other rules
    
    def __post_init__(self):
        if self.requires is None:
            self.requires = []


class TreeSitterFuzzer:
    """Tree-sitter grammar-aware code fuzzer."""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
        # Grammar definitions for different languages
        self.grammars = self._initialize_grammars()
        
        # Track generated constructs to avoid infinite recursion
        self.generation_depth = 0
        self.max_depth = 10
    
    def _initialize_grammars(self) -> Dict[LanguageGrammar, Dict[str, GrammarRule]]:
        """Initialize grammar rules for supported languages."""
        grammars = {}
        
        # Python grammar rules
        grammars[LanguageGrammar.PYTHON] = {
            'module': GrammarRule(
                'module',
                [
                    '{imports}{classes}{functions}{statements}',
                    '{imports}{functions}{statements}',
                    '{functions}{statements}',
                    '{statements}'
                ]
            ),
            'imports': GrammarRule(
                'imports',
                [
                    'import os\nimport sys\n',
                    'from typing import Dict, List, Optional\n',
                    'import random\nfrom dataclasses import dataclass\n',
                    'import json\nimport pathlib\nfrom collections import defaultdict\n'
                ]
            ),
            'functions': GrammarRule(
                'functions',
                [
                    'def {function_name}({parameters}):\n    {docstring}{statements}\n\n',
                    'def {function_name}({parameters}) -> {return_type}:\n    {docstring}{statements}\n\n',
                    'async def {function_name}({parameters}):\n    {docstring}{statements}\n\n'
                ],
                requires=['function_name', 'parameters', 'return_type', 'docstring', 'statements']
            ),
            'classes': GrammarRule(
                'classes',
                [
                    'class {class_name}:\n    {docstring}{class_body}\n\n',
                    'class {class_name}({base_class}):\n    {docstring}{class_body}\n\n',
                    '@dataclass\nclass {class_name}:\n    {docstring}{class_body}\n\n'
                ],
                requires=['class_name', 'base_class', 'docstring', 'class_body']
            ),
            'class_body': GrammarRule(
                'class_body',
                [
                    '    def __init__(self{parameters}):\n        {statements}\n\n    {methods}',
                    '    {class_variables}\n\n    {methods}',
                    '    {methods}'
                ],
                requires=['parameters', 'statements', 'methods', 'class_variables']
            ),
            'methods': GrammarRule(
                'methods',
                [
                    '    def {method_name}(self{parameters}):\n        {docstring}        {statements}\n',
                    '    def {method_name}(self{parameters}) -> {return_type}:\n        {docstring}        {statements}\n',
                    '    @property\n    def {method_name}(self):\n        {docstring}        {statements}\n'
                ],
                requires=['method_name', 'parameters', 'return_type', 'docstring', 'statements']
            ),
            'statements': GrammarRule(
                'statements',
                [
                    '    {variable} = {expression}\n',
                    '    if {condition}:\n        {statements}',
                    '    for {variable} in {iterable}:\n        {statements}',
                    '    while {condition}:\n        {statements}',
                    '    try:\n        {statements}    except Exception as e:\n        {statements}',
                    '    return {expression}\n',
                    '    {function_call}\n',
                    '    pass\n'
                ],
                requires=['variable', 'expression', 'condition', 'iterable', 'function_call']
            ),
            'function_name': GrammarRule(
                'function_name',
                ['process_data', 'calculate_score', 'validate_input', 'handle_error', 'main', 'helper']
            ),
            'class_name': GrammarRule(
                'class_name', 
                ['DataProcessor', 'Calculator', 'Validator', 'Handler', 'Manager', 'Service']
            ),
            'method_name': GrammarRule(
                'method_name',
                ['process', 'calculate', 'validate', 'handle', 'get', 'set', 'update', 'delete']
            ),
            'variable': GrammarRule(
                'variable',
                ['data', 'result', 'value', 'item', 'config', 'params', 'output', 'temp']
            ),
            'parameters': GrammarRule(
                'parameters',
                ['', ', data', ', value: int', ', config: Dict[str, Any]', ', *args', ', **kwargs']
            ),
            'return_type': GrammarRule(
                'return_type',
                ['int', 'str', 'Dict[str, Any]', 'List[str]', 'Optional[str]', 'bool']
            ),
            'expression': GrammarRule(
                'expression',
                ['42', '"hello"', 'True', 'None', '[]', '{}', 'len(data)', 'str(value)', 'data[0]']
            ),
            'condition': GrammarRule(
                'condition',
                ['True', 'data is not None', 'len(data) > 0', 'value == 42', 'isinstance(data, str)']
            ),
            'iterable': GrammarRule(
                'iterable',
                ['range(10)', 'data', '["a", "b", "c"]', 'data.items()', 'enumerate(data)']
            ),
            'function_call': GrammarRule(
                'function_call',
                ['print(data)', 'len(data)', 'str(value)', 'process_data()', 'helper(data)']
            ),
            'docstring': GrammarRule(
                'docstring',
                [
                    '    """Function description."""\n',
                    '    """Class description."""\n',
                    '    """\n    Function description.\n    \n    Args:\n        data: Input data.\n    """\n',
                    ''  # No docstring
                ]
            ),
            'base_class': GrammarRule(
                'base_class',
                ['object', 'BaseClass', 'ABC', 'Exception']
            ),
            'class_variables': GrammarRule(
                'class_variables',
                ['    name: str', '    value: int = 0', '    data: Dict[str, Any] = field(default_factory=dict)']
            )
        }
        
        # JavaScript grammar rules (simplified)
        grammars[LanguageGrammar.JAVASCRIPT] = {
            'module': GrammarRule(
                'module',
                [
                    '{imports}{classes}{functions}{statements}',
                    '{functions}{statements}',
                    '{statements}'
                ]
            ),
            'imports': GrammarRule(
                'imports', 
                [
                    "const fs = require('fs');\nconst path = require('path');\n",
                    "import { useState, useEffect } from 'react';\n",
                    "const express = require('express');\n"
                ]
            ),
            'functions': GrammarRule(
                'functions',
                [
                    'function {function_name}({parameters}) {\n{statements}}\n\n',
                    'const {function_name} = ({parameters}) => {\n{statements}};\n\n',
                    'async function {function_name}({parameters}) {\n{statements}}\n\n'
                ],
                requires=['function_name', 'parameters', 'statements']
            ),
            'classes': GrammarRule(
                'classes',
                [
                    'class {class_name} {\n{class_body}}\n\n',
                    'class {class_name} extends {base_class} {\n{class_body}}\n\n'
                ],
                requires=['class_name', 'base_class', 'class_body']
            ),
            'class_body': GrammarRule(
                'class_body',
                [
                    '  constructor({parameters}) {\n{statements}  }\n\n{methods}',
                    '  {methods}'
                ],
                requires=['parameters', 'statements', 'methods']
            ),
            'methods': GrammarRule(
                'methods',
                [
                    '  {method_name}({parameters}) {\n{statements}  }\n',
                    '  async {method_name}({parameters}) {\n{statements}  }\n'
                ],
                requires=['method_name', 'parameters', 'statements']
            ),
            'statements': GrammarRule(
                'statements',
                [
                    '  const {variable} = {expression};\n',
                    '  let {variable} = {expression};\n',
                    '  if ({condition}) {\n{statements}  }\n',
                    '  for (const {variable} of {iterable}) {\n{statements}  }\n',
                    '  return {expression};\n',
                    '  {function_call};\n'
                ],
                requires=['variable', 'expression', 'condition', 'iterable', 'function_call']
            ),
            'function_name': GrammarRule(
                'function_name',
                ['processData', 'calculateScore', 'validateInput', 'handleError', 'main', 'helper']
            ),
            'class_name': GrammarRule(
                'class_name',
                ['DataProcessor', 'Calculator', 'Validator', 'Handler', 'Manager', 'Service']
            ),
            'method_name': GrammarRule(
                'method_name',
                ['process', 'calculate', 'validate', 'handle', 'get', 'set', 'update', 'delete']
            ),
            'variable': GrammarRule(
                'variable',
                ['data', 'result', 'value', 'item', 'config', 'params', 'output', 'temp']
            ),
            'parameters': GrammarRule(
                'parameters',
                ['', 'data', 'value', 'config', '...args']
            ),
            'expression': GrammarRule(
                'expression',
                ['42', '"hello"', 'true', 'null', '[]', '{}', 'data.length', 'String(value)', 'data[0]']
            ),
            'condition': GrammarRule(
                'condition',
                ['true', 'data != null', 'data.length > 0', 'value === 42', 'typeof data === "string"']
            ),
            'iterable': GrammarRule(
                'iterable',
                ['[1, 2, 3]', 'data', '["a", "b", "c"]', 'Object.keys(data)']
            ),
            'function_call': GrammarRule(
                'function_call',
                ['console.log(data)', 'processData()', 'helper(data)']
            ),
            'base_class': GrammarRule(
                'base_class',
                ['Object', 'Error', 'EventEmitter']
            )
        }
        
        return grammars
    
    def generate_code(self, 
                     language: LanguageGrammar, 
                     target_lines: Optional[int] = None,
                     corruption_probability: float = 0.0) -> str:
        """Generate code using grammar rules."""
        
        if language not in self.grammars:
            raise ValueError(f"Unsupported language: {language}")
        
        grammar = self.grammars[language]
        
        # Reset generation state
        self.generation_depth = 0
        
        # Start with module rule
        code = self._expand_rule('module', grammar, {})
        
        # Apply corruption if requested
        if corruption_probability > 0:
            code = self._apply_corruption(code, corruption_probability)
        
        # Truncate or expand to target lines if specified
        if target_lines:
            code = self._adjust_to_target_lines(code, target_lines, language)
        
        return code
    
    def _expand_rule(self, 
                    rule_name: str, 
                    grammar: Dict[str, GrammarRule], 
                    context: Dict[str, str]) -> str:
        """Expand a grammar rule recursively."""
        
        # Prevent infinite recursion
        self.generation_depth += 1
        if self.generation_depth > self.max_depth:
            return ""
        
        if rule_name not in grammar:
            # If rule not found, return the rule name as-is (might be literal)
            return rule_name
        
        rule = grammar[rule_name]
        
        # Select pattern based on weights
        pattern = self._weighted_choice(rule.patterns, [rule.weight] * len(rule.patterns))
        
        # Find all placeholders in pattern
        placeholders = self._find_placeholders(pattern)
        
        # Expand each placeholder
        expanded = pattern
        for placeholder in placeholders:
            if placeholder in context:
                # Use cached expansion
                replacement = context[placeholder]
            else:
                # Recursively expand
                replacement = self._expand_rule(placeholder, grammar, context)
                context[placeholder] = replacement  # Cache for consistency
            
            expanded = expanded.replace('{' + placeholder + '}', replacement)
        
        self.generation_depth -= 1
        return expanded
    
    def _find_placeholders(self, pattern: str) -> List[str]:
        """Find all placeholder names in a pattern."""
        placeholders = []
        i = 0
        while i < len(pattern):
            if pattern[i] == '{':
                # Find closing brace
                j = pattern.find('}', i)
                if j != -1:
                    placeholders.append(pattern[i+1:j])
                    i = j + 1
                else:
                    i += 1
            else:
                i += 1
        return placeholders
    
    def _weighted_choice(self, choices: List[str], weights: List[float]) -> str:
        """Choose randomly from weighted choices."""
        if not choices:
            return ""
        
        if len(choices) == 1:
            return choices[0]
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(choices)
        
        normalized_weights = [w / total_weight for w in weights]
        
        # Random selection
        r = random.random()
        cumulative = 0
        for choice, weight in zip(choices, normalized_weights):
            cumulative += weight
            if r <= cumulative:
                return choice
        
        return choices[-1]  # Fallback
    
    def _apply_corruption(self, code: str, corruption_probability: float) -> str:
        """Apply random corruption to code for robustness testing."""
        if random.random() > corruption_probability:
            return code
        
        corruptions = [
            self._corrupt_indentation,
            self._corrupt_syntax,
            self._corrupt_encoding,
            self._corrupt_line_endings
        ]
        
        corruption_func = random.choice(corruptions)
        return corruption_func(code)
    
    def _corrupt_indentation(self, code: str) -> str:
        """Corrupt indentation."""
        lines = code.split('\n')
        corrupted_lines = []
        
        for line in lines:
            if random.random() < 0.2 and line.strip():  # 20% chance
                # Randomly change indentation
                stripped = line.lstrip()
                if stripped:
                    new_indent = random.randint(0, 8)
                    corrupted_lines.append(' ' * new_indent + stripped)
                else:
                    corrupted_lines.append(line)
            else:
                corrupted_lines.append(line)
        
        return '\n'.join(corrupted_lines)
    
    def _corrupt_syntax(self, code: str) -> str:
        """Corrupt syntax elements."""
        corruption_rules = [
            (r':', ''),  # Remove colons
            (r'\(', ''),  # Remove opening parens
            (r'\)', ''),  # Remove closing parens
            (r'"', "'"),  # Change quotes
            (r'    ', '\t'),  # Mix tabs and spaces
            (r'=', '=='),  # Change assignment to comparison
        ]
        
        rule_pattern, replacement = random.choice(corruption_rules)
        
        # Apply corruption to random lines
        lines = code.split('\n')
        for i in range(len(lines)):
            if random.random() < 0.1:  # 10% chance per line
                import re
                lines[i] = re.sub(rule_pattern, replacement, lines[i], count=1)
        
        return '\n'.join(lines)
    
    def _corrupt_encoding(self, code: str) -> str:
        """Corrupt character encoding."""
        corrupted = bytearray(code.encode('utf-8'))
        
        # Randomly corrupt a few bytes
        for _ in range(random.randint(1, 3)):
            if corrupted:
                pos = random.randint(0, len(corrupted) - 1)
                corrupted[pos] = random.randint(0, 255)
        
        # Try to decode, using replacement characters for invalid bytes
        try:
            return corrupted.decode('utf-8', errors='replace')
        except:
            return code  # Return original if corruption fails
    
    def _corrupt_line_endings(self, code: str) -> str:
        """Corrupt line endings."""
        # Mix different line ending styles
        endings = ['\n', '\r\n', '\r']
        
        lines = code.split('\n')
        result = ""
        
        for i, line in enumerate(lines):
            result += line
            if i < len(lines) - 1:  # Don't add ending to last line
                result += random.choice(endings)
        
        return result
    
    def _adjust_to_target_lines(self, code: str, target_lines: int, language: LanguageGrammar) -> str:
        """Adjust generated code to approximately target number of lines."""
        current_lines = len(code.split('\n'))
        
        if current_lines < target_lines:
            # Need to add more content
            return self._expand_code(code, target_lines - current_lines, language)
        elif current_lines > target_lines:
            # Need to truncate
            lines = code.split('\n')
            return '\n'.join(lines[:target_lines])
        else:
            return code
    
    def _expand_code(self, code: str, additional_lines: int, language: LanguageGrammar) -> str:
        """Add more code to reach target line count."""
        grammar = self.grammars[language]
        
        # Generate additional functions/statements
        additions = []
        lines_added = 0
        
        while lines_added < additional_lines:
            if random.random() < 0.7:  # 70% chance for function
                additional = self._expand_rule('functions', grammar, {})
            else:
                additional = self._expand_rule('statements', grammar, {})
            
            additions.append(additional)
            lines_added += len(additional.split('\n'))
        
        return code + '\n' + '\n'.join(additions)
    
    def generate_fuzzing_samples(self, 
                                language: LanguageGrammar,
                                num_samples: int = 10,
                                corruption_probability: float = 0.1) -> Iterator[Dict[str, Any]]:
        """Generate multiple fuzzing samples."""
        
        for i in range(num_samples):
            # Vary parameters for each sample
            target_lines = random.choice([None, 10, 50, 100, 500])
            corruption_prob = corruption_probability * random.uniform(0.5, 1.5)
            
            try:
                code = self.generate_code(language, target_lines, corruption_prob)
                
                yield {
                    'sample_id': i,
                    'language': language.value,
                    'target_lines': target_lines,
                    'actual_lines': len(code.split('\n')),
                    'corruption_probability': corruption_prob,
                    'content': code,
                    'file_path': f'fuzz_sample_{i}.{self._get_file_extension(language)}'
                }
                
            except Exception as e:
                logger.warning(f"Failed to generate sample {i} for {language.value}: {e}")
                continue
    
    def _get_file_extension(self, language: LanguageGrammar) -> str:
        """Get appropriate file extension for language."""
        extensions = {
            LanguageGrammar.PYTHON: 'py',
            LanguageGrammar.JAVASCRIPT: 'js',
            LanguageGrammar.TYPESCRIPT: 'ts',
            LanguageGrammar.RUST: 'rs',
            LanguageGrammar.GO: 'go',
            LanguageGrammar.JAVA: 'java',
            LanguageGrammar.CPP: 'cpp',
            LanguageGrammar.HTML: 'html',
            LanguageGrammar.CSS: 'css',
            LanguageGrammar.JSON: 'json',
            LanguageGrammar.MARKDOWN: 'md'
        }
        return extensions.get(language, 'txt')
    
    def generate_edge_cases(self, language: LanguageGrammar) -> Iterator[Dict[str, Any]]:
        """Generate edge case samples for testing."""
        
        edge_cases = [
            # Empty file
            {
                'name': 'empty_file',
                'content': '',
                'description': 'Empty file'
            },
            # Single line
            {
                'name': 'single_line',
                'content': 'print("hello")' if language == LanguageGrammar.PYTHON else 'console.log("hello");',
                'description': 'Single line of code'
            },
            # Very long line
            {
                'name': 'very_long_line',
                'content': 'x = ' + '"' + 'a' * 10000 + '"',
                'description': 'Very long line'
            },
            # Many short lines
            {
                'name': 'many_short_lines',
                'content': '\n'.join([f'x{i} = {i}' for i in range(1000)]),
                'description': 'Many short lines'
            },
            # Unicode content
            {
                'name': 'unicode_content',
                'content': '# 测试中文注释\n变量 = "中文字符串"\nprint(变量)' if language == LanguageGrammar.PYTHON 
                          else '// 测试中文注释\nconst 变量 = "中文字符串";\nconsole.log(变量);',
                'description': 'Unicode characters'
            },
            # Mixed indentation
            {
                'name': 'mixed_indentation',
                'content': 'def func():\n    if True:\n\t\treturn 42\n  else:\n      return 0',
                'description': 'Mixed tabs and spaces'
            },
            # Nested structure
            {
                'name': 'deeply_nested',
                'content': self._generate_deeply_nested_code(language),
                'description': 'Deeply nested code structure'
            },
            # Binary/control characters
            {
                'name': 'control_chars',
                'content': 'print("test")\x00\x01\x02\nprint("after binary")',
                'description': 'Control and binary characters'
            }
        ]
        
        for i, case in enumerate(edge_cases):
            yield {
                'sample_id': f'edge_{i}',
                'language': language.value,
                'edge_case_name': case['name'],
                'description': case['description'],
                'content': case['content'],
                'file_path': f'edge_{case["name"]}.{self._get_file_extension(language)}',
                'actual_lines': len(case['content'].split('\n'))
            }
    
    def _generate_deeply_nested_code(self, language: LanguageGrammar) -> str:
        """Generate deeply nested code structure."""
        if language == LanguageGrammar.PYTHON:
            base = "def func():\n"
            indent = "    "
            for i in range(20):  # 20 levels deep
                base += indent * (i + 1) + f"if x > {i}:\n"
            base += indent * 21 + "return 42"
            return base
        elif language == LanguageGrammar.JAVASCRIPT:
            base = "function func() {\n"
            indent = "  "
            for i in range(20):
                base += indent * (i + 1) + f"if (x > {i}) {{\n"
            base += indent * 21 + "return 42;\n"
            for i in range(20):
                base += indent * (20 - i) + "}\n"
            base += "}"
            return base
        else:
            return "// Nested structure not implemented for this language"


# Convenience function
def generate_tree_sitter_fuzzing_samples(language: str, 
                                        num_samples: int = 10,
                                        include_edge_cases: bool = True,
                                        corruption_probability: float = 0.1) -> List[Dict[str, Any]]:
    """Generate fuzzing samples for a given language."""
    
    try:
        lang_enum = LanguageGrammar(language.lower())
    except ValueError:
        raise ValueError(f"Unsupported language: {language}")
    
    fuzzer = TreeSitterFuzzer()
    samples = []
    
    # Generate regular samples
    for sample in fuzzer.generate_fuzzing_samples(lang_enum, num_samples, corruption_probability):
        samples.append(sample)
    
    # Generate edge cases
    if include_edge_cases:
        for edge_case in fuzzer.generate_edge_cases(lang_enum):
            samples.append(edge_case)
    
    return samples


if __name__ == "__main__":
    # Demo usage
    fuzzer = TreeSitterFuzzer(seed=42)
    
    # Generate Python samples
    print("=== Python Samples ===")
    for i, sample in enumerate(fuzzer.generate_fuzzing_samples(LanguageGrammar.PYTHON, 3)):
        print(f"Sample {i}:")
        print(sample['content'][:200] + "..." if len(sample['content']) > 200 else sample['content'])
        print("-" * 50)
    
    # Generate edge cases
    print("\n=== Edge Cases ===")
    for sample in fuzzer.generate_edge_cases(LanguageGrammar.PYTHON):
        if sample['edge_case_name'] in ['empty_file', 'single_line']:
            print(f"{sample['edge_case_name']}: '{sample['content']}'")