"""
Core fuzzing engine for PackRepo testing.

Implements the main fuzzing orchestration with support for:
- Multi-strategy fuzzing (content, structure, boundary, concolic)
- Crash detection and classification  
- Coverage-guided fuzzing
- Time-based fuzzing campaigns
- Regression crash detection
"""

from __future__ import annotations

import os
import sys
import time
import random
import hashlib
import tempfile
import subprocess
import traceback
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Callable, Iterator, NamedTuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

# Configure logging for fuzzer
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrashSeverity(Enum):
    """Crash severity levels."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class FuzzingStrategy(Enum):
    """Available fuzzing strategies."""
    FILE_CONTENT = "file_content"
    REPOSITORY_STRUCTURE = "repository_structure"  
    BOUNDARY_CONDITIONS = "boundary_conditions"
    CONCOLIC_TESTING = "concolic_testing"
    RANDOM_GENERATION = "random_generation"


@dataclass
class CrashInfo:
    """Information about a discovered crash."""
    crash_id: str
    severity: CrashSeverity
    exception_type: str
    exception_message: str
    stack_trace: str
    input_hash: str
    strategy: FuzzingStrategy
    timestamp: float
    reproduction_steps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'crash_id': self.crash_id,
            'severity': self.severity.value,
            'exception_type': self.exception_type,
            'exception_message': self.exception_message,
            'stack_trace': self.stack_trace,
            'input_hash': self.input_hash,
            'strategy': self.strategy.value,
            'timestamp': self.timestamp,
            'reproduction_steps': self.reproduction_steps
        }


@dataclass
class FuzzingResult:
    """Results from a fuzzing campaign."""
    campaign_id: str
    start_time: float
    end_time: float
    total_executions: int
    crashes_found: List[CrashInfo] = field(default_factory=list)
    coverage_data: Dict[str, Any] = field(default_factory=dict)
    strategy_stats: Dict[FuzzingStrategy, int] = field(default_factory=dict)
    
    @property
    def duration_minutes(self) -> float:
        """Campaign duration in minutes."""
        return (self.end_time - self.start_time) / 60.0
    
    @property
    def executions_per_minute(self) -> float:
        """Execution rate."""
        if self.duration_minutes == 0:
            return 0.0
        return self.total_executions / self.duration_minutes
    
    def crashes_by_severity(self) -> Dict[CrashSeverity, int]:
        """Count crashes by severity."""
        counts = {severity: 0 for severity in CrashSeverity}
        for crash in self.crashes_found:
            counts[crash.severity] += 1
        return counts
    
    def has_medium_plus_crashes(self) -> bool:
        """Check if any medium or higher severity crashes were found."""
        severe_crashes = [c for c in self.crashes_found 
                         if c.severity in [CrashSeverity.MEDIUM, CrashSeverity.HIGH, CrashSeverity.CRITICAL]]
        return len(severe_crashes) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'campaign_id': self.campaign_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_minutes': self.duration_minutes,
            'total_executions': self.total_executions,
            'executions_per_minute': self.executions_per_minute,
            'crashes_found': [crash.to_dict() for crash in self.crashes_found],
            'crashes_by_severity': {sev.value: count for sev, count in self.crashes_by_severity().items()},
            'has_medium_plus_crashes': self.has_medium_plus_crashes(),
            'coverage_data': self.coverage_data,
            'strategy_stats': {strat.value: count for strat, count in self.strategy_stats.items()}
        }


class FuzzTarget:
    """Base class for fuzzing targets."""
    
    def __init__(self, name: str):
        self.name = name
    
    def execute(self, input_data: Any) -> Any:
        """Execute the target with given input."""
        raise NotImplementedError
    
    def classify_crash(self, exception: Exception, input_data: Any) -> CrashSeverity:
        """Classify the severity of a crash."""
        # Default classification logic
        if isinstance(exception, (MemoryError, RecursionError)):
            return CrashSeverity.CRITICAL
        elif isinstance(exception, (ValueError, TypeError, KeyError, IndexError)):
            return CrashSeverity.MEDIUM
        elif isinstance(exception, (AssertionError, RuntimeError)):
            return CrashSeverity.HIGH
        else:
            return CrashSeverity.LOW


class PackRepoSelectorTarget(FuzzTarget):
    """Fuzzing target for PackRepo selector algorithms."""
    
    def __init__(self):
        super().__init__("packrepo_selector")
    
    def execute(self, input_data: Dict[str, Any]) -> Any:
        """Execute selector with fuzzed input."""
        try:
            from packrepo.packer.selector.selector import Selector
            
            # Extract fuzzing parameters
            chunks = input_data.get('chunks', [])
            budget = input_data.get('budget', 1000)
            selector_config = input_data.get('config', {})
            
            # Create selector instance
            selector = Selector(**selector_config)
            
            # Execute selection
            result = selector.select(chunks, budget)
            
            # Validate result constraints
            self._validate_selection_result(result, budget)
            
            return result
            
        except Exception as e:
            logger.debug(f"Selector execution failed: {e}")
            raise
    
    def _validate_selection_result(self, result: Any, budget: int):
        """Validate selection result meets basic constraints."""
        if not isinstance(result, (list, set)):
            raise ValueError(f"Invalid result type: {type(result)}")
        
        # Check budget constraint if result has cost information
        total_cost = 0
        for item in result:
            if hasattr(item, 'cost'):
                total_cost += item.cost
            elif isinstance(item, dict) and 'cost' in item:
                total_cost += item['cost']
        
        if total_cost > budget * 1.01:  # Allow small tolerance
            raise ValueError(f"Budget constraint violated: {total_cost} > {budget}")


class PackRepoChunkerTarget(FuzzTarget):
    """Fuzzing target for PackRepo chunker algorithms."""
    
    def __init__(self):
        super().__init__("packrepo_chunker")
    
    def execute(self, input_data: Dict[str, Any]) -> Any:
        """Execute chunker with fuzzed input."""
        try:
            from packrepo.packer.chunker.chunker import CodeChunker
            
            # Extract parameters
            file_content = input_data.get('content', '')
            file_path = input_data.get('file_path', 'test.py')
            chunker_config = input_data.get('config', {})
            
            # Create chunker
            chunker = CodeChunker(**chunker_config)
            
            # Execute chunking
            chunks = chunker.chunk_file(file_content, file_path)
            
            # Validate chunks
            self._validate_chunks(chunks, file_content)
            
            return chunks
            
        except Exception as e:
            logger.debug(f"Chunker execution failed: {e}")
            raise
    
    def _validate_chunks(self, chunks: List[Any], original_content: str):
        """Validate chunking results."""
        if not isinstance(chunks, list):
            raise ValueError(f"Invalid chunks type: {type(chunks)}")
        
        # Check that chunk line numbers are valid
        original_lines = original_content.split('\n')
        max_line = len(original_lines)
        
        for chunk in chunks:
            if hasattr(chunk, 'start_line') and hasattr(chunk, 'end_line'):
                if chunk.start_line < 1 or chunk.end_line > max_line:
                    raise ValueError(f"Invalid chunk line range: {chunk.start_line}-{chunk.end_line}")
                if chunk.start_line > chunk.end_line:
                    raise ValueError(f"Invalid chunk range: start > end")


class PackRepoTokenizerTarget(FuzzTarget):
    """Fuzzing target for PackRepo tokenizer."""
    
    def __init__(self):
        super().__init__("packrepo_tokenizer")
    
    def execute(self, input_data: Dict[str, Any]) -> Any:
        """Execute tokenizer with fuzzed input."""
        try:
            from packrepo.packer.tokenizer.implementations import get_tokenizer
            
            # Extract parameters
            text = input_data.get('text', '')
            tokenizer_name = input_data.get('tokenizer', 'cl100k')
            
            # Get tokenizer
            tokenizer = get_tokenizer(tokenizer_name)
            
            # Execute tokenization
            token_count = tokenizer.count_tokens(text)
            
            # Validate result
            if token_count < 0:
                raise ValueError(f"Invalid token count: {token_count}")
            
            # For very long inputs, token count should be reasonable
            if len(text) > 100000 and token_count > len(text) * 2:
                raise ValueError(f"Unreasonably high token count: {token_count} for {len(text)} chars")
            
            return token_count
            
        except Exception as e:
            logger.debug(f"Tokenizer execution failed: {e}")
            raise


class FuzzerEngine:
    """Main fuzzing engine orchestrating all strategies."""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed or int(time.time())
        random.seed(self.seed)
        
        self.targets: Dict[str, FuzzTarget] = {}
        self.strategies: Dict[FuzzingStrategy, Callable] = {}
        self.known_crashes: Set[str] = set()
        
        # Register default targets
        self._register_default_targets()
        
        # Register default strategies
        self._register_default_strategies()
    
    def _register_default_targets(self):
        """Register default fuzzing targets."""
        self.register_target(PackRepoSelectorTarget())
        self.register_target(PackRepoChunkerTarget()) 
        self.register_target(PackRepoTokenizerTarget())
    
    def _register_default_strategies(self):
        """Register default fuzzing strategies."""
        self.strategies[FuzzingStrategy.RANDOM_GENERATION] = self._random_generation_strategy
        self.strategies[FuzzingStrategy.BOUNDARY_CONDITIONS] = self._boundary_conditions_strategy
        self.strategies[FuzzingStrategy.FILE_CONTENT] = self._file_content_strategy
    
    def register_target(self, target: FuzzTarget):
        """Register a fuzzing target."""
        self.targets[target.name] = target
    
    def register_strategy(self, strategy: FuzzingStrategy, generator_func: Callable):
        """Register a fuzzing strategy."""
        self.strategies[strategy] = generator_func
    
    def run_campaign(self, 
                    duration_minutes: float,
                    target_names: Optional[List[str]] = None,
                    strategies: Optional[List[FuzzingStrategy]] = None) -> FuzzingResult:
        """Run a fuzzing campaign for specified duration."""
        
        campaign_id = f"fuzz_{int(time.time())}_{random.randint(1000, 9999)}"
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        # Select targets
        if target_names is None:
            target_names = list(self.targets.keys())
        
        # Select strategies 
        if strategies is None:
            strategies = list(self.strategies.keys())
        
        result = FuzzingResult(
            campaign_id=campaign_id,
            start_time=start_time,
            end_time=end_time,
            total_executions=0
        )
        
        logger.info(f"Starting fuzzing campaign {campaign_id} for {duration_minutes:.1f} minutes")
        logger.info(f"Targets: {target_names}")
        logger.info(f"Strategies: {[s.value for s in strategies]}")
        
        execution_count = 0
        
        # Main fuzzing loop
        while time.time() < end_time:
            try:
                # Select target and strategy
                target_name = random.choice(target_names)
                strategy = random.choice(strategies)
                target = self.targets[target_name]
                
                # Generate input using strategy
                input_data = self.strategies[strategy](target_name)
                
                # Calculate input hash for deduplication
                input_hash = hashlib.sha256(str(input_data).encode()).hexdigest()
                
                # Execute target
                try:
                    target.execute(input_data)
                    execution_count += 1
                    
                    # Update strategy stats
                    result.strategy_stats[strategy] = result.strategy_stats.get(strategy, 0) + 1
                    
                except Exception as e:
                    # Classify and record crash
                    if input_hash not in self.known_crashes:
                        crash_info = self._create_crash_info(e, input_hash, strategy, target, input_data)
                        result.crashes_found.append(crash_info)
                        self.known_crashes.add(input_hash)
                        
                        logger.warning(f"New crash found: {crash_info.exception_type} - {crash_info.severity.value}")
                    
                    execution_count += 1
                    result.strategy_stats[strategy] = result.strategy_stats.get(strategy, 0) + 1
                
                # Periodic progress updates
                if execution_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = execution_count / max(elapsed / 60, 0.001)  # per minute
                    crashes = len(result.crashes_found)
                    logger.info(f"Progress: {execution_count} executions, {crashes} crashes, {rate:.1f} exec/min")
                
            except KeyboardInterrupt:
                logger.info("Fuzzing campaign interrupted by user")
                break
            except Exception as e:
                logger.error(f"Fuzzing engine error: {e}")
                # Continue fuzzing despite internal errors
                continue
        
        # Finalize results
        result.end_time = time.time()
        result.total_executions = execution_count
        
        logger.info(f"Fuzzing campaign {campaign_id} completed:")
        logger.info(f"  Duration: {result.duration_minutes:.1f} minutes")
        logger.info(f"  Executions: {result.total_executions}")
        logger.info(f"  Rate: {result.executions_per_minute:.1f} exec/min")
        logger.info(f"  Crashes found: {len(result.crashes_found)}")
        
        crash_counts = result.crashes_by_severity()
        for severity, count in crash_counts.items():
            if count > 0:
                logger.info(f"    {severity.value}: {count}")
        
        return result
    
    def _create_crash_info(self, 
                          exception: Exception, 
                          input_hash: str,
                          strategy: FuzzingStrategy,
                          target: FuzzTarget,
                          input_data: Any) -> CrashInfo:
        """Create crash information record."""
        
        crash_id = f"crash_{int(time.time())}_{random.randint(100, 999)}"
        severity = target.classify_crash(exception, input_data)
        
        return CrashInfo(
            crash_id=crash_id,
            severity=severity,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            stack_trace=traceback.format_exc(),
            input_hash=input_hash,
            strategy=strategy,
            timestamp=time.time(),
            reproduction_steps=[
                f"Target: {target.name}",
                f"Strategy: {strategy.value}", 
                f"Input: {str(input_data)[:200]}..."
            ]
        )
    
    def _random_generation_strategy(self, target_name: str) -> Dict[str, Any]:
        """Generate random inputs for targets."""
        
        if target_name == "packrepo_selector":
            return self._generate_random_selector_input()
        elif target_name == "packrepo_chunker":
            return self._generate_random_chunker_input()
        elif target_name == "packrepo_tokenizer":
            return self._generate_random_tokenizer_input()
        else:
            return {}
    
    def _boundary_conditions_strategy(self, target_name: str) -> Dict[str, Any]:
        """Generate boundary condition test cases."""
        
        if target_name == "packrepo_selector":
            return self._generate_boundary_selector_input()
        elif target_name == "packrepo_chunker":
            return self._generate_boundary_chunker_input()
        elif target_name == "packrepo_tokenizer":
            return self._generate_boundary_tokenizer_input()
        else:
            return {}
    
    def _file_content_strategy(self, target_name: str) -> Dict[str, Any]:
        """Generate file content-based test cases."""
        
        if target_name == "packrepo_chunker":
            return self._generate_file_content_chunker_input()
        elif target_name == "packrepo_tokenizer":
            return self._generate_file_content_tokenizer_input()
        else:
            return self._random_generation_strategy(target_name)
    
    def _generate_random_selector_input(self) -> Dict[str, Any]:
        """Generate random selector inputs."""
        num_chunks = random.randint(0, 100)
        chunks = []
        
        for i in range(num_chunks):
            chunks.append({
                'id': f'chunk_{i}',
                'cost': random.randint(1, 1000),
                'score': random.uniform(0, 1),
                'content': 'x' * random.randint(10, 500)
            })
        
        return {
            'chunks': chunks,
            'budget': random.randint(100, 10000),
            'config': {'strategy': random.choice(['greedy', 'submodular'])}
        }
    
    def _generate_random_chunker_input(self) -> Dict[str, Any]:
        """Generate random chunker inputs."""
        # Generate random code-like content
        content_patterns = [
            "def function():\n    pass\n",
            "class MyClass:\n    def __init__(self):\n        self.x = 1\n",
            "# Comment\nprint('hello')\n",
            "import os\nimport sys\n",
            "x = 1\ny = 2\nz = x + y\n"
        ]
        
        content = random.choice(content_patterns) * random.randint(1, 20)
        
        return {
            'content': content,
            'file_path': f"test_{random.randint(1, 100)}.py",
            'config': {'max_chunk_size': random.randint(100, 2000)}
        }
    
    def _generate_random_tokenizer_input(self) -> Dict[str, Any]:
        """Generate random tokenizer inputs."""
        # Generate various text types
        text_types = [
            lambda: 'hello world ' * random.randint(1, 100),
            lambda: ''.join(chr(random.randint(32, 126)) for _ in range(random.randint(1, 1000))),
            lambda: 'def func():\n    return "test"\n' * random.randint(1, 50),
            lambda: '中文测试' * random.randint(1, 20),  # Unicode
            lambda: '🚀🎯🔥' * random.randint(1, 10)      # Emojis
        ]
        
        text_gen = random.choice(text_types)
        
        return {
            'text': text_gen(),
            'tokenizer': random.choice(['cl100k', 'o200k', 'gpt2'])
        }
    
    def _generate_boundary_selector_input(self) -> Dict[str, Any]:
        """Generate boundary condition selector inputs."""
        boundary_cases = [
            {'chunks': [], 'budget': 0},  # Empty
            {'chunks': [], 'budget': 1000000},  # Empty with large budget
            {'chunks': [{'id': 'single', 'cost': 0, 'score': 1}], 'budget': 0},  # Zero cost/budget
            {'chunks': [{'id': 'expensive', 'cost': 1000000, 'score': 1}], 'budget': 1000},  # Impossible selection
        ]
        
        case = random.choice(boundary_cases)
        case['config'] = {}
        return case
    
    def _generate_boundary_chunker_input(self) -> Dict[str, Any]:
        """Generate boundary condition chunker inputs."""
        boundary_cases = [
            {'content': '', 'file_path': 'empty.py'},  # Empty file
            {'content': 'x', 'file_path': 'tiny.py'},  # Single character
            {'content': 'x' * 1000000, 'file_path': 'huge.py'},  # Very large file
            {'content': '\n' * 1000, 'file_path': 'newlines.py'},  # Many newlines
            {'content': '中文\n测试\n代码', 'file_path': 'unicode.py'},  # Unicode
            {'content': '\x00\x01\x02', 'file_path': 'binary.py'},  # Binary data
        ]
        
        case = random.choice(boundary_cases)
        case['config'] = {}
        return case
    
    def _generate_boundary_tokenizer_input(self) -> Dict[str, Any]:
        """Generate boundary condition tokenizer inputs."""
        boundary_cases = [
            {'text': ''},  # Empty
            {'text': 'x'},  # Single char
            {'text': 'x' * 1000000},  # Very long
            {'text': '\n' * 1000},  # Many newlines
            {'text': ' ' * 1000},  # Many spaces
            {'text': '🚀' * 1000},  # Unicode emojis
            {'text': '\x00\x01\x02'},  # Control chars
        ]
        
        case = random.choice(boundary_cases)
        case['tokenizer'] = random.choice(['cl100k', 'o200k'])
        return case
    
    def _generate_file_content_chunker_input(self) -> Dict[str, Any]:
        """Generate realistic file content for chunking."""
        
        file_templates = {
            'python': [
                "#!/usr/bin/env python3\n",
                "\"\"\"Module docstring.\"\"\"\n\n",
                "import os\nimport sys\nfrom typing import Dict, List\n\n",
                "class ExampleClass:\n    \"\"\"Example class.\"\"\"\n    \n    def __init__(self):\n        self.data = {}\n\n",
                "    def method(self, param: str) -> Dict[str, Any]:\n        \"\"\"Example method.\"\"\"\n        return {'result': param}\n\n",
                "def function(x: int, y: int) -> int:\n    \"\"\"Example function.\"\"\"\n    if x > y:\n        return x + y\n    else:\n        return x - y\n\n",
                "if __name__ == '__main__':\n    main()\n"
            ],
            'javascript': [
                "// Module header\n",
                "const express = require('express');\nconst path = require('path');\n\n",
                "class APIHandler {\n  constructor(config) {\n    this.config = config;\n  }\n\n",
                "  async handleRequest(req, res) {\n    try {\n      const result = await this.process(req.body);\n      res.json(result);\n    } catch (error) {\n      res.status(500).json({error: error.message});\n    }\n  }\n}\n\n",
                "function processData(data) {\n  return data.map(item => ({\n    ...item,\n    processed: true\n  }));\n}\n\n",
                "module.exports = { APIHandler, processData };\n"
            ]
        }
        
        language = random.choice(['python', 'javascript'])
        extension = 'py' if language == 'python' else 'js'
        
        # Randomly combine templates
        templates = file_templates[language]
        num_sections = random.randint(1, len(templates))
        selected_templates = random.sample(templates, num_sections)
        
        content = ''.join(selected_templates)
        
        return {
            'content': content,
            'file_path': f'generated_{random.randint(1, 1000)}.{extension}',
            'config': {'language': language}
        }
    
    def _generate_file_content_tokenizer_input(self) -> Dict[str, Any]:
        """Generate realistic content for tokenizer testing."""
        return self._generate_file_content_chunker_input()
    
    def save_results(self, result: FuzzingResult, output_path: str):
        """Save fuzzing results to file."""
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Fuzzing results saved to {output_path}")
    
    def load_known_crashes(self, crashes_file: str):
        """Load known crashes to avoid duplicate reporting."""
        try:
            with open(crashes_file, 'r') as f:
                crash_data = json.load(f)
                
            for crash_record in crash_data.get('crashes_found', []):
                self.known_crashes.add(crash_record['input_hash'])
                
            logger.info(f"Loaded {len(self.known_crashes)} known crashes from {crashes_file}")
        except FileNotFoundError:
            logger.info(f"No existing crashes file found at {crashes_file}")
        except Exception as e:
            logger.error(f"Error loading crashes file: {e}")


# Convenience function for CLI usage
def run_fuzzing_campaign(duration_minutes: float, 
                        output_dir: str = "fuzzing_results",
                        seed: Optional[int] = None) -> FuzzingResult:
    """Run a complete fuzzing campaign."""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create fuzzer
    fuzzer = FuzzerEngine(seed=seed)
    
    # Load known crashes if available
    known_crashes_file = os.path.join(output_dir, "known_crashes.json")
    fuzzer.load_known_crashes(known_crashes_file)
    
    # Run campaign
    result = fuzzer.run_campaign(duration_minutes)
    
    # Save results
    output_file = os.path.join(output_dir, f"{result.campaign_id}.json")
    fuzzer.save_results(result, output_file)
    
    # Update known crashes
    if result.crashes_found:
        fuzzer.save_results(result, known_crashes_file)
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fuzzer_engine.py <duration_minutes> [output_dir] [seed]")
        sys.exit(1)
    
    duration = float(sys.argv[1])
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "fuzzing_results"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    result = run_fuzzing_campaign(duration, output_dir, seed)
    
    print(f"\nFuzzing completed:")
    print(f"Executions: {result.total_executions}")
    print(f"Crashes: {len(result.crashes_found)}")
    print(f"Medium+ crashes: {result.has_medium_plus_crashes()}")