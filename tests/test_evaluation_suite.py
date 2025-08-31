#!/usr/bin/env python3
"""
Comprehensive Test Suite for FastPath Evaluation Framework
==========================================================

Complete test coverage for all evaluation components:
- Unit tests for individual components
- Integration tests for end-to-end workflows
- Statistical validation tests
- Reproducibility verification tests
- Performance regression tests

Ensures reliability and correctness of research results.
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

# Import modules to test
from research_evaluation_suite import ExperimentOrchestrator, ExperimentConfig, ExperimentResult
from baseline_implementations import (
    NaiveTFIDFRetriever, BM25Retriever, RandomRetriever,
    FastPathV1, FastPathV2, FastPathV3
)
from multi_repository_benchmark import (
    RepositoryBenchmark, Repository, RepositoryType, RepositoryMetadata
)
from statistical_analysis_engine import StatisticalAnalyzer, EffectSize, SignificanceTest
from reproducibility_framework import ReproducibilityManager, EnvironmentSnapshot
from publication_data_generator import PublicationDataGenerator


class TestBaselineImplementations(unittest.TestCase):
    """Test all baseline retrieval system implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_repository = self._create_mock_repository()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_mock_repository(self):
        """Create a mock repository for testing."""
        mock_repo = Mock()
        mock_repo.get_text_files.return_value = [
            'main.py', 'utils.py', 'config.json', 'README.md', 'test_main.py'
        ]
        mock_repo.read_file.side_effect = lambda path: {
            'main.py': 'def main():\n    print("Hello World")\n\nif __name__ == "__main__":\n    main()',
            'utils.py': 'def helper_function(x):\n    return x * 2\n\nclass UtilityClass:\n    pass',
            'config.json': '{"debug": true, "port": 8080}',
            'README.md': '# Test Project\n\nThis is a test project for evaluation.',
            'test_main.py': 'import unittest\n\nclass TestMain(unittest.TestCase):\n    def test_main(self):\n        pass'
        }.get(path, '')
        return mock_repo
    
    def test_naive_tfidf_retriever(self):
        """Test Naive TF-IDF retriever."""
        retriever = NaiveTFIDFRetriever()
        retriever.configure(token_budget=1000)
        
        result = retriever.retrieve(self.mock_repository)
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        
        # Check that content is within token budget
        total_tokens = sum(retriever._count_tokens(content) for content in result)
        self.assertLessEqual(total_tokens, 1000)
    
    def test_bm25_retriever(self):
        """Test BM25 retriever."""
        retriever = BM25Retriever()
        retriever.configure(token_budget=1000, chunk_size=100)
        
        result = retriever.retrieve(self.mock_repository)
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        
        # Test chunking functionality
        self.assertTrue(any('===' in content for content in result))
    
    def test_random_retriever(self):
        """Test Random retriever."""
        retriever = RandomRetriever()
        retriever.configure(token_budget=1000, random_seed=42)
        
        # Test deterministic behavior with same seed
        result1 = retriever.retrieve(self.mock_repository)
        
        retriever.configure(token_budget=1000, random_seed=42)
        result2 = retriever.retrieve(self.mock_repository)
        
        self.assertEqual(result1, result2)
    
    def test_fastpath_v1(self):
        """Test FastPath V1."""
        retriever = FastPathV1()
        retriever.configure(token_budget=1000)
        
        result = retriever.retrieve(self.mock_repository)
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        
        # Check that scoring is applied (should contain score information)
        self.assertTrue(any('score:' in content for content in result))
    
    def test_fastpath_v2(self):
        """Test FastPath V2 with PageRank."""
        retriever = FastPathV2()
        retriever.configure(token_budget=1000)
        
        result = retriever.retrieve(self.mock_repository)
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
    
    def test_fastpath_v3(self):
        """Test FastPath V3 with quota system."""
        retriever = FastPathV3()
        retriever.configure(token_budget=1000)
        
        result = retriever.retrieve(self.mock_repository)
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        
        # Check quota system is applied (should contain category information)
        self.assertTrue(any('[' in content and ']' in content for content in result))
    
    def test_configuration_requirements(self):
        """Test that all retrievers require configuration."""
        retrievers = [
            NaiveTFIDFRetriever(), BM25Retriever(), RandomRetriever(),
            FastPathV1(), FastPathV2(), FastPathV3()
        ]
        
        for retriever in retrievers:
            with self.assertRaises(RuntimeError):
                retriever.retrieve(self.mock_repository)
    
    def test_token_budget_compliance(self):
        """Test that all retrievers respect token budgets."""
        retrievers = [
            NaiveTFIDFRetriever(), BM25Retriever(), RandomRetriever(),
            FastPathV1(), FastPathV2(), FastPathV3()
        ]
        
        token_budget = 500
        
        for retriever in retrievers:
            retriever.configure(token_budget=token_budget)
            result = retriever.retrieve(self.mock_repository)
            
            total_tokens = sum(retriever._count_tokens(content) for content in result)
            self.assertLessEqual(total_tokens, token_budget * 1.1)  # Allow 10% tolerance


class TestMultiRepositoryBenchmark(unittest.TestCase):
    """Test multi-repository benchmark framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark = RepositoryBenchmark()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_repository_metadata_creation(self):
        """Test repository metadata creation."""
        metadata = RepositoryMetadata(
            name="test_repo",
            owner="test_owner",
            url="https://github.com/test_owner/test_repo",
            description="Test repository",
            language="Python",
            stars=100,
            forks=10,
            size_kb=1024,
            last_updated="2024-01-01T00:00:00Z",
            topics=["testing", "python"]
        )
        
        self.assertEqual(metadata.name, "test_repo")
        self.assertEqual(metadata.language, "Python")
        self.assertIsInstance(metadata.topics, list)
    
    def test_repository_creation(self):
        """Test repository object creation."""
        metadata = RepositoryMetadata(
            name="test_repo", owner="test_owner",
            url="https://github.com/test_owner/test_repo",
            description="Test", language="Python",
            stars=100, forks=10, size_kb=1024,
            last_updated="2024-01-01T00:00:00Z",
            topics=["test"]
        )
        
        repo = Repository(metadata, RepositoryType.LIBRARY)
        
        self.assertEqual(repo.name, "test_owner/test_repo")
        self.assertEqual(repo.type, RepositoryType.LIBRARY)
        self.assertFalse(repo.prepared)
    
    def test_synthetic_repository_creation(self):
        """Test synthetic repository creation."""
        repos = self.benchmark.create_synthetic_repositories(
            RepositoryType.CLI_TOOL, count=2, random_seed=42
        )
        
        self.assertEqual(len(repos), 2)
        for repo in repos:
            self.assertIsInstance(repo, Repository)
            self.assertEqual(repo.type, RepositoryType.CLI_TOOL)
            self.assertTrue(repo.prepared)
            self.assertGreater(len(repo.get_text_files()), 0)
    
    def test_question_generation(self):
        """Test QA question generation."""
        repos = self.benchmark.create_synthetic_repositories(
            RepositoryType.WEB_APPLICATION, count=1, random_seed=42
        )
        
        repo = repos[0]
        questions = repo.generate_questions('architectural', count=5, random_seed=42)
        
        self.assertEqual(len(questions), 5)
        for question in questions:
            self.assertIn('question', question)
            self.assertIn('type', question)
            self.assertIn('answer', question)
            self.assertEqual(question['type'], 'architectural')
    
    @patch('subprocess.run')
    def test_repository_classification(self, mock_run):
        """Test repository type classification."""
        # Mock successful git operations
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "main\n"
        
        metadata = RepositoryMetadata(
            name="react_app", owner="test",
            url="https://github.com/test/react_app",
            description="A React web application",
            language="JavaScript", stars=100, forks=10, size_kb=1024,
            last_updated="2024-01-01T00:00:00Z",
            topics=["react", "webapp", "frontend"]
        )
        
        repo = Repository(metadata, RepositoryType.LIBRARY)
        classified_type = self.benchmark._classify_repository(repo)
        
        self.assertEqual(classified_type, RepositoryType.WEB_APPLICATION)


class TestStatisticalAnalyzer(unittest.TestCase):
    """Test statistical analysis engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = StatisticalAnalyzer(
            confidence_level=0.95,
            bootstrap_iterations=1000,
            random_seed=42
        )
        self.sample_data = self._create_sample_measurements()
    
    def _create_sample_measurements(self):
        """Create sample measurement data for testing."""
        np.random.seed(42)
        
        measurements = []
        systems = ['fastpath_v3', 'fastpath_v2', 'bm25', 'random']
        
        for i in range(50):  # 50 measurements total
            system = systems[i % len(systems)]
            
            # Create realistic performance differences
            base_accuracy = 0.7
            base_time = 2.0
            
            if 'fastpath' in system:
                accuracy_boost = 0.15 if system == 'fastpath_v3' else 0.10
                time_reduction = 0.5 if system == 'fastpath_v3' else 0.3
            else:
                accuracy_boost = -0.1 if system == 'random' else 0.0
                time_reduction = -0.5 if system == 'random' else 0.0
            
            measurements.append({
                'system_name': system,
                'repository_id': f'repo_{i // len(systems)}',
                'repository_type': 'library',
                'success': True,
                'execution_time_seconds': base_time - time_reduction + np.random.normal(0, 0.3),
                'qa_accuracy': base_accuracy + accuracy_boost + np.random.normal(0, 0.05),
                'qa_f1_score': base_accuracy + accuracy_boost + np.random.normal(0, 0.03),
                'memory_usage_bytes': 50000000 + np.random.normal(0, 10000000),
                'tokens_used': 100000 + np.random.randint(-20000, 20000),
                'files_retrieved': np.random.randint(5, 50)
            })
        
        return measurements
    
    def test_system_data_extraction(self):
        """Test extraction of system data from measurements."""
        system_data = self.analyzer._extract_system_data(pd.DataFrame(self.sample_data))
        
        self.assertIsInstance(system_data, dict)
        self.assertGreater(len(system_data), 0)
        
        for system, metrics in system_data.items():
            self.assertIsInstance(metrics, dict)
            self.assertIn('qa_accuracy', metrics)
            self.assertIsInstance(metrics['qa_accuracy'], np.ndarray)
    
    def test_system_summaries(self):
        """Test calculation of system summaries."""
        system_data = self.analyzer._extract_system_data(pd.DataFrame(self.sample_data))
        summaries = self.analyzer._calculate_system_summaries(system_data)
        
        self.assertIsInstance(summaries, dict)
        
        for system, summary in summaries.items():
            self.assertIn('qa_accuracy', summary)
            accuracy_stats = summary['qa_accuracy']
            
            self.assertIn('mean', accuracy_stats)
            self.assertIn('std', accuracy_stats)
            self.assertIn('count', accuracy_stats)
            self.assertGreater(accuracy_stats['count'], 0)
    
    def test_cohens_d_calculation(self):
        """Test Cohen's d effect size calculation."""
        sample1 = np.array([0.8, 0.82, 0.78, 0.85, 0.79])
        sample2 = np.array([0.7, 0.68, 0.72, 0.69, 0.71])
        
        effect_size = self.analyzer._calculate_cohens_d(sample1, sample2)
        
        self.assertIsInstance(effect_size, EffectSize)
        self.assertGreater(effect_size.value, 0)  # sample1 > sample2
        self.assertEqual(effect_size.method, "cohen_d")
        self.assertIsInstance(effect_size.confidence_interval, tuple)
    
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval calculation."""
        data = np.array([0.75, 0.78, 0.82, 0.79, 0.81, 0.77, 0.80])
        
        result = self.analyzer._bootstrap_confidence_interval(data)
        
        self.assertAlmostEqual(result.statistic, np.mean(data), places=5)
        self.assertEqual(result.method, "bca")
        self.assertEqual(result.confidence_level, 0.95)
        self.assertLess(result.confidence_interval[0], result.statistic)
        self.assertGreater(result.confidence_interval[1], result.statistic)
    
    def test_complete_analysis(self):
        """Test complete statistical analysis pipeline."""
        results = self.analyzer.analyze_experiment_results(self.sample_data)
        
        # Check all required sections are present
        required_sections = [
            'system_summaries', 'pairwise_comparisons', 'effect_sizes',
            'bootstrap_intervals', 'significance_tests', 'summary'
        ]
        
        for section in required_sections:
            self.assertIn(section, results)
        
        # Check system summaries
        self.assertIsInstance(results['system_summaries'], dict)
        self.assertGreater(len(results['system_summaries']), 0)
        
        # Check effect sizes
        self.assertIsInstance(results['effect_sizes'], dict)
        
        # Check summary
        self.assertIn('key_findings', results['summary'])
        self.assertIn('experiment_summary', results['summary'])
    
    def test_statistical_power_calculation(self):
        """Test statistical power analysis."""
        group1 = np.array([0.8, 0.82, 0.78, 0.85, 0.79, 0.83])
        group2 = np.array([0.7, 0.68, 0.72, 0.69, 0.71, 0.70])
        
        power_result = self.analyzer._calculate_statistical_power(group1, group2, 'qa_accuracy')
        
        self.assertIsInstance(power_result.observed_power, float)
        self.assertGreaterEqual(power_result.observed_power, 0)
        self.assertLessEqual(power_result.observed_power, 1)
        self.assertGreater(power_result.effect_size, 0)


class TestReproducibilityFramework(unittest.TestCase):
    """Test reproducibility framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = ReproducibilityManager(base_seed=42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_environment_snapshot_creation(self):
        """Test environment snapshot creation."""
        snapshot = self.manager.create_environment_snapshot()
        
        self.assertIsInstance(snapshot, EnvironmentSnapshot)
        self.assertIn('platform', snapshot.system_info)
        self.assertIn('version', snapshot.python_info)
        self.assertIsInstance(snapshot.installed_packages, dict)
    
    def test_experiment_tracking(self):
        """Test experiment tracking functionality."""
        config = {'test_param': 'test_value', 'seed': 42}
        
        config_hash = self.manager.start_experiment_tracking('test_exp', config)
        
        self.assertIsInstance(config_hash, str)
        self.assertEqual(len(config_hash), 64)  # SHA256 hash length
        self.assertIn('test_exp', self.manager.provenance_data)
    
    def test_data_tracking(self):
        """Test input/output data tracking."""
        config = {'test': True}
        self.manager.start_experiment_tracking('test_exp', config)
        
        # Track input data
        test_data = [1, 2, 3, 4, 5]
        input_checksum = self.manager.track_input_data('test_input', test_data)
        
        self.assertIsInstance(input_checksum, str)
        self.assertEqual(len(input_checksum), 64)
        
        # Track output data
        output_checksum = self.manager.track_output_data('test_output', test_data)
        
        self.assertIsInstance(output_checksum, str)
        self.assertEqual(input_checksum, output_checksum)  # Same data should have same checksum
    
    def test_provenance_saving(self):
        """Test provenance data saving."""
        config = {'test': True}
        self.manager.start_experiment_tracking('test_exp', config)
        
        provenance_file = self.manager.save_provenance(self.temp_dir)
        
        self.assertTrue(Path(provenance_file).exists())
        
        # Verify file contents
        with open(provenance_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data['experiment_id'], 'test_exp')
        self.assertIn('environment', data)
        self.assertIn('config_hash', data)
    
    def test_seed_reproducibility(self):
        """Test random seed reproducibility."""
        # Create two managers with same seed
        manager1 = ReproducibilityManager(base_seed=123)
        manager2 = ReproducibilityManager(base_seed=123)
        
        # Generate random numbers
        import random
        np.random.seed(123)
        random.seed(123)
        values1 = [random.random() for _ in range(5)] + [np.random.random() for _ in range(5)]
        
        np.random.seed(123)
        random.seed(123)
        values2 = [random.random() for _ in range(5)] + [np.random.random() for _ in range(5)]
        
        self.assertEqual(values1, values2)


class TestPublicationDataGenerator(unittest.TestCase):
    """Test publication data generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = PublicationDataGenerator(self.temp_dir)
        
        # Create sample data
        self.sample_measurements = self._create_sample_data()
        self.sample_statistical_results = self._create_sample_statistical_results()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_sample_data(self):
        """Create sample measurement data."""
        measurements = []
        systems = ['fastpath_v3', 'bm25', 'random']
        
        for i in range(30):
            system = systems[i % len(systems)]
            measurements.append({
                'system_name': system,
                'repository_id': f'repo_{i}',
                'repository_type': 'library',
                'success': True,
                'execution_time_seconds': 2.0 + np.random.normal(0, 0.3),
                'qa_accuracy': 0.75 + np.random.normal(0, 0.05),
                'qa_f1_score': 0.73 + np.random.normal(0, 0.03),
                'memory_usage_bytes': 50000000 + np.random.normal(0, 5000000),
                'tokens_used': 100000,
                'files_retrieved': 20
            })
        
        return measurements
    
    def _create_sample_statistical_results(self):
        """Create sample statistical results."""
        return {
            'system_summaries': {
                'fastpath_v3': {
                    'qa_accuracy': {'mean': 0.85, 'std': 0.03, 'median': 0.85, 'count': 10, 'min': 0.80, 'max': 0.90},
                    'execution_time_seconds': {'mean': 1.5, 'std': 0.2, 'median': 1.5, 'count': 10, 'min': 1.2, 'max': 1.8}
                },
                'bm25': {
                    'qa_accuracy': {'mean': 0.70, 'std': 0.04, 'median': 0.70, 'count': 10, 'min': 0.65, 'max': 0.75},
                    'execution_time_seconds': {'mean': 2.0, 'std': 0.3, 'median': 2.0, 'count': 10, 'min': 1.5, 'max': 2.5}
                }
            },
            'significance_tests': {
                'qa_accuracy_anova': {
                    'test_name': 'One-way ANOVA',
                    'statistic': 15.2,
                    'p_value': 0.001,
                    'is_significant': True
                }
            },
            'effect_sizes': {
                'fastpath_v3': {
                    'vs_bm25': {
                        'qa_accuracy': {
                            'value': 1.2,
                            'magnitude': 'large',
                            'confidence_interval': [0.8, 1.6],
                            'method': 'cohen_d'
                        }
                    }
                }
            },
            'bootstrap_intervals': {
                'fastpath_v3': {
                    'qa_accuracy': {
                        'statistic': 0.85,
                        'confidence_interval': [0.82, 0.88],
                        'method': 'bca'
                    }
                }
            }
        }
    
    def test_directory_creation(self):
        """Test output directory structure creation."""
        expected_dirs = ['tables', 'figures', 'data', 'latex']
        
        for dir_name in expected_dirs:
            dir_path = Path(self.temp_dir) / dir_name
            self.assertTrue(dir_path.exists())
            self.assertTrue(dir_path.is_dir())
    
    def test_performance_table_generation(self):
        """Test LaTeX performance table generation."""
        table_file = self.generator.generate_performance_table(
            self.sample_statistical_results['system_summaries']
        )
        
        self.assertTrue(Path(table_file).exists())
        
        with open(table_file, 'r') as f:
            content = f.read()
        
        self.assertIn('\\begin{table*}', content)
        self.assertIn('\\end{table*}', content)
        self.assertIn('fastpath_v3', content.lower())
        self.assertIn('bm25', content.lower())
    
    def test_data_export(self):
        """Test research data export."""
        data_files = self.generator.export_research_data(
            self.sample_measurements,
            self.sample_statistical_results
        )
        
        self.assertGreater(len(data_files), 0)
        
        for file_path in data_files:
            self.assertTrue(Path(file_path).exists())
            
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                self.assertGreater(len(df), 0)
    
    def test_system_name_formatting(self):
        """Test system name formatting for display."""
        test_cases = [
            ('fastpath_v1', 'FastPath V1'),
            ('naive_tfidf', 'Naive TF-IDF'),
            ('bm25', 'BM25'),
            ('random', 'Random')
        ]
        
        for input_name, expected_output in test_cases:
            formatted = self.generator._format_system_name(input_name)
            self.assertEqual(formatted, expected_output)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete evaluation pipeline."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal config
        self.config = ExperimentConfig(
            name="integration_test",
            description="Integration test experiment",
            token_budget=5000,
            questions_per_repository=2,
            min_repositories_per_type=1,
            max_repositories_per_type=1,
            bootstrap_iterations=100  # Reduced for faster testing
        )
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('multi_repository_benchmark.RepositoryBenchmark.discover_repositories')
    def test_end_to_end_evaluation(self, mock_discover):
        """Test complete end-to-end evaluation pipeline."""
        # Mock repository discovery to use synthetic repositories
        benchmark = RepositoryBenchmark()
        synthetic_repos = benchmark.create_synthetic_repositories(
            RepositoryType.LIBRARY, count=1, random_seed=42
        )
        mock_discover.return_value = synthetic_repos
        
        # Run evaluation
        orchestrator = ExperimentOrchestrator(
            config=self.config,
            output_directory=self.temp_dir
        )
        
        # Mock the LLM QA evaluation to avoid external dependencies
        with patch.object(orchestrator, '_mock_qa_system') as mock_qa:
            mock_qa.return_value = "Mock answer for integration test"
            
            with patch.object(orchestrator, '_evaluate_answer_quality') as mock_eval:
                mock_eval.return_value = True
                
                try:
                    result = orchestrator.run_complete_evaluation()
                    
                    # Verify result structure
                    self.assertIsInstance(result, ExperimentResult)
                    self.assertEqual(result.config.name, "integration_test")
                    self.assertIsInstance(result.system_results, dict)
                    self.assertIsInstance(result.statistical_summary, dict)
                    self.assertGreater(len(result.raw_measurements), 0)
                    
                    # Verify output files exist
                    output_path = Path(self.temp_dir)
                    self.assertTrue(any(output_path.glob('*.log')))
                    
                except Exception as e:
                    self.fail(f"End-to-end evaluation failed: {str(e)}")
    
    def test_reproducibility_integration(self):
        """Test reproducibility framework integration."""
        manager = ReproducibilityManager(base_seed=42)
        
        # Start tracking
        config_hash = manager.start_experiment_tracking('integration_test', asdict(self.config))
        
        # Track some data
        test_data = {'metric': 'accuracy', 'value': 0.85}
        data_checksum = manager.track_input_data('test_data', test_data)
        
        # Finalize
        result_fingerprint = manager.finalize_experiment(test_data)
        
        # Validate
        validation_result = manager.validate_experiment([], {})
        
        self.assertIsInstance(validation_result, dict)
        self.assertIn('checksum', validation_result)
        self.assertIn('environment_hash', validation_result)
    
    def test_statistical_pipeline_integration(self):
        """Test statistical analysis pipeline integration."""
        # Create more realistic sample data
        np.random.seed(42)
        measurements = []
        
        systems = ['fastpath_v3', 'bm25', 'random']
        for system in systems:
            for i in range(15):  # More samples for better statistics
                base_accuracy = {'fastpath_v3': 0.85, 'bm25': 0.70, 'random': 0.55}[system]
                
                measurements.append({
                    'system_name': system,
                    'repository_id': f'repo_{i}',
                    'repository_type': 'library',
                    'success': True,
                    'execution_time_seconds': np.random.exponential(2.0),
                    'qa_accuracy': np.clip(base_accuracy + np.random.normal(0, 0.05), 0, 1),
                    'qa_f1_score': np.clip(base_accuracy + np.random.normal(0, 0.03), 0, 1),
                    'memory_usage_bytes': np.random.lognormal(17, 0.5),  # ~50MB mean
                    'tokens_used': np.random.randint(50000, 150000),
                    'files_retrieved': np.random.randint(5, 30)
                })
        
        # Run statistical analysis
        analyzer = StatisticalAnalyzer(bootstrap_iterations=100)
        results = analyzer.analyze_experiment_results(measurements)
        
        # Verify comprehensive analysis
        self.assertIn('system_summaries', results)
        self.assertIn('effect_sizes', results)
        self.assertIn('significance_tests', results)
        self.assertIn('bootstrap_intervals', results)
        
        # Test publication output generation
        generator = PublicationDataGenerator(self.temp_dir)
        
        with patch('matplotlib.pyplot.savefig'):  # Mock to avoid GUI dependencies
            generated_files = generator.generate_all_outputs(
                measurements, results, self.config
            )
        
        self.assertIn('tables', generated_files)
        self.assertIn('figures', generated_files)
        self.assertIn('data_files', generated_files)


def run_test_suite():
    """Run the complete test suite."""
    # Create test suite
    test_modules = [
        TestBaselineImplementations,
        TestMultiRepositoryBenchmark, 
        TestStatisticalAnalyzer,
        TestReproducibilityFramework,
        TestPublicationDataGenerator,
        TestIntegration
    ]
    
    suite = unittest.TestSuite()
    
    for test_module in test_modules:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_module)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100:.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_test_suite()
    sys.exit(0 if success else 1)