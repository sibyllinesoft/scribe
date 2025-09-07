#!/usr/bin/env python3
"""
Targeted tests for fastpath module to increase coverage.
Focus on covering the main uncovered lines in fastpath.py.
"""

import pytest
import pathlib
import tempfile
from unittest.mock import patch, MagicMock

from scribe.fastpath import should_use_intelligent_mode, select_files_fastpath, FASTPATH_AVAILABLE


class TestFastpathModule:
    """Test fastpath module functions for coverage."""
    
    def test_should_use_intelligent_mode_fastpath_unavailable(self):
        """Test should_use_intelligent_mode when FASTPATH_AVAILABLE is False."""
        with patch('scribe.fastpath.FASTPATH_AVAILABLE', False):
            with tempfile.TemporaryDirectory() as temp_dir:
                result = should_use_intelligent_mode(pathlib.Path(temp_dir))
                assert result is False
    
    def test_should_use_intelligent_mode_git_success(self):
        """Test should_use_intelligent_mode with successful git ls-files."""
        with patch('scribe.fastpath.FASTPATH_AVAILABLE', True):
            with patch('scribe.fastpath.run') as mock_run:
                # Mock git ls-files returning many files
                mock_run.return_value.stdout = "file1.py\nfile2.py\n" + "\n".join([f"file{i}.py" for i in range(3, 60)])
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    result = should_use_intelligent_mode(pathlib.Path(temp_dir))
                    assert result is True  # Should be True because >50 files
    
    def test_should_use_intelligent_mode_git_failure_fallback(self):
        """Test should_use_intelligent_mode with git failure, fallback to filesystem."""
        with patch('scribe.fastpath.FASTPATH_AVAILABLE', True):
            with patch('scribe.fastpath.run', side_effect=Exception("Git failed")):
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Create many files to trigger intelligent mode
                    for i in range(60):
                        (pathlib.Path(temp_dir) / f"file{i}.py").write_text("test")
                    
                    result = should_use_intelligent_mode(pathlib.Path(temp_dir))
                    assert result is True  # Should be True because >50 files in filesystem
    
    def test_should_use_intelligent_mode_small_repo(self):
        """Test should_use_intelligent_mode with small repository."""
        with patch('scribe.fastpath.FASTPATH_AVAILABLE', True):
            with patch('scribe.fastpath.run') as mock_run:
                # Mock git ls-files returning few files
                mock_run.return_value.stdout = "file1.py\nfile2.py\nfile3.py"
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    result = should_use_intelligent_mode(pathlib.Path(temp_dir))
                    assert result is False  # Should be False because <50 files


class TestSelectFilesFastpath:
    """Test the select_files_fastpath function for coverage."""
    
    def test_select_files_fastpath_unavailable(self):
        """Test select_files_fastpath when FASTPATH_AVAILABLE is False."""
        with patch('scribe.fastpath.FASTPATH_AVAILABLE', False):
            with tempfile.TemporaryDirectory() as temp_dir:
                with pytest.raises(RuntimeError, match="Scribe intelligent selection not available"):
                    select_files_fastpath(pathlib.Path(temp_dir), 10000)
    
    @patch('scribe.fastpath.FASTPATH_AVAILABLE', True)
    def test_select_files_fastpath_basic_execution(self):
        """Test basic execution path of select_files_fastpath."""
        # Mock all the imports and dependencies
        with patch('scribe.fastpath.FastScanner') as mock_scanner_class, \
             patch('scribe.fastpath.create_fastpath_engine') as mock_create_engine, \
             patch('scribe.fastpath.get_variant_flag_configuration') as mock_get_config, \
             patch('scribe.fastpath.estimate_tokens_scan_result') as mock_estimate_tokens:
            
            # Setup mocks
            mock_scanner = MagicMock()
            mock_scanner_class.return_value = mock_scanner
            mock_scanner.scan_repository.return_value = MagicMock()
            
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            mock_engine.select_files.return_value = ([], None)  # Empty selection
            
            mock_get_config.return_value = MagicMock()
            mock_estimate_tokens.return_value = 100
            
            # Mock FastPathVariant enum
            with patch('scribe.fastpath.FastPathVariant') as mock_variant:
                mock_variant.V5_INTEGRATED = "v5_integrated"
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    files, diff_content = select_files_fastpath(pathlib.Path(temp_dir), 10000)
                    
                    assert files == []
                    assert diff_content is None
                    mock_scanner.scan_repository.assert_called_once()
                    mock_create_engine.assert_called_once()
    
    @patch('scribe.fastpath.FASTPATH_AVAILABLE', True)
    def test_select_files_fastpath_with_entry_points(self):
        """Test select_files_fastpath with entry points."""
        with patch('scribe.fastpath.FastScanner') as mock_scanner_class, \
             patch('scribe.fastpath.create_fastpath_engine') as mock_create_engine, \
             patch('scribe.fastpath.get_variant_flag_configuration') as mock_get_config, \
             patch('scribe.fastpath.estimate_tokens_scan_result') as mock_estimate_tokens:
            
            # Setup mocks
            mock_scanner = MagicMock()
            mock_scanner_class.return_value = mock_scanner
            mock_scanner.scan_repository.return_value = MagicMock()
            
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            mock_engine.select_files.return_value = ([], None)
            
            mock_get_config.return_value = MagicMock()
            mock_estimate_tokens.return_value = 100
            
            # Mock EntryPointSpec
            with patch('packrepo.fastpath.types.EntryPointSpec') as mock_entry_spec, \
                 patch('scribe.fastpath.FastPathVariant') as mock_variant:
                mock_variant.V5_INTEGRATED = "v5_integrated"
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    files, diff_content = select_files_fastpath(
                        pathlib.Path(temp_dir), 
                        10000,
                        entry_points=["main.py"],
                        entry_functions=["main.py:main_function"]
                    )
                    
                    assert files == []
                    assert diff_content is None
                    # Verify entry points were processed
                    assert mock_entry_spec.call_count == 2  # One for entry_points, one for entry_functions
    
    @patch('scribe.fastpath.FASTPATH_AVAILABLE', True)
    def test_select_files_fastpath_with_diffs(self):
        """Test select_files_fastpath with diff inclusion."""
        with patch('scribe.fastpath.FastScanner') as mock_scanner_class, \
             patch('scribe.fastpath.create_fastpath_engine') as mock_create_engine, \
             patch('scribe.fastpath.get_variant_flag_configuration') as mock_get_config, \
             patch('scribe.fastpath.estimate_tokens_scan_result') as mock_estimate_tokens:
            
            # Setup mocks
            mock_scanner = MagicMock()
            mock_scanner_class.return_value = mock_scanner
            mock_scanner.scan_repository.return_value = MagicMock()
            
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            mock_engine.select_files.return_value = ([], None)
            
            mock_get_config.return_value = MagicMock()
            mock_estimate_tokens.return_value = 100
            
            # Mock DiffPackingOptions
            with patch('packrepo.fastpath.types.DiffPackingOptions') as mock_diff_options, \
                 patch('scribe.fastpath.FastPathVariant') as mock_variant:
                mock_variant.V5_INTEGRATED = "v5_integrated"
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Test with diff_branch
                    files, diff_content = select_files_fastpath(
                        pathlib.Path(temp_dir), 
                        10000,
                        include_diffs=True,
                        diff_branch="feature-branch"
                    )
                    
                    assert files == []
                    mock_diff_options.assert_called()
                    
                    # Test with diff_commits
                    files, diff_content = select_files_fastpath(
                        pathlib.Path(temp_dir), 
                        10000,
                        include_diffs=True,
                        diff_commits=3
                    )
                    
                    assert files == []
                    assert mock_diff_options.call_count >= 2
    
    @patch('scribe.fastpath.FASTPATH_AVAILABLE', True)
    def test_select_files_fastpath_variant_mapping(self):
        """Test that all variant strings are handled correctly."""
        variants = ['v1_baseline', 'v2_quotas', 'v3_centrality', 'v4_demotion', 'v5_integrated']
        
        for variant_str in variants:
            with patch('scribe.fastpath.FastScanner') as mock_scanner_class, \
                 patch('scribe.fastpath.create_fastpath_engine') as mock_create_engine, \
                 patch('scribe.fastpath.get_variant_flag_configuration') as mock_get_config, \
                 patch('scribe.fastpath.estimate_tokens_scan_result') as mock_estimate_tokens:
                
                # Setup mocks
                mock_scanner = MagicMock()
                mock_scanner_class.return_value = mock_scanner
                mock_scanner.scan_repository.return_value = MagicMock()
                
                mock_engine = MagicMock()
                mock_create_engine.return_value = mock_engine
                mock_engine.select_files.return_value = ([], None)
                
                mock_get_config.return_value = MagicMock()
                mock_estimate_tokens.return_value = 100
                
                # Mock FastPathVariant enum with the variant
                with patch('scribe.fastpath.FastPathVariant') as mock_variant:
                    setattr(mock_variant, variant_str.upper(), variant_str)
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        files, diff_content = select_files_fastpath(
                            pathlib.Path(temp_dir), 
                            10000,
                            variant=variant_str
                        )
                        
                        assert files == []
                        assert diff_content is None