"""Centralized entry point processing utilities.

This module provides standardized functionality for processing and converting
entry point specifications across the FastPath system, eliminating code
duplication in types.py, personalized_centrality.py, and diff_packer.py.
"""

from typing import List, Union, Optional
from ..base_types import EntryPointSpec

# Alias for backward compatibility
EntryPoint = EntryPointSpec


class EntryPointConverter:
    """Centralized utility for normalizing entry point inputs.
    
    Handles conversion of mixed string/EntryPointSpec lists to standardized
    EntryPointSpec objects, with consistent validation and error handling.
    """
    
    @staticmethod
    def normalize_entry_points(
        entry_points: List[Union[str, EntryPointSpec]]
    ) -> List[EntryPointSpec]:
        """Convert mixed entry point inputs to standardized EntryPointSpec objects.
        
        Args:
            entry_points: List containing strings (file paths) or EntryPointSpec objects
            
        Returns:
            List of EntryPointSpec objects with normalized attributes
            
        Raises:
            ValueError: If entry point inputs are invalid or empty
        """
        if not entry_points:
            return []
            
        processed_entry_points = []
        
        for ep in entry_points:
            if isinstance(ep, str):
                if not ep.strip():
                    continue  # Skip empty strings
                processed_entry_points.append(EntryPointSpec(file_path=ep.strip()))
            elif isinstance(ep, EntryPointSpec):
                processed_entry_points.append(ep)
            else:
                raise ValueError(f"Invalid entry point type: {type(ep)}. Expected str or EntryPointSpec")
                
        return processed_entry_points
    
    @staticmethod
    def validate_entry_points(entry_points: List[EntryPointSpec]) -> None:
        """Validate that entry point specifications are well-formed.
        
        Args:
            entry_points: List of EntryPointSpec objects to validate
            
        Raises:
            ValueError: If any entry point specification is invalid
        """
        for i, ep in enumerate(entry_points):
            if not ep.file_path:
                raise ValueError(f"Entry point {i} has empty file_path")
            
            if ep.weight <= 0:
                raise ValueError(f"Entry point {i} has invalid weight: {ep.weight}. Weight must be positive")
            
            # Function/class names should be valid Python identifiers if specified
            if ep.function_name and not ep.function_name.isidentifier():
                raise ValueError(f"Entry point {i} has invalid function_name: {ep.function_name}")
                
            if ep.class_name and not ep.class_name.isidentifier():
                raise ValueError(f"Entry point {i} has invalid class_name: {ep.class_name}")
    
    @staticmethod
    def create_from_strings(file_paths: List[str]) -> List[EntryPointSpec]:
        """Create EntryPointSpec objects from a list of file path strings.
        
        Args:
            file_paths: List of file path strings
            
        Returns:
            List of EntryPointSpec objects with default weights
        """
        return EntryPointConverter.normalize_entry_points(file_paths)
    
    @staticmethod
    def get_total_weight(entry_points: List[EntryPointSpec]) -> float:
        """Calculate the total weight of all entry points.
        
        Args:
            entry_points: List of EntryPointSpec objects
            
        Returns:
            Sum of all entry point weights
        """
        return sum(ep.weight for ep in entry_points)
    
    @staticmethod
    def filter_by_file_patterns(
        entry_points: List[EntryPointSpec], 
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[EntryPointSpec]:
        """Filter entry points by file path patterns.
        
        Args:
            entry_points: List of EntryPointSpec objects to filter
            include_patterns: Patterns that file paths must match (if specified)
            exclude_patterns: Patterns that file paths must not match
            
        Returns:
            Filtered list of EntryPointSpec objects
        """
        from .file_patterns import FilePatternMatcher
        
        filtered = []
        matcher = FilePatternMatcher()
        
        for ep in entry_points:
            # Check include patterns
            if include_patterns and not any(
                matcher.matches(ep.file_path, pattern) for pattern in include_patterns
            ):
                continue
                
            # Check exclude patterns  
            if exclude_patterns and any(
                matcher.matches(ep.file_path, pattern) for pattern in exclude_patterns
            ):
                continue
                
            filtered.append(ep)
            
        return filtered