"""
Configuration manager with repomix compatibility.

Handles both scribe.config.json and repomix.config.json with automatic fallback
and format conversion between repomix and scribe configuration schemas.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass


@dataclass
class ScribeConfig:
    """Scribe configuration with repomix compatibility."""
    
    # Core settings
    input_max_file_size: int = 50_000_000
    
    # Output settings
    output_style: str = "plain"  # "plain", "json", "markdown", "xml"
    output_file_path: Optional[str] = None
    output_parsable_style: bool = False
    output_header_text: Optional[str] = None
    output_show_line_numbers: bool = False
    output_file_summary: bool = True
    output_directory_structure: bool = True
    output_files: bool = True
    output_copy_to_clipboard: bool = False
    
    # Pattern settings (repomix-compatible)
    include: List[str] = None
    ignore_use_gitignore: bool = True
    ignore_use_default_patterns: bool = True
    ignore_custom_patterns: List[str] = None
    
    # Git integration
    git_sort_by_changes: bool = False
    git_sort_by_changes_max_commits: int = 100
    git_include_diffs: bool = False
    git_include_logs: bool = False
    git_include_logs_count: int = 50
    
    # Remote repository
    remote_url: Optional[str] = None
    remote_branch: Optional[str] = None
    
    # Token settings
    token_count_encoding: str = "o200k_base"
    
    # Security (repomix-compatible)
    security_enable_security_check: bool = True
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.include is None:
            self.include = ["**/*"]
        if self.ignore_custom_patterns is None:
            self.ignore_custom_patterns = []


class ConfigManager:
    """
    Configuration manager with repomix compatibility.
    
    Supports both scribe.config.json and repomix.config.json with automatic
    fallback and format conversion.
    """
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self._config: Optional[ScribeConfig] = None
        
    def load_config(self, custom_config_path: Optional[Path] = None) -> ScribeConfig:
        """
        Load configuration with fallback priority:
        1. Custom config path (if provided)
        2. scribe.config.json
        3. repomix.config.json  
        4. Default configuration
        """
        if custom_config_path and custom_config_path.exists():
            config_data = self._load_config_file(custom_config_path)
            self._config = self._convert_config(config_data, custom_config_path)
            return self._config
            
        # Try scribe.config.json first
        scribe_config_path = self.repo_path / "scribe.config.json"
        if scribe_config_path.exists():
            config_data = self._load_config_file(scribe_config_path)
            self._config = self._convert_config(config_data, scribe_config_path)
            return self._config
            
        # Fallback to repomix.config.json
        repomix_config_path = self.repo_path / "repomix.config.json"
        if repomix_config_path.exists():
            config_data = self._load_config_file(repomix_config_path)
            self._config = self._convert_repomix_config(config_data)
            return self._config
            
        # Default configuration
        self._config = ScribeConfig()
        return self._config
        
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            return {}
            
    def _convert_config(self, config_data: Dict[str, Any], config_path: Path) -> ScribeConfig:
        """Convert configuration data to ScribeConfig."""
        # Check if this is a repomix-style config based on structure
        if self._is_repomix_style_config(config_data):
            return self._convert_repomix_config(config_data)
        else:
            return self._convert_scribe_config(config_data)
            
    def _is_repomix_style_config(self, config_data: Dict[str, Any]) -> bool:
        """Detect if config uses repomix-style structure."""
        repomix_indicators = [
            "output.filePath",
            "output.style", 
            "ignore.customPatterns",
            "ignore.useGitignore",
            "tokenCount.encoding"
        ]
        
        # Check for nested repomix structure
        for indicator in repomix_indicators:
            if self._has_nested_key(config_data, indicator):
                return True
                
        return False
        
    def _has_nested_key(self, data: Dict[str, Any], key: str) -> bool:
        """Check if nested key exists in dictionary."""
        keys = key.split('.')
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return False
                
        return True
        
    def _convert_repomix_config(self, config_data: Dict[str, Any]) -> ScribeConfig:
        """Convert repomix configuration to ScribeConfig."""
        config = ScribeConfig()
        
        # Input settings
        if "input" in config_data:
            input_config = config_data["input"]
            config.input_max_file_size = input_config.get("maxFileSize", config.input_max_file_size)
            
        # Output settings
        if "output" in config_data:
            output_config = config_data["output"]
            config.output_style = output_config.get("style", config.output_style)
            config.output_file_path = output_config.get("filePath", config.output_file_path)
            config.output_parsable_style = output_config.get("parsableStyle", config.output_parsable_style)
            config.output_header_text = output_config.get("headerText", config.output_header_text)
            config.output_show_line_numbers = output_config.get("showLineNumbers", config.output_show_line_numbers)
            config.output_file_summary = output_config.get("fileSummary", config.output_file_summary)
            config.output_directory_structure = output_config.get("directoryStructure", config.output_directory_structure)
            config.output_files = output_config.get("files", config.output_files)
            config.output_copy_to_clipboard = output_config.get("copyToClipboard", config.output_copy_to_clipboard)
            
            # Git settings within output
            if "git" in output_config:
                git_config = output_config["git"]
                config.git_sort_by_changes = git_config.get("sortByChanges", config.git_sort_by_changes)
                config.git_sort_by_changes_max_commits = git_config.get("sortByChangesMaxCommits", config.git_sort_by_changes_max_commits)
                config.git_include_diffs = git_config.get("includeDiffs", config.git_include_diffs)
                config.git_include_logs = git_config.get("includeLogs", config.git_include_logs)
                config.git_include_logs_count = git_config.get("includeLogsCount", config.git_include_logs_count)
                
        # Include/ignore patterns
        config.include = config_data.get("include", config.include)
        
        if "ignore" in config_data:
            ignore_config = config_data["ignore"]
            config.ignore_use_gitignore = ignore_config.get("useGitignore", config.ignore_use_gitignore)
            config.ignore_use_default_patterns = ignore_config.get("useDefaultPatterns", config.ignore_use_default_patterns)
            config.ignore_custom_patterns = ignore_config.get("customPatterns", config.ignore_custom_patterns)
            
        # Remote repository
        if "remote" in config_data:
            remote_config = config_data["remote"]
            config.remote_url = remote_config.get("url", config.remote_url)
            config.remote_branch = remote_config.get("branch", config.remote_branch)
            
        # Token settings
        if "tokenCount" in config_data:
            token_config = config_data["tokenCount"]
            config.token_count_encoding = token_config.get("encoding", config.token_count_encoding)
            
        # Security settings
        if "security" in config_data:
            security_config = config_data["security"]
            config.security_enable_security_check = security_config.get("enableSecurityCheck", config.security_enable_security_check)
            
        return config
        
    def _convert_scribe_config(self, config_data: Dict[str, Any]) -> ScribeConfig:
        """Convert scribe-native configuration to ScribeConfig."""
        config = ScribeConfig()
        
        # Direct mapping for scribe-native format
        for field_name, default_value in config.__dataclass_fields__.items():
            if field_name in config_data:
                setattr(config, field_name, config_data[field_name])
                
        return config
        
    def get_config(self) -> ScribeConfig:
        """Get the loaded configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config
        
    def create_repomix_compatible_config(self, output_path: Path):
        """Create a repomix-compatible configuration file."""
        config = self.get_config()
        
        repomix_config = {
            "input": {
                "maxFileSize": config.input_max_file_size
            },
            "output": {
                "style": config.output_style,
                "filePath": config.output_file_path,
                "parsableStyle": config.output_parsable_style,
                "headerText": config.output_header_text,
                "showLineNumbers": config.output_show_line_numbers,
                "fileSummary": config.output_file_summary,
                "directoryStructure": config.output_directory_structure,
                "files": config.output_files,
                "copyToClipboard": config.output_copy_to_clipboard,
                "git": {
                    "sortByChanges": config.git_sort_by_changes,
                    "sortByChangesMaxCommits": config.git_sort_by_changes_max_commits,
                    "includeDiffs": config.git_include_diffs,
                    "includeLogs": config.git_include_logs,
                    "includeLogsCount": config.git_include_logs_count
                }
            },
            "include": config.include,
            "ignore": {
                "useGitignore": config.ignore_use_gitignore,
                "useDefaultPatterns": config.ignore_use_default_patterns,
                "customPatterns": config.ignore_custom_patterns
            },
            "remote": {
                "url": config.remote_url,
                "branch": config.remote_branch
            },
            "tokenCount": {
                "encoding": config.token_count_encoding
            },
            "security": {
                "enableSecurityCheck": config.security_enable_security_check
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(repomix_config, f, indent=2)
            
    def create_scribe_config(self, output_path: Path):
        """Create a scribe-native configuration file."""
        config = self.get_config()
        
        # Convert to dictionary
        config_dict = {}
        for field_name in config.__dataclass_fields__:
            value = getattr(config, field_name)
            if value is not None:  # Only include non-None values
                config_dict[field_name] = value
                
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)


def load_config(repo_path: Path, custom_config_path: Optional[Path] = None) -> ScribeConfig:
    """Convenience function to load configuration."""
    manager = ConfigManager(repo_path)
    return manager.load_config(custom_config_path)


# Example usage and testing
if __name__ == "__main__":
    # Test configuration loading
    repo_path = Path.cwd()
    config_manager = ConfigManager(repo_path)
    
    # Test repomix-style config
    repomix_config = {
        "output": {
            "style": "json",
            "filePath": "output.json",
            "showLineNumbers": True,
            "git": {
                "sortByChanges": True,
                "includeDiffs": True
            }
        },
        "include": ["**/*.py", "**/*.md"],
        "ignore": {
            "useGitignore": True,
            "customPatterns": ["**/tests/**", "**/*.tmp"]
        }
    }
    
    converted_config = config_manager._convert_repomix_config(repomix_config)
    print(f"Converted config: style={converted_config.output_style}, "
          f"git_sort={converted_config.git_sort_by_changes}")
    
    # Test scribe-native config
    scribe_config = {
        "output_style": "markdown",
        "include": ["**/*.py"],
        "git_sort_by_changes": True
    }
    
    converted_scribe = config_manager._convert_scribe_config(scribe_config)
    print(f"Scribe config: style={converted_scribe.output_style}")