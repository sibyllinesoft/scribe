"""
Prompt Management System for PackRepo Evaluation

Handles versioned prompt templates with SHA tracking and immutability enforcement.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PromptTemplate:
    """Versioned prompt template with immutability tracking."""
    name: str
    content: str
    version: str
    sha256: str
    created_at: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PromptRegistry:
    """Registry for managing prompt templates with versioning."""
    
    def __init__(self, prompts_dir: Path):
        self.prompts_dir = Path(prompts_dir)
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all prompt templates from directory."""
        for prompt_file in self.prompts_dir.glob("*.md"):
            template = self._load_template(prompt_file)
            self.templates[template.name] = template
    
    def _load_template(self, prompt_file: Path) -> PromptTemplate:
        """Load a single prompt template."""
        content = prompt_file.read_text(encoding='utf-8')
        sha256 = hashlib.sha256(content.encode()).hexdigest()
        
        # Use file modification time as version
        mtime = prompt_file.stat().st_mtime
        version = datetime.fromtimestamp(mtime).isoformat()
        
        return PromptTemplate(
            name=prompt_file.stem,
            content=content,
            version=version,
            sha256=sha256,
            created_at=datetime.now().isoformat(),
            metadata={
                "file_path": str(prompt_file),
                "file_size": len(content)
            }
        )
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name."""
        return self.templates.get(name)
    
    def format_template(self, name: str, **kwargs) -> str:
        """Format a prompt template with provided variables."""
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        
        try:
            return template.content.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")
    
    def get_template_sha(self, name: str) -> Optional[str]:
        """Get SHA256 hash for a template."""
        template = self.get_template(name)
        return template.sha256 if template else None
    
    def validate_immutability(self, recorded_shas: Dict[str, str]) -> Dict[str, bool]:
        """
        Validate that templates haven't changed since recording.
        
        Args:
            recorded_shas: Dictionary mapping template names to their recorded SHA hashes
            
        Returns:
            Dictionary mapping template names to validation status (True = unchanged)
        """
        results = {}
        for name, recorded_sha in recorded_shas.items():
            current_template = self.get_template(name)
            if current_template:
                results[name] = current_template.sha256 == recorded_sha
            else:
                results[name] = False
        return results
    
    def get_manifest(self) -> Dict[str, Any]:
        """Get manifest of all templates with their SHAs."""
        return {
            name: {
                "sha256": template.sha256,
                "version": template.version,
                "created_at": template.created_at,
                "metadata": template.metadata
            }
            for name, template in self.templates.items()
        }
    
    def save_manifest(self, output_file: Path):
        """Save template manifest to file."""
        manifest = self.get_manifest()
        with open(output_file, 'w') as f:
            json.dump(manifest, f, indent=2)


# Global registry instance
_prompt_registry = None


def get_prompt_registry(prompts_dir: Optional[Path] = None) -> PromptRegistry:
    """Get global prompt registry instance."""
    global _prompt_registry
    
    if _prompt_registry is None:
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent
        _prompt_registry = PromptRegistry(prompts_dir)
    
    return _prompt_registry


def get_prompt(name: str, **kwargs) -> str:
    """Convenience function to get and format a prompt."""
    registry = get_prompt_registry()
    return registry.format_template(name, **kwargs)


def get_prompt_sha(name: str) -> Optional[str]:
    """Get SHA256 hash for a prompt template."""
    registry = get_prompt_registry()
    return registry.get_template_sha(name)


__all__ = [
    "PromptTemplate",
    "PromptRegistry", 
    "get_prompt_registry",
    "get_prompt",
    "get_prompt_sha"
]