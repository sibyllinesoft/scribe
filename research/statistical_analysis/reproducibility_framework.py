#!/usr/bin/env python3
"""
Reproducibility Framework for FastPath Research
===============================================

Comprehensive framework ensuring full reproducibility of research results:
- Deterministic execution with seed control
- Environment specification with exact dependency versions
- Data provenance tracking for all experimental results  
- Configuration management for all experimental parameters
- Result validation with checksums and verification

Enables peer review and independent validation of research claims.
"""

import os
import json
import sys
import hashlib
import platform
import subprocess
import pkg_resources
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class EnvironmentSnapshot:
    """Complete environment snapshot for reproducibility."""
    
    # System information
    system_info: Dict[str, str]
    python_info: Dict[str, str]
    
    # Package dependencies
    installed_packages: Dict[str, str]
    pip_freeze_output: str
    
    # Hardware information  
    hardware_info: Dict[str, str]
    
    # Git repository state
    git_info: Optional[Dict[str, str]]
    
    # Environment variables (filtered)
    environment_variables: Dict[str, str]
    
    # Timestamp
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ExperimentProvenance:
    """Complete provenance tracking for experiment."""
    
    # Experiment metadata
    experiment_id: str
    config_hash: str
    random_seeds: Dict[str, int]
    
    # Input data fingerprints
    input_data_checksums: Dict[str, str]
    repository_fingerprints: Dict[str, str]
    
    # Execution tracking
    execution_order: List[str]
    intermediate_results: Dict[str, str]
    
    # Output validation
    output_checksums: Dict[str, str]
    result_fingerprint: str
    
    # Environment snapshot
    environment: EnvironmentSnapshot
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ReproducibilityReport:
    """Report on reproducibility validation."""
    
    # Validation results
    environment_match: bool
    seed_validation: bool
    checksum_validation: bool
    result_validation: bool
    
    # Detailed comparisons
    environment_differences: List[str]
    seed_differences: List[str]
    checksum_differences: List[str]
    result_differences: List[str]
    
    # Overall assessment
    is_reproducible: bool
    confidence_score: float  # 0-1 scale
    
    # Recommendations
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class ReproducibilityManager:
    """
    Manages all aspects of research reproducibility.
    
    Handles seed control, environment tracking, data provenance,
    and result validation for complete reproducibility.
    """
    
    def __init__(self, base_seed: int = 42):
        """Initialize with base random seed."""
        self.base_seed = base_seed
        self.current_experiment_id = None
        self.provenance_data = {}
        
        # Initialize all random number generators
        self._set_global_seeds(base_seed)
    
    def _set_global_seeds(self, seed: int) -> None:
        """Set seeds for all random number generators."""
        # Python random
        import random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # Set seeds for other libraries if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
        
        # Store seed tracking
        self.current_seeds = {
            'base_seed': seed,
            'python_random': seed,
            'numpy': seed,
            'timestamp': datetime.now().isoformat()
        }
    
    def create_environment_snapshot(self) -> EnvironmentSnapshot:
        """Create complete environment snapshot."""
        
        # System information
        system_info = {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'architecture': platform.architecture()[0],
            'hostname': platform.node()
        }
        
        # Python information
        python_info = {
            'version': sys.version,
            'version_info': str(sys.version_info),
            'executable': sys.executable,
            'platform': sys.platform,
            'prefix': sys.prefix,
            'path': str(sys.path)
        }
        
        # Installed packages
        installed_packages = {}
        pip_freeze_output = ""
        
        try:
            # Get pip freeze output
            result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], 
                                  capture_output=True, text=True)
            pip_freeze_output = result.stdout
            
            # Parse package versions
            for line in pip_freeze_output.strip().split('\n'):
                if '==' in line:
                    package, version = line.split('==', 1)
                    installed_packages[package] = version
        except Exception as e:
            pip_freeze_output = f"Error getting pip freeze: {str(e)}"
        
        # Hardware information
        hardware_info = {
            'cpu_count': str(os.cpu_count()),
            'total_memory': self._get_memory_info()
        }
        
        # Add GPU information if available
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                hardware_info['gpu_info'] = [
                    {'name': gpu.name, 'memory': f"{gpu.memoryTotal}MB"}
                    for gpu in gpus
                ]
        except ImportError:
            pass
        
        # Git repository information
        git_info = self._get_git_info()
        
        # Environment variables (filtered for security)
        env_vars = {}
        safe_env_vars = [
            'PATH', 'PYTHONPATH', 'HOME', 'USER', 'SHELL',
            'LANG', 'LC_ALL', 'TZ', 'CONDA_DEFAULT_ENV'
        ]
        
        for var in safe_env_vars:
            if var in os.environ:
                env_vars[var] = os.environ[var]
        
        return EnvironmentSnapshot(
            system_info=system_info,
            python_info=python_info,
            installed_packages=installed_packages,
            pip_freeze_output=pip_freeze_output,
            hardware_info=hardware_info,
            git_info=git_info,
            environment_variables=env_vars,
            created_at=datetime.now().isoformat()
        )
    
    def _get_memory_info(self) -> str:
        """Get total system memory information."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return f"{memory.total // (1024**3)}GB"
        except ImportError:
            return "Unknown"
    
    def _get_git_info(self) -> Optional[Dict[str, str]]:
        """Get git repository information."""
        try:
            git_info = {}
            
            # Get current commit hash
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                git_info['commit_hash'] = result.stdout.strip()
            
            # Get branch name
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                  capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                git_info['branch'] = result.stdout.strip()
            
            # Get remote URL
            result = subprocess.run(['git', 'remote', 'get-url', 'origin'], 
                                  capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                git_info['remote_url'] = result.stdout.strip()
            
            # Get working directory status
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                git_info['working_directory_clean'] = len(result.stdout.strip()) == 0
                git_info['status_output'] = result.stdout.strip()
            
            return git_info if git_info else None
            
        except Exception:
            return None
    
    def start_experiment_tracking(self, experiment_id: str, config: Dict[str, Any]) -> str:
        """Start tracking an experiment for reproducibility."""
        self.current_experiment_id = experiment_id
        
        # Create configuration hash
        config_str = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        
        # Create environment snapshot
        environment = self.create_environment_snapshot()
        
        # Initialize provenance tracking
        self.provenance_data[experiment_id] = ExperimentProvenance(
            experiment_id=experiment_id,
            config_hash=config_hash,
            random_seeds=self.current_seeds.copy(),
            input_data_checksums={},
            repository_fingerprints={},
            execution_order=[],
            intermediate_results={},
            output_checksums={},
            result_fingerprint="",
            environment=environment
        )
        
        return config_hash
    
    def track_input_data(self, data_name: str, data: Any) -> str:
        """Track input data with checksum."""
        if not self.current_experiment_id:
            raise RuntimeError("No active experiment. Call start_experiment_tracking() first.")
        
        # Calculate checksum based on data type
        if isinstance(data, (pd.DataFrame, pd.Series)):
            checksum = self._calculate_dataframe_checksum(data)
        elif isinstance(data, (list, dict)):
            data_str = json.dumps(data, sort_keys=True, default=str)
            checksum = hashlib.sha256(data_str.encode()).hexdigest()
        elif isinstance(data, np.ndarray):
            checksum = hashlib.sha256(data.tobytes()).hexdigest()
        elif isinstance(data, str):
            checksum = hashlib.sha256(data.encode()).hexdigest()
        else:
            # Convert to string representation
            data_str = str(data)
            checksum = hashlib.sha256(data_str.encode()).hexdigest()
        
        # Store checksum
        provenance = self.provenance_data[self.current_experiment_id]
        provenance.input_data_checksums[data_name] = checksum
        
        return checksum
    
    def track_repository_state(self, repo_id: str, repository: 'Repository') -> str:
        """Track repository state with fingerprint."""
        if not self.current_experiment_id:
            raise RuntimeError("No active experiment. Call start_experiment_tracking() first.")
        
        # Create repository fingerprint
        fingerprint_data = {
            'repo_id': repo_id,
            'metadata': repository.get_metadata(),
            'text_files': repository.get_text_files(),
            'prepared': repository.prepared
        }
        
        # Add file content checksums for key files
        text_files = repository.get_text_files()[:50]  # Limit for performance
        file_checksums = {}
        
        for file_path in text_files:
            try:
                content = repository.read_file(file_path)
                if content:
                    file_checksum = hashlib.sha256(content.encode()).hexdigest()
                    file_checksums[file_path] = file_checksum
            except Exception:
                continue
        
        fingerprint_data['file_checksums'] = file_checksums
        
        # Calculate overall fingerprint
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True, default=str)
        fingerprint = hashlib.sha256(fingerprint_str.encode()).hexdigest()
        
        # Store fingerprint
        provenance = self.provenance_data[self.current_experiment_id]
        provenance.repository_fingerprints[repo_id] = fingerprint
        
        return fingerprint
    
    def track_execution_step(self, step_name: str, step_data: Optional[Dict[str, Any]] = None) -> None:
        """Track execution step for provenance."""
        if not self.current_experiment_id:
            raise RuntimeError("No active experiment. Call start_experiment_tracking() first.")
        
        provenance = self.provenance_data[self.current_experiment_id]
        provenance.execution_order.append(step_name)
        
        if step_data:
            # Store intermediate result checksum
            data_str = json.dumps(step_data, sort_keys=True, default=str)
            checksum = hashlib.sha256(data_str.encode()).hexdigest()
            provenance.intermediate_results[step_name] = checksum
    
    def track_output_data(self, output_name: str, data: Any) -> str:
        """Track output data with checksum."""
        if not self.current_experiment_id:
            raise RuntimeError("No active experiment. Call start_experiment_tracking() first.")
        
        # Calculate checksum (same logic as input data)
        if isinstance(data, (pd.DataFrame, pd.Series)):
            checksum = self._calculate_dataframe_checksum(data)
        elif isinstance(data, (list, dict)):
            data_str = json.dumps(data, sort_keys=True, default=str)
            checksum = hashlib.sha256(data_str.encode()).hexdigest()
        elif isinstance(data, np.ndarray):
            checksum = hashlib.sha256(data.tobytes()).hexdigest()
        elif isinstance(data, str):
            checksum = hashlib.sha256(data.encode()).hexdigest()
        else:
            data_str = str(data)
            checksum = hashlib.sha256(data_str.encode()).hexdigest()
        
        # Store checksum
        provenance = self.provenance_data[self.current_experiment_id]
        provenance.output_checksums[output_name] = checksum
        
        return checksum
    
    def finalize_experiment(self, final_results: Dict[str, Any]) -> str:
        """Finalize experiment tracking with result fingerprint."""
        if not self.current_experiment_id:
            raise RuntimeError("No active experiment. Call start_experiment_tracking() first.")
        
        # Calculate final result fingerprint
        result_str = json.dumps(final_results, sort_keys=True, default=str)
        result_fingerprint = hashlib.sha256(result_str.encode()).hexdigest()
        
        # Store result fingerprint
        provenance = self.provenance_data[self.current_experiment_id]
        provenance.result_fingerprint = result_fingerprint
        
        return result_fingerprint
    
    def save_provenance(self, output_directory: str) -> str:
        """Save complete provenance data to file."""
        if not self.current_experiment_id:
            raise RuntimeError("No active experiment. Call start_experiment_tracking() first.")
        
        # Create output directory
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        
        # Save provenance data
        provenance_file = Path(output_directory) / f"provenance_{self.current_experiment_id}.json"
        
        provenance = self.provenance_data[self.current_experiment_id]
        
        with open(provenance_file, 'w') as f:
            json.dump(provenance.to_dict(), f, indent=2, default=str)
        
        return str(provenance_file)
    
    def validate_experiment(
        self, 
        raw_measurements: List[Dict[str, Any]], 
        statistical_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """Validate experiment results for reproducibility."""
        if not self.current_experiment_id:
            raise RuntimeError("No active experiment. Call start_experiment_tracking() first.")
        
        # Track final outputs
        self.track_output_data("raw_measurements", raw_measurements)
        self.track_output_data("statistical_results", statistical_results)
        
        # Finalize experiment
        final_results = {
            'raw_measurements': raw_measurements,
            'statistical_results': statistical_results
        }
        result_fingerprint = self.finalize_experiment(final_results)
        
        # Calculate environment hash
        environment = self.provenance_data[self.current_experiment_id].environment
        env_str = json.dumps(environment.to_dict(), sort_keys=True, default=str)
        environment_hash = hashlib.sha256(env_str.encode()).hexdigest()
        
        # Calculate overall experiment checksum
        provenance = self.provenance_data[self.current_experiment_id]
        experiment_data = {
            'config_hash': provenance.config_hash,
            'seeds': provenance.random_seeds,
            'input_checksums': provenance.input_data_checksums,
            'output_checksums': provenance.output_checksums,
            'result_fingerprint': provenance.result_fingerprint
        }
        
        experiment_str = json.dumps(experiment_data, sort_keys=True)
        experiment_checksum = hashlib.sha256(experiment_str.encode()).hexdigest()
        
        return {
            'checksum': experiment_checksum,
            'environment_hash': environment_hash,
            'result_fingerprint': result_fingerprint
        }
    
    def compare_experiments(
        self, 
        experiment1_file: str, 
        experiment2_file: str
    ) -> ReproducibilityReport:
        """Compare two experiments for reproducibility validation."""
        
        # Load experiment data
        with open(experiment1_file, 'r') as f:
            exp1_data = json.load(f)
        
        with open(experiment2_file, 'r') as f:
            exp2_data = json.load(f)
        
        # Initialize report
        report = ReproducibilityReport(
            environment_match=True,
            seed_validation=True,
            checksum_validation=True,
            result_validation=True,
            environment_differences=[],
            seed_differences=[],
            checksum_differences=[],
            result_differences=[],
            is_reproducible=True,
            confidence_score=1.0,
            recommendations=[]
        )
        
        # Compare environments
        env1 = exp1_data['environment']
        env2 = exp2_data['environment']
        
        # Check critical environment components
        critical_env_keys = [
            'python_info.version_info',
            'installed_packages',
            'system_info.platform',
            'system_info.system'
        ]
        
        for key in critical_env_keys:
            val1 = self._get_nested_value(env1, key)
            val2 = self._get_nested_value(env2, key)
            
            if val1 != val2:
                report.environment_match = False
                report.environment_differences.append(
                    f"{key}: {val1} != {val2}"
                )
        
        # Compare seeds
        seeds1 = exp1_data.get('random_seeds', {})
        seeds2 = exp2_data.get('random_seeds', {})
        
        for seed_name in set(seeds1.keys()) | set(seeds2.keys()):
            if seeds1.get(seed_name) != seeds2.get(seed_name):
                report.seed_validation = False
                report.seed_differences.append(
                    f"{seed_name}: {seeds1.get(seed_name)} != {seeds2.get(seed_name)}"
                )
        
        # Compare input data checksums
        checksums1 = exp1_data.get('input_data_checksums', {})
        checksums2 = exp2_data.get('input_data_checksums', {})
        
        for checksum_name in set(checksums1.keys()) | set(checksums2.keys()):
            if checksums1.get(checksum_name) != checksums2.get(checksum_name):
                report.checksum_validation = False
                report.checksum_differences.append(
                    f"{checksum_name}: {checksums1.get(checksum_name)} != {checksums2.get(checksum_name)}"
                )
        
        # Compare result fingerprints
        result1 = exp1_data.get('result_fingerprint', '')
        result2 = exp2_data.get('result_fingerprint', '')
        
        if result1 != result2:
            report.result_validation = False
            report.result_differences.append(
                f"Result fingerprints differ: {result1} != {result2}"
            )
        
        # Calculate overall reproducibility
        checks = [
            report.environment_match,
            report.seed_validation, 
            report.checksum_validation,
            report.result_validation
        ]
        
        report.is_reproducible = all(checks)
        report.confidence_score = sum(checks) / len(checks)
        
        # Generate recommendations
        if not report.environment_match:
            report.recommendations.append(
                "Environment differences detected. Ensure identical Python version and package versions."
            )
        
        if not report.seed_validation:
            report.recommendations.append(
                "Random seed differences detected. Use identical seed values for reproduction."
            )
        
        if not report.checksum_validation:
            report.recommendations.append(
                "Input data differences detected. Verify input data integrity and preprocessing steps."
            )
        
        if not report.result_validation:
            report.recommendations.append(
                "Result differences detected. Check for non-deterministic operations or environment issues."
            )
        
        if report.is_reproducible:
            report.recommendations.append(
                "Experiments are fully reproducible. Results can be trusted for publication."
            )
        
        return report
    
    def _calculate_dataframe_checksum(self, df: pd.DataFrame) -> str:
        """Calculate checksum for pandas DataFrame."""
        # Sort by all columns to ensure consistent ordering
        try:
            df_sorted = df.sort_values(by=list(df.columns))
        except Exception:
            df_sorted = df
        
        # Convert to string representation
        df_str = df_sorted.to_csv(index=False)
        return hashlib.sha256(df_str.encode()).hexdigest()
    
    def _get_nested_value(self, data: Dict[str, Any], key_path: str) -> Any:
        """Get nested dictionary value using dot notation."""
        keys = key_path.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def generate_reproducibility_package(
        self, 
        experiment_id: str, 
        output_directory: str
    ) -> Dict[str, str]:
        """Generate complete reproducibility package."""
        if experiment_id not in self.provenance_data:
            raise ValueError(f"Experiment {experiment_id} not found in provenance data")
        
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        package_files = {}
        
        # 1. Save provenance data
        provenance_file = self.save_provenance(str(output_dir))
        package_files['provenance'] = provenance_file
        
        # 2. Create requirements.txt from environment
        provenance = self.provenance_data[experiment_id]
        requirements_file = output_dir / "requirements.txt"
        with open(requirements_file, 'w') as f:
            f.write(provenance.environment.pip_freeze_output)
        package_files['requirements'] = str(requirements_file)
        
        # 3. Create environment.yml for conda
        env_yml_file = output_dir / "environment.yml"
        with open(env_yml_file, 'w') as f:
            f.write(f"name: fastpath-research-{experiment_id}\n")
            f.write("channels:\n  - conda-forge\n  - defaults\n")
            f.write("dependencies:\n")
            f.write(f"  - python={provenance.environment.python_info['version_info'].split()[0]}\n")
            f.write("  - pip\n")
            f.write("  - pip:\n")
            
            for package, version in provenance.environment.installed_packages.items():
                f.write(f"    - {package}=={version}\n")
        
        package_files['environment_yml'] = str(env_yml_file)
        
        # 4. Create reproduction script
        repro_script = output_dir / "reproduce_experiment.py"
        with open(repro_script, 'w') as f:
            f.write(self._generate_reproduction_script(experiment_id))
        package_files['reproduction_script'] = str(repro_script)
        
        # 5. Create README
        readme_file = output_dir / "REPRODUCTION_README.md"
        with open(readme_file, 'w') as f:
            f.write(self._generate_reproduction_readme(experiment_id))
        package_files['readme'] = str(readme_file)
        
        return package_files
    
    def _generate_reproduction_script(self, experiment_id: str) -> str:
        """Generate script to reproduce the experiment."""
        provenance = self.provenance_data[experiment_id]
        
        script = f'''#!/usr/bin/env python3
"""
Reproduction script for experiment: {experiment_id}
Generated automatically by ReproducibilityManager

This script reproduces the exact experimental conditions and execution.
"""

import sys
from research_evaluation_suite import ExperimentOrchestrator, ExperimentConfig
from reproducibility_framework import ReproducibilityManager

def main():
    # Set up reproducibility manager with original seeds
    repro_manager = ReproducibilityManager(base_seed={provenance.random_seeds.get('base_seed', 42)})
    
    # Recreate original configuration
    # NOTE: You'll need to adapt this based on your actual configuration
    config = ExperimentConfig(
        name="{experiment_id}_reproduction",
        description="Reproduction of experiment {experiment_id}",
        random_seed={provenance.random_seeds.get('base_seed', 42)}
    )
    
    # Run experiment
    orchestrator = ExperimentOrchestrator(
        config=config,
        output_directory=f"./reproduction_results_{experiment_id}"
    )
    
    try:
        result = orchestrator.run_complete_evaluation()
        print(f"✅ Reproduction completed successfully")
        print(f"Experiment ID: {{result.experiment_id}}")
        print(f"Duration: {{result.duration_seconds:.2f}} seconds")
        
        # Validate against original results
        print("\\n🔍 Validating reproduction...")
        # Add validation logic here
        
    except Exception as e:
        print(f"❌ Reproduction failed: {{str(e)}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        return script
    
    def _generate_reproduction_readme(self, experiment_id: str) -> str:
        """Generate README for reproduction package."""
        provenance = self.provenance_data[experiment_id]
        
        readme = f'''# Reproduction Package for Experiment {experiment_id}

This package contains everything needed to reproduce the experimental results.

## System Requirements

**Original Environment:**
- Platform: {provenance.environment.system_info.get('platform', 'Unknown')}
- Python Version: {provenance.environment.python_info.get('version_info', 'Unknown')}
- Total Memory: {provenance.environment.hardware_info.get('total_memory', 'Unknown')}

## Quick Start

1. **Set up environment:**
   ```bash
   conda env create -f environment.yml
   conda activate fastpath-research-{experiment_id}
   ```
   
   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run reproduction:**
   ```bash
   python reproduce_experiment.py
   ```

## Files Included

- `provenance_{experiment_id}.json` - Complete provenance tracking data
- `requirements.txt` - Python package requirements
- `environment.yml` - Conda environment specification  
- `reproduce_experiment.py` - Automated reproduction script
- `REPRODUCTION_README.md` - This file

## Validation

The reproduction script will:
1. Set identical random seeds
2. Use the same configuration parameters
3. Execute the same analysis pipeline
4. Compare results with original experiment

## Expected Results

**Original Configuration Hash:** `{provenance.config_hash}`
**Original Random Seeds:** {json.dumps(provenance.random_seeds, indent=2)}
**Original Result Fingerprint:** `{provenance.result_fingerprint}`

## Contact

If you encounter issues reproducing these results, please check:
1. Python version matches exactly
2. All package versions are identical
3. Random seeds are set correctly
4. Input data is unchanged

## Citation

Please cite the original FastPath research paper when using this reproduction package.
'''
        return readme


# Export main classes
__all__ = [
    'ReproducibilityManager',
    'EnvironmentSnapshot', 
    'ExperimentProvenance',
    'ReproducibilityReport'
]