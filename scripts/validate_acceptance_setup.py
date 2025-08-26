#!/usr/bin/env python3
"""
PackRepo Acceptance Setup Validator

Validates that all components of the acceptance gate system are properly configured:
- Gate configuration files exist and are valid
- Required scripts are executable
- Directory structure is correct
- Dependencies are available
- Test data and artifacts are present

This ensures the acceptance pipeline can run successfully.
"""

import json
import yaml
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
import importlib.util
from datetime import datetime


class AcceptanceSetupValidator:
    """Validates the complete acceptance gate setup."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        
    def validate_all(self) -> bool:
        """Run all validation checks."""
        print("🔍 PackRepo Acceptance Gate Setup Validation")
        print(f"Project Root: {self.project_root}")
        print(f"Validation Time: {datetime.utcnow().isoformat()}")
        print("=" * 60)
        
        checks = [
            ("Directory Structure", self._validate_directory_structure),
            ("Configuration Files", self._validate_configuration_files),
            ("Python Scripts", self._validate_python_scripts), 
            ("Dependencies", self._validate_dependencies),
            ("Permissions", self._validate_permissions),
            ("Test Data", self._validate_test_data),
            ("Integration", self._validate_integration)
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            print(f"\n📋 {check_name}...")
            try:
                result = check_func()
                self.validation_results[check_name] = result
                
                if result["passed"]:
                    print(f"✅ {check_name}: PASSED")
                    if result.get("details"):
                        for detail in result["details"]:
                            print(f"   • {detail}")
                else:
                    print(f"❌ {check_name}: FAILED")
                    all_passed = False
                    for error in result.get("errors", []):
                        print(f"   🚨 {error}")
                        
                for warning in result.get("warnings", []):
                    print(f"   ⚠️  {warning}")
                    
            except Exception as e:
                print(f"❌ {check_name}: EXCEPTION - {e}")
                all_passed = False
                self.errors.append(f"{check_name}: {e}")
        
        # Summary
        print("\n" + "=" * 60)
        if all_passed:
            print("✅ ALL VALIDATION CHECKS PASSED")
            print("🚀 Acceptance gate system is ready to run!")
        else:
            print("❌ VALIDATION FAILED") 
            print(f"Errors: {len(self.errors)}")
            print(f"Warnings: {len(self.warnings)}")
            print("\nFix the issues above before running the acceptance pipeline.")
        
        # Save validation report
        self._save_validation_report(all_passed)
        
        return all_passed
    
    def _validate_directory_structure(self) -> Dict[str, Any]:
        """Validate required directory structure exists."""
        required_dirs = [
            "scripts",
            "artifacts",
            "artifacts/metrics", 
            "logs"
        ]
        
        optional_dirs = [
            "artifacts/reports",
            "locks",
            "tests/properties",
            "tests/metamorphic",
            "tests/mutation",
            "tests/fuzzing"
        ]
        
        errors = []
        warnings = []
        details = []
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                errors.append(f"Required directory missing: {dir_path}")
            elif not full_path.is_dir():
                errors.append(f"Path exists but is not a directory: {dir_path}")
            else:
                details.append(f"Required directory exists: {dir_path}")
        
        for dir_path in optional_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                warnings.append(f"Optional directory missing: {dir_path}")
            else:
                details.append(f"Optional directory exists: {dir_path}")
        
        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "details": details
        }
    
    def _validate_configuration_files(self) -> Dict[str, Any]:
        """Validate configuration files exist and are valid."""
        config_files = [
            ("scripts/gates.yaml", "YAML", True),
            ("TODO.md", "Markdown", True),
            ("requirements.txt", "Text", False)
        ]
        
        errors = []
        warnings = []
        details = []
        
        for file_path, file_type, required in config_files:
            full_path = self.project_root / file_path
            
            if not full_path.exists():
                if required:
                    errors.append(f"Required config file missing: {file_path}")
                else:
                    warnings.append(f"Optional config file missing: {file_path}")
                continue
            
            # Validate file content based on type
            try:
                if file_type == "YAML":
                    with open(full_path) as f:
                        yaml.safe_load(f)
                    details.append(f"Valid YAML config: {file_path}")
                    
                elif file_type == "JSON":
                    with open(full_path) as f:
                        json.load(f)
                    details.append(f"Valid JSON config: {file_path}")
                    
                else:
                    # Just check it's readable
                    with open(full_path) as f:
                        f.read()
                    details.append(f"Readable config file: {file_path}")
                    
            except Exception as e:
                errors.append(f"Invalid config file {file_path}: {e}")
        
        # Validate gates.yaml structure specifically
        gates_file = self.project_root / "scripts/gates.yaml"
        if gates_file.exists():
            try:
                with open(gates_file) as f:
                    gates_config = yaml.safe_load(f)
                
                required_sections = ["gates", "promotion_rules", "risk_assessment"]
                for section in required_sections:
                    if section not in gates_config:
                        errors.append(f"Missing section in gates.yaml: {section}")
                    else:
                        details.append(f"Found gates.yaml section: {section}")
                        
            except Exception as e:
                errors.append(f"Could not validate gates.yaml structure: {e}")
        
        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "details": details
        }
    
    def _validate_python_scripts(self) -> Dict[str, Any]:
        """Validate Python scripts exist and are syntactically correct."""
        scripts = [
            "scripts/acceptance_gates.py",
            "scripts/gatekeeper.py",
            "scripts/run_acceptance_pipeline.py", 
            "scripts/bootstrap_bca.py",
            "scripts/fdr.py"
        ]
        
        errors = []
        warnings = []
        details = []
        
        for script_path in scripts:
            full_path = self.project_root / script_path
            
            if not full_path.exists():
                errors.append(f"Required script missing: {script_path}")
                continue
                
            # Check if file is executable
            if not full_path.stat().st_mode & 0o111:
                warnings.append(f"Script not executable: {script_path}")
            
            # Basic syntax check
            try:
                with open(full_path) as f:
                    source = f.read()
                
                compile(source, str(full_path), 'exec')
                details.append(f"Valid Python syntax: {script_path}")
                
            except SyntaxError as e:
                errors.append(f"Syntax error in {script_path}: {e}")
            except Exception as e:
                warnings.append(f"Could not validate {script_path}: {e}")
        
        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "details": details
        }
    
    def _validate_dependencies(self) -> Dict[str, Any]:
        """Validate required Python dependencies are available."""
        required_modules = [
            "json",
            "yaml", 
            "pathlib",
            "subprocess",
            "datetime",
            "dataclasses",
            "typing",
            "logging",
            "hashlib"
        ]
        
        optional_modules = [
            "numpy",
            "scipy", 
            "pandas"
        ]
        
        errors = []
        warnings = []
        details = []
        
        for module in required_modules:
            try:
                importlib.import_module(module)
                details.append(f"Required module available: {module}")
            except ImportError:
                errors.append(f"Required module missing: {module}")
        
        for module in optional_modules:
            try:
                importlib.import_module(module)
                details.append(f"Optional module available: {module}")
            except ImportError:
                warnings.append(f"Optional module missing: {module}")
        
        # Check Python version
        import sys
        if sys.version_info < (3, 7):
            errors.append(f"Python 3.7+ required, found: {sys.version}")
        else:
            details.append(f"Python version OK: {sys.version.split()[0]}")
        
        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "details": details
        }
    
    def _validate_permissions(self) -> Dict[str, Any]:
        """Validate file and directory permissions."""
        errors = []
        warnings = []
        details = []
        
        # Check if we can write to artifacts directory
        artifacts_dir = self.project_root / "artifacts"
        if artifacts_dir.exists():
            try:
                test_file = artifacts_dir / "permission_test.tmp"
                test_file.write_text("test")
                test_file.unlink()
                details.append("Write permission OK: artifacts directory")
            except Exception as e:
                errors.append(f"Cannot write to artifacts directory: {e}")
        
        # Check script executability
        scripts_dir = self.project_root / "scripts"
        if scripts_dir.exists():
            python_scripts = list(scripts_dir.glob("*.py"))
            executable_count = sum(1 for script in python_scripts 
                                 if script.stat().st_mode & 0o111)
            
            if executable_count == 0 and python_scripts:
                warnings.append("No Python scripts are executable (may need chmod +x)")
            else:
                details.append(f"Executable scripts found: {executable_count}/{len(python_scripts)}")
        
        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "details": details
        }
    
    def _validate_test_data(self) -> Dict[str, Any]:
        """Validate test data and sample artifacts exist."""
        errors = []
        warnings = []
        details = []
        
        # Check for sample configuration
        sample_configs = [
            "scripts/gates.yaml"
        ]
        
        for config in sample_configs:
            config_path = self.project_root / config
            if config_path.exists():
                details.append(f"Sample config available: {config}")
            else:
                warnings.append(f"Sample config missing: {config}")
        
        # Check if we have any existing test artifacts
        metrics_dir = self.project_root / "artifacts" / "metrics"
        if metrics_dir.exists():
            artifact_count = len(list(metrics_dir.glob("*.json")))
            if artifact_count > 0:
                details.append(f"Existing test artifacts found: {artifact_count}")
            else:
                warnings.append("No existing test artifacts found")
        
        return {
            "passed": True,  # Test data is not required for basic validation
            "errors": errors,
            "warnings": warnings,
            "details": details
        }
    
    def _validate_integration(self) -> Dict[str, Any]:
        """Validate that components can work together."""
        errors = []
        warnings = []
        details = []
        
        # Test if we can import main modules
        scripts_dir = self.project_root / "scripts"
        sys.path.insert(0, str(scripts_dir))
        
        try:
            # Test basic imports
            spec = importlib.util.spec_from_file_location(
                "acceptance_gates", 
                scripts_dir / "acceptance_gates.py"
            )
            if spec and spec.loader:
                acceptance_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(acceptance_module)
                details.append("Can import acceptance_gates module")
            
            spec = importlib.util.spec_from_file_location(
                "gatekeeper",
                scripts_dir / "gatekeeper.py" 
            )
            if spec and spec.loader:
                gatekeeper_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gatekeeper_module)
                details.append("Can import gatekeeper module")
                
        except Exception as e:
            errors.append(f"Module import failed: {e}")
        
        # Test if basic command execution works
        try:
            result = subprocess.run(
                ["python", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                details.append("Python command execution works")
            else:
                warnings.append("Python command execution issues")
        except Exception as e:
            warnings.append(f"Could not test command execution: {e}")
        
        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "warnings": warnings, 
            "details": details
        }
    
    def _save_validation_report(self, passed: bool):
        """Save validation report to artifacts."""
        report = {
            "validation_metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "project_root": str(self.project_root),
                "passed": passed
            },
            "check_results": self.validation_results,
            "summary": {
                "total_errors": len(self.errors),
                "total_warnings": len(self.warnings),
                "checks_passed": sum(1 for r in self.validation_results.values() if r["passed"]),
                "total_checks": len(self.validation_results)
            },
            "errors": self.errors,
            "warnings": self.warnings
        }
        
        artifacts_dir = self.project_root / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        report_file = artifacts_dir / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Validation report saved: {report_file}")


def main():
    """Main validation execution."""
    project_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    
    validator = AcceptanceSetupValidator(project_root)
    passed = validator.validate_all()
    
    if passed:
        print("\n🎉 Validation complete! You can now run the acceptance pipeline:")
        print(f"   python scripts/run_acceptance_pipeline.py V2")
        sys.exit(0)
    else:
        print("\n💥 Validation failed. Fix the issues above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()