#!/usr/bin/env python3
"""
Research-Grade System Validation Script

Validates that the research-grade acceptance gates and gatekeeper system
is properly installed and configured for academic publication validation.
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

def check_python_requirements() -> Tuple[bool, List[str]]:
    """Check Python version and required packages."""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 10):
        issues.append(f"Python 3.10+ required, found {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check required packages
    required_packages = ['numpy', 'scipy', 'yaml']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Missing required package: {package}")
    
    return len(issues) == 0, issues

def check_system_files() -> Tuple[bool, List[str]]:
    """Check that all required system files are present."""
    issues = []
    
    required_files = [
        "scripts/research_grade_acceptance_gates.py",
        "scripts/research_grade_gatekeeper.py", 
        "scripts/research_grade_pipeline.py",
        "config/research_gates_config.yaml",
        "RESEARCH_GRADE_ACCEPTANCE_SYSTEM.md"
    ]
    
    for file_path in required_files:
        path = Path(file_path)
        if not path.exists():
            issues.append(f"Missing required file: {file_path}")
        elif not path.is_file():
            issues.append(f"Path exists but is not a file: {file_path}")
    
    return len(issues) == 0, issues

def check_script_executability() -> Tuple[bool, List[str]]:
    """Check that key scripts are executable and can import successfully."""
    issues = []
    
    scripts = [
        "scripts/research_grade_acceptance_gates.py",
        "scripts/research_grade_gatekeeper.py",
        "scripts/research_grade_pipeline.py"
    ]
    
    for script_path in scripts:
        try:
            # Test that script can be imported (syntax check)
            result = subprocess.run([
                sys.executable, "-c", 
                f"import ast; ast.parse(open('{script_path}').read())"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                issues.append(f"Syntax error in {script_path}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            issues.append(f"Timeout checking {script_path}")
        except Exception as e:
            issues.append(f"Error checking {script_path}: {e}")
    
    return len(issues) == 0, issues

def check_configuration_validity() -> Tuple[bool, List[str]]:
    """Check that configuration files are valid."""
    issues = []
    
    config_path = Path("config/research_gates_config.yaml")
    
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Check for required configuration sections
            required_sections = [
                'research_thresholds',
                'gates', 
                'statistical_methods',
                'publication_requirements',
                'decision_routing'
            ]
            
            for section in required_sections:
                if section not in config:
                    issues.append(f"Missing configuration section: {section}")
                    
        except yaml.YAMLError as e:
            issues.append(f"Invalid YAML syntax in config: {e}")
        except Exception as e:
            issues.append(f"Error reading configuration: {e}")
    else:
        issues.append("Configuration file not found: config/research_gates_config.yaml")
    
    return len(issues) == 0, issues

def run_basic_functionality_test() -> Tuple[bool, List[str]]:
    """Run basic functionality tests to ensure system works."""
    issues = []
    
    # Create minimal test directories
    test_dirs = ["artifacts", "publication_artifacts"]
    for test_dir in test_dirs:
        Path(test_dir).mkdir(exist_ok=True)
    
    # Test configuration loading
    try:
        import yaml
        config_path = Path("config/research_gates_config.yaml")
        
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                
            # Verify key thresholds are reasonable
            thresholds = config.get('research_thresholds', {})
            
            mutation_threshold = thresholds.get('mutation_score_threshold', 0)
            if mutation_threshold < 0.5 or mutation_threshold > 1.0:
                issues.append(f"Unreasonable mutation score threshold: {mutation_threshold}")
                
            effect_threshold = thresholds.get('effect_size_threshold', 0)
            if effect_threshold < 0.05 or effect_threshold > 0.5:
                issues.append(f"Unreasonable effect size threshold: {effect_threshold}")
                
        else:
            issues.append("Cannot test configuration - file not found")
            
    except Exception as e:
        issues.append(f"Configuration validation failed: {e}")
    
    # Test basic imports (components should be importable)
    try:
        sys.path.insert(0, str(Path("scripts").resolve()))
        
        # Test that we can import the main classes
        from research_grade_acceptance_gates import ResearchGradeAcceptanceEngine
        from research_grade_gatekeeper import ResearchGradeGatekeeper
        from research_grade_pipeline import ResearchGradePipeline
        
        # Test basic instantiation
        engine = ResearchGradeAcceptanceEngine()
        gatekeeper = ResearchGradeGatekeeper()
        
        # Basic validation that they have expected methods
        if not hasattr(engine, 'evaluate_all_research_gates'):
            issues.append("AcceptanceEngine missing required method")
            
        if not hasattr(gatekeeper, 'make_publication_decision'):
            issues.append("Gatekeeper missing required method")
            
    except ImportError as e:
        issues.append(f"Import error in system components: {e}")
    except Exception as e:
        issues.append(f"Basic functionality test failed: {e}")
    finally:
        if str(Path("scripts").resolve()) in sys.path:
            sys.path.remove(str(Path("scripts").resolve()))
    
    return len(issues) == 0, issues

def main():
    """Main validation routine."""
    print("\n" + "="*70)
    print("RESEARCH-GRADE FASTPATH SYSTEM VALIDATION")
    print("="*70)
    
    all_checks_passed = True
    
    # Run all validation checks
    checks = [
        ("Python Requirements", check_python_requirements),
        ("System Files", check_system_files),
        ("Script Executability", check_script_executability),
        ("Configuration Validity", check_configuration_validity),
        ("Basic Functionality", run_basic_functionality_test)
    ]
    
    for check_name, check_function in checks:
        print(f"\n🔍 Checking {check_name}...")
        
        try:
            passed, issues = check_function()
            
            if passed:
                print(f"   ✅ {check_name}: PASSED")
            else:
                print(f"   ❌ {check_name}: FAILED")
                for issue in issues:
                    print(f"      - {issue}")
                all_checks_passed = False
                
        except Exception as e:
            print(f"   ⚠️  {check_name}: ERROR - {e}")
            all_checks_passed = False
    
    # Summary
    print(f"\n" + "="*70)
    if all_checks_passed:
        print("✅ VALIDATION COMPLETE - System ready for research-grade validation")
        print("\n🚀 Next steps:")
        print("   1. Run: python scripts/research_grade_pipeline.py V2")
        print("   2. Check results in: artifacts/gatekeeper_decision.json")
        print("   3. Review publication package: publication_artifacts/")
        exit_code = 0
    else:
        print("❌ VALIDATION FAILED - Issues must be resolved before use")
        print("\n🔧 Required actions:")
        print("   1. Install missing dependencies: pip install numpy scipy pyyaml")
        print("   2. Check file permissions and paths")
        print("   3. Verify configuration syntax")
        print("   4. Re-run validation: python scripts/validate_research_system.py")
        exit_code = 1
    
    print("="*70)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
