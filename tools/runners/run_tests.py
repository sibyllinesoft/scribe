#!/usr/bin/env python3
"""
Test Runner for FastPath Research Evaluation Suite
==================================================

Runs the complete test suite and validates the evaluation framework.
Use this to verify the system works correctly before running research evaluations.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print("\nOutput:")
                print(result.stdout[-1000:])  # Last 1000 chars
        else:
            print("❌ FAILED")
            if result.stderr:
                print("\nError:")
                print(result.stderr[-1000:])
            return False
            
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        return False
        
    return True


def main():
    """Main test runner."""
    print("FastPath Research Evaluation Suite - Test Runner")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Python syntax check
    if not run_command([sys.executable, "-m", "py_compile", "research_evaluation_suite.py"], 
                      "Python Syntax Check - Main Orchestrator"):
        all_passed = False
    
    # Test 2: Import checks  
    test_imports = [
        "research_evaluation_suite",
        "baseline_implementations", 
        "multi_repository_benchmark",
        "statistical_analysis_engine",
        "reproducibility_framework",
        "publication_data_generator"
    ]
    
    for module in test_imports:
        if not run_command([sys.executable, "-c", f"import {module}; print(f'✅ {module} imported successfully')"],
                          f"Import Check - {module}"):
            all_passed = False
    
    # Test 3: Unit tests
    if not run_command([sys.executable, "test_evaluation_suite.py"], 
                      "Unit Test Suite"):
        all_passed = False
    
    # Test 4: Example data generation (quick version)
    if not run_command([sys.executable, "-c", """
import sys
sys.path.append('.')
from generate_example_research_data import ExampleDataGenerator
import tempfile
import shutil

print('Testing example data generation...')
generator = ExampleDataGenerator(random_seed=42)

# Generate small dataset for testing
temp_dir = tempfile.mkdtemp()
try:
    results = generator.generate_comprehensive_dataset(
        repositories_per_type=2,  # Very small for testing
        output_directory=temp_dir
    )
    print(f'✅ Generated {len(results["measurements"])} measurements')
    print(f'✅ Created {sum(len(files) for files in results["generated_files"].values())} files')
    
    # Validate key claims
    validation = results['validation_results']
    print(f'QA Improvement: {validation["qa_improvement_percentage"]:.1f}%')
    print(f'Speed Improvement: {validation["speed_improvement_percentage"]:.1f}%')
    print('✅ Example data generation successful')
    
finally:
    shutil.rmtree(temp_dir, ignore_errors=True)
"""], "Example Data Generation Test"):
        all_passed = False
    
    # Final summary
    print(f"\n{'='*60}")
    print("🏁 FINAL RESULTS")
    print(f"{'='*60}")
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nThe FastPath Research Evaluation Suite is ready for use.")
        print("\nNext steps:")
        print("1. Run: python generate_example_research_data.py")
        print("2. Check output in: ./fastpath_example_results/")
        print("3. Review the research_data_report.md for validation results")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before running research evaluations.")
        print("\nCommon issues:")
        print("- Missing dependencies: pip install -r requirements_research.txt")
        print("- Python version: Requires Python 3.8+")
        print("- System resources: Ensure sufficient memory for data processing")
        return 1


if __name__ == "__main__":
    sys.exit(main())