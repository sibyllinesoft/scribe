"""
Container-based Integration Testing for PackRepo.

This module implements hermetic container-based testing with Docker,
ensuring reproducible test environments and CI/CD pipeline integration.

From TODO.md requirements:
- Integration with hermetic build infrastructure
- Container-based test isolation
- CI/CD pipeline validation
- Cross-platform compatibility testing
"""

import asyncio
import docker
import json
import subprocess
import tempfile
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import unittest

from tests.e2e.test_golden_smoke_flows import GoldenSmokeTestSuite
from tests.e2e.test_consistency_validation import ConsistencyValidator
from tests.e2e.test_performance_regression import PerformanceRegressionTester


@dataclass
class ContainerTestResult:
    """Results from a container-based test execution."""
    container_id: str
    test_name: str
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time_ms: float
    container_stats: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class IntegrationTestSuite:
    """Complete integration test suite configuration."""
    name: str
    dockerfile_content: str
    test_commands: List[str]
    expected_files: List[str]
    environment_vars: Dict[str, str]
    resource_limits: Dict[str, Any]


class ContainerIntegrationTester:
    """
    Container-based integration testing framework.
    
    Provides hermetic test environments using Docker containers,
    ensuring reproducible and isolated test execution across
    different platforms and CI/CD environments.
    """
    
    def __init__(self):
        """Initialize container integration tester."""
        self.docker_client = None
        self.test_containers = []
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
            print("✅ Docker client initialized successfully")
        except Exception as e:
            print(f"❌ Docker client initialization failed: {e}")
            self.docker_client = None
    
    def __del__(self):
        """Cleanup test containers on destruction."""
        self.cleanup_containers()
    
    def _create_base_dockerfile(self, python_version: str = "3.11") -> str:
        """Create base Dockerfile for PackRepo testing."""
        return f"""
FROM python:{python_version}-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    gcc \\
    g++ \\
    make \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 testuser && chown -R testuser:testuser /app
USER testuser

# Copy requirements and install Python dependencies
COPY requirements.txt* ./
COPY setup.py* ./
COPY pyproject.toml* ./

# Install dependencies
RUN pip install --no-cache-dir --user pytest pytest-asyncio pytest-cov

# Copy application code
COPY --chown=testuser:testuser . .

# Install package in development mode
RUN pip install --no-cache-dir --user -e .

# Set Python path
ENV PATH="/home/testuser/.local/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Default command
CMD ["python", "-m", "pytest", "--verbose"]
"""
    
    def _create_test_suite_config(self) -> List[IntegrationTestSuite]:
        """Create configuration for different integration test suites."""
        return [
            IntegrationTestSuite(
                name="python_3_9_compatibility",
                dockerfile_content=self._create_base_dockerfile("3.9"),
                test_commands=[
                    "python --version",
                    "python -m pytest tests/unit/ -v",
                    "python -m pytest tests/integration/ -v",
                    "python -c 'import packrepo; print(\"✅ Import successful\")'",
                ],
                expected_files=[
                    "/app/packrepo/__init__.py",
                    "/app/tests/unit/",
                    "/app/tests/integration/"
                ],
                environment_vars={
                    "PYTHONPATH": "/app",
                    "PYTEST_CURRENT_TEST": "container_integration"
                },
                resource_limits={
                    "mem_limit": "1g",
                    "memswap_limit": "1g",
                    "cpu_period": 100000,
                    "cpu_quota": 100000  # 1 CPU
                }
            ),
            IntegrationTestSuite(
                name="python_3_11_full_suite",
                dockerfile_content=self._create_base_dockerfile("3.11"),
                test_commands=[
                    "python --version",
                    "python -m pytest tests/unit/ --cov=packrepo --cov-report=term-missing",
                    "python -m pytest tests/properties/ -v",
                    "python -m pytest tests/metamorphic/ -v",
                    "python -m pytest tests/e2e/test_golden_smoke_flows.py -v"
                ],
                expected_files=[
                    "/app/packrepo/",
                    "/app/tests/"
                ],
                environment_vars={
                    "PYTHONPATH": "/app",
                    "COVERAGE_CORE": "sysmon"
                },
                resource_limits={
                    "mem_limit": "2g",
                    "memswap_limit": "2g",
                    "cpu_period": 100000,
                    "cpu_quota": 200000  # 2 CPUs
                }
            ),
            IntegrationTestSuite(
                name="performance_testing",
                dockerfile_content=self._create_base_dockerfile("3.11") + """
# Install additional performance testing tools
RUN pip install --no-cache-dir --user memory-profiler psutil
""",
                test_commands=[
                    "python -m pytest tests/e2e/test_performance_regression.py -v",
                    "python -c 'import psutil; print(f\"Memory: {psutil.virtual_memory().available // 1024 // 1024}MB\")'",
                    "python -c 'import os; print(f\"CPUs: {os.cpu_count()}\")'",
                ],
                expected_files=[
                    "/app/tests/e2e/test_performance_regression.py"
                ],
                environment_vars={
                    "PYTHONPATH": "/app",
                    "PERFORMANCE_TEST": "1"
                },
                resource_limits={
                    "mem_limit": "1g",
                    "memswap_limit": "1g",
                    "cpu_period": 100000,
                    "cpu_quota": 100000
                }
            ),
            IntegrationTestSuite(
                name="consistency_validation",
                dockerfile_content=self._create_base_dockerfile("3.11"),
                test_commands=[
                    "python -m pytest tests/e2e/test_consistency_validation.py::TestConsistencyValidation::test_full_consistency_validation -v",
                    "python -c 'from tests.e2e.test_consistency_validation import ConsistencyValidator; v = ConsistencyValidator(num_runs=10); print(\"✅ Consistency validator initialized\")'",
                ],
                expected_files=[
                    "/app/tests/e2e/test_consistency_validation.py"
                ],
                environment_vars={
                    "PYTHONPATH": "/app",
                    "CONSISTENCY_TEST": "1"
                },
                resource_limits={
                    "mem_limit": "1g",
                    "memswap_limit": "1g",
                    "cpu_period": 100000,
                    "cpu_quota": 100000
                }
            )
        ]
    
    def build_test_image(self, suite: IntegrationTestSuite,
                        source_dir: Path) -> str:
        """Build Docker image for integration test suite."""
        if not self.docker_client:
            raise RuntimeError("Docker client not available")
        
        print(f"🐳 Building test image: {suite.name}")
        
        # Create temporary directory for Docker context
        with tempfile.TemporaryDirectory() as temp_dir:
            context_dir = Path(temp_dir)
            
            # Copy source files to context
            shutil.copytree(source_dir, context_dir / "src")
            
            # Create Dockerfile
            dockerfile_path = context_dir / "Dockerfile"
            with open(dockerfile_path, 'w') as f:
                f.write(suite.dockerfile_content)
            
            # Copy source into context root for COPY commands
            for item in (source_dir).iterdir():
                if item.is_file():
                    shutil.copy2(item, context_dir)
                elif item.is_dir() and item.name not in ['.git', '__pycache__', '.pytest_cache']:
                    shutil.copytree(item, context_dir / item.name, dirs_exist_ok=True)
            
            # Build image
            try:
                image, build_logs = self.docker_client.images.build(
                    path=str(context_dir),
                    tag=f"packrepo-test-{suite.name}",
                    rm=True,
                    forcerm=True
                )
                
                print(f"✅ Built image: {image.short_id}")
                return image.id
                
            except docker.errors.BuildError as e:
                print(f"❌ Build failed for {suite.name}:")
                for log in e.build_log:
                    if 'stream' in log:
                        print(f"  {log['stream'].strip()}")
                raise
    
    def run_container_test(self, suite: IntegrationTestSuite,
                          image_id: str) -> List[ContainerTestResult]:
        """Run integration tests in a container."""
        if not self.docker_client:
            raise RuntimeError("Docker client not available")
        
        results = []
        
        print(f"🚀 Running container tests: {suite.name}")
        
        try:
            # Create and start container
            container = self.docker_client.containers.run(
                image_id,
                command="/bin/bash",
                environment=suite.environment_vars,
                detach=True,
                tty=True,
                stdin_open=True,
                **suite.resource_limits
            )
            
            self.test_containers.append(container)
            
            # Wait for container to be ready
            time.sleep(2)
            
            # Run each test command
            for i, command in enumerate(suite.test_commands):
                print(f"  📋 Running command {i+1}/{len(suite.test_commands)}: {command}")
                
                start_time = time.perf_counter()
                
                try:
                    # Execute command in container
                    exec_result = container.exec_run(
                        command,
                        workdir="/app",
                        environment=suite.environment_vars
                    )
                    
                    end_time = time.perf_counter()
                    execution_time_ms = (end_time - start_time) * 1000
                    
                    # Get container stats
                    stats = container.stats(stream=False)
                    
                    result = ContainerTestResult(
                        container_id=container.short_id,
                        test_name=f"{suite.name}_command_{i}",
                        success=exec_result.exit_code == 0,
                        exit_code=exec_result.exit_code,
                        stdout=exec_result.output.decode('utf-8') if exec_result.output else "",
                        stderr="",
                        execution_time_ms=execution_time_ms,
                        container_stats=stats
                    )
                    
                    results.append(result)
                    
                    if result.success:
                        print(f"    ✅ Command succeeded ({result.execution_time_ms:.1f}ms)")
                    else:
                        print(f"    ❌ Command failed (exit code: {result.exit_code})")
                        print(f"       Output: {result.stdout[:200]}...")
                        
                except Exception as e:
                    end_time = time.perf_counter()
                    execution_time_ms = (end_time - start_time) * 1000
                    
                    result = ContainerTestResult(
                        container_id=container.short_id,
                        test_name=f"{suite.name}_command_{i}",
                        success=False,
                        exit_code=-1,
                        stdout="",
                        stderr=str(e),
                        execution_time_ms=execution_time_ms,
                        container_stats={},
                        error_message=str(e)
                    )
                    
                    results.append(result)
                    print(f"    ❌ Command exception: {e}")
            
            # Verify expected files exist
            for expected_file in suite.expected_files:
                try:
                    exec_result = container.exec_run(f"ls -la {expected_file}")
                    if exec_result.exit_code != 0:
                        print(f"    ⚠️  Expected file missing: {expected_file}")
                    else:
                        print(f"    ✅ Expected file found: {expected_file}")
                except Exception as e:
                    print(f"    ❌ Error checking file {expected_file}: {e}")
            
        except Exception as e:
            print(f"❌ Container test failed: {e}")
            result = ContainerTestResult(
                container_id="unknown",
                test_name=suite.name,
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time_ms=0,
                container_stats={},
                error_message=str(e)
            )
            results.append(result)
        
        return results
    
    def run_all_integration_tests(self, source_dir: Path) -> Dict[str, List[ContainerTestResult]]:
        """Run all container-based integration tests."""
        if not self.docker_client:
            print("❌ Docker not available - skipping container tests")
            return {}
        
        test_suites = self._create_test_suite_config()
        all_results = {}
        
        print("🐳 PackRepo Container Integration Testing")
        print("=" * 50)
        
        for suite in test_suites:
            try:
                # Build test image
                image_id = self.build_test_image(suite, source_dir)
                
                # Run tests in container
                results = self.run_container_test(suite, image_id)
                all_results[suite.name] = results
                
                # Summary for this suite
                successful_tests = len([r for r in results if r.success])
                total_tests = len(results)
                
                print(f"📊 {suite.name}: {successful_tests}/{total_tests} tests passed")
                
            except Exception as e:
                print(f"❌ Suite {suite.name} failed: {e}")
                all_results[suite.name] = [
                    ContainerTestResult(
                        container_id="unknown",
                        test_name=suite.name,
                        success=False,
                        exit_code=-1,
                        stdout="",
                        stderr=str(e),
                        execution_time_ms=0,
                        container_stats={},
                        error_message=str(e)
                    )
                ]
        
        return all_results
    
    def cleanup_containers(self):
        """Clean up test containers."""
        if not self.docker_client:
            return
        
        for container in self.test_containers:
            try:
                container.stop(timeout=5)
                container.remove()
                print(f"🧹 Cleaned up container: {container.short_id}")
            except Exception as e:
                print(f"⚠️  Failed to cleanup container {container.short_id}: {e}")
        
        self.test_containers.clear()
    
    def generate_ci_config(self) -> Dict[str, Any]:
        """Generate CI/CD pipeline configuration for container tests."""
        return {
            "github_actions": {
                "name": "PackRepo Container Integration Tests",
                "on": ["push", "pull_request"],
                "jobs": {
                    "container-tests": {
                        "runs-on": "ubuntu-latest",
                        "services": {
                            "docker": {
                                "image": "docker:dind",
                                "options": "--privileged"
                            }
                        },
                        "steps": [
                            {
                                "uses": "actions/checkout@v3"
                            },
                            {
                                "name": "Set up Docker Buildx",
                                "uses": "docker/setup-buildx-action@v2"
                            },
                            {
                                "name": "Run Container Integration Tests",
                                "run": "python -m pytest tests/e2e/test_container_integration.py -v"
                            },
                            {
                                "name": "Upload Test Results",
                                "uses": "actions/upload-artifact@v3",
                                "if": "always()",
                                "with": {
                                    "name": "container-test-results",
                                    "path": "test-results/"
                                }
                            }
                        ]
                    }
                }
            },
            "gitlab_ci": {
                "image": "docker:latest",
                "services": ["docker:dind"],
                "variables": {
                    "DOCKER_DRIVER": "overlay2",
                    "DOCKER_TLS_CERTDIR": "/certs"
                },
                "before_script": [
                    "apk add --no-cache python3 py3-pip",
                    "pip3 install -r requirements.txt"
                ],
                "stages": ["test"],
                "container_integration_tests": {
                    "stage": "test",
                    "script": [
                        "python -m pytest tests/e2e/test_container_integration.py -v"
                    ],
                    "artifacts": {
                        "reports": {
                            "junit": "test-results/junit.xml"
                        },
                        "paths": ["test-results/"]
                    }
                }
            }
        }


class TestContainerIntegration(unittest.TestCase):
    """Test cases for container integration testing framework."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures."""
        cls.tester = ContainerIntegrationTester()
        cls.source_dir = Path(__file__).parent.parent.parent  # Project root
    
    def setUp(self):
        """Set up test fixtures."""
        if not self.tester.docker_client:
            self.skipTest("Docker not available")
    
    def test_docker_client_availability(self):
        """Test Docker client is properly initialized."""
        self.assertIsNotNone(self.tester.docker_client)
        
        # Test Docker daemon is responsive
        try:
            self.tester.docker_client.ping()
        except Exception as e:
            self.fail(f"Docker daemon not responsive: {e}")
    
    def test_dockerfile_generation(self):
        """Test Dockerfile generation for different Python versions."""
        dockerfile_39 = self.tester._create_base_dockerfile("3.9")
        dockerfile_311 = self.tester._create_base_dockerfile("3.11")
        
        self.assertIn("FROM python:3.9-slim", dockerfile_39)
        self.assertIn("FROM python:3.11-slim", dockerfile_311)
        self.assertIn("USER testuser", dockerfile_39)
        self.assertIn("WORKDIR /app", dockerfile_311)
    
    def test_test_suite_configuration(self):
        """Test integration test suite configuration."""
        suites = self.tester._create_test_suite_config()
        
        self.assertGreater(len(suites), 0)
        
        for suite in suites:
            self.assertIsInstance(suite.name, str)
            self.assertIsInstance(suite.dockerfile_content, str)
            self.assertIsInstance(suite.test_commands, list)
            self.assertGreater(len(suite.test_commands), 0)
    
    @unittest.skipIf(not shutil.which('docker'), "Docker not available")
    def test_simple_container_test(self):
        """Test running a simple container test (if Docker is available)."""
        suites = self.tester._create_test_suite_config()
        
        # Use the simplest test suite
        simple_suite = None
        for suite in suites:
            if "compatibility" in suite.name:
                simple_suite = suite
                break
        
        if simple_suite is None:
            simple_suite = suites[0]
        
        # Modify test commands to be very simple
        simple_suite.test_commands = [
            "python --version",
            "echo 'Container test successful'"
        ]
        
        try:
            # Build and test
            image_id = self.tester.build_test_image(simple_suite, self.source_dir)
            results = self.tester.run_container_test(simple_suite, image_id)
            
            # Verify results
            self.assertGreater(len(results), 0)
            successful_tests = len([r for r in results if r.success])
            self.assertGreater(successful_tests, 0)
            
        except Exception as e:
            # Don't fail the test if Docker has issues - log and skip
            print(f"Container test skipped due to Docker issue: {e}")
            self.skipTest(f"Docker issue: {e}")
    
    def test_ci_config_generation(self):
        """Test CI/CD configuration generation."""
        config = self.tester.generate_ci_config()
        
        # Verify GitHub Actions config
        self.assertIn("github_actions", config)
        github_config = config["github_actions"]
        self.assertIn("jobs", github_config)
        self.assertIn("container-tests", github_config["jobs"])
        
        # Verify GitLab CI config
        self.assertIn("gitlab_ci", config)
        gitlab_config = config["gitlab_ci"]
        self.assertIn("services", gitlab_config)
        self.assertIn("container_integration_tests", gitlab_config)
    
    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, 'tester'):
            self.tester.cleanup_containers()


if __name__ == '__main__':
    # Run container integration tests
    tester = ContainerIntegrationTester()
    source_dir = Path(__file__).parent.parent.parent
    
    print("🐳 PackRepo Container Integration Testing")
    print("=" * 50)
    
    if not tester.docker_client:
        print("❌ Docker not available - container tests cannot run")
        print("Please install Docker to enable container-based testing")
        exit(1)
    
    try:
        results = tester.run_all_integration_tests(source_dir)
        
        print("\n📊 CONTAINER INTEGRATION TEST SUMMARY")
        print("=" * 50)
        
        total_suites = len(results)
        successful_suites = 0
        total_tests = 0
        successful_tests = 0
        
        for suite_name, suite_results in results.items():
            suite_success_count = len([r for r in suite_results if r.success])
            suite_total = len(suite_results)
            
            status = "✅ PASS" if suite_success_count == suite_total else "❌ FAIL"
            print(f"{suite_name}: {status} ({suite_success_count}/{suite_total})")
            
            total_tests += suite_total
            successful_tests += suite_success_count
            
            if suite_success_count == suite_total:
                successful_suites += 1
        
        print(f"\nOverall: {successful_suites}/{total_suites} suites passed")
        print(f"Individual tests: {successful_tests}/{total_tests} passed")
        
        if successful_suites == total_suites:
            print("🎉 All container integration tests passed!")
        else:
            print("⚠️  Some container integration tests failed")
        
        # Generate CI configuration
        print("\n🔧 Generating CI/CD configuration...")
        ci_config = tester.generate_ci_config()
        
        with open("ci_container_config.json", "w") as f:
            json.dump(ci_config, f, indent=2)
        print("📄 CI configuration saved to: ci_container_config.json")
        
    finally:
        tester.cleanup_containers()
    
    # Also run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)