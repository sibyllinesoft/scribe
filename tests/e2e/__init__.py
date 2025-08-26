"""
E2E Integration Tests for PackRepo.

This module implements comprehensive end-to-end testing:
- Golden smoke flows with real repositories
- Multi-run consistency validation
- Performance regression testing
- Container-based integration testing
- CI/CD pipeline integration

From TODO.md requirements:
- Golden smoke flows pass
- 100× rerun flakiness test with ≤1% instability
- Performance p50/p95 within objectives
- Integration with hermetic build infrastructure
"""

from .test_golden_smoke_flows import GoldenSmokeTestSuite
from .test_consistency_validation import ConsistencyValidator, ConsistencyAnalysis
from .test_performance_regression import PerformanceRegressionTester, PerformanceBenchmark, PerformanceObjectives
from .test_container_integration import ContainerIntegrationTester, ContainerTestResult

__all__ = [
    'GoldenSmokeTestSuite',
    'ConsistencyValidator',
    'ConsistencyAnalysis', 
    'PerformanceRegressionTester',
    'PerformanceBenchmark',
    'PerformanceObjectives',
    'ContainerIntegrationTester',
    'ContainerTestResult'
]