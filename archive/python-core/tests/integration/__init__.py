"""
Integration Testing Module for PackRepo.

This module provides integration between the comprehensive testing framework
and production systems including Oracle V1 and hermetic build infrastructure.

From TODO.md requirements:
- Integration with existing V1 oracle system
- Hermetic build infrastructure integration
- All oracles pass on DATASET_REPOS
- CI/CD pipeline orchestration
"""

from .oracle_integration import (
    OracleIntegrationTester,
    ComprehensiveIntegrationTester,
    HermeticBuildManager,
    DatasetRepositoryManager,
    OracleTestResult,
    HermeticBuildResult,
    IntegrationTestSuite
)

__all__ = [
    'OracleIntegrationTester',
    'ComprehensiveIntegrationTester',
    'HermeticBuildManager', 
    'DatasetRepositoryManager',
    'OracleTestResult',
    'HermeticBuildResult',
    'IntegrationTestSuite'
]