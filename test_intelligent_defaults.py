#!/usr/bin/env python3
"""
Test script for Intelligent Scribe Defaults System

This script validates the complete intelligent defaults system across multiple
repositories to ensure 100% success rate and optimal performance.
"""

import pathlib
import time
import sys
from typing import List, Dict, Any

from intelligent_scribe import IntelligentScribe


def run_comprehensive_test(limit: int = 10) -> Dict[str, Any]:
    """Run comprehensive test of the intelligent defaults system."""
    
    print("🧪 Starting comprehensive test of Intelligent Scribe Defaults")
    print("=" * 60)
    
    start_time = time.time()
    results = {
        'test_start_time': start_time,
        'test_results': [],
        'summary': {},
        'success': False
    }
    
    try:
        # Initialize intelligent scribe
        intelligent_scribe = IntelligentScribe()
        
        # Test 1: Repository Discovery
        print("📁 Test 1: Repository Discovery")
        repositories = intelligent_scribe.validator.discover_repositories()
        test_repos = repositories[:limit] if limit else repositories
        
        print(f"   Found {len(repositories)} total repositories")
        print(f"   Testing {len(test_repos)} repositories")
        results['repositories_found'] = len(repositories)
        results['repositories_tested'] = len(test_repos)
        
        # Test 2: Individual Repository Testing
        print(f"\n🔍 Test 2: Individual Repository Analysis")
        
        test_results = []
        for i, repo_path in enumerate(test_repos, 1):
            print(f"   [{i}/{len(test_repos)}] Testing {repo_path.name}...")
            
            repo_start_time = time.time()
            try:
                # Test analysis and configuration generation
                analysis, config = intelligent_scribe.analyze_and_configure(repo_path)
                
                # Test dry run execution
                success = intelligent_scribe.run_with_intelligent_defaults(
                    repo_path, dry_run=True
                )
                
                repo_time = time.time() - repo_start_time
                
                result = {
                    'repo_name': repo_path.name,
                    'repo_path': str(repo_path),
                    'success': success,
                    'analysis_time': repo_time,
                    'source_files': len(analysis.source_files),
                    'estimated_tokens': analysis.estimated_source_tokens,
                    'token_budget': config.token_budget,
                    'confidence': config.confidence_score,
                    'algorithm': config.algorithm,
                    'languages': list(analysis.languages.keys()),
                    'is_monorepo': analysis.is_monorepo,
                    'error': None
                }
                
                if success:
                    print(f"      ✅ Success: {config.token_budget:,} tokens, {config.confidence_score:.1%} confidence")
                else:
                    print(f"      ❌ Failed")
                    
            except Exception as e:
                repo_time = time.time() - repo_start_time
                result = {
                    'repo_name': repo_path.name,
                    'repo_path': str(repo_path),
                    'success': False,
                    'analysis_time': repo_time,
                    'error': str(e)
                }
                print(f"      💥 Exception: {e}")
            
            test_results.append(result)
        
        results['test_results'] = test_results
        
        # Test 3: System Validation
        print(f"\n🔬 Test 3: System-wide Validation")
        
        validation_start = time.time()
        summary = intelligent_scribe.validator.validate_all_repositories(limit=limit)
        validation_time = time.time() - validation_start
        
        print(f"   Validation completed in {validation_time:.1f}s")
        print(f"   Success rate: {summary.success_rate:.1%}")
        print(f"   Average token utilization: {summary.avg_token_utilization:.1%}")
        
        # Calculate overall results
        successful_tests = sum(1 for r in test_results if r['success'])
        success_rate = successful_tests / len(test_results) if test_results else 0
        
        results['summary'] = {
            'success_rate': success_rate,
            'successful_repositories': successful_tests,
            'total_repositories': len(test_results),
            'validation_success_rate': summary.success_rate,
            'avg_token_utilization': summary.avg_token_utilization,
            'avg_confidence': summary.avg_confidence_score,
            'total_test_time': time.time() - start_time,
            'validation_time': validation_time,
        }
        
        # Determine overall success
        results['success'] = success_rate >= 0.95 and summary.success_rate >= 0.95
        
        return results
        
    except Exception as e:
        results['error'] = str(e)
        results['success'] = False
        return results


def print_test_summary(results: Dict[str, Any]) -> None:
    """Print comprehensive test summary."""
    
    print("\n" + "=" * 60)
    print("🎯 INTELLIGENT SCRIBE DEFAULTS - TEST SUMMARY")
    print("=" * 60)
    
    if not results.get('success', False):
        print("❌ OVERALL TEST RESULT: FAILED")
        if 'error' in results:
            print(f"Error: {results['error']}")
    else:
        print("✅ OVERALL TEST RESULT: PASSED")
    
    if 'summary' in results:
        summary = results['summary']
        print(f"\n📊 Test Results:")
        print(f"   Repositories Found: {results.get('repositories_found', 0)}")
        print(f"   Repositories Tested: {results.get('repositories_tested', 0)}")
        print(f"   Individual Test Success Rate: {summary['success_rate']:.1%}")
        print(f"   System Validation Success Rate: {summary['validation_success_rate']:.1%}")
        print(f"   Average Token Utilization: {summary['avg_token_utilization']:.1%}")
        print(f"   Average Confidence: {summary['avg_confidence']:.1%}")
        print(f"   Total Test Time: {summary['total_test_time']:.1f}s")
    
    # Show detailed results for failed repositories
    if 'test_results' in results:
        failed_results = [r for r in results['test_results'] if not r['success']]
        if failed_results:
            print(f"\n❌ Failed Repositories ({len(failed_results)}):")
            for result in failed_results[:10]:  # Show first 10 failures
                print(f"   - {result['repo_name']}: {result.get('error', 'Unknown error')}")
    
    # Performance analysis
    if 'test_results' in results:
        successful_results = [r for r in results['test_results'] if r['success']]
        if successful_results:
            avg_time = sum(r['analysis_time'] for r in successful_results) / len(successful_results)
            avg_tokens = sum(r.get('estimated_tokens', 0) for r in successful_results) / len(successful_results)
            avg_budget = sum(r.get('token_budget', 0) for r in successful_results) / len(successful_results)
            
            print(f"\n⚡ Performance Analysis:")
            print(f"   Average Analysis Time: {avg_time:.2f}s")
            print(f"   Average Source Tokens: {avg_tokens:,.0f}")
            print(f"   Average Token Budget: {avg_budget:,.0f}")
            print(f"   Average Utilization: {(avg_tokens / avg_budget * 100) if avg_budget > 0 else 0:.1f}%")
    
    print("\n" + "=" * 60)


def test_specific_repositories() -> None:
    """Test specific known repositories for detailed validation."""
    
    print("\n🎯 Testing Specific Repositories")
    print("-" * 40)
    
    # Test repositories that should work well
    test_repos = [
        "/home/nathan/Projects/echo",
        "/home/nathan/Projects/scribe", 
        "/home/nathan/Projects/arbiter",
        "/home/nathan/Projects/arachne",
        "/home/nathan/Projects/conclave",
    ]
    
    intelligent_scribe = IntelligentScribe()
    
    for repo_path_str in test_repos:
        repo_path = pathlib.Path(repo_path_str)
        if not repo_path.exists():
            print(f"⚠️  Skipping non-existent repository: {repo_path}")
            continue
        
        print(f"\n🔍 Testing {repo_path.name}:")
        
        try:
            start_time = time.time()
            
            # Analyze and configure
            analysis, config = intelligent_scribe.analyze_and_configure(repo_path)
            
            # Run dry run
            success = intelligent_scribe.run_with_intelligent_defaults(
                repo_path, dry_run=True
            )
            
            elapsed = time.time() - start_time
            
            print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
            print(f"   Time: {elapsed:.2f}s")
            print(f"   Source Files: {len(analysis.source_files)}")
            print(f"   Estimated Tokens: {analysis.estimated_source_tokens:,}")
            print(f"   Token Budget: {config.token_budget:,}")
            print(f"   Algorithm: {config.algorithm}")
            print(f"   Confidence: {config.confidence_score:.1%}")
            print(f"   Languages: {', '.join(analysis.languages.keys())}")
            
        except Exception as e:
            print(f"   ❌ Exception: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Intelligent Scribe Defaults System")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of repositories to test")
    parser.add_argument("--specific", action="store_true", help="Test specific known repositories")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive test suite")
    
    args = parser.parse_args()
    
    if args.specific:
        test_specific_repositories()
        return 0
    
    if args.comprehensive or not any([args.specific]):
        # Run comprehensive test by default
        results = run_comprehensive_test(args.limit)
        print_test_summary(results)
        
        # Return appropriate exit code
        return 0 if results.get('success', False) else 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())