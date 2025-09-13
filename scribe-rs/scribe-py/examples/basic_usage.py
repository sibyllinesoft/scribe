#!/usr/bin/env python3
"""
Basic usage example of the Scribe-RS Python bindings.

This example demonstrates how to use the Scribe library for comprehensive
code repository analysis including file scanning, heuristic scoring, 
and dependency graph analysis.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the scribe_rs package to the path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from scribe_rs import (
    Repository, HeuristicScorer, PageRankAnalyzer, PatternMatcher,
    AnalysisConfig, analyze_repository_complete, get_info
)


async def basic_repository_analysis(repo_path: str):
    """Demonstrate basic repository analysis workflow."""
    print(f"🔍 Analyzing repository: {repo_path}")
    
    try:
        # Create repository instance
        repo = Repository(repo_path)
        print(f"✅ Repository loaded: {repo.path}")
        
        # Check if it's a Git repository
        if repo.has_git():
            print("📁 Git repository detected")
        
        # Scan files with progress callback
        def progress_callback(current, total):
            print(f"📄 Scanning files: {current}/{total}")
            return True  # Continue processing
        
        print("\n🔎 Scanning files...")
        files = await repo.scan_files(
            max_files=100,  # Limit for demo
            exclude_patterns=["*.pyc", "__pycache__", ".git"],
            progress_callback=progress_callback
        )
        
        print(f"✅ Found {len(files) if hasattr(files, '__len__') else 'N/A'} files")
        
        # Get repository statistics
        print("\n📊 Repository Statistics:")
        lang_stats = await repo.get_language_stats()
        size_stats = await repo.get_size_stats()
        
        print("Language distribution:")
        for lang, stats in lang_stats.items():
            print(f"  {lang}: {stats['file_count']} files, {stats['line_count']} lines")
        
        print("\nSize statistics:")
        for key, value in size_stats.items():
            if key.endswith('_size'):
                print(f"  {key}: {value / 1024:.1f} KB")
            else:
                print(f"  {key}: {value}")
        
        return repo
        
    except Exception as e:
        print(f"❌ Error analyzing repository: {e}")
        return None


async def heuristic_scoring_demo(repo_path: str):
    """Demonstrate heuristic scoring capabilities."""
    print(f"\n🎯 Heuristic Scoring Demo: {repo_path}")
    
    try:
        # Create repository and get files
        repo = Repository(repo_path)
        files = await repo.scan_files(max_files=50)
        
        # Create scorer with custom weights
        custom_weights = {
            "documentation": 0.25,  # Emphasize documentation
            "complexity": 0.20,     # Code complexity
            "functions": 0.15,      # Function definitions
            "imports": 0.10,        # Import statements
            "centrality": 0.30,     # Graph centrality (will be added later)
        }
        
        scorer = HeuristicScorer(weights=custom_weights)
        
        print("📈 Scoring files...")
        scored_files = await scorer.score_files(files, batch_size=20)
        
        # Get top files by score
        top_files = scorer.get_top_files(scored_files, n=5)
        
        print("🏆 Top 5 files by score:")
        for path, score in top_files:
            print(f"  {score:.3f}: {path}")
        
        # Calculate scoring statistics
        stats = scorer.calculate_score_statistics(scored_files)
        print(f"\n📊 Scoring statistics:")
        print(f"  Mean score: {stats.get('mean', 0):.3f}")
        print(f"  Median score: {stats.get('median', 0):.3f}")
        print(f"  Std deviation: {stats.get('std_dev', 0):.3f}")
        
        return scored_files
        
    except Exception as e:
        print(f"❌ Error in heuristic scoring: {e}")
        return None


async def pagerank_analysis_demo(repo_path: str):
    """Demonstrate PageRank dependency analysis."""
    print(f"\n🕸️ PageRank Dependency Analysis: {repo_path}")
    
    try:
        # Create repository and get files  
        repo = Repository(repo_path)
        files = await repo.scan_files(max_files=50)
        
        # Create PageRank analyzer with custom config
        analyzer = PageRankAnalyzer(
            damping_factor=0.85,
            max_iterations=100,
            tolerance=1e-6
        )
        
        print("🔍 Analyzing dependencies...")
        centrality_scores = await analyzer.analyze_dependencies(
            files,
            include_external=False
        )
        
        # Get graph statistics
        graph_stats = await analyzer.get_graph_statistics()
        
        print("📈 Graph statistics:")
        for key, value in graph_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        # Find circular dependencies
        circular_deps = await analyzer.find_circular_dependencies()
        if circular_deps:
            print(f"⚠️ Found {len(circular_deps)} circular dependencies")
            for cycle in circular_deps[:3]:  # Show first 3
                print(f"  Cycle: {' -> '.join(cycle)}")
        else:
            print("✅ No circular dependencies found")
        
        # Find strongly connected components
        components = await analyzer.find_strongly_connected_components()
        print(f"🔗 Found {len(components)} strongly connected components")
        
        return centrality_scores
        
    except Exception as e:
        print(f"❌ Error in PageRank analysis: {e}")
        return None


async def pattern_matching_demo(repo_path: str):
    """Demonstrate pattern matching capabilities."""
    print(f"\n🔍 Pattern Matching Demo: {repo_path}")
    
    try:
        # Create pattern matcher
        matcher = PatternMatcher()
        
        # Add custom pattern rules
        custom_rules = [
            {
                "name": "todo_comments",
                "pattern": r"(?i)\b(TODO|FIXME|BUG|HACK)\b.*",
                "language": "python",
                "description": "Find TODO/FIXME comments",
                "category": "maintenance",
                "severity": "low"
            },
            {
                "name": "async_function",
                "pattern": r"async\s+def\s+(\w+)",
                "language": "python", 
                "description": "Find async function definitions",
                "category": "async",
                "severity": "info"
            },
            {
                "name": "error_handling",
                "pattern": r"except\s+(\w+):",
                "language": "python",
                "description": "Find exception handling blocks", 
                "category": "error_handling",
                "severity": "info"
            }
        ]
        
        # Add rules
        for rule in custom_rules:
            matcher.add_rule(rule)
        
        print(f"✅ Added {len(custom_rules)} custom pattern rules")
        
        # Get repository files
        repo = Repository(repo_path)
        files = await repo.scan_files(max_files=20)
        
        # Find pattern matches
        print("🔍 Finding pattern matches...")
        matches = await matcher.find_matches_batch(
            files,
            rule_filter=None,  # Apply all rules
            batch_size=10
        )
        
        total_matches = sum(len(file_matches) for file_matches in matches.values())
        print(f"📊 Found {total_matches} total matches across {len(matches)} files")
        
        # Show sample matches
        match_count = 0
        for file_path, file_matches in matches.items():
            if not file_matches:
                continue
                
            print(f"\n📄 {file_path}:")
            for match in file_matches[:3]:  # Show first 3 matches per file
                print(f"  🎯 {match['rule_name']}: {match['matched_text'][:50]}...")
                match_count += 1
                if match_count >= 10:  # Limit total output
                    break
            
            if match_count >= 10:
                break
        
        return matches
        
    except Exception as e:
        print(f"❌ Error in pattern matching: {e}")
        return None


async def comprehensive_analysis_demo(repo_path: str):
    """Demonstrate the high-level comprehensive analysis function."""
    print(f"\n🎯 Comprehensive Analysis Demo: {repo_path}")
    
    try:
        # Create custom configuration
        config = AnalysisConfig()
        config.max_files = 100
        config.batch_size = 20
        config.scoring_weights["documentation"] = 0.25  # Emphasize docs
        config.pagerank_damping = 0.90  # Higher damping factor
        
        # Progress callback
        def progress_callback(current, total):
            percentage = (current / total) * 100
            print(f"📊 Progress: {percentage:.1f}% ({current}/{total})")
            return True
        
        print("🚀 Running comprehensive analysis...")
        results = await analyze_repository_complete(
            repo_path,
            config=config,
            progress_callback=progress_callback
        )
        
        print("✅ Analysis complete!")
        
        # Display summary results
        metadata = results["analysis_metadata"]
        print(f"\n📋 Analysis Summary:")
        print(f"  Files analyzed: {metadata['files_analyzed']}")
        print(f"  Scribe version: {metadata['scribe_version']}")
        
        if "language_stats" in results:
            print(f"  Languages found: {len(results['language_stats'])}")
        
        if "git_stats" in results:
            git_stats = results["git_stats"] 
            print(f"  Git commits: {git_stats.get('total_commits', 'N/A')}")
            print(f"  Git authors: {git_stats.get('total_authors', 'N/A')}")
        
        # Show top scored files
        if results["file_scores"]:
            print("\n🏆 Top files by final score:")
            # Note: In real implementation, we'd need to sort the scores
            # This is a simplified example
            file_count = 0
            for file_path in list(results["file_scores"].keys())[:5]:
                scores = results["file_scores"][file_path]
                final_score = scores.get("final_score", 0)
                print(f"  {final_score:.3f}: {file_path}")
                file_count += 1
                if file_count >= 5:
                    break
        
        return results
        
    except Exception as e:
        print(f"❌ Error in comprehensive analysis: {e}")
        return None


async def main():
    """Main demonstration function."""
    print("🦀 Scribe-RS Python Bindings Demo")
    print("=" * 50)
    
    # Show library information
    info = get_info()
    print(f"📚 Scribe-RS version: {info['version']['version']}")
    print(f"🏗️ Build target: {info['build'].get('target', 'unknown')}")
    print(f"🚀 Capabilities: {', '.join(info['capabilities'].keys())}")
    
    # Get repository path from command line or use current directory
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        repo_path = os.getcwd()
    
    if not os.path.exists(repo_path):
        print(f"❌ Repository path does not exist: {repo_path}")
        return
    
    print(f"\n🎯 Target repository: {repo_path}")
    
    # Run all demonstrations
    await basic_repository_analysis(repo_path)
    await heuristic_scoring_demo(repo_path)
    await pagerank_analysis_demo(repo_path)
    await pattern_matching_demo(repo_path)
    await comprehensive_analysis_demo(repo_path)
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    # Run the async demo
    asyncio.run(main())