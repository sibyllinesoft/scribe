"""
Rule-based text centrality system (README++).

Implements sophisticated document prioritization without ML models:
- Filename-based priority signals
- Path depth and location analysis  
- Document structure scoring
- Cross-references and link analysis integration
- Reserves 15% of token budget for text layer
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..fastpath.fast_scan import ScanResult, DocumentAnalysis
from .link_graph import LinkAnalysisResult
from ..packer.tokenizer import estimate_tokens_scan_result


# Priority filename patterns (ordered by importance)
FILENAME_PRIORITIES = [
    # Tier 1: Must-have documentation (priority 3.0)
    ("readme", 3.0), ("read_me", 3.0), ("readme.md", 3.0), ("readme.txt", 3.0),
    
    # Tier 2: Architecture and design (priority 2.5)  
    ("architecture", 2.5), ("architecture.md", 2.5), ("design.md", 2.5),
    ("adr", 2.5), ("adrs", 2.5), ("decision-records", 2.5),
    
    # Tier 3: Getting started guides (priority 2.0)
    ("getting-started", 2.0), ("getting_started", 2.0), ("quickstart", 2.0),
    ("tutorial", 2.0), ("guide", 2.0), ("overview", 2.0),
    
    # Tier 4: Project information (priority 1.5)
    ("contributing", 1.5), ("contributing.md", 1.5), ("changelog", 1.5),
    ("api", 1.5), ("api.md", 1.5), ("specification", 1.5), ("spec.md", 1.5),
    
    # Tier 5: Development docs (priority 1.0)
    ("development", 1.0), ("developer", 1.0), ("dev", 1.0),
    ("setup", 1.0), ("installation", 1.0), ("install", 1.0),
]

# Path location boosters
PATH_BOOSTS = [
    # Root level gets maximum boost
    (lambda depth, parts: depth == 1, 2.0, "root_level"),
    
    # Documentation directories
    (lambda depth, parts: any(p.lower() in {'doc', 'docs', 'documentation'} for p in parts), 1.5, "docs_directory"),
    
    # Project root subdirectories
    (lambda depth, parts: depth == 2 and parts[0].lower() not in {'node_modules', 'build', 'dist', '.git'}, 1.2, "project_subdir"),
    
    # Guide/tutorial directories  
    (lambda depth, parts: any(p.lower() in {'guide', 'guides', 'tutorial', 'tutorials', 'examples'} for p in parts), 1.3, "guide_directory"),
    
    # API documentation
    (lambda depth, parts: any(p.lower() in {'api', 'reference', 'manual'} for p in parts), 1.1, "api_directory"),
]


@dataclass
class CentralityResult:
    """Result of text centrality analysis."""
    centrality_score: float
    filename_boost: float
    path_boost: float
    structure_score: float
    link_centrality: float
    final_priority: float
    priority_tier: str
    reasoning: List[str]


class TextPriorityScorer:
    """
    Rule-based document priority scorer for README++ system.
    
    Uses linguistic and structural signals to identify central documentation
    without requiring ML models or embeddings.
    """
    
    def __init__(self, token_budget_reserve: float = 0.15):
        self.token_budget_reserve = token_budget_reserve
        
        # Compile filename patterns for efficiency
        self.filename_patterns = []
        for pattern, priority in FILENAME_PRIORITIES:
            self.filename_patterns.append((pattern.lower(), priority))
            
    def _calculate_filename_boost(self, file_path: str) -> Tuple[float, List[str]]:
        """Calculate priority boost based on filename patterns."""
        path_obj = Path(file_path)
        filename = path_obj.name.lower()
        stem = path_obj.stem.lower()
        
        max_boost = 0.0
        reasoning = []
        
        # Check exact filename matches
        for pattern, priority in self.filename_patterns:
            if filename == pattern or stem == pattern:
                if priority > max_boost:
                    max_boost = priority
                    reasoning = [f"Exact filename match: {pattern} (priority {priority})"]
                    
        # Check partial matches for README-like files
        readme_indicators = ['readme', 'read_me', 'read-me']
        for indicator in readme_indicators:
            if indicator in filename:
                boost = 2.8  # Slightly less than exact match
                if boost > max_boost:
                    max_boost = boost
                    reasoning = [f"README indicator: {indicator} (priority {boost})"]
                    
        # Check for architecture/design keywords
        arch_keywords = ['architecture', 'design', 'adr', 'decision']
        for keyword in arch_keywords:
            if keyword in filename:
                boost = 2.2
                if boost > max_boost:
                    max_boost = boost
                    reasoning = [f"Architecture keyword: {keyword} (priority {boost})"]
                    
        return max_boost, reasoning
        
    def _calculate_path_boost(self, file_path: str) -> Tuple[float, List[str]]:
        """Calculate priority boost based on file location."""
        path_obj = Path(file_path)
        parts = list(path_obj.parts)
        depth = len(parts)
        
        total_boost = 1.0  # Base multiplier
        reasoning = []
        
        # Apply path-based boosts
        for condition, boost, description in PATH_BOOSTS:
            if condition(depth, parts):
                total_boost *= boost
                reasoning.append(f"{description}: {boost}x boost")
                
        return total_boost, reasoning
        
    def _calculate_structure_score(self, doc_analysis: Optional[DocumentAnalysis]) -> Tuple[float, List[str]]:
        """Calculate score based on document structure quality."""
        if not doc_analysis:
            return 0.0, ["No document analysis available"]
            
        score = 0.0
        reasoning = []
        
        # Heading structure indicates well-organized document
        if doc_analysis.heading_count > 0:
            # Optimal heading count is 3-10 for readability
            if 3 <= doc_analysis.heading_count <= 10:
                heading_score = 1.0
            elif doc_analysis.heading_count < 3:
                heading_score = doc_analysis.heading_count / 3.0
            else:
                # Diminishing returns for too many headings
                heading_score = 1.0 - (doc_analysis.heading_count - 10) * 0.05
                heading_score = max(heading_score, 0.2)
                
            score += heading_score * 0.4
            reasoning.append(f"Heading structure: {doc_analysis.heading_count} headings (score: {heading_score:.2f})")
            
        # Table of contents indicates comprehensive documentation
        if doc_analysis.toc_indicators > 0:
            toc_score = min(doc_analysis.toc_indicators * 0.5, 1.0)
            score += toc_score * 0.3
            reasoning.append(f"TOC indicators: {doc_analysis.toc_indicators} (score: {toc_score:.2f})")
            
        # Internal links indicate reference quality
        if doc_analysis.link_count > 0:
            # Optimal link count is 5-20 for reference docs
            if 5 <= doc_analysis.link_count <= 20:
                link_score = 1.0
            elif doc_analysis.link_count < 5:
                link_score = doc_analysis.link_count / 5.0
            else:
                link_score = 1.0 - (doc_analysis.link_count - 20) * 0.02
                link_score = max(link_score, 0.3)
                
            score += link_score * 0.2
            reasoning.append(f"Internal links: {doc_analysis.link_count} (score: {link_score:.2f})")
            
        # Code blocks in documentation indicate technical depth
        if doc_analysis.code_block_count > 0:
            code_score = min(doc_analysis.code_block_count / 5.0, 1.0)
            score += code_score * 0.1
            reasoning.append(f"Code blocks: {doc_analysis.code_block_count} (score: {code_score:.2f})")
            
        return min(score, 2.0), reasoning  # Cap at 2.0
        
    def _calculate_link_centrality(self, file_path: str, 
                                 link_analysis: Optional[LinkAnalysisResult]) -> Tuple[float, List[str]]:
        """Calculate centrality score based on link graph analysis."""
        if not link_analysis:
            return 0.0, ["No link analysis available"]
            
        score = 0.0
        reasoning = []
        
        # In-degree boost (how many files link to this one)
        in_degree = link_analysis.in_degree.get(file_path, 0)
        if in_degree > 0:
            # Logarithmic scaling for in-degree
            in_score = math.log(in_degree + 1) * 0.5
            score += min(in_score, 2.0)
            reasoning.append(f"In-degree centrality: {in_degree} links (score: {in_score:.2f})")
            
        # PageRank boost
        pagerank = link_analysis.pagerank_scores.get(file_path, 0.0)
        if pagerank > 0:
            # Normalize PageRank (typical values are small)
            pr_score = pagerank * 100  # Scale up for visibility
            score += min(pr_score, 1.0)
            reasoning.append(f"PageRank centrality: {pagerank:.4f} (score: {pr_score:.2f})")
            
        # Authority file boost
        if file_path in link_analysis.authority_files:
            score += 1.0
            reasoning.append("Authority file boost: +1.0")
            
        # Hub file boost (files that link to many others)
        if file_path in link_analysis.hub_files:
            score += 0.5
            reasoning.append("Hub file boost: +0.5")
            
        return min(score, 3.0), reasoning  # Cap at 3.0
        
    def score_document_centrality(self, result: ScanResult, 
                                 link_analysis: Optional[LinkAnalysisResult] = None) -> CentralityResult:
        """
        Calculate comprehensive centrality score for a document.
        
        Combines multiple rule-based signals to identify central documentation.
        """
        file_path = result.stats.path
        
        # Calculate individual components
        filename_boost, filename_reasoning = self._calculate_filename_boost(file_path)
        path_boost, path_reasoning = self._calculate_path_boost(file_path)
        structure_score, structure_reasoning = self._calculate_structure_score(result.doc_analysis)
        link_centrality, link_reasoning = self._calculate_link_centrality(file_path, link_analysis)
        
        # Base centrality score (weighted combination)
        centrality_score = (
            0.4 * filename_boost +      # Filename is most important
            0.2 * path_boost +          # Path location matters
            0.2 * structure_score +     # Document quality
            0.2 * link_centrality       # Network centrality
        )
        
        # Apply multipliers for exceptional cases
        final_priority = centrality_score
        
        # README files get additional multiplier
        if result.stats.is_readme:
            final_priority *= 1.5
            
        # Root-level files get boost
        if len(Path(file_path).parts) == 1:
            final_priority *= 1.2
            
        # Determine priority tier
        if final_priority >= 4.0:
            tier = "CRITICAL"
        elif final_priority >= 3.0:
            tier = "HIGH"
        elif final_priority >= 2.0:
            tier = "MEDIUM"
        elif final_priority >= 1.0:
            tier = "LOW"
        else:
            tier = "MINIMAL"
            
        # Combine all reasoning
        all_reasoning = []
        all_reasoning.extend(filename_reasoning)
        all_reasoning.extend(path_reasoning)
        all_reasoning.extend(structure_reasoning)
        all_reasoning.extend(link_reasoning)
        
        return CentralityResult(
            centrality_score=centrality_score,
            filename_boost=filename_boost,
            path_boost=path_boost,
            structure_score=structure_score,
            link_centrality=link_centrality,
            final_priority=final_priority,
            priority_tier=tier,
            reasoning=all_reasoning
        )
        
    def select_priority_documents(self, scan_results: List[ScanResult], 
                                 token_budget: int,
                                 link_analysis: Optional[LinkAnalysisResult] = None) -> List[Tuple[ScanResult, CentralityResult]]:
        """
        Select high-priority documents within text layer budget.
        
        Reserves specified percentage of token budget for documentation.
        """
        # Calculate text layer budget
        text_budget = int(token_budget * self.token_budget_reserve)
        
        # Score all documentation files
        doc_files = [result for result in scan_results if result.stats.is_docs]
        scored_docs = []
        
        for result in doc_files:
            centrality = self.score_document_centrality(result, link_analysis)
            scored_docs.append((result, centrality))
            
        # Sort by priority (descending)
        scored_docs.sort(key=lambda x: x[1].final_priority, reverse=True)
        
        # Select files within budget
        selected = []
        used_tokens = 0
        
        # Always include README files first (if they exist and fit)
        readme_files = [(r, c) for r, c in scored_docs if r.stats.is_readme]
        for result, centrality in readme_files:
            # Estimate tokens using centralized utility
            estimated_tokens = estimate_tokens_scan_result(result, use_lines=True)
            
            if used_tokens + estimated_tokens <= text_budget:
                selected.append((result, centrality))
                used_tokens += estimated_tokens
                
        # Add other high-priority documents
        remaining = [(r, c) for r, c in scored_docs if not r.stats.is_readme]
        for result, centrality in remaining:
            estimated_tokens = estimate_tokens_scan_result(result, use_lines=True)
            
            if used_tokens + estimated_tokens <= text_budget:
                selected.append((result, centrality))
                used_tokens += estimated_tokens
            else:
                break
                
        return selected
        
    def get_priority_statistics(self, scored_docs: List[Tuple[ScanResult, CentralityResult]]) -> Dict[str, any]:
        """Get statistics about document priority distribution."""
        if not scored_docs:
            return {}
            
        tier_counts = {}
        avg_scores = {}
        
        for result, centrality in scored_docs:
            tier = centrality.priority_tier
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            
            if tier not in avg_scores:
                avg_scores[tier] = []
            avg_scores[tier].append(centrality.final_priority)
            
        # Calculate averages
        tier_averages = {tier: sum(scores) / len(scores) 
                        for tier, scores in avg_scores.items()}
        
        return {
            'total_documents': len(scored_docs),
            'tier_distribution': tier_counts,
            'tier_average_scores': tier_averages,
            'readme_count': sum(1 for r, c in scored_docs if r.stats.is_readme),
            'highest_priority': max(c.final_priority for r, c in scored_docs),
            'lowest_priority': min(c.final_priority for r, c in scored_docs),
        }


def create_text_priority_scorer(budget_reserve: float = 0.15) -> TextPriorityScorer:
    """Create a text priority scorer instance."""
    return TextPriorityScorer(budget_reserve)