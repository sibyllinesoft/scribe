"""
Intra-repository markdown link analysis for Extended mode.

Builds link graph from markdown documents to identify central documentation:
- Parses internal links in markdown files
- Computes in-degree and PageRank-style centrality
- Identifies documentation hubs and authoritative sources
- Designed for <5s execution in Extended mode
"""

from __future__ import annotations

import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import unquote, urlparse

from ..fastpath.fast_scan import ScanResult


@dataclass
class LinkAnalysisResult:
    """Result of link analysis for document centrality."""
    in_degree: Dict[str, int]           # How many docs link to each file
    out_degree: Dict[str, int]          # How many links each file contains
    pagerank_scores: Dict[str, float]   # PageRank-style centrality scores
    authority_files: List[str]          # Files with high in-degree
    hub_files: List[str]               # Files with high out-degree
    link_clusters: Dict[str, List[str]] # Related document clusters


class LinkGraphAnalyzer:
    """
    Analyzes intra-repository links to identify central documentation.
    
    Focuses on markdown files and builds lightweight link graph for
    centrality analysis without heavy graph algorithms.
    """
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        
        # Link patterns for markdown
        self.link_patterns = [
            r'\[([^\]]+)\]\(([^)]+)\)',        # Standard markdown links
            r'\[([^\]]+)\]:\s*([^\s]+)',       # Reference-style links  
            r'<([^>]+\.md[^>]*)>',             # Direct markdown links
            r'!\[([^\]]*)\]\(([^)]+)\)',       # Image links (may point to docs)
        ]
        
        # Compile patterns
        self.compiled_patterns = [re.compile(pattern, re.MULTILINE) for pattern in self.link_patterns]
        
    def _extract_links_from_content(self, content: str, source_path: str) -> List[Tuple[str, str]]:
        """Extract all internal links from markdown content."""
        links = []
        
        for pattern in self.compiled_patterns:
            matches = pattern.findall(content)
            
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        link_text, link_url = match
                    else:
                        link_text = match[0] if match[0] else "unnamed"
                        link_url = match[1] if len(match) > 1 else match[0]
                else:
                    link_text = "unnamed"
                    link_url = match
                    
                # Process the link URL
                resolved_path = self._resolve_link_path(link_url, source_path)
                if resolved_path:
                    links.append((link_text, resolved_path))
                    
        return links
        
    def _resolve_link_path(self, link_url: str, source_path: str) -> Optional[str]:
        """Resolve a link URL to an absolute repository path."""
        # Skip external links
        if link_url.startswith(('http://', 'https://', 'ftp://', 'mailto:')):
            return None
            
        # Skip anchors without paths
        if link_url.startswith('#'):
            return None
            
        # Remove URL fragments and query parameters
        parsed = urlparse(link_url)
        clean_path = unquote(parsed.path)
        
        if not clean_path:
            return None
            
        # Resolve relative paths
        source_dir = Path(source_path).parent
        
        if clean_path.startswith('/'):
            # Absolute path from repo root
            target_path = self.repo_path / clean_path.lstrip('/')
        else:
            # Relative path from source file
            target_path = source_dir / clean_path
            
        try:
            # Normalize and make relative to repo root
            normalized = target_path.resolve()
            relative_path = normalized.relative_to(self.repo_path)
            
            # Only include existing files within the repository
            if normalized.exists() and normalized.is_file():
                return str(relative_path)
                
        except (ValueError, OSError):
            # Path is outside repository or invalid
            pass
            
        return None
        
    def _is_documentation_file(self, file_path: str) -> bool:
        """Check if file is a documentation file worth analyzing."""
        path_lower = file_path.lower()
        
        # Include markdown files
        if path_lower.endswith(('.md', '.markdown', '.rst')):
            return True
            
        # Include text files that look like documentation
        if path_lower.endswith('.txt') and any(
            doc_indicator in path_lower for doc_indicator in 
            ['readme', 'doc', 'guide', 'manual', 'help', 'faq']
        ):
            return True
            
        return False
        
    def analyze_link_graph(self, scan_results: List[ScanResult]) -> LinkAnalysisResult:
        """
        Analyze link graph from scan results.
        
        Builds link graph and computes centrality measures for documentation files.
        """
        # Filter to documentation files
        doc_files = [result for result in scan_results 
                    if self._is_documentation_file(result.stats.path)]
        
        if not doc_files:
            return LinkAnalysisResult({}, {}, {}, [], [], {})
            
        # Build link graph
        link_graph = defaultdict(list)  # source -> [targets]
        reverse_graph = defaultdict(list)  # target -> [sources]
        all_files = {result.stats.path for result in doc_files}
        
        for result in doc_files:
            source_path = result.stats.path
            
            # Read file content to extract links
            try:
                full_path = self.repo_path / source_path
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # Extract links from content
                links = self._extract_links_from_content(content, source_path)
                
                for link_text, target_path in links:
                    # Only include links to files in our document set
                    if target_path in all_files:
                        link_graph[source_path].append(target_path)
                        reverse_graph[target_path].append(source_path)
                        
            except (OSError, UnicodeDecodeError):
                # Skip files we can't read
                continue
                
        # Compute degree measures
        in_degree = {}
        out_degree = {}
        
        for file_path in all_files:
            in_degree[file_path] = len(reverse_graph[file_path])
            out_degree[file_path] = len(link_graph[file_path])
            
        # Compute simplified PageRank scores
        pagerank_scores = self._compute_pagerank(link_graph, reverse_graph, all_files)
        
        # Identify authority and hub files
        authority_files = self._identify_authorities(in_degree, pagerank_scores)
        hub_files = self._identify_hubs(out_degree, all_files)
        
        # Find document clusters
        link_clusters = self._find_clusters(link_graph, reverse_graph, all_files)
        
        return LinkAnalysisResult(
            in_degree=in_degree,
            out_degree=out_degree,
            pagerank_scores=pagerank_scores,
            authority_files=authority_files,
            hub_files=hub_files,
            link_clusters=link_clusters
        )
        
    def _compute_pagerank(self, link_graph: Dict[str, List[str]], 
                         reverse_graph: Dict[str, List[str]], 
                         all_files: Set[str], 
                         damping: float = 0.85, 
                         iterations: int = 10) -> Dict[str, float]:
        """Compute simplified PageRank scores (limited iterations for speed)."""
        num_files = len(all_files)
        if num_files == 0:
            return {}
            
        # Initialize scores uniformly
        scores = {file_path: 1.0 / num_files for file_path in all_files}
        
        # Power iteration (limited for speed)
        for _ in range(iterations):
            new_scores = {}
            
            for file_path in all_files:
                # Base score from random walk
                score = (1 - damping) / num_files
                
                # Add contributions from incoming links
                for source in reverse_graph[file_path]:
                    out_links = len(link_graph[source])
                    if out_links > 0:
                        score += damping * scores[source] / out_links
                        
                new_scores[file_path] = score
                
            scores = new_scores
            
        return scores
        
    def _identify_authorities(self, in_degree: Dict[str, int], 
                            pagerank_scores: Dict[str, float], 
                            top_k: int = 10) -> List[str]:
        """Identify authoritative files based on in-degree and PageRank."""
        # Combine in-degree and PageRank for authority score
        authority_scores = {}
        
        max_in_degree = max(in_degree.values()) if in_degree else 1
        max_pagerank = max(pagerank_scores.values()) if pagerank_scores else 1
        
        for file_path in in_degree:
            # Normalized combination of metrics
            norm_in_degree = in_degree[file_path] / max_in_degree
            norm_pagerank = pagerank_scores.get(file_path, 0) / max_pagerank
            
            authority_scores[file_path] = 0.6 * norm_in_degree + 0.4 * norm_pagerank
            
        # Return top authorities
        sorted_authorities = sorted(authority_scores.items(), 
                                  key=lambda x: x[1], reverse=True)
        return [file_path for file_path, score in sorted_authorities[:top_k]]
        
    def _identify_hubs(self, out_degree: Dict[str, int], 
                      all_files: Set[str], 
                      top_k: int = 10) -> List[str]:
        """Identify hub files with many outgoing links."""
        # Filter to files with significant out-degree
        hub_candidates = {file_path: degree for file_path, degree in out_degree.items() 
                         if degree >= 3}  # At least 3 outgoing links
        
        # Sort by out-degree
        sorted_hubs = sorted(hub_candidates.items(), key=lambda x: x[1], reverse=True)
        return [file_path for file_path, degree in sorted_hubs[:top_k]]
        
    def _find_clusters(self, link_graph: Dict[str, List[str]], 
                      reverse_graph: Dict[str, List[str]], 
                      all_files: Set[str]) -> Dict[str, List[str]]:
        """Find clusters of related documents using link patterns."""
        clusters = {}
        
        # Simple clustering based on mutual links and common neighbors
        for file_path in all_files:
            cluster = [file_path]
            
            # Add directly connected files
            connected = set()
            connected.update(link_graph[file_path])
            connected.update(reverse_graph[file_path])
            
            # Add files with common neighbors (simplified)
            for neighbor in list(connected):
                neighbor_links = set(link_graph[neighbor]) | set(reverse_graph[neighbor])
                current_links = set(link_graph[file_path]) | set(reverse_graph[file_path])
                
                # If significant overlap, include in cluster
                overlap = len(neighbor_links & current_links)
                if overlap >= 2:  # At least 2 common connections
                    cluster.append(neighbor)
                    
            # Only include non-trivial clusters
            if len(cluster) > 1:
                clusters[file_path] = cluster
                
        return clusters


def create_link_analyzer(repo_path: Path) -> LinkGraphAnalyzer:
    """Create a link graph analyzer instance."""
    return LinkGraphAnalyzer(repo_path)