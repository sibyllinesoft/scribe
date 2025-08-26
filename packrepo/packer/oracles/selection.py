"""Selection properties validation oracle for PackRepo."""

from __future__ import annotations

import time
import numpy as np
from typing import Dict, Any, Optional, List, Set, Tuple
from collections import defaultdict

from . import Oracle, OracleReport, OracleResult
from ..packfmt.base import PackFormat


class SelectionPropertiesOracle(Oracle):
    """Oracle for validating selection algorithm properties."""
    
    category = "selection"
    
    def name(self) -> str:
        return "selection_properties"
    
    def validate(self, pack: PackFormat, context: Optional[Dict[str, Any]] = None) -> OracleReport:
        """Validate selection algorithm properties and quality metrics."""
        start_time = time.time()
        
        try:
            errors = []
            details = {}
            
            index = pack.index
            chunks = index.chunks or []
            
            details["selected_chunks"] = len(chunks)
            details["coverage_score"] = index.coverage_score
            details["diversity_score"] = index.diversity_score
            
            if not chunks:
                return OracleReport(
                    oracle_name=self.name(),
                    result=OracleResult.SKIP,
                    message="No chunks selected, skipping selection properties validation",
                    details=details,
                    execution_time=time.time() - start_time
                )
            
            # Validate coverage score is reasonable
            coverage_score = index.coverage_score
            if coverage_score < 0 or coverage_score > 1:
                errors.append(f"Invalid coverage score: {coverage_score} (must be 0-1)")
            
            # Validate diversity score is reasonable
            diversity_score = index.diversity_score
            if diversity_score < 0 or diversity_score > 1:
                errors.append(f"Invalid diversity score: {diversity_score} (must be 0-1)")
            
            # Validate facility location properties
            file_coverage = self._check_file_coverage(chunks)
            details.update(file_coverage)
            
            if file_coverage["unique_files"] == 0:
                errors.append("No files covered by selection")
            elif file_coverage["coverage_ratio"] < 0.1:  # Less than 10% of available files
                errors.append(f"Very low file coverage: {file_coverage['coverage_ratio']:.1%}")
            
            # Validate MMR diversity properties
            diversity_check = self._check_diversity_properties(chunks)
            details.update(diversity_check)
            
            if diversity_check["avg_similarity"] > 0.9:  # Very high similarity
                errors.append(f"Low diversity: average chunk similarity {diversity_check['avg_similarity']:.1%}")
            
            # Validate selection scores are monotone (if available)
            if context and "selection_scores" in context:
                monotone_check = self._check_monotone_scores(context["selection_scores"])
                details.update(monotone_check)
                
                if not monotone_check["is_monotone"]:
                    errors.append("Selection scores not monotonically decreasing")
            
            # Validate chunk importance distribution
            importance_check = self._check_importance_distribution(chunks)
            details.update(importance_check)
            
            # Check for selection bias issues
            bias_check = self._check_selection_bias(chunks)
            details.update(bias_check)
            
            for bias_error in bias_check["bias_issues"]:
                errors.append(bias_error)
            
            # Overall assessment
            if errors:
                result = OracleResult.FAIL
                message = f"Selection properties validation failed: {len(errors)} issues found"
                details["errors"] = errors[:5]  # Show first 5 errors
            else:
                result = OracleResult.PASS
                message = f"Selection properties validated: {len(chunks)} chunks with good coverage ({file_coverage['coverage_ratio']:.1%}) and diversity"
            
            return OracleReport(
                oracle_name=self.name(),
                result=result,
                message=message,
                details=details,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return OracleReport(
                oracle_name=self.name(),
                result=OracleResult.ERROR,
                message=f"Selection properties oracle failed: {str(e)}",
                details={"exception": type(e).__name__, "error": str(e)},
                execution_time=time.time() - start_time
            )
    
    def _check_file_coverage(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check file coverage properties."""
        files = set()
        file_chunk_counts = defaultdict(int)
        
        for chunk in chunks:
            file_path = chunk.get('rel_path', '')
            if file_path:
                files.add(file_path)
                file_chunk_counts[file_path] += 1
        
        unique_files = len(files)
        max_chunks_per_file = max(file_chunk_counts.values()) if file_chunk_counts else 0
        avg_chunks_per_file = len(chunks) / max(1, unique_files)
        
        # Estimate total files if provided in context
        total_files = len(file_chunk_counts)  # Minimum estimate
        coverage_ratio = unique_files / max(1, total_files)
        
        return {
            "unique_files": unique_files,
            "max_chunks_per_file": max_chunks_per_file,
            "avg_chunks_per_file": avg_chunks_per_file,
            "coverage_ratio": coverage_ratio,
            "file_distribution": dict(list(file_chunk_counts.items())[:5])  # Sample
        }
    
    def _check_diversity_properties(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check diversity properties of selected chunks."""
        if len(chunks) <= 1:
            return {"avg_similarity": 0.0, "diversity_issues": []}
        
        # Simple diversity check based on available metadata
        file_types = defaultdict(int)
        chunk_kinds = defaultdict(int)
        
        for chunk in chunks:
            file_path = chunk.get('rel_path', '')
            chunk_kind = chunk.get('kind', 'unknown')
            
            # Extract file extension
            if '.' in file_path:
                ext = file_path.split('.')[-1].lower()
                file_types[ext] += 1
            
            chunk_kinds[chunk_kind] += 1
        
        # Diversity metrics
        unique_file_types = len(file_types)
        unique_chunk_kinds = len(chunk_kinds)
        
        # Estimate average similarity (heuristic based on file types and kinds)
        total_chunks = len(chunks)
        dominant_type_count = max(file_types.values()) if file_types else total_chunks
        dominant_kind_count = max(chunk_kinds.values()) if chunk_kinds else total_chunks
        
        # Higher values = less diversity
        type_concentration = dominant_type_count / total_chunks
        kind_concentration = dominant_kind_count / total_chunks
        
        avg_similarity = (type_concentration + kind_concentration) / 2
        
        return {
            "avg_similarity": avg_similarity,
            "unique_file_types": unique_file_types,
            "unique_chunk_kinds": unique_chunk_kinds,
            "type_distribution": dict(file_types),
            "kind_distribution": dict(chunk_kinds)
        }
    
    def _check_monotone_scores(self, selection_scores: Dict[str, float]) -> Dict[str, Any]:
        """Check if selection scores are monotonically decreasing."""
        if not selection_scores:
            return {"is_monotone": True, "violations": 0}
        
        scores = list(selection_scores.values())
        violations = 0
        
        for i in range(1, len(scores)):
            if scores[i] > scores[i-1]:  # Should be non-increasing
                violations += 1
        
        is_monotone = violations == 0
        
        return {
            "is_monotone": is_monotone,
            "violations": violations,
            "score_range": [min(scores), max(scores)] if scores else [0, 0]
        }
    
    def _check_importance_distribution(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check distribution of chunk importance scores."""
        importance_scores = []
        
        for chunk in chunks:
            # Try different importance score field names
            score = chunk.get('importance_score') or chunk.get('centrality_score') or chunk.get('score', 0.0)
            if isinstance(score, (int, float)):
                importance_scores.append(float(score))
        
        if not importance_scores:
            return {"has_importance_scores": False}
        
        scores_array = np.array(importance_scores)
        
        return {
            "has_importance_scores": True,
            "mean_importance": float(np.mean(scores_array)),
            "std_importance": float(np.std(scores_array)),
            "min_importance": float(np.min(scores_array)),
            "max_importance": float(np.max(scores_array)),
            "score_distribution_quartiles": [
                float(np.percentile(scores_array, q)) for q in [25, 50, 75]
            ]
        }
    
    def _check_selection_bias(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for potential selection biases."""
        bias_issues = []
        
        # Check for file size bias
        token_counts = [chunk.get('selected_tokens', 0) for chunk in chunks]
        if token_counts:
            min_tokens = min(token_counts)
            max_tokens = max(token_counts)
            
            if min_tokens > 0 and (max_tokens / min_tokens) > 100:  # Very large variance
                bias_issues.append(f"Extreme token count variance: {min_tokens}-{max_tokens}")
        
        # Check for path bias (e.g., only selecting from certain directories)
        paths = [chunk.get('rel_path', '') for chunk in chunks]
        root_dirs = set()
        
        for path in paths:
            if '/' in path:
                root_dir = path.split('/')[0]
                root_dirs.add(root_dir)
        
        if len(root_dirs) == 1 and len(chunks) > 5:
            bias_issues.append(f"All chunks from single directory: {list(root_dirs)[0]}")
        
        # Check for file type bias
        extensions = defaultdict(int)
        for path in paths:
            if '.' in path:
                ext = path.split('.')[-1].lower()
                extensions[ext] += 1
        
        if extensions:
            dominant_ext = max(extensions, key=extensions.get)
            if extensions[dominant_ext] / len(chunks) > 0.8:  # >80% of one type
                bias_issues.append(f"Dominant file type bias: {dominant_ext} ({extensions[dominant_ext]}/{len(chunks)})")
        
        return {
            "bias_issues": bias_issues,
            "token_range": [min(token_counts), max(token_counts)] if token_counts else [0, 0],
            "unique_root_dirs": len(root_dirs),
            "extension_distribution": dict(extensions)
        }


class BudgetEfficiencyOracle(Oracle):
    """Oracle for validating budget utilization efficiency."""
    
    category = "selection"
    
    def name(self) -> str:
        return "budget_efficiency"
    
    def validate(self, pack: PackFormat, context: Optional[Dict[str, Any]] = None) -> OracleReport:
        """Validate efficient use of token budget."""
        start_time = time.time()
        
        try:
            errors = []
            details = {}
            
            index = pack.index
            target_budget = index.target_budget
            actual_tokens = index.actual_tokens
            chunks = index.chunks or []
            
            if target_budget <= 0:
                return OracleReport(
                    oracle_name=self.name(),
                    result=OracleResult.SKIP,
                    message="No target budget set",
                    details=details,
                    execution_time=time.time() - start_time
                )
            
            utilization = actual_tokens / target_budget
            details["utilization"] = utilization
            details["target_budget"] = target_budget
            details["actual_tokens"] = actual_tokens
            
            # Check utilization efficiency
            if utilization < 0.5:
                errors.append(f"Very low budget utilization: {utilization:.1%}")
            elif utilization < 0.8:
                # This might be acceptable, but worth noting
                details["low_utilization_warning"] = True
            
            # Analyze token efficiency per chunk
            if chunks:
                token_counts = [chunk.get('selected_tokens', 0) for chunk in chunks]
                details["avg_tokens_per_chunk"] = np.mean(token_counts) if token_counts else 0
                details["token_efficiency_distribution"] = {
                    "min": min(token_counts) if token_counts else 0,
                    "max": max(token_counts) if token_counts else 0,
                    "std": float(np.std(token_counts)) if token_counts else 0
                }
                
                # Check for very small chunks that might indicate poor selection
                small_chunks = [t for t in token_counts if t < 50]  # Arbitrary threshold
                if len(small_chunks) > len(chunks) * 0.3:  # >30% very small chunks
                    errors.append(f"Many very small chunks: {len(small_chunks)}/{len(chunks)} < 50 tokens")
            
            # Check budget utilization pattern
            expected_utilization_min = 0.95  # Should use at least 95% of budget
            expected_utilization_max = 1.0   # Should not exceed budget
            
            if utilization < expected_utilization_min:
                underutilization = expected_utilization_min - utilization
                if underutilization > 0.05:  # More than 5% underutilization
                    errors.append(f"Significant budget underutilization: {underutilization:.1%} unused")
            
            # Overall assessment
            if errors:
                result = OracleResult.FAIL
                message = f"Budget efficiency issues: {'; '.join(errors)}"
            else:
                result = OracleResult.PASS
                message = f"Budget efficiently utilized: {utilization:.1%} of {target_budget} tokens"
            
            return OracleReport(
                oracle_name=self.name(),
                result=result,
                message=message,
                details=details,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return OracleReport(
                oracle_name=self.name(),
                result=OracleResult.ERROR,
                message=f"Budget efficiency oracle failed: {str(e)}",
                details={"exception": type(e).__name__, "error": str(e)},
                execution_time=time.time() - start_time
            )