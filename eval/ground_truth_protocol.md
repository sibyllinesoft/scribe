# Ground-Truth Protocol for FastPath V5 Evaluation
**ICSE 2025 Submission - Workstream B: Reproducible File Relevance Annotation**

## Executive Summary

This protocol establishes a rigorous, reproducible methodology for generating high-quality ground-truth annotations for repository file relevance evaluation. The approach leverages Pull Request (PR) modified files as objective relevance signals, combined with systematic human annotation to achieve Cohen's κ ≥ 0.7 (substantial inter-rater agreement) as required for academic credibility.

## 1. Protocol Overview

### 1.1 Objective
Establish explicit, reproducible ground-truth labels for repository file relevance using:
- **Primary Signal**: PR-modified files as objective relevance indicators
- **Validation Layer**: Multi-annotator human judgment with statistical validation
- **Quality Gate**: Cohen's κ ≥ 0.7 inter-rater reliability threshold

### 1.2 Academic Rigor Requirements
- **Reproducibility**: All procedures must be fully replicable by independent researchers
- **Statistical Validation**: Inter-rater reliability measured using Cohen's kappa
- **Bias Mitigation**: Multiple annotation rounds with disagreement resolution
- **Audit Trail**: Complete documentation of all annotation decisions

## 2. Ground-Truth Generation Methodology

### 2.1 PR-Modified File Extraction Protocol

#### 2.1.1 Repository Selection Criteria
```yaml
selection_criteria:
  size_diversity:
    - small: 50-500 files
    - medium: 500-2000 files  
    - large: 2000-10000 files
    - very_large: >10000 files
  
  language_coverage:
    - typescript: web/backend systems
    - python: data science/backend
    - rust: systems programming
    - go: microservices/infrastructure
    - java: enterprise applications
  
  domain_diversity:
    - web_frameworks: React, Vue, Angular applications
    - backend_apis: REST/GraphQL services
    - data_processing: ETL, analytics pipelines
    - cli_tools: command-line utilities
    - libraries: reusable components/SDKs
    
  activity_requirements:
    - min_commits: 100
    - min_contributors: 5
    - active_period: last 12 months
    - min_prs: 25 with meaningful file modifications
```

#### 2.1.2 PR Analysis Procedure
```python
# Pseudo-code for PR analysis
def extract_pr_relevance_signals(repo, pr_id):
    """Extract objective relevance signals from PR modifications."""
    pr_data = fetch_pr(repo, pr_id)
    
    relevance_signals = {
        'modified_files': [],
        'change_patterns': {},
        'semantic_context': {}
    }
    
    for file_change in pr_data.files:
        signal = {
            'file_path': file_change.filename,
            'change_type': file_change.status,  # added/modified/deleted
            'lines_changed': file_change.additions + file_change.deletions,
            'change_density': calculate_change_density(file_change),
            'semantic_role': infer_semantic_role(file_change),
            'dependency_impact': analyze_dependency_impact(file_change)
        }
        relevance_signals['modified_files'].append(signal)
    
    return relevance_signals
```

#### 2.1.3 Relevance Score Calculation
Each PR-modified file receives an objective relevance score:

```yaml
relevance_scoring:
  primary_indicators:
    - change_density: weight=0.3  # lines changed / file size
    - semantic_importance: weight=0.25  # core logic vs utility
    - dependency_centrality: weight=0.25  # import/usage frequency
    - change_frequency: weight=0.2  # historical modification patterns
  
  relevance_tiers:
    - high: score >= 0.7 (definitely relevant)
    - medium: score 0.4-0.69 (likely relevant) 
    - low: score 0.2-0.39 (possibly relevant)
    - minimal: score < 0.2 (minimally relevant)
```

### 2.2 Human Annotation Framework

#### 2.2.1 Annotator Selection and Training

**Annotator Qualifications:**
- Minimum 2+ years software development experience
- Familiarity with target programming languages
- Understanding of software engineering practices
- No prior exposure to specific evaluation repositories

**Training Protocol:**
1. **Overview Session (30 min)**: Annotation goals and quality standards
2. **Practice Phase (60 min)**: Annotate 10 example scenarios with feedback
3. **Calibration Test (30 min)**: Annotate pre-scored examples, achieve ≥85% agreement
4. **Qualification Gate**: Must pass calibration before production annotation

#### 2.2.2 Annotation Task Design

**Task Structure:**
```yaml
annotation_task:
  context:
    repository_overview:
      - name: "example-web-app"
      - description: "React frontend with Node.js API"
      - primary_languages: ["typescript", "javascript"]
      - size_metrics: {files: 1247, lines: 45680}
    
    query_scenario:
      type: "feature_implementation"
      description: "Add user authentication system"
      specific_requirements:
        - login/logout functionality
        - session management
        - user profile editing
        - password reset flow
  
  annotation_items:
    - file_path: "src/components/AuthForm.tsx"
      content_preview: "React component for login form..."
      relevance_scale: [1, 2, 3, 4, 5]
      confidence_scale: [1, 2, 3, 4, 5]
      reasoning_required: true
```

**Relevance Scale Definition:**
- **5 - Critical**: File is essential for task completion
- **4 - High**: File likely requires modification or deep understanding
- **3 - Medium**: File provides important context or may need minor changes
- **2 - Low**: File provides some context but unlikely to need changes
- **1 - Minimal**: File is largely irrelevant to the task

#### 2.2.3 Multi-Round Annotation Process

**Round 1 - Independent Annotation:**
- Each annotator works independently
- No communication between annotators
- Complete reasoning documentation required
- Time limit: 45 minutes per 100 files

**Round 2 - Disagreement Resolution:**
- Identify files with score variance > 1.5
- Provide additional context from PR analysis
- Re-annotate disagreement cases
- Document reasoning for changes

**Round 3 - Final Validation:**
- Calculate inter-rater reliability
- Apply Cohen's κ ≥ 0.7 threshold
- If below threshold: additional annotator round
- Final consensus scoring for datasets

## 3. Statistical Validation Framework

### 3.1 Inter-Rater Reliability Calculation

#### 3.1.1 Cohen's Kappa Implementation
```python
def calculate_cohens_kappa(annotations_a, annotations_b):
    """Calculate Cohen's kappa for inter-rater reliability."""
    from sklearn.metrics import cohen_kappa_score
    
    # Convert 5-point scale to binary for primary analysis
    binary_a = [1 if score >= 3 else 0 for score in annotations_a]
    binary_b = [1 if score >= 3 else 0 for score in annotations_b]
    
    # Calculate both binary and weighted kappa
    kappa_binary = cohen_kappa_score(binary_a, binary_b)
    kappa_weighted = cohen_kappa_score(annotations_a, annotations_b, weights='linear')
    
    return {
        'binary_kappa': kappa_binary,
        'weighted_kappa': kappa_weighted,
        'interpretation': interpret_kappa(kappa_binary)
    }

def interpret_kappa(kappa):
    """Interpret Cohen's kappa according to Landis & Koch (1977)."""
    if kappa >= 0.81: return "almost_perfect"
    elif kappa >= 0.61: return "substantial" 
    elif kappa >= 0.41: return "moderate"
    elif kappa >= 0.21: return "fair"
    else: return "poor"
```

#### 3.1.2 Quality Gates
```yaml
quality_thresholds:
  minimum_acceptable:
    cohens_kappa: 0.70  # Substantial agreement
    confidence_score: 3.5  # Average annotator confidence
    completion_rate: 0.95  # Percentage of items annotated
  
  target_excellence:
    cohens_kappa: 0.80  # Near perfect agreement
    confidence_score: 4.0  # High annotator confidence
    completion_rate: 0.98  # Near complete annotation
    
  failure_conditions:
    cohens_kappa: < 0.60  # Requires additional training
    completion_rate: < 0.90  # Insufficient data coverage
```

### 3.2 Bias Detection and Mitigation

#### 3.2.1 Systematic Bias Analysis
```python
def analyze_annotation_bias(annotations, metadata):
    """Detect systematic biases in annotation patterns."""
    biases = {}
    
    # Language bias: Different scoring patterns by file type
    biases['language_bias'] = calculate_bias_by_language(annotations, metadata)
    
    # Size bias: Tendency to rate larger/smaller files differently
    biases['size_bias'] = calculate_bias_by_file_size(annotations, metadata)
    
    # Position bias: Rating files differently based on list position
    biases['position_bias'] = calculate_bias_by_position(annotations, metadata)
    
    # Annotator consistency: Individual scoring patterns
    biases['annotator_consistency'] = calculate_annotator_consistency(annotations)
    
    return biases
```

#### 3.2.2 Bias Mitigation Strategies
- **Randomized Presentation**: Shuffle file order for each annotator
- **Blinded Annotation**: Hide file names/paths when possible
- **Balanced Assignment**: Distribute file types evenly across annotators
- **Calibration Anchors**: Include pre-scored reference files in each batch

## 4. Quality Assurance Procedures

### 4.1 Annotation Quality Control

#### 4.1.1 Real-time Quality Monitoring
```python
class AnnotationQualityMonitor:
    def __init__(self, quality_thresholds):
        self.thresholds = quality_thresholds
        self.alerts = []
    
    def monitor_annotation_session(self, annotations):
        """Monitor annotation quality in real-time."""
        # Check annotation speed (too fast = low quality)
        if self.check_annotation_speed(annotations):
            self.alerts.append("annotation_speed_warning")
        
        # Check response variance (no variance = low engagement)
        if self.check_response_variance(annotations):
            self.alerts.append("low_variance_warning")
        
        # Check confidence alignment (low confidence + extreme scores)
        if self.check_confidence_alignment(annotations):
            self.alerts.append("confidence_misalignment")
        
        return self.alerts
```

#### 4.1.2 Post-Annotation Validation
```yaml
validation_checks:
  logical_consistency:
    - test_confidence_score_alignment
    - test_reasoning_completeness
    - test_score_justification_match
  
  statistical_validity:
    - test_score_distribution_normality
    - test_annotator_bias_detection
    - test_inter_rater_reliability_threshold
  
  completeness_validation:
    - test_all_items_annotated
    - test_reasoning_text_quality
    - test_confidence_score_provided
```

### 4.2 Data Integrity and Audit Trail

#### 4.2.1 Audit Trail Schema
```json
{
  "annotation_session": {
    "session_id": "ann_20250101_001",
    "annotator_id": "annotator_alpha_001",
    "start_time": "2025-01-01T10:00:00Z",
    "end_time": "2025-01-01T10:45:00Z",
    "repository": "example-web-app",
    "task_batch": "auth_implementation_001"
  },
  "annotations": [
    {
      "item_id": "file_001",
      "file_path": "src/components/AuthForm.tsx",
      "relevance_score": 5,
      "confidence_score": 4,
      "reasoning": "Central component for authentication UI...",
      "annotation_time": "2025-01-01T10:05:23Z",
      "revision_count": 0
    }
  ],
  "quality_metrics": {
    "completion_rate": 0.98,
    "average_confidence": 3.8,
    "annotation_speed_wpm": 45
  }
}
```

#### 4.2.2 Version Control and Reproducibility
```bash
# Annotation data versioning
git tag annotation-v1.0.0 "Initial annotation dataset - κ=0.73"
git tag annotation-v1.1.0 "Refined annotations after disagreement resolution - κ=0.78"

# Cryptographic integrity
sha256sum annotations_v1.0.0.json > annotations_v1.0.0.sha256
gpg --armor --detach-sig annotations_v1.0.0.json
```

## 5. Reproducibility Framework

### 5.1 Environment Specification

#### 5.1.1 Computational Environment
```yaml
environment:
  containerization:
    base_image: "python:3.11-slim"
    dependencies:
      - pandas==2.1.0
      - scikit-learn==1.3.0
      - numpy==1.24.3
      - jupyter==1.0.0
  
  annotation_platform:
    framework: "streamlit"
    version: "1.28.0"
    database: "postgresql:15"
    authentication: "oauth2"
  
  analysis_tools:
    statistics: "R:4.3.0"
    visualization: "matplotlib:3.7.0"
    reporting: "pandoc:3.1.0"
```

#### 5.1.2 Hermetic Execution Protocol
```python
def create_hermetic_environment():
    """Create reproducible annotation environment."""
    config = {
        'random_seed': 42,
        'python_version': '3.11.5',
        'package_versions': load_locked_requirements(),
        'dataset_hash': calculate_dataset_hash(),
        'annotation_config_hash': calculate_config_hash()
    }
    
    # Generate boot transcript
    boot_transcript = {
        'timestamp': datetime.utcnow().isoformat(),
        'environment_hash': hashlib.sha256(str(config).encode()).hexdigest(),
        'reproducibility_signature': sign_environment(config)
    }
    
    return config, boot_transcript
```

### 5.2 Artifact Generation and Verification

#### 5.2.1 Ground-Truth Dataset Schema
```json
{
  "dataset_metadata": {
    "version": "1.0.0",
    "creation_date": "2025-01-01T00:00:00Z",
    "protocol_version": "ground_truth_v1.0",
    "statistical_validation": {
      "cohens_kappa": 0.76,
      "annotator_count": 3,
      "item_count": 2847,
      "quality_gate_passed": true
    }
  },
  "repositories": [
    {
      "repo_id": "web_app_001",
      "name": "example-web-app", 
      "url": "https://github.com/example/web-app",
      "commit_hash": "abc123...",
      "language_primary": "typescript",
      "file_count": 1247,
      "annotation_tasks": 5
    }
  ],
  "annotations": [
    {
      "file_id": "file_001",
      "repository": "web_app_001", 
      "file_path": "src/components/AuthForm.tsx",
      "ground_truth_score": 4.7,
      "annotator_agreement": 0.85,
      "pr_signal_score": 0.82,
      "final_relevance_tier": "high"
    }
  ]
}
```

#### 5.2.2 Verification and Integrity Checks
```python
def verify_ground_truth_integrity(dataset_path):
    """Verify ground-truth dataset integrity and quality."""
    dataset = load_dataset(dataset_path)
    
    checks = {
        'schema_validation': validate_schema(dataset),
        'statistical_thresholds': check_kappa_threshold(dataset),
        'completeness_check': verify_annotation_completeness(dataset),
        'bias_analysis': detect_systematic_bias(dataset),
        'hash_verification': verify_cryptographic_hashes(dataset)
    }
    
    return all(checks.values()), checks
```

## 6. Implementation Timeline and Deliverables

### 6.1 Phase 1: Infrastructure Setup (Week 1-2)
- [ ] Develop PR analysis extraction tools
- [ ] Build annotation platform interface
- [ ] Create annotator training materials
- [ ] Establish quality monitoring systems

### 6.2 Phase 2: Pilot Annotation (Week 3-4)
- [ ] Recruit and train annotators
- [ ] Conduct pilot annotation on 2 repositories
- [ ] Validate Cohen's κ threshold achievement
- [ ] Refine annotation procedures based on pilot results

### 6.3 Phase 3: Full Dataset Creation (Week 5-8)
- [ ] Annotate complete repository dataset (≥5 repositories)
- [ ] Execute multi-round annotation with disagreement resolution
- [ ] Generate final ground-truth dataset with quality validation
- [ ] Create reproducibility artifacts and documentation

### 6.4 Phase 4: Validation and Documentation (Week 9-10)
- [ ] Independent verification of annotation quality
- [ ] Generate comprehensive audit trail and documentation
- [ ] Prepare ICSE submission materials
- [ ] Package reproducibility kit for peer review

## 7. Success Criteria and Risk Mitigation

### 7.1 Primary Success Criteria
- **Cohen's κ ≥ 0.70**: Achieve substantial inter-rater agreement
- **Dataset Coverage**: ≥5 repositories with diverse characteristics
- **Annotation Quality**: ≥95% completion rate with high confidence scores
- **Reproducibility**: Independent teams can replicate results

### 7.2 Risk Mitigation Strategies

#### 7.2.1 Low Inter-Rater Reliability Risk
**Mitigation:**
- Enhanced annotator training with domain experts
- Additional calibration rounds with feedback
- Refined annotation guidelines based on pilot results
- Emergency protocol: recruit additional expert annotators

#### 7.2.2 Annotator Availability/Quality Risk  
**Mitigation:**
- Over-recruit annotators (4+ for 3 needed)
- Implement real-time quality monitoring
- Backup annotation platform ready for deployment
- Budget contingency for expert annotator compensation

#### 7.2.3 Technical Infrastructure Risk
**Mitigation:**
- Containerized, version-controlled environment
- Automated backup and recovery procedures
- Multi-cloud deployment for redundancy
- Offline-capable annotation tools

## 8. Conclusion

This ground-truth protocol establishes a rigorous, academically sound methodology for generating high-quality file relevance annotations. The combination of objective PR signals and systematic human annotation, validated through statistical measures, provides the credibility required for ICSE submission while ensuring complete reproducibility for peer review validation.

The protocol's emphasis on bias mitigation, quality control, and comprehensive audit trails addresses the key concerns of the academic community regarding annotation reliability in software engineering research.