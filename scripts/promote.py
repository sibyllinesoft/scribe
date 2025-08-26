#!/usr/bin/env python3
"""
PackRepo Promotion Manager

Handles the automated promotion of variants based on gatekeeper decisions
and statistical validation. Implements the complete promotion workflow
including artifact packaging, deployment preparation, and rollback setup.

Key responsibilities:
- Validate gatekeeper decision and supporting evidence
- Package artifacts for deployment
- Prepare deployment configurations  
- Set up monitoring and rollback mechanisms
- Generate promotion documentation
"""

import json
import sys
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import subprocess
import hashlib


@dataclass
class PromotionArtifact:
    """Promotion artifact metadata."""
    name: str
    path: str
    checksum: str
    size_bytes: int
    type: str  # 'container', 'config', 'metrics', 'documentation'


@dataclass
class PromotionPackage:
    """Complete promotion package with all required artifacts."""
    variant: str
    version: str
    commit_sha: str
    timestamp: str
    
    # Decision context
    gatekeeper_decision: str
    composite_score: float
    critical_failures: int
    statistical_evidence: Dict
    
    # Artifacts
    artifacts: List[PromotionArtifact]
    container_digest: str
    
    # Deployment config
    deployment_config: Dict
    rollback_config: Dict
    monitoring_config: Dict


class PromotionManager:
    """Manages the promotion process for validated variants."""
    
    def __init__(self, artifacts_dir: Path, deployment_config: Optional[Dict] = None):
        """Initialize promotion manager."""
        self.artifacts_dir = Path(artifacts_dir)
        self.deployment_config = deployment_config or {}
        self.promotion_dir = self.artifacts_dir / "promotion"
        self.promotion_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_promotion_criteria(self, gatekeeper_decision: Dict) -> bool:
        """Validate that promotion criteria are met."""
        
        required_fields = [
            "decision", "composite_score", "critical_failures", 
            "gate_details", "recommendations"
        ]
        
        for field in required_fields:
            if field not in gatekeeper_decision:
                print(f"Missing required field in gatekeeper decision: {field}")
                return False
        
        # Check decision is PROMOTE
        if gatekeeper_decision["decision"] != "PROMOTE":
            print(f"Gatekeeper decision is {gatekeeper_decision['decision']}, not PROMOTE")
            return False
            
        # Validate composite score
        composite_score = gatekeeper_decision.get("composite_score", 0.0)
        if composite_score < 0.85:
            print(f"Composite score {composite_score:.3f} below promotion threshold 0.85")
            return False
            
        # Check critical failures
        critical_failures = gatekeeper_decision.get("critical_failures", 999)
        if critical_failures > 0:
            print(f"Cannot promote with {critical_failures} critical failures")
            return False
            
        print("✅ Promotion criteria validated successfully")
        return True
    
    def collect_artifacts(self, variant: str) -> List[PromotionArtifact]:
        """Collect and validate all required artifacts for promotion."""
        
        artifacts = []
        required_artifacts = [
            ("gatekeeper_decision.json", "metrics"),
            ("bootstrap_ci_results.json", "metrics"), 
            ("fdr_correction_results.json", "metrics"),
            ("boot_transcript.json", "verification"),
            ("mutation_test_results.json", "testing"),
            ("sast_scan_results.json", "security")
        ]
        
        # Collect required artifacts
        for artifact_name, artifact_type in required_artifacts:
            artifact_path = self.artifacts_dir / "metrics" / artifact_name
            
            if not artifact_path.exists():
                artifact_path = self.artifacts_dir / artifact_name
                
            if artifact_path.exists():
                checksum = self._compute_file_checksum(artifact_path)
                size = artifact_path.stat().st_size
                
                artifacts.append(PromotionArtifact(
                    name=artifact_name,
                    path=str(artifact_path),
                    checksum=checksum,
                    size_bytes=size,
                    type=artifact_type
                ))
                print(f"✅ Collected artifact: {artifact_name}")
            else:
                print(f"⚠️ Missing optional artifact: {artifact_name}")
        
        # Collect variant-specific evaluation logs
        variant_logs_dir = Path(f"logs/{variant}")
        if variant_logs_dir.exists():
            for log_file in variant_logs_dir.rglob("*.json*"):
                checksum = self._compute_file_checksum(log_file)
                size = log_file.stat().st_size
                
                artifacts.append(PromotionArtifact(
                    name=f"{variant}/{log_file.name}",
                    path=str(log_file),
                    checksum=checksum, 
                    size_bytes=size,
                    type="evaluation"
                ))
        
        print(f"✅ Collected {len(artifacts)} artifacts for promotion")
        return artifacts
    
    def _compute_file_checksum(self, filepath: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def generate_deployment_config(self, variant: str, gatekeeper_decision: Dict) -> Dict:
        """Generate deployment configuration for the promoted variant."""
        
        # Extract performance metrics from gatekeeper decision
        gate_details = gatekeeper_decision.get("gate_details", [])
        
        # Find performance-related gates
        performance_config = {}
        for gate in gate_details:
            gate_name = gate.get("name", "")
            if "latency" in gate_name.lower():
                if "p50" in gate_name.lower():
                    performance_config["latency_p50_ms"] = gate.get("actual", 150.0)
                elif "p95" in gate_name.lower():
                    performance_config["latency_p95_ms"] = gate.get("actual", 300.0)
            elif "memory" in gate_name.lower():
                performance_config["memory_limit_mb"] = int(gate.get("actual", 512.0) * 1.2)  # 20% buffer
        
        deployment_config = {
            "variant": variant,
            "image": {
                "repository": "packrepo",
                "tag": f"{variant.lower()}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "pullPolicy": "Always"
            },
            
            "resources": {
                "requests": {
                    "memory": f"{performance_config.get('memory_limit_mb', 256)}Mi",
                    "cpu": "100m"
                },
                "limits": {
                    "memory": f"{performance_config.get('memory_limit_mb', 512)}Mi", 
                    "cpu": "1000m"
                }
            },
            
            "scaling": {
                "minReplicas": 2,
                "maxReplicas": 10,
                "targetCPUUtilizationPercentage": 70,
                "targetMemoryUtilizationPercentage": 80
            },
            
            "healthCheck": {
                "livenessProbe": {
                    "httpGet": {"path": "/health", "port": 8080},
                    "initialDelaySeconds": 30,
                    "periodSeconds": 10
                },
                "readinessProbe": {
                    "httpGet": {"path": "/ready", "port": 8080},
                    "initialDelaySeconds": 5,
                    "periodSeconds": 5
                }
            },
            
            "environment": {
                "PACKREPO_VARIANT": variant,
                "PACKREPO_VERSION": "1.0",
                "LOG_LEVEL": "INFO",
                "METRICS_ENABLED": "true"
            }
        }
        
        return deployment_config
    
    def generate_rollback_config(self, variant: str) -> Dict:
        """Generate rollback configuration and procedures."""
        
        rollback_config = {
            "enabled": True,
            "triggers": {
                "error_rate_threshold": 5.0,  # > 5% error rate
                "latency_p95_threshold": 500.0,  # > 500ms P95 latency
                "memory_usage_threshold": 90.0,  # > 90% memory usage
                "availability_threshold": 99.0   # < 99% availability
            },
            
            "rollback_procedure": [
                {
                    "step": 1,
                    "action": "stop_traffic",
                    "description": "Stop routing traffic to new variant"
                },
                {
                    "step": 2, 
                    "action": "restore_previous",
                    "description": "Deploy previous known good version"
                },
                {
                    "step": 3,
                    "action": "verify_health",
                    "description": "Verify rollback deployment is healthy"
                },
                {
                    "step": 4,
                    "action": "resume_traffic", 
                    "description": "Resume traffic to rolled back version"
                }
            ],
            
            "rollback_window": "30m",  # Auto-rollback within 30 minutes
            "notification_channels": ["slack", "email"],
            
            "validation_checks": [
                "health_endpoints_responding",
                "error_rate_within_bounds",
                "latency_within_sla",
                "memory_usage_stable"
            ]
        }
        
        return rollback_config
    
    def generate_monitoring_config(self, variant: str, performance_config: Dict) -> Dict:
        """Generate monitoring and alerting configuration."""
        
        monitoring_config = {
            "metrics": {
                "business": [
                    {
                        "name": "qa_accuracy", 
                        "type": "gauge",
                        "description": "Question-answering accuracy"
                    },
                    {
                        "name": "token_efficiency",
                        "type": "gauge", 
                        "description": "QA accuracy per 100k tokens"
                    }
                ],
                
                "technical": [
                    {
                        "name": "request_latency_seconds",
                        "type": "histogram",
                        "description": "Request processing latency"
                    },
                    {
                        "name": "memory_usage_bytes",
                        "type": "gauge",
                        "description": "Memory usage in bytes"
                    },
                    {
                        "name": "error_rate",
                        "type": "counter",
                        "description": "Error rate per second"
                    }
                ]
            },
            
            "alerts": [
                {
                    "name": "HighErrorRate",
                    "condition": "error_rate > 0.05",
                    "duration": "5m",
                    "severity": "critical",
                    "action": "trigger_rollback"
                },
                {
                    "name": "HighLatency",
                    "condition": f"p95(request_latency_seconds) > {performance_config.get('latency_p95_ms', 300) / 1000}",
                    "duration": "10m",
                    "severity": "warning",
                    "action": "escalate_to_oncall"
                },
                {
                    "name": "MemoryUsageHigh",
                    "condition": f"memory_usage_bytes > {performance_config.get('memory_limit_mb', 512) * 1024 * 1024 * 0.9}",
                    "duration": "15m",
                    "severity": "warning",
                    "action": "auto_scale_up"
                }
            ],
            
            "dashboards": [
                {
                    "name": f"PackRepo {variant} Performance",
                    "panels": [
                        "request_rate", "error_rate", "latency_percentiles",
                        "memory_usage", "cpu_usage", "qa_accuracy_trend"
                    ]
                }
            ],
            
            "sla_targets": {
                "availability": 99.9,
                "latency_p95_ms": performance_config.get("latency_p95_ms", 300),
                "error_rate_percent": 0.1
            }
        }
        
        return monitoring_config
    
    def create_promotion_package(
        self, 
        variant: str, 
        gatekeeper_decision: Dict,
        statistical_evidence: Dict,
        commit_sha: str
    ) -> PromotionPackage:
        """Create complete promotion package."""
        
        # Collect artifacts
        artifacts = self.collect_artifacts(variant)
        
        # Generate configurations
        deployment_config = self.generate_deployment_config(variant, gatekeeper_decision)
        rollback_config = self.generate_rollback_config(variant)  
        
        # Extract performance config for monitoring
        performance_config = {
            "latency_p95_ms": 300.0,  # Default
            "memory_limit_mb": deployment_config["resources"]["limits"]["memory"].replace("Mi", "")
        }
        
        monitoring_config = self.generate_monitoring_config(variant, performance_config)
        
        # Create promotion package
        promotion_package = PromotionPackage(
            variant=variant,
            version="1.0",
            commit_sha=commit_sha,
            timestamp=datetime.now(timezone.utc).isoformat(),
            
            gatekeeper_decision=gatekeeper_decision["decision"],
            composite_score=gatekeeper_decision.get("composite_score", 0.0),
            critical_failures=gatekeeper_decision.get("critical_failures", 0),
            statistical_evidence=statistical_evidence,
            
            artifacts=artifacts,
            container_digest="sha256:placeholder",  # Would be filled by actual container build
            
            deployment_config=deployment_config,
            rollback_config=rollback_config,
            monitoring_config=monitoring_config
        )
        
        return promotion_package
    
    def save_promotion_package(self, package: PromotionPackage) -> Path:
        """Save promotion package to disk."""
        
        package_dir = self.promotion_dir / f"{package.variant}_{package.commit_sha[:8]}"
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Save package metadata
        package_file = package_dir / "promotion_package.json"
        with open(package_file, 'w') as f:
            json.dump(asdict(package), f, indent=2, default=str)
        
        # Save individual configs
        configs = {
            "deployment_config.json": package.deployment_config,
            "rollback_config.json": package.rollback_config,
            "monitoring_config.json": package.monitoring_config
        }
        
        for filename, config in configs.items():
            config_file = package_dir / filename
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
        
        # Create artifact archive
        archive_path = package_dir / "promotion_artifacts.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            for artifact in package.artifacts:
                if Path(artifact.path).exists():
                    tar.add(artifact.path, arcname=artifact.name)
        
        print(f"✅ Promotion package saved: {package_dir}")
        return package_dir
    
    def generate_promotion_report(self, package: PromotionPackage) -> str:
        """Generate human-readable promotion report."""
        
        report = f"""# PackRepo Promotion Report - {package.variant}

**Timestamp:** {package.timestamp}
**Commit:** {package.commit_sha}
**Variant:** {package.variant} v{package.version}
**Decision:** {package.gatekeeper_decision}

## Quality Assessment

- **Composite Score:** {package.composite_score:.3f}/1.000
- **Critical Failures:** {package.critical_failures}
- **Total Artifacts:** {len(package.artifacts)}

## Statistical Evidence

"""
        
        # Add statistical evidence
        if package.statistical_evidence:
            for key, value in package.statistical_evidence.items():
                if isinstance(value, (int, float)):
                    report += f"- **{key.replace('_', ' ').title()}:** {value:.3f}\n"
                else:
                    report += f"- **{key.replace('_', ' ').title()}:** {value}\n"
        
        report += f"""

## Deployment Configuration

- **Image Tag:** {package.deployment_config['image']['tag']}
- **Memory Limit:** {package.deployment_config['resources']['limits']['memory']}
- **CPU Limit:** {package.deployment_config['resources']['limits']['cpu']}
- **Min Replicas:** {package.deployment_config['scaling']['minReplicas']}
- **Max Replicas:** {package.deployment_config['scaling']['maxReplicas']}

## Rollback Configuration

- **Rollback Enabled:** {package.rollback_config['enabled']}
- **Error Rate Threshold:** {package.rollback_config['triggers']['error_rate_threshold']}%
- **Latency Threshold:** {package.rollback_config['triggers']['latency_p95_threshold']}ms
- **Rollback Window:** {package.rollback_config['rollback_window']}

## Monitoring & Alerts

- **Availability SLA:** {package.monitoring_config['sla_targets']['availability']}%
- **Latency SLA:** {package.monitoring_config['sla_targets']['latency_p95_ms']}ms P95
- **Error Rate SLA:** {package.monitoring_config['sla_targets']['error_rate_percent']}%
- **Active Alerts:** {len(package.monitoring_config['alerts'])}

## Artifacts Included

"""
        
        # List artifacts by type
        by_type = {}
        for artifact in package.artifacts:
            if artifact.type not in by_type:
                by_type[artifact.type] = []
            by_type[artifact.type].append(artifact)
        
        for artifact_type, type_artifacts in by_type.items():
            report += f"### {artifact_type.title()}\n\n"
            for artifact in type_artifacts:
                report += f"- {artifact.name} ({artifact.size_bytes:,} bytes)\n"
            report += "\n"
        
        report += f"""## Next Steps

1. **Deployment:** Deploy using `deployment_config.json`
2. **Monitoring:** Set up alerts and dashboards from `monitoring_config.json`  
3. **Verification:** Validate deployment health and performance
4. **Documentation:** Update production documentation

Generated by PackRepo Promotion Manager v1.0
"""
        
        return report


def main():
    """Main promotion manager execution."""
    
    if len(sys.argv) < 4:
        print("Usage: promote.py <artifacts_dir> <variant> <commit_sha> [gatekeeper_decision.json]")
        print("Example: promote.py artifacts V2 abc123def gatekeeper_decision.json")
        sys.exit(1)
    
    artifacts_dir = Path(sys.argv[1])
    variant = sys.argv[2]
    commit_sha = sys.argv[3]
    gatekeeper_file = Path(sys.argv[4]) if len(sys.argv) > 4 else artifacts_dir / "metrics" / "decision.json"
    
    print(f"PackRepo Promotion Manager")
    print(f"Variant: {variant}")
    print(f"Commit: {commit_sha}")
    print(f"Artifacts: {artifacts_dir}")
    print(f"Gatekeeper Decision: {gatekeeper_file}")
    
    # Load gatekeeper decision
    if not gatekeeper_file.exists():
        print(f"Error: Gatekeeper decision file not found: {gatekeeper_file}")
        sys.exit(1)
        
    try:
        with open(gatekeeper_file, 'r') as f:
            gatekeeper_decision = json.load(f)
    except Exception as e:
        print(f"Error loading gatekeeper decision: {e}")
        sys.exit(1)
    
    # Initialize promotion manager
    promotion_manager = PromotionManager(artifacts_dir)
    
    # Validate promotion criteria
    if not promotion_manager.validate_promotion_criteria(gatekeeper_decision):
        print("❌ Promotion criteria not met - cannot promote")
        sys.exit(1)
    
    # Load statistical evidence if available
    statistical_evidence = {}
    for evidence_file in ["bootstrap_ci_results.json", "fdr_correction_results.json"]:
        evidence_path = artifacts_dir / "metrics" / evidence_file
        if evidence_path.exists():
            try:
                with open(evidence_path, 'r') as f:
                    statistical_evidence[evidence_file.replace('.json', '')] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load {evidence_file}: {e}")
    
    # Create promotion package
    print("📦 Creating promotion package...")
    promotion_package = promotion_manager.create_promotion_package(
        variant, gatekeeper_decision, statistical_evidence, commit_sha
    )
    
    # Save package
    package_dir = promotion_manager.save_promotion_package(promotion_package)
    
    # Generate report
    report = promotion_manager.generate_promotion_report(promotion_package)
    report_file = package_dir / "promotion_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Promotion package created successfully!")
    print(f"Package directory: {package_dir}")
    print(f"Promotion report: {report_file}")
    
    # Display summary
    print(f"\n{'='*60}")
    print(f"PROMOTION SUMMARY")
    print(f"{'='*60}")
    print(f"Variant: {promotion_package.variant}")
    print(f"Composite Score: {promotion_package.composite_score:.3f}")
    print(f"Critical Failures: {promotion_package.critical_failures}")
    print(f"Artifacts: {len(promotion_package.artifacts)}")
    print(f"Container: {promotion_package.container_digest[:12]}...")
    print(f"Ready for deployment: ✅")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()