#!/usr/bin/env python3
"""
PackRepo QA Evaluation Runner

Complete implementation of TODO.md Workstreams B & C:
- LLM QA Harness with real model integration
- Blind A/B Judge with rubric-based scoring  
- Statistical validation with consistency checks
- Comprehensive telemetry and cost tracking

Usage:
    python run_evaluation.py --config configs/example_config.json --output qa_results/
    python run_evaluation.py --smoke-test  # Quick validation
    python run_evaluation.py --validate-setup  # Check dependencies and connectivity
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from harness.llm_client import create_llm_client, LLMProviderError
from harness.runner import QARunner, QARunConfig, QATask, run_qa_evaluation
from harness.judge import create_judge
from harness.scorers import create_scorer, ScoringCriteria, ScoreType
from prompts import get_prompt_registry


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None):
    """Setup structured logging."""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


async def validate_setup(config: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate that all required components are available and functional.
    
    Returns:
        Dictionary mapping component names to validation status
    """
    logger = logging.getLogger(__name__)
    validation_results = {}
    
    # Check prompt templates
    logger.info("Validating prompt templates...")
    try:
        registry = get_prompt_registry()
        required_prompts = ["answerer_system", "answerer_user_template", "judge_rubric"]
        
        for prompt_name in required_prompts:
            template = registry.get_template(prompt_name)
            validation_results[f"prompt_{prompt_name}"] = template is not None
            if not template:
                logger.error(f"Missing required prompt template: {prompt_name}")
            else:
                logger.debug(f"Found prompt template: {prompt_name} (SHA: {template.sha256[:8]})")
    
    except Exception as e:
        logger.error(f"Prompt validation failed: {e}")
        validation_results["prompt_system"] = False
    
    # Check LLM connectivity
    logger.info("Validating LLM connectivity...")
    try:
        llm_client = create_llm_client(
            config.get("llm_config", {}),
            log_dir=Path("validation_logs")
        )
        
        # Test basic connectivity
        response = await llm_client.generate(
            prompt="Say 'connection test successful'",
            temperature=0.0,
            max_tokens=10
        )
        
        validation_results["llm_connectivity"] = len(response.text) > 0
        validation_results["llm_cost_tracking"] = response.cost_usd >= 0.0
        validation_results["llm_latency_tracking"] = response.latency_ms > 0
        
        logger.info(f"LLM test successful - Model: {response.model}, Cost: ${response.cost_usd:.6f}")
        
        await llm_client.close()
        
    except Exception as e:
        logger.error(f"LLM validation failed: {e}")
        validation_results["llm_connectivity"] = False
        validation_results["llm_cost_tracking"] = False
        validation_results["llm_latency_tracking"] = False
    
    # Check pack files
    logger.info("Validating pack files...")
    pack_paths = config.get("pack_paths", {})
    for variant, pack_path in pack_paths.items():
        pack_file = Path(pack_path)
        exists = pack_file.exists()
        validation_results[f"pack_{variant}"] = exists
        
        if not exists:
            logger.warning(f"Pack file missing: {pack_path}")
        else:
            # Try to load and validate structure
            try:
                with open(pack_file, 'r') as f:
                    content = f.read()
                    
                # Check if it's valid JSON or plain text
                try:
                    json.loads(content)
                    validation_results[f"pack_{variant}_format"] = True
                except json.JSONDecodeError:
                    # Plain text is also valid
                    validation_results[f"pack_{variant}_format"] = len(content) > 0
                    
                logger.debug(f"Pack validated: {variant} ({len(content)} chars)")
                
            except Exception as e:
                logger.error(f"Pack validation failed for {variant}: {e}")
                validation_results[f"pack_{variant}_format"] = False
    
    # Check output directory permissions
    output_dir = Path(config.get("output_dir", "qa_outputs"))
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / "test_write.txt"
        test_file.write_text("test")
        test_file.unlink()
        validation_results["output_directory"] = True
        logger.debug(f"Output directory validated: {output_dir}")
    except Exception as e:
        logger.error(f"Output directory validation failed: {e}")
        validation_results["output_directory"] = False
    
    return validation_results


def load_config(config_path: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load and validate configuration file."""
    logger = logging.getLogger(__name__)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Resolve environment variables in config
    config_str = json.dumps(config)
    for env_var in os.environ:
        config_str = config_str.replace(f"${{{env_var}}}", os.environ[env_var])
        config_str = config_str.replace(f"${env_var}", os.environ[env_var])
    
    config = json.loads(config_str)
    
    # Override output directory if specified
    if output_dir:
        config["output_dir"] = str(output_dir)
    
    # Validate required sections
    required_sections = ["pack_paths", "tasks", "llm_config"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Convert pack paths to Path objects
    config["pack_paths"] = {k: Path(v) for k, v in config["pack_paths"].items()}
    
    # Convert tasks to QATask objects
    config["tasks"] = [QATask(**task) for task in config["tasks"]]
    
    logger.info(f"Loaded configuration: {len(config['pack_paths'])} variants, {len(config['tasks'])} tasks")
    
    return config


async def run_smoke_test() -> bool:
    """
    Run a quick smoke test to validate the evaluation pipeline.
    
    Returns:
        True if smoke test passes, False otherwise
    """
    logger = logging.getLogger(__name__)
    logger.info("Running smoke test...")
    
    # Create minimal test configuration
    temp_dir = Path("smoke_test_temp")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Create test pack file
        test_pack = temp_dir / "test.pack"
        test_content = {
            "index": {"total_tokens": 100},
            "body": "This is a test repository for smoke testing the QA evaluation system."
        }
        test_pack.write_text(json.dumps(test_content))
        
        # Create minimal config
        config = {
            "pack_paths": {"test": test_pack},
            "tasks": [
                QATask(
                    question_id="smoke",
                    question="What is this repository for?",
                    context_budget=500,
                    difficulty="easy",
                    category="test"
                )
            ],
            "llm_config": {
                "providers": {
                    "local": {"base_url": "http://localhost:11434", "model": "llama3.1"}
                },
                "default_provider": "local",
                "rate_limit_rpm": 10
            },
            "seeds": [42],
            "temperature": 0.0,
            "output_dir": temp_dir / "output",
            "max_concurrent": 1
        }
        
        # Create QA config
        qa_config = QARunConfig(
            pack_paths=config["pack_paths"],
            tasks=config["tasks"],
            llm_config=config["llm_config"],
            seeds=config["seeds"],
            temperature=config["temperature"],
            output_dir=config["output_dir"],
            max_concurrent=config["max_concurrent"]
        )
        
        # Run evaluation
        results = await run_qa_evaluation(qa_config)
        
        # Validate results
        success = (
            results["overall_stats"]["total_evaluations"] > 0 and
            results["overall_stats"]["success_rate"] > 0 and
            results["overall_stats"]["errors"] == 0
        )
        
        if success:
            logger.info("✅ Smoke test PASSED")
        else:
            logger.error("❌ Smoke test FAILED")
            logger.error(f"Results: {results}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Smoke test FAILED with exception: {e}")
        return False
    
    finally:
        # Cleanup
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


async def main():
    """Main entry point for QA evaluation."""
    parser = argparse.ArgumentParser(
        description="PackRepo QA Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full evaluation
    python run_evaluation.py --config configs/example_config.json --output qa_results/
    
    # Validate setup without running evaluation
    python run_evaluation.py --validate-setup --config configs/example_config.json
    
    # Quick smoke test
    python run_evaluation.py --smoke-test
    
    # Run with debug logging
    python run_evaluation.py --config configs/example_config.json --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Configuration JSON file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="qa_outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--validate-setup",
        action="store_true",
        help="Validate setup without running evaluation"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true", 
        help="Run quick smoke test"
    )
    parser.add_argument(
        "--no-prompt-validation",
        action="store_true",
        help="Skip prompt immutability validation"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.output / "evaluation.log" if args.output else None
    setup_logging(args.log_level, log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting PackRepo QA Evaluation System")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Handle smoke test
        if args.smoke_test:
            success = await run_smoke_test()
            return 0 if success else 1
        
        # Require config for other operations
        if not args.config:
            logger.error("Configuration file required (use --config)")
            parser.print_help()
            return 1
        
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config, args.output)
        
        # Handle setup validation
        if args.validate_setup:
            logger.info("Validating system setup...")
            validation_results = await validate_setup(config)
            
            # Print validation report
            print("\n" + "="*60)
            print("SETUP VALIDATION REPORT")
            print("="*60)
            
            all_passed = True
            for component, status in validation_results.items():
                status_str = "✅ PASS" if status else "❌ FAIL"
                print(f"{component:30} {status_str}")
                if not status:
                    all_passed = False
            
            print("="*60)
            if all_passed:
                print("✅ ALL VALIDATIONS PASSED - System ready for evaluation")
                return 0
            else:
                print("❌ VALIDATION FAILURES DETECTED - Fix issues before running evaluation")
                return 1
        
        # Run full evaluation
        logger.info("Starting QA evaluation...")
        start_time = datetime.now()
        
        # Create QA configuration
        qa_config = QARunConfig(
            pack_paths=config["pack_paths"],
            tasks=config["tasks"],
            llm_config=config["llm_config"],
            seeds=config.get("seeds", [42]),
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 2048),
            output_dir=args.output,
            log_level=args.log_level,
            enforce_budget=config.get("enforce_budget", True),
            validate_prompts=not args.no_prompt_validation,
            max_concurrent=config.get("max_concurrent", 3),
            timeout_seconds=config.get("timeout_seconds", 300)
        )
        
        # Run evaluation
        results = await run_qa_evaluation(qa_config)
        
        # Calculate total time
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Print results summary
        print("\n" + "="*80)
        print("QA EVALUATION COMPLETED")
        print("="*80)
        print(f"Total Time: {total_time:.1f} seconds")
        print(f"Total Evaluations: {results['overall_stats']['total_evaluations']}")
        print(f"Success Rate: {results['overall_stats']['success_rate']:.2%}")
        print(f"Total Cost: ${results['overall_stats']['total_cost_usd']:.4f}")
        print(f"Average Latency: {results['overall_stats']['avg_latency_ms']:.1f}ms")
        print(f"Budget Violations: {results['overall_stats']['budget_violations']}")
        print(f"Errors: {results['overall_stats']['errors']}")
        
        # Variant breakdown
        print(f"\nVARIANT PERFORMANCE:")
        for variant, stats in results["variant_stats"].items():
            efficiency = stats["total_tokens"] / max(1, stats["evaluations"])
            print(f"  {variant:20} {stats['evaluations']:3d} evals, "
                  f"{stats['avg_latency_ms']:6.1f}ms avg, "
                  f"${stats['total_cost_usd']:8.4f}, "
                  f"{efficiency:6.0f} tok/eval")
        
        # Provider breakdown
        print(f"\nPROVIDER USAGE:")
        for provider, stats in results["provider_stats"].items():
            print(f"  {provider:15} {stats['evaluations']:3d} evals, "
                  f"${stats['total_cost_usd']:8.4f}, "
                  f"{stats['avg_latency_ms']:6.1f}ms avg")
        
        print(f"\nResults saved to: {args.output}")
        print("="*80)
        
        # Return success/failure code
        if results["overall_stats"]["success_rate"] >= 0.9:  # 90% success threshold
            logger.info("✅ Evaluation completed successfully")
            return 0
        else:
            logger.warning("⚠️  Evaluation completed with issues (success rate < 90%)")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))