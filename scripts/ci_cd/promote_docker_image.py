#!/usr/bin/env python3
"""
Docker Image Promotion Pipeline
Implements candidate → stable tag promotion with validation
"""

import sys
import os
import json
import time
import subprocess
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import argparse

# Import monitoring integration
try:
    from monitoring_integration import MonitoringIntegration
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImagePromotionPipeline:
    """
    Handles Docker image promotion from candidate to stable tags
    Pipeline: build → smoke test → mini-benchmark → promote to stable
    """

    def __init__(self, registry: str = "ghcr.io/gwleee/spectravision", enable_monitoring: bool = True):
        self.registry = registry
        self.promotion_history = []

        # Initialize monitoring if available
        self.monitor = None
        if enable_monitoring and MONITORING_AVAILABLE:
            try:
                self.monitor = MonitoringIntegration("outputs/promotion_monitoring")
                self.monitor.start_monitoring()
                logger.info("Monitoring integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize monitoring: {e}")
        elif not MONITORING_AVAILABLE:
            logger.info("Monitoring integration not available")

    def run_smoke_test(self, image_tag: str) -> Tuple[bool, Dict[str, Any]]:
        """Run smoke test on candidate image"""
        logger.info(f"🔥 Running smoke test on {image_tag}...")

        # Log to monitoring system
        if self.monitor:
            self.monitor.log_event("INFO", "smoke_test_start", f"Starting smoke test for {image_tag}",
                                 {"image_tag": image_tag, "step": "smoke_test"})

        try:
            # Run the runtime GPU smoke test in the candidate image
            smoke_test_cmd = [
                "docker", "run", "--rm", "--gpus", "all",
                "-v", f"{Path.cwd()}/scripts:/workspace/scripts",
                image_tag,
                "python3", "/workspace/scripts/runtime_gpu_smoke_test.py", "--json"
            ]

            result = subprocess.run(
                smoke_test_cmd,
                capture_output=True,
                text=True,
                timeout=180  # 3 minutes timeout
            )

            if result.returncode == 0:
                # Parse JSON output to get detailed results
                try:
                    test_results = json.loads(result.stdout)
                    logger.info(f"✅ Smoke test PASSED for {image_tag}")

                    # Log key hardware info
                    hw_info = test_results.get('hardware_info', {})
                    if hw_info:
                        logger.info(f"GPU: {hw_info.get('gpu_name', 'Unknown')} ({hw_info.get('gpu_memory_gb', 0)}GB)")

                    # Log success to monitoring
                    if self.monitor:
                        self.monitor.log_event("INFO", "smoke_test_success", f"Smoke test passed for {image_tag}",
                                             {"image_tag": image_tag, "test_results": test_results})

                    return True, test_results

                except json.JSONDecodeError:
                    logger.warning("Could not parse smoke test JSON output")
                    return True, {"status": "passed", "details": "unknown"}

            else:
                logger.error(f"❌ Smoke test FAILED for {image_tag}")
                logger.error(f"Exit code: {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")

                # Log failure to monitoring
                if self.monitor:
                    self.monitor.log_event("ERROR", "smoke_test_failure", f"Smoke test failed for {image_tag}",
                                         {"image_tag": image_tag, "exit_code": result.returncode,
                                          "stdout": result.stdout, "stderr": result.stderr})

                return False, {"status": "failed", "error": result.stderr}

        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Smoke test TIMEOUT for {image_tag}")
            return False, {"status": "timeout", "error": "Smoke test timed out after 3 minutes"}
        except Exception as e:
            logger.error(f"💥 Smoke test ERROR for {image_tag}: {e}")
            return False, {"status": "error", "error": str(e)}

    def run_mini_benchmark(self, image_tag: str) -> Tuple[bool, Dict[str, Any]]:
        """Run mini-benchmark validation on candidate image"""
        logger.info(f"🏃 Running mini-benchmark validation on {image_tag}...")

        # Extract version from image tag (e.g., ghcr.io/gwleee/spectravision:4.49-candidate -> 4.49)
        version = image_tag.split(":")[-1].replace("-candidate", "")

        # Define mini-benchmark: 1 fast model + 1 small benchmark per transformer version
        # Note: Only using versions that have actual Docker images built
        mini_benchmarks = {
            "4.33": {"model": "InstructBLIP-7B", "benchmark": "MMBench"},
            "4.37": {"model": "InternVL2-2B", "benchmark": "MMBench"},
            "4.43": {"model": "Moondream2", "benchmark": "MMBench"},
            "4.49": {"model": "SmolVLM-256M", "benchmark": "MMBench"},
            "4.51": {"model": "Phi-4-Vision", "benchmark": "MMBench"}
        }

        if version not in mini_benchmarks:
            logger.warning(f"No mini-benchmark defined for version {version}")
            return True, {"status": "skipped", "reason": f"No mini-benchmark for version {version}"}

        bench_config = mini_benchmarks[version]
        model = bench_config["model"]
        benchmark = bench_config["benchmark"]

        logger.info(f"Running mini-benchmark: {model} on {benchmark}")

        try:
            # Prepare output directory
            output_dir = f"/tmp/promotion_validation_{int(time.time())}"

            # Run mini-benchmark using the candidate image
            benchmark_cmd = [
                "docker", "run", "--rm", "--gpus", "all",
                "-v", f"{output_dir}:/workspace/outputs",
                "-v", f"{Path.cwd()}/.env:/workspace/.env",  # Mount .env if exists
                "-e", f"HF_TOKEN={os.environ.get('HF_TOKEN', '')}",
                "-e", f"CUDA_VISIBLE_DEVICES=0",
                "--network", "host",
                image_tag,
                "python3", "-m", "vlmeval.run",
                "--model", model.replace('-', '_').lower(),  # VLMEvalKit format
                "--data", f"{benchmark}_DEV_EN",
                "--work-dir", "/workspace/outputs",
                "--verbose"
            ]

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            logger.info(f"Executing: {' '.join(benchmark_cmd)}")

            result = subprocess.run(
                benchmark_cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout for mini-benchmark
            )

            if result.returncode == 0:
                # Check if output files were generated
                output_files = list(Path(output_dir).rglob("*.csv"))
                output_files.extend(list(Path(output_dir).rglob("*.xlsx")))

                if output_files:
                    logger.info(f"✅ Mini-benchmark PASSED for {image_tag}")
                    logger.info(f"Generated {len(output_files)} output files")

                    # Clean up
                    subprocess.run(["rm", "-rf", output_dir], check=False)

                    return True, {
                        "status": "passed",
                        "model": model,
                        "benchmark": benchmark,
                        "output_files": len(output_files)
                    }
                else:
                    logger.error(f"❌ Mini-benchmark FAILED for {image_tag} - No output files generated")
                    return False, {
                        "status": "failed",
                        "error": "No output files generated",
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
            else:
                logger.error(f"❌ Mini-benchmark FAILED for {image_tag}")
                logger.error(f"Exit code: {result.returncode}")
                logger.error(f"STDOUT: {result.stdout[-1000:]}")  # Last 1000 chars
                logger.error(f"STDERR: {result.stderr[-1000:]}")

                # Clean up
                subprocess.run(["rm", "-rf", output_dir], check=False)

                return False, {
                    "status": "failed",
                    "exit_code": result.returncode,
                    "stdout": result.stdout[-1000:],
                    "stderr": result.stderr[-1000:]
                }

        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Mini-benchmark TIMEOUT for {image_tag}")
            subprocess.run(["rm", "-rf", output_dir], check=False)
            return False, {"status": "timeout", "error": "Mini-benchmark timed out after 10 minutes"}
        except Exception as e:
            logger.error(f"💥 Mini-benchmark ERROR for {image_tag}: {e}")
            subprocess.run(["rm", "-rf", output_dir], check=False)
            return False, {"status": "error", "error": str(e)}

    def promote_image(self, candidate_tag: str) -> Tuple[bool, str]:
        """Promote candidate image to stable tag"""

        # Generate stable tag
        if not candidate_tag.endswith("-candidate"):
            logger.error(f"Invalid candidate tag: {candidate_tag} (must end with -candidate)")
            return False, "Invalid candidate tag format"

        # Extract version and create stable tag
        version = candidate_tag.split(":")[-1].replace("-candidate", "")
        stable_tag = f"{self.registry}:{version}-stable"

        logger.info(f"🏷️  Promoting {candidate_tag} → {stable_tag}")

        try:
            # Pull candidate image
            logger.info(f"Pulling candidate image: {candidate_tag}")
            subprocess.run(["docker", "pull", candidate_tag], check=True)

            # Tag as stable
            logger.info(f"Tagging as stable: {stable_tag}")
            subprocess.run(["docker", "tag", candidate_tag, stable_tag], check=True)

            # Push stable tag
            logger.info(f"Pushing stable tag: {stable_tag}")
            subprocess.run(["docker", "push", stable_tag], check=True)

            logger.info(f"✅ Successfully promoted {candidate_tag} to {stable_tag}")
            return True, stable_tag

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to promote image: {e}")
            return False, f"Docker command failed: {e}"

    def run_promotion_pipeline(self, candidate_tag: str, skip_tests: bool = False) -> Dict[str, Any]:
        """
        Run complete promotion pipeline
        Steps: smoke test → mini-benchmark → promote to stable
        """

        pipeline_start = time.time()
        result = {
            "candidate_tag": candidate_tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline_duration": 0,
            "steps": {},
            "final_status": "unknown",
            "stable_tag": None,
            "errors": []
        }

        logger.info(f"🚀 Starting promotion pipeline for {candidate_tag}")

        # Log pipeline start to monitoring
        if self.monitor:
            self.monitor.log_event("INFO", "promotion_pipeline_start",
                                 f"Starting promotion pipeline for {candidate_tag}",
                                 {"candidate_tag": candidate_tag, "skip_tests": skip_tests})

        try:
            # Step 1: Smoke Test
            if not skip_tests:
                logger.info("📍 Step 1: Smoke Test")
                smoke_start = time.time()
                smoke_success, smoke_result = self.run_smoke_test(candidate_tag)
                smoke_duration = time.time() - smoke_start

                result["steps"]["smoke_test"] = {
                    "success": smoke_success,
                    "duration": round(smoke_duration, 2),
                    "result": smoke_result
                }

                if not smoke_success:
                    result["final_status"] = "failed_smoke_test"
                    result["errors"].append("Smoke test failed")
                    logger.error("🚫 Pipeline STOPPED: Smoke test failed")
                    return result

                # Step 2: Mini-benchmark
                logger.info("📍 Step 2: Mini-benchmark Validation")
                bench_start = time.time()
                bench_success, bench_result = self.run_mini_benchmark(candidate_tag)
                bench_duration = time.time() - bench_start

                result["steps"]["mini_benchmark"] = {
                    "success": bench_success,
                    "duration": round(bench_duration, 2),
                    "result": bench_result
                }

                if not bench_success:
                    result["final_status"] = "failed_mini_benchmark"
                    result["errors"].append("Mini-benchmark failed")
                    logger.error("🚫 Pipeline STOPPED: Mini-benchmark failed")
                    return result
            else:
                logger.warning("⚠️  SKIPPING smoke test and mini-benchmark (--skip-tests)")
                result["steps"]["tests_skipped"] = {"reason": "User requested skip"}

            # Step 3: Promote to Stable
            logger.info("📍 Step 3: Promote to Stable")
            promote_start = time.time()
            promote_success, stable_tag = self.promote_image(candidate_tag)
            promote_duration = time.time() - promote_start

            result["steps"]["promotion"] = {
                "success": promote_success,
                "duration": round(promote_duration, 2),
                "stable_tag": stable_tag if promote_success else None
            }

            if promote_success:
                result["final_status"] = "success"
                result["stable_tag"] = stable_tag
                logger.info(f"🎉 PROMOTION SUCCESS: {candidate_tag} → {stable_tag}")
            else:
                result["final_status"] = "failed_promotion"
                result["errors"].append(f"Promotion failed: {stable_tag}")
                logger.error("🚫 Pipeline FAILED: Promotion step failed")

        except Exception as e:
            result["final_status"] = "pipeline_error"
            result["errors"].append(f"Pipeline exception: {str(e)}")
            logger.error(f"💥 Pipeline ERROR: {e}")

        finally:
            result["pipeline_duration"] = round(time.time() - pipeline_start, 2)
            self.promotion_history.append(result)

            # Log pipeline completion to monitoring
            if self.monitor:
                self.monitor.log_event(
                    "INFO" if result["final_status"] == "success" else "ERROR",
                    "promotion_pipeline_complete",
                    f"Promotion pipeline completed for {candidate_tag} with status {result['final_status']}",
                    {
                        "candidate_tag": candidate_tag,
                        "final_status": result["final_status"],
                        "pipeline_duration": result["pipeline_duration"],
                        "stable_tag": result.get("stable_tag"),
                        "errors": result["errors"]
                    }
                )

                # Stop monitoring for this operation
                self.monitor.stop_monitoring()

        return result

def main():
    parser = argparse.ArgumentParser(description="Docker Image Promotion Pipeline")
    parser.add_argument("candidate_tag", help="Candidate image tag to promote (e.g., ghcr.io/gwleee/spectravision:4.49-candidate)")
    parser.add_argument("--skip-tests", action="store_true", help="Skip smoke test and mini-benchmark validation")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--registry", default="ghcr.io/gwleee/spectravision", help="Docker registry base")

    args = parser.parse_args()

    # Initialize promotion pipeline
    pipeline = ImagePromotionPipeline(registry=args.registry)

    # Run promotion pipeline
    result = pipeline.run_promotion_pipeline(args.candidate_tag, skip_tests=args.skip_tests)

    # Output results
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"PROMOTION PIPELINE RESULTS")
        print(f"{'='*60}")
        print(f"Candidate: {result['candidate_tag']}")
        print(f"Status: {result['final_status'].upper()}")
        print(f"Duration: {result['pipeline_duration']}s")

        if result.get('stable_tag'):
            print(f"Stable Tag: {result['stable_tag']}")

        if result['errors']:
            print(f"Errors: {', '.join(result['errors'])}")

        print(f"{'='*60}")

    # Exit with appropriate code
    sys.exit(0 if result["final_status"] == "success" else 1)

if __name__ == "__main__":
    main()