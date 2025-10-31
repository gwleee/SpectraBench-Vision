#!/usr/bin/env python3
"""
Automated Tag Promotion Script
Handles automated promotion from candidate tags to stable tags
Integrates with CI/CD systems and handles scheduling
"""

import os
import sys
import json
import time
import logging
import subprocess
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import local modules
try:
    from promote_docker_image import ImagePromotionPipeline
    from monitoring_integration import MonitoringIntegration
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutomatedPromotion:
    """
    Handles automated Docker image promotion workflows
    Supports scheduled promotions, CI/CD integration, and batch processing
    """

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "configs/promotion_automation.json"
        self.config = self.load_config()
        self.promotion_pipeline = None
        self.monitor = None

        # Initialize components if available
        if DEPENDENCIES_AVAILABLE:
            self.promotion_pipeline = ImagePromotionPipeline(
                registry=self.config.get("registry", "ghcr.io/gwleee/spectravision")
            )

            if self.config.get("monitoring", {}).get("enabled", True):
                try:
                    self.monitor = MonitoringIntegration("outputs/automated_promotion_monitoring")
                    self.monitor.start_monitoring()
                    logger.info("Monitoring enabled for automated promotion")
                except Exception as e:
                    logger.warning(f"Failed to initialize monitoring: {e}")
        else:
            logger.warning("Dependencies not available - limited functionality")

    def load_config(self) -> Dict[str, Any]:
        """Load automation configuration"""
        default_config = {
            "registry": "ghcr.io/gwleee/spectravision",
            "promotion_schedule": {
                "enabled": False,
                "cron_schedule": "0 2 * * 1",  # Monday 2 AM
                "check_interval_hours": 24
            },
            "ci_cd": {
                "enabled": True,
                "trigger_on_push": True,
                "require_all_tests_pass": True,
                "slack_notifications": False
            },
            "monitoring": {
                "enabled": True,
                "alert_thresholds": {
                    "failure_rate_percent": 20,
                    "consecutive_failures": 3
                }
            },
            "promotion_rules": {
                "min_age_hours": 4,  # Candidate must exist for at least 4 hours
                "max_age_days": 7,   # Don't promote candidates older than 7 days
                "require_smoke_test": True,
                "require_mini_benchmark": True
            },
            "versions": ["4.33", "4.37", "4.43", "4.49", "4.51"]
        }

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                        elif isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if subkey not in config[key]:
                                    config[key][subkey] = subvalue
                    return config
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_file}: {e}")

        return default_config

    def save_config(self):
        """Save current configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_file}: {e}")

    def get_candidate_images(self) -> List[Dict[str, Any]]:
        """Get list of candidate images available for promotion"""
        candidates = []

        for version in self.config["versions"]:
            candidate_tag = f"{self.config['registry']}:{version}-candidate"

            try:
                # Check if candidate image exists
                result = subprocess.run([
                    "docker", "manifest", "inspect", candidate_tag
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    # Get image creation time
                    manifest = json.loads(result.stdout)
                    created_time = datetime.fromisoformat(
                        manifest.get("created", datetime.now().isoformat()).replace('Z', '+00:00')
                    )

                    candidates.append({
                        "tag": candidate_tag,
                        "version": version,
                        "created": created_time,
                        "age_hours": (datetime.now() - created_time.replace(tzinfo=None)).total_seconds() / 3600
                    })

            except Exception as e:
                logger.debug(f"Could not inspect {candidate_tag}: {e}")
                continue

        return candidates

    def filter_candidates_by_rules(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter candidates based on promotion rules"""
        filtered = []
        rules = self.config["promotion_rules"]

        for candidate in candidates:
            age_hours = candidate["age_hours"]

            # Check age requirements
            if age_hours < rules["min_age_hours"]:
                logger.debug(f"Candidate {candidate['tag']} too young: {age_hours:.1f}h < {rules['min_age_hours']}h")
                continue

            if age_hours > rules["max_age_days"] * 24:
                logger.warning(f"Candidate {candidate['tag']} too old: {age_hours:.1f}h > {rules['max_age_days'] * 24}h")
                continue

            # Check if stable tag already exists
            stable_tag = candidate["tag"].replace("-candidate", "-stable")
            try:
                result = subprocess.run([
                    "docker", "manifest", "inspect", stable_tag
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    logger.debug(f"Stable tag already exists: {stable_tag}")
                    continue

            except Exception:
                pass  # Stable tag doesn't exist, which is what we want

            filtered.append(candidate)

        return filtered

    def promote_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Promote a single candidate image"""
        if not DEPENDENCIES_AVAILABLE or not self.promotion_pipeline:
            return {
                "success": False,
                "error": "Dependencies not available",
                "candidate": candidate
            }

        candidate_tag = candidate["tag"]
        logger.info(f"Starting promotion for {candidate_tag}")

        if self.monitor:
            self.monitor.log_event("INFO", "automated_promotion_start",
                                 f"Starting automated promotion for {candidate_tag}",
                                 {"candidate": candidate})

        try:
            # Run promotion pipeline
            skip_tests = not (
                self.config["promotion_rules"]["require_smoke_test"] and
                self.config["promotion_rules"]["require_mini_benchmark"]
            )

            result = self.promotion_pipeline.run_promotion_pipeline(
                candidate_tag, skip_tests=skip_tests
            )

            success = result["final_status"] == "success"

            if self.monitor:
                self.monitor.log_event(
                    "INFO" if success else "ERROR",
                    "automated_promotion_complete",
                    f"Automated promotion {'succeeded' if success else 'failed'} for {candidate_tag}",
                    {"candidate": candidate, "result": result}
                )

            return {
                "success": success,
                "result": result,
                "candidate": candidate
            }

        except Exception as e:
            logger.error(f"Exception during promotion of {candidate_tag}: {e}")

            if self.monitor:
                self.monitor.log_event("ERROR", "automated_promotion_exception",
                                     f"Exception during promotion of {candidate_tag}: {e}",
                                     {"candidate": candidate, "error": str(e)})

            return {
                "success": False,
                "error": str(e),
                "candidate": candidate
            }

    def run_batch_promotion(self, force: bool = False) -> Dict[str, Any]:
        """Run batch promotion of all eligible candidates"""
        logger.info("Starting batch promotion process")

        batch_result = {
            "timestamp": datetime.now().isoformat(),
            "total_candidates": 0,
            "eligible_candidates": 0,
            "successful_promotions": 0,
            "failed_promotions": 0,
            "promotions": [],
            "errors": []
        }

        try:
            # Get all candidate images
            candidates = self.get_candidate_images()
            batch_result["total_candidates"] = len(candidates)

            if not candidates:
                logger.info("No candidate images found")
                return batch_result

            # Filter by promotion rules
            if not force:
                eligible_candidates = self.filter_candidates_by_rules(candidates)
            else:
                eligible_candidates = candidates
                logger.info("Force mode enabled - bypassing age filters")

            batch_result["eligible_candidates"] = len(eligible_candidates)

            if not eligible_candidates:
                logger.info("No candidates eligible for promotion")
                return batch_result

            # Promote each eligible candidate
            for candidate in eligible_candidates:
                promotion_result = self.promote_candidate(candidate)
                batch_result["promotions"].append(promotion_result)

                if promotion_result["success"]:
                    batch_result["successful_promotions"] += 1
                    logger.info(f"✅ Successfully promoted {candidate['tag']}")
                else:
                    batch_result["failed_promotions"] += 1
                    batch_result["errors"].append(f"Failed to promote {candidate['tag']}: {promotion_result.get('error', 'Unknown error')}")
                    logger.error(f"❌ Failed to promote {candidate['tag']}")

        except Exception as e:
            logger.error(f"Exception during batch promotion: {e}")
            batch_result["errors"].append(f"Batch promotion exception: {str(e)}")

        # Log batch completion
        if self.monitor:
            self.monitor.log_event("INFO", "batch_promotion_complete",
                                 f"Batch promotion completed: {batch_result['successful_promotions']}/{batch_result['eligible_candidates']} successful",
                                 batch_result)

        logger.info(f"Batch promotion completed: {batch_result['successful_promotions']}/{batch_result['eligible_candidates']} successful")

        return batch_result

    def check_promotion_schedule(self) -> bool:
        """Check if it's time to run scheduled promotion"""
        if not self.config["promotion_schedule"]["enabled"]:
            return False

        # Simple implementation - check if enough time has passed since last run
        state_file = "outputs/automated_promotion_state.json"
        check_interval = self.config["promotion_schedule"]["check_interval_hours"]

        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    last_run = datetime.fromisoformat(state.get("last_run", "2000-01-01"))

                    if (datetime.now() - last_run).total_seconds() < check_interval * 3600:
                        return False
            except Exception as e:
                logger.warning(f"Failed to read state file: {e}")

        return True

    def update_promotion_state(self):
        """Update the promotion state file"""
        state_file = "outputs/automated_promotion_state.json"
        state = {
            "last_run": datetime.now().isoformat(),
            "version": "1.0"
        }

        try:
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update state file: {e}")

    def run_scheduled_promotion(self):
        """Run scheduled promotion if it's time"""
        if not self.check_promotion_schedule():
            logger.info("Scheduled promotion not due yet")
            return

        logger.info("Running scheduled promotion")
        result = self.run_batch_promotion()
        self.update_promotion_state()

        # Send notifications if configured
        if self.config["ci_cd"]["slack_notifications"]:
            self.send_slack_notification(result)

        return result

    def send_slack_notification(self, batch_result: Dict[str, Any]):
        """Send Slack notification about promotion results"""
        # Placeholder for Slack integration
        logger.info(f"Would send Slack notification: {batch_result['successful_promotions']} promotions completed")

    def run_ci_cd_hook(self, event: str, context: Dict[str, Any] = None):
        """Handle CI/CD webhook events"""
        if not self.config["ci_cd"]["enabled"]:
            logger.info("CI/CD integration disabled")
            return

        logger.info(f"Processing CI/CD event: {event}")

        if event == "push" and self.config["ci_cd"]["trigger_on_push"]:
            # Check if this is a push to a candidate tag
            if context and "ref" in context:
                ref = context["ref"]
                if "candidate" in ref:
                    logger.info("Push to candidate branch detected - triggering promotion check")
                    return self.run_batch_promotion()

        elif event == "schedule":
            return self.run_scheduled_promotion()

def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description="Automated Docker Image Promotion")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--mode", choices=["batch", "schedule", "ci-cd", "list"], default="batch",
                       help="Operation mode")
    parser.add_argument("--force", action="store_true", help="Force promotion bypassing age rules")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--event", help="CI/CD event type (for ci-cd mode)")
    parser.add_argument("--context", help="CI/CD event context JSON (for ci-cd mode)")

    args = parser.parse_args()

    # Initialize automation
    automation = AutomatedPromotion(config_file=args.config)

    try:
        if args.mode == "list":
            # List available candidates
            candidates = automation.get_candidate_images()
            eligible = automation.filter_candidates_by_rules(candidates)

            result = {
                "total_candidates": len(candidates),
                "eligible_candidates": len(eligible),
                "candidates": candidates,
                "eligible": eligible
            }

            if args.json:
                print(json.dumps(result, indent=2, default=str))
            else:
                print(f"Found {len(candidates)} candidate images:")
                for candidate in candidates:
                    status = "✅ ELIGIBLE" if candidate in eligible else "❌ NOT ELIGIBLE"
                    print(f"  {candidate['tag']} (age: {candidate['age_hours']:.1f}h) - {status}")

        elif args.mode == "batch":
            result = automation.run_batch_promotion(force=args.force)

            if args.json:
                print(json.dumps(result, indent=2, default=str))
            else:
                print(f"\nBatch Promotion Results:")
                print(f"Total candidates: {result['total_candidates']}")
                print(f"Eligible candidates: {result['eligible_candidates']}")
                print(f"Successful promotions: {result['successful_promotions']}")
                print(f"Failed promotions: {result['failed_promotions']}")

                if result['errors']:
                    print(f"\nErrors:")
                    for error in result['errors']:
                        print(f"  - {error}")

        elif args.mode == "schedule":
            result = automation.run_scheduled_promotion()
            if result and args.json:
                print(json.dumps(result, indent=2, default=str))

        elif args.mode == "ci-cd":
            context = {}
            if args.context:
                try:
                    context = json.loads(args.context)
                except json.JSONDecodeError:
                    logger.error("Invalid JSON context")
                    sys.exit(1)

            result = automation.run_ci_cd_hook(args.event or "push", context)
            if result and args.json:
                print(json.dumps(result, indent=2, default=str))

        # Exit with appropriate code
        if args.mode in ["batch", "ci-cd"] and isinstance(result, dict):
            sys.exit(0 if result.get("failed_promotions", 0) == 0 else 1)
        else:
            sys.exit(0)

    finally:
        # Clean up monitoring
        if automation.monitor:
            automation.monitor.stop_monitoring()

if __name__ == "__main__":
    main()