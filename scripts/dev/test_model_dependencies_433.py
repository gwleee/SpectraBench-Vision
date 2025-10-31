#!/usr/bin/env python3
"""
Automated Model Dependency Testing for Transformers 4.33
Tests each model individually and records missing dependencies
"""

import subprocess
import json
import sys
from datetime import datetime
from pathlib import Path

# Models to test (from configs/models.yaml transformers_4_33)
MODELS_TO_TEST = [
    {"name": "InstructBLIP-7B", "vlm_id": "instructblip_7b"},
    {"name": "VisualGLM-6B", "vlm_id": "VisualGLM_6b"},
    {"name": "Qwen-VL-Chat", "vlm_id": "qwen_chat"},
    {"name": "mPLUG-Owl2", "vlm_id": "mPLUG-Owl2"},
    {"name": "Monkey-Chat", "vlm_id": "monkey-chat"},
    {"name": "MiniMonkey", "vlm_id": "minimonkey"},
    {"name": "InternLM-XComposer2", "vlm_id": "XComposer2"},
    {"name": "InternLM-XComposer", "vlm_id": "XComposer"},
    {"name": "IDEFICS-9B", "vlm_id": "idefics_9b_instruct"},
    {"name": "ShareCaptioner", "vlm_id": "sharecaptioner"},
    {"name": "InstructBLIP-13B", "vlm_id": "instructblip_13b"},
]

# Simple benchmark for quick testing
TEST_BENCHMARK = "MMBench_DEV_EN"

# Docker image to test
DOCKER_IMAGE = "ghcr.io/gwleee/spectravision:4.33"

# Output directory
OUTPUT_DIR = Path("/home/gwlee/Benchmark/SpectraBench-Vision/outputs/dependency_tests")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def test_model(model_info: dict, timeout_seconds: int = 180) -> dict:
    """
    Test a single model and capture any dependency errors

    Returns dict with:
    - success: bool
    - error_message: str (if failed)
    - missing_modules: list (if ModuleNotFoundError)
    - output: str (full output)
    """
    model_name = model_info["name"]
    vlm_id = model_info["vlm_id"]

    print(f"\n{'='*60}")
    print(f"Testing: {model_name} ({vlm_id})")
    print(f"{'='*60}")

    # Docker command with host network mode
    cmd = [
        "docker", "run", "--rm",
        "--network=host",
        "--gpus", "device=0",
        "-v", f"{OUTPUT_DIR.parent}:/workspace/outputs",
        "-v", "/home/gwlee/Benchmark/SpectraBench-Vision/.env:/workspace/VLMEvalKit/.env",
        "-v", "/home/gwlee/Benchmark/SpectraBench-Vision/scripts:/tmp/scripts",
        DOCKER_IMAGE,
        "bash", "-c",
        f"python /tmp/scripts/patch_vlmeval_433.py && "
        f"cd /workspace/VLMEvalKit && "
        f"timeout {timeout_seconds} python run.py --data {TEST_BENCHMARK} --model {vlm_id} --verbose"
    ]

    try:
        print(f"Command: {' '.join(cmd[:10])}...")
        print(f"Timeout: {timeout_seconds}s")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 10  # Extra buffer
        )

        output = result.stdout + result.stderr

        # Check for success indicators
        success_indicators = [
            "Infer ",  # Inference started
            "Loading checkpoint shards: 100%",  # Model loaded
            "it/s]",  # Processing samples
        ]

        # Check for error patterns
        module_not_found_errors = []
        import_errors = []
        other_errors = []

        for line in output.split('\n'):
            if "ModuleNotFoundError: No module named" in line:
                # Extract module name
                try:
                    module = line.split("No module named")[1].strip().strip("'\"")
                    module_not_found_errors.append(module)
                except:
                    module_not_found_errors.append("Unknown module")

            if "ImportError:" in line:
                import_errors.append(line.strip())

            if "ERROR" in line or "CRITICAL" in line:
                if "ModuleNotFoundError" not in line and "ImportError" not in line:
                    other_errors.append(line.strip())

        # Determine success
        has_success_indicator = any(indicator in output for indicator in success_indicators)
        has_module_error = len(module_not_found_errors) > 0
        has_import_error = len(import_errors) > 0
        has_other_error = len(other_errors) > 0

        if has_success_indicator and not has_module_error:
            status = "SUCCESS"
            success = True
            error_msg = None
        elif has_module_error:
            status = "MISSING_DEPENDENCIES"
            success = False
            error_msg = f"Missing modules: {', '.join(set(module_not_found_errors))}"
        elif has_import_error:
            status = "IMPORT_ERROR"
            success = False
            error_msg = import_errors[0] if import_errors else "Unknown import error"
        elif has_other_error:
            status = "OTHER_ERROR"
            success = False
            error_msg = other_errors[0] if other_errors else "Unknown error"
        else:
            status = "TIMEOUT_OR_INCOMPLETE"
            success = False
            error_msg = "Timeout or incomplete execution"

        result_dict = {
            "model_name": model_name,
            "vlm_id": vlm_id,
            "status": status,
            "success": success,
            "error_message": error_msg,
            "missing_modules": list(set(module_not_found_errors)),
            "import_errors": import_errors,
            "other_errors": other_errors,
            "return_code": result.returncode,
            "output_sample": output[:1000] + "..." if len(output) > 1000 else output,
            "full_output_lines": len(output.split('\n')),
        }

        # Print result
        print(f"\nStatus: {status}")
        if error_msg:
            print(f"Error: {error_msg}")
        if module_not_found_errors:
            print(f"Missing modules: {', '.join(set(module_not_found_errors))}")

        return result_dict

    except subprocess.TimeoutExpired:
        print(f"\n⏱️  Timeout after {timeout_seconds}s")
        return {
            "model_name": model_name,
            "vlm_id": vlm_id,
            "status": "TIMEOUT",
            "success": False,
            "error_message": f"Timeout after {timeout_seconds}s",
            "missing_modules": [],
            "import_errors": [],
            "other_errors": [],
            "return_code": -1,
            "output_sample": "Timeout",
            "full_output_lines": 0,
        }

    except Exception as e:
        print(f"\n❌ Exception: {e}")
        return {
            "model_name": model_name,
            "vlm_id": vlm_id,
            "status": "EXCEPTION",
            "success": False,
            "error_message": str(e),
            "missing_modules": [],
            "import_errors": [],
            "other_errors": [],
            "return_code": -1,
            "output_sample": str(e),
            "full_output_lines": 0,
        }


def main():
    print("=" * 60)
    print("Transformers 4.33 Model Dependency Testing")
    print("=" * 60)
    print(f"Docker Image: {DOCKER_IMAGE}")
    print(f"Test Benchmark: {TEST_BENCHMARK}")
    print(f"Models to test: {len(MODELS_TO_TEST)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Test each model
    results = []
    for i, model_info in enumerate(MODELS_TO_TEST, 1):
        print(f"\n[{i}/{len(MODELS_TO_TEST)}]", end=" ")
        result = test_model(model_info, timeout_seconds=180)
        results.append(result)

        # Small delay between tests
        if i < len(MODELS_TO_TEST):
            print("\nWaiting 5 seconds before next test...")
            import time
            time.sleep(5)

    # Generate summary
    print("\n" + "=" * 60)
    print("TESTING SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    missing_deps = [r for r in results if r["missing_modules"]]

    print(f"Total models tested: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Missing dependencies: {len(missing_deps)}")

    if successful:
        print("\n✅ Successful models:")
        for r in successful:
            print(f"  - {r['model_name']}")

    if missing_deps:
        print("\n⚠️  Models with missing dependencies:")
        for r in missing_deps:
            modules = ', '.join(r['missing_modules'])
            print(f"  - {r['model_name']}: {modules}")

    if failed:
        print("\n❌ Failed models (other errors):")
        for r in failed:
            if not r['missing_modules']:
                print(f"  - {r['model_name']}: {r['status']}")

    # Collect all missing modules
    all_missing_modules = set()
    for r in missing_deps:
        all_missing_modules.update(r['missing_modules'])

    if all_missing_modules:
        print("\n📦 All missing modules:")
        for module in sorted(all_missing_modules):
            print(f"  - {module}")

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = OUTPUT_DIR / f"dependency_test_results_{timestamp}.json"

    summary = {
        "timestamp": timestamp,
        "docker_image": DOCKER_IMAGE,
        "test_benchmark": TEST_BENCHMARK,
        "total_models": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "missing_dependencies": len(missing_deps),
        "all_missing_modules": sorted(list(all_missing_modules)),
        "results": results,
    }

    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    # Generate requirements suggestions
    if all_missing_modules:
        print("\n" + "=" * 60)
        print("SUGGESTED REQUIREMENTS TO ADD")
        print("=" * 60)
        print("\nAdd these to docker/requirements/transformers-4.33.in:")
        for module in sorted(all_missing_modules):
            # Common module to package mappings
            package_map = {
                "lavis": "salesforce-lavis",
                "timm": "timm",
                "einops": "einops",
                "sentencepiece": "sentencepiece",
                "protobuf": "protobuf",
            }
            package = package_map.get(module, module)
            print(f"  {package}")
        print("\nThen run: pip-compile docker/requirements/transformers-4.33.in")

    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
