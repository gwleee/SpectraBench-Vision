#!/usr/bin/env python3
"""
Build-time smoke tests for Docker images
Validates that newly built images have proper dependencies and functionality
"""

import sys
import json
import time
import docker
import argparse
from typing import Dict, List, Tuple


def run_smoke_test(image_name: str, transformer_version: str) -> Tuple[bool, Dict]:
    """
    Run comprehensive smoke test on a Docker image
    Returns (success, test_results)
    """
    client = docker.from_env()
    test_results = {
        'image': image_name,
        'transformer_version': transformer_version,
        'tests': {},
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0
    }

    def run_test(test_name: str, command: List[str], expected_exit_code: int = 0) -> bool:
        """Run a single test command in container"""
        test_results['total_tests'] += 1

        try:
            print(f"  🧪 Running test: {test_name}")

            container = client.containers.run(
                image_name,
                command=command,
                detach=True,
                remove=True
            )

            # Wait for completion with timeout
            try:
                result = container.wait(timeout=60)
                exit_code = result['StatusCode']

                if exit_code == expected_exit_code:
                    test_results['tests'][test_name] = {'status': 'PASS', 'exit_code': exit_code}
                    test_results['passed_tests'] += 1
                    print(f"    ✅ PASS")
                    return True
                else:
                    test_results['tests'][test_name] = {'status': 'FAIL', 'exit_code': exit_code}
                    test_results['failed_tests'] += 1
                    print(f"    ❌ FAIL (exit code: {exit_code})")
                    return False

            except Exception as e:
                test_results['tests'][test_name] = {'status': 'TIMEOUT', 'error': str(e)}
                test_results['failed_tests'] += 1
                print(f"    ⏰ TIMEOUT: {e}")
                return False

        except Exception as e:
            test_results['tests'][test_name] = {'status': 'ERROR', 'error': str(e)}
            test_results['failed_tests'] += 1
            print(f"    💥 ERROR: {e}")
            return False

    print(f"🔬 Running smoke tests for {image_name}")
    print(f"   Expected transformer version: {transformer_version}")

    # Test 1: Python availability
    run_test("python_availability", ["python3", "--version"])

    # Test 2: Transformers version check
    run_test("transformers_version", [
        "python3", "-c",
        f"import transformers; assert transformers.__version__ == '{transformer_version}', f'Expected {transformer_version}, got {{transformers.__version__}}'"
    ])

    # Test 3: Core ML packages
    run_test("pytorch_import", ["python3", "-c", "import torch; print(f'PyTorch {torch.__version__}')"])

    # Test 4: CUDA availability (informational)
    run_test("cuda_check", ["python3", "-c", "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"])

    # Test 5: VLMEvalKit import
    run_test("vlmevalkit_import", ["python3", "-c", "from vlmeval.config import supported_VLM; print(f'VLMEvalKit loaded with {len(supported_VLM)} models')"])

    # Test 6: Essential dependencies
    essential_packages = [
        "numpy", "pandas", "pillow", "requests", "tqdm",
        "matplotlib", "seaborn", "scipy", "scikit-learn"
    ]

    for package in essential_packages:
        run_test(f"{package}_import", ["python3", "-c", f"import {package}"])

    # Test 7: Version-specific packages
    version_specific_tests = {
        "4.33": ["qwen-vl-utils", "SwissArmyTransformer"],
        "4.37": ["deepspeed", "accelerate"],
        "4.49": ["flash-attn", "xformers"],
        "4.51": ["bitsandbytes"]
    }

    if transformer_version in version_specific_tests:
        for package in version_specific_tests[transformer_version]:
            # Convert package names for import
            import_name = package.replace("-", "_")
            run_test(f"{package}_import", ["python3", "-c", f"import {import_name}"])

    # Test 8: File system structure
    run_test("workspace_structure", ["test", "-d", "/workspace"])
    run_test("vlmevalkit_structure", ["test", "-d", "/workspace/VLMEvalKit"])

    # Test 9: Environment variables
    run_test("pythonpath_check", ["python3", "-c", "import os; assert '/workspace' in os.environ.get('PYTHONPATH', ''), 'PYTHONPATH not set correctly'"])

    # Test 10: Write permissions
    run_test("write_permissions", ["touch", "/workspace/test_write_permission"])
    run_test("cleanup_test_file", ["rm", "-f", "/workspace/test_write_permission"])

    # Calculate success rate
    success_rate = test_results['passed_tests'] / test_results['total_tests'] if test_results['total_tests'] > 0 else 0
    test_results['success_rate'] = success_rate

    print(f"\n📊 Test Results Summary:")
    print(f"   Total tests: {test_results['total_tests']}")
    print(f"   Passed: {test_results['passed_tests']} ✅")
    print(f"   Failed: {test_results['failed_tests']} ❌")
    print(f"   Success rate: {success_rate:.1%}")

    # Determine overall success
    overall_success = test_results['failed_tests'] == 0

    if overall_success:
        print(f"🎉 SMOKE TEST PASSED: {image_name} is ready for deployment!")
    else:
        print(f"💥 SMOKE TEST FAILED: {image_name} has issues that need to be resolved")

    return overall_success, test_results


def main():
    parser = argparse.ArgumentParser(description="Run Docker image smoke tests")
    parser.add_argument("image_name", help="Docker image name to test")
    parser.add_argument("transformer_version", help="Expected transformer version (e.g., '4.33.2')")
    parser.add_argument("--output-file", help="JSON file to save test results")
    parser.add_argument("--fail-fast", action="store_true", help="Exit on first test failure")

    args = parser.parse_args()

    try:
        success, results = run_smoke_test(args.image_name, args.transformer_version)

        # Save results to file if specified
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"📝 Test results saved to: {args.output_file}")

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("❌ Smoke test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Smoke test error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()