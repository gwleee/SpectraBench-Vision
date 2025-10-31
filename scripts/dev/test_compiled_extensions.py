#!/usr/bin/env python3
"""
Test compiled extensions compatibility across different transformer versions
Verifies CUDA/Torch/Flash-Attn/DeepSpeed/ONNX Runtime compatibility
"""

import sys
import json
import argparse
from typing import Dict, List, Optional


def test_base_imports() -> Dict[str, any]:
    """Test base torch and transformers imports"""
    results = {"status": "success", "errors": []}

    try:
        import torch
        import transformers

        results["torch_version"] = torch.__version__
        results["transformers_version"] = transformers.__version__
        results["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            results["cuda_version"] = torch.version.cuda
            results["cuda_arch_list"] = torch.cuda.get_arch_list()
            results["gpu_count"] = torch.cuda.device_count()
            results["gpu_name"] = torch.cuda.get_device_name(0)

        print(f"✅ Torch {torch.__version__} + Transformers {transformers.__version__}")

    except Exception as e:
        results["status"] = "error"
        results["errors"].append(f"Base imports failed: {e}")
        print(f"❌ Base imports failed: {e}")

    return results


def test_flash_attention(version: str) -> Dict[str, any]:
    """Test Flash Attention import and compatibility"""
    results = {"status": "success", "errors": [], "module": "flash-attn"}

    try:
        import flash_attn

        results["version"] = flash_attn.__version__
        print(f"✅ Flash Attention {flash_attn.__version__}")

        # Version-specific checks
        version_requirements = {
            "4.33": (2.0, 2.4),
            "4.37": (2.0, 2.4),
            "4.43": (2.0, 2.5),
            "4.49": (2.4, 3.0),
            "4.51": (2.5, 10.0),
        }

        if version in version_requirements:
            min_ver, max_ver = version_requirements[version]
            actual_ver = float(".".join(flash_attn.__version__.split(".")[:2]))

            if min_ver <= actual_ver < max_ver:
                print(f"✅ Flash Attention version compatible with {version}")
            else:
                warning = f"⚠️ Flash Attention {actual_ver} may not be optimal for {version} (expected {min_ver}-{max_ver})"
                results["warnings"] = [warning]
                print(warning)

    except ImportError as e:
        results["status"] = "error"
        results["errors"].append(f"Flash Attention not available: {e}")
        print(f"❌ Flash Attention import failed: {e}")
    except Exception as e:
        results["status"] = "error"
        results["errors"].append(f"Flash Attention test failed: {e}")
        print(f"❌ Flash Attention test failed: {e}")

    return results


def test_deepspeed(version: str) -> Dict[str, any]:
    """Test DeepSpeed import and compatibility"""
    results = {"status": "success", "errors": [], "module": "deepspeed"}

    try:
        import deepspeed

        results["version"] = deepspeed.__version__
        print(f"✅ DeepSpeed {deepspeed.__version__}")

        # Check CUDA ops compilation
        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            results["cpu_adam_available"] = True
        except:
            results["cpu_adam_available"] = False

        # Version requirements
        version_requirements = {
            "4.33": "0.10.0",
            "4.37": "0.10.0",
            "4.43": "0.10.0",
            "4.49": "0.12.0",
            "4.51": "0.14.0",
        }

        if version in version_requirements:
            required = version_requirements[version]
            if deepspeed.__version__ >= required:
                print(f"✅ DeepSpeed version sufficient for {version}")
            else:
                warning = f"⚠️ DeepSpeed {deepspeed.__version__} may be too old for {version} (recommended: {required}+)"
                results["warnings"] = [warning]
                print(warning)

    except ImportError as e:
        results["status"] = "error"
        results["errors"].append(f"DeepSpeed not available: {e}")
        print(f"❌ DeepSpeed import failed: {e}")
    except Exception as e:
        results["status"] = "error"
        results["errors"].append(f"DeepSpeed test failed: {e}")
        print(f"❌ DeepSpeed test failed: {e}")

    return results


def test_xformers() -> Dict[str, any]:
    """Test XFormers import and compatibility"""
    results = {"status": "success", "errors": [], "module": "xformers"}

    try:
        import xformers

        results["version"] = xformers.__version__
        print(f"✅ XFormers {xformers.__version__}")

        # Check memory efficient attention availability
        try:
            from xformers.ops import memory_efficient_attention
            results["memory_efficient_attention"] = True
            print("✅ Memory efficient attention available")
        except:
            results["memory_efficient_attention"] = False
            print("⚠️ Memory efficient attention not available")

    except ImportError as e:
        results["status"] = "error"
        results["errors"].append(f"XFormers not available: {e}")
        print(f"❌ XFormers import failed: {e}")
    except Exception as e:
        results["status"] = "error"
        results["errors"].append(f"XFormers test failed: {e}")
        print(f"❌ XFormers test failed: {e}")

    return results


def test_bitsandbytes() -> Dict[str, any]:
    """Test BitsAndBytes import and compatibility"""
    results = {"status": "success", "errors": [], "module": "bitsandbytes"}

    try:
        import bitsandbytes as bnb

        results["version"] = bnb.__version__
        print(f"✅ BitsAndBytes {bnb.__version__}")

        # Check CUDA ops availability
        try:
            import torch
            if torch.cuda.is_available():
                # Test 8-bit optimizer
                from bitsandbytes.optim import Adam8bit
                results["cuda_ops_available"] = True
                print("✅ BitsAndBytes CUDA ops available")
        except:
            results["cuda_ops_available"] = False
            print("⚠️ BitsAndBytes CUDA ops not fully available")

    except ImportError as e:
        results["status"] = "error"
        results["errors"].append(f"BitsAndBytes not available: {e}")
        print(f"❌ BitsAndBytes import failed: {e}")
    except Exception as e:
        results["status"] = "error"
        results["errors"].append(f"BitsAndBytes test failed: {e}")
        print(f"❌ BitsAndBytes test failed: {e}")

    return results


def test_onnx_runtime(version: str) -> Dict[str, any]:
    """Test ONNX Runtime and Optimum (4.51 specific)"""
    results = {"status": "success", "errors": [], "module": "onnxruntime"}

    # Only test for 4.51
    if version != "4.51":
        results["status"] = "skipped"
        results["reason"] = f"ONNX Runtime not required for version {version}"
        print(f"⏭️  ONNX Runtime test skipped for {version}")
        return results

    try:
        import onnxruntime

        results["version"] = onnxruntime.__version__
        print(f"✅ ONNX Runtime {onnxruntime.__version__}")

        # Check available providers
        providers = onnxruntime.get_available_providers()
        results["providers"] = providers

        if "CUDAExecutionProvider" in providers:
            print("✅ ONNX Runtime CUDA provider available")
            results["cuda_provider"] = True
        else:
            warning = "⚠️ ONNX Runtime CUDA provider not available"
            results["warnings"] = [warning]
            results["cuda_provider"] = False
            print(warning)

        # Test Optimum
        try:
            import optimum
            results["optimum_version"] = optimum.__version__
            print(f"✅ Optimum {optimum.__version__}")
        except ImportError as e:
            results["errors"].append(f"Optimum not available: {e}")
            print(f"❌ Optimum import failed: {e}")

    except ImportError as e:
        results["status"] = "error"
        results["errors"].append(f"ONNX Runtime not available: {e}")
        print(f"❌ ONNX Runtime import failed (required for 4.51): {e}")
    except Exception as e:
        results["status"] = "error"
        results["errors"].append(f"ONNX Runtime test failed: {e}")
        print(f"❌ ONNX Runtime test failed: {e}")

    return results


def run_full_test_suite(version: str, json_output: bool = False) -> Dict[str, any]:
    """Run full test suite for compiled extensions"""
    print(f"\n{'='*80}")
    print(f"Compiled Extensions Test Suite - Transformers {version}")
    print(f"{'='*80}\n")

    results = {
        "version": version,
        "timestamp": None,
        "overall_status": "success",
        "tests": {},
    }

    # Import datetime here to avoid issues if not available
    try:
        from datetime import datetime
        results["timestamp"] = datetime.now().isoformat()
    except:
        pass

    # Run all tests
    tests = [
        ("base_imports", lambda: test_base_imports()),
        ("flash_attention", lambda: test_flash_attention(version)),
        ("deepspeed", lambda: test_deepspeed(version)),
        ("xformers", lambda: test_xformers()),
        ("bitsandbytes", lambda: test_bitsandbytes()),
        ("onnx_runtime", lambda: test_onnx_runtime(version)),
    ]

    for test_name, test_func in tests:
        print(f"\n--- Testing: {test_name} ---")
        try:
            test_result = test_func()
            results["tests"][test_name] = test_result

            if test_result.get("status") == "error":
                results["overall_status"] = "error"
        except Exception as e:
            results["tests"][test_name] = {
                "status": "error",
                "errors": [f"Test execution failed: {e}"],
            }
            results["overall_status"] = "error"
            print(f"❌ Test {test_name} crashed: {e}")

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}\n")

    success_count = sum(
        1 for t in results["tests"].values() if t.get("status") == "success"
    )
    error_count = sum(
        1 for t in results["tests"].values() if t.get("status") == "error"
    )
    skip_count = sum(
        1 for t in results["tests"].values() if t.get("status") == "skipped"
    )

    print(f"✅ Passed: {success_count}")
    print(f"❌ Failed: {error_count}")
    print(f"⏭️  Skipped: {skip_count}")
    print(f"\nOverall Status: {results['overall_status'].upper()}")

    if json_output:
        print(f"\n{'='*80}")
        print("JSON OUTPUT")
        print(f"{'='*80}")
        print(json.dumps(results, indent=2))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test compiled extensions compatibility"
    )
    parser.add_argument(
        "--version",
        required=True,
        choices=["4.33", "4.37", "4.43", "4.49", "4.51"],
        help="Transformers version to test",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results in JSON format"
    )

    args = parser.parse_args()

    results = run_full_test_suite(args.version, args.json)

    # Exit with error code if any tests failed
    sys.exit(0 if results["overall_status"] == "success" else 1)


if __name__ == "__main__":
    main()
