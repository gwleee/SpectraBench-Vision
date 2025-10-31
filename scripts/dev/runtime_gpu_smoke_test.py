#!/usr/bin/env python3
"""
Runtime GPU Smoke Test for Docker Containers
Validates CUDA availability, basic torch functionality, and hardware capabilities
"""

import sys
import json
import time
from typing import Dict, Tuple, Any

def run_gpu_smoke_test() -> Tuple[bool, Dict[str, Any]]:
    """
    Run comprehensive GPU smoke test inside container
    Returns (success, test_results)
    """
    test_results = {
        'timestamp': time.time(),
        'test_id': f"gpu_smoke_{int(time.time())}",
        'tests': {},
        'summary': {'passed': 0, 'failed': 0, 'errors': []},
        'hardware_info': {}
    }

    all_passed = True

    # Test 1: Basic Python and imports
    try:
        import torch
        import transformers
        test_results['tests']['python_imports'] = {
            'status': 'PASS',
            'torch_version': torch.__version__,
            'transformers_version': transformers.__version__,
            'message': f"torch={torch.__version__}, transformers={transformers.__version__}"
        }
        test_results['summary']['passed'] += 1
        print(f"✅ Python imports: torch={torch.__version__}, transformers={transformers.__version__}")
    except Exception as e:
        test_results['tests']['python_imports'] = {
            'status': 'FAIL',
            'error': str(e),
            'message': f"Import error: {e}"
        }
        test_results['summary']['failed'] += 1
        test_results['summary']['errors'].append(f"Python imports: {e}")
        all_passed = False
        print(f"❌ Python imports failed: {e}")

    # Test 2: CUDA availability
    try:
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()

        if cuda_available and device_count > 0:
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            device_props = torch.cuda.get_device_properties(current_device)

            test_results['tests']['cuda_availability'] = {
                'status': 'PASS',
                'device_count': device_count,
                'current_device': current_device,
                'device_name': device_name,
                'memory_total_gb': round(device_props.total_memory / (1024**3), 2),
                'message': f"CUDA available: {device_name} ({device_count} devices)"
            }
            test_results['summary']['passed'] += 1

            # Store hardware info
            test_results['hardware_info'] = {
                'gpu_name': device_name,
                'gpu_memory_gb': round(device_props.total_memory / (1024**3), 2),
                'compute_capability': f"{device_props.major}.{device_props.minor}",
                'multiprocessor_count': device_props.multiprocessor_count
            }

            print(f"✅ CUDA: {device_name} with {round(device_props.total_memory / (1024**3), 2)}GB")
        else:
            test_results['tests']['cuda_availability'] = {
                'status': 'FAIL',
                'cuda_available': cuda_available,
                'device_count': device_count,
                'message': f"CUDA not available: available={cuda_available}, devices={device_count}"
            }
            test_results['summary']['failed'] += 1
            test_results['summary']['errors'].append("CUDA not available")
            all_passed = False
            print(f"❌ CUDA not available: available={cuda_available}, devices={device_count}")

    except Exception as e:
        test_results['tests']['cuda_availability'] = {
            'status': 'FAIL',
            'error': str(e),
            'message': f"CUDA check error: {e}"
        }
        test_results['summary']['failed'] += 1
        test_results['summary']['errors'].append(f"CUDA check: {e}")
        all_passed = False
        print(f"❌ CUDA check failed: {e}")

    # Test 3: Simple tensor operations on GPU (if CUDA available)
    if test_results['tests'].get('cuda_availability', {}).get('status') == 'PASS':
        try:
            # Create small tensors on GPU and perform basic operations
            device = torch.device('cuda')
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.matmul(x, y)
            result_sum = z.sum().item()

            test_results['tests']['gpu_tensor_ops'] = {
                'status': 'PASS',
                'tensor_shape': list(x.shape),
                'result_sum': round(result_sum, 4),
                'message': f"GPU tensor operations successful"
            }
            test_results['summary']['passed'] += 1
            print(f"✅ GPU tensor ops: matrix mult successful")

        except Exception as e:
            test_results['tests']['gpu_tensor_ops'] = {
                'status': 'FAIL',
                'error': str(e),
                'message': f"GPU tensor operations failed: {e}"
            }
            test_results['summary']['failed'] += 1
            test_results['summary']['errors'].append(f"GPU tensor ops: {e}")
            all_passed = False
            print(f"❌ GPU tensor operations failed: {e}")

    # Test 4: Basic model loading (minimal test)
    try:
        from transformers import AutoTokenizer, AutoModel
        model_name = 'distilbert-base-uncased'  # Small, fast model

        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        load_time = time.time() - start_time

        # Test basic tokenization
        test_input = "This is a test sentence."
        tokens = tokenizer(test_input, return_tensors='pt')
        token_count = len(tokens['input_ids'][0])

        test_results['tests']['model_loading'] = {
            'status': 'PASS',
            'model_name': model_name,
            'load_time_seconds': round(load_time, 2),
            'token_count': token_count,
            'message': f"Model loading successful in {load_time:.2f}s"
        }
        test_results['summary']['passed'] += 1
        print(f"✅ Model loading: {model_name} in {load_time:.2f}s")

    except Exception as e:
        test_results['tests']['model_loading'] = {
            'status': 'FAIL',
            'error': str(e),
            'message': f"Model loading failed: {e}"
        }
        test_results['summary']['failed'] += 1
        test_results['summary']['errors'].append(f"Model loading: {e}")
        all_passed = False
        print(f"❌ Model loading failed: {e}")

    # Test 5: Optional - Flash Attention availability (version-dependent)
    try:
        import flash_attn
        test_results['tests']['flash_attention'] = {
            'status': 'PASS',
            'version': getattr(flash_attn, '__version__', 'unknown'),
            'message': f"Flash attention available"
        }
        test_results['summary']['passed'] += 1
        print(f"✅ Flash Attention available")
    except ImportError:
        # This is expected in some transformer versions, not a failure
        test_results['tests']['flash_attention'] = {
            'status': 'SKIP',
            'message': "Flash attention not installed (expected for some versions)"
        }
        print(f"⚠️  Flash Attention not available (expected for some versions)")
    except Exception as e:
        test_results['tests']['flash_attention'] = {
            'status': 'FAIL',
            'error': str(e),
            'message': f"Flash attention check error: {e}"
        }
        test_results['summary']['failed'] += 1
        all_passed = False
        print(f"❌ Flash Attention check failed: {e}")

    # Final summary
    total_tests = test_results['summary']['passed'] + test_results['summary']['failed']
    success_rate = (test_results['summary']['passed'] / total_tests * 100) if total_tests > 0 else 0

    test_results['summary']['total_tests'] = total_tests
    test_results['summary']['success_rate'] = round(success_rate, 1)
    test_results['summary']['overall_status'] = 'PASS' if all_passed else 'FAIL'

    print(f"\n{'='*50}")
    print(f"GPU Smoke Test Results:")
    print(f"Passed: {test_results['summary']['passed']}/{total_tests} ({success_rate:.1f}%)")
    if test_results['summary']['errors']:
        print(f"Errors: {', '.join(test_results['summary']['errors'])}")
    print(f"Overall Status: {test_results['summary']['overall_status']}")
    print(f"{'='*50}")

    return all_passed, test_results

def main():
    """Main function for command-line usage"""
    print("🔥 Starting Runtime GPU Smoke Test...")
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    success, results = run_gpu_smoke_test()

    # Output JSON results for programmatic use
    if len(sys.argv) > 1 and sys.argv[1] == '--json':
        print(json.dumps(results, indent=2))

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()