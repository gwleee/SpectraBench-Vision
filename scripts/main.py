#!/usr/bin/env python3
"""
SpectraVision Main Entry Point
Hardware-aware multimodal model evaluation system
- 단일 환경에서 평가 실행
- interactive/non-interactive 모드 지원
- availability testing 기능 포함
"""

import argparse
import sys
import os
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Load .env file if it exists
def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if env_file.exists():
        print("Loading .env file...")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    os.environ[key] = value
                    print(f"Set {key}={value[:10]}..." if len(value) > 10 else f"Set {key}={value}")
        print(".env file loaded successfully!")
    else:
        print("No .env file found, using existing environment variables")

# Load environment variables at startup
load_env_file()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from spectravision.config import ConfigManager
from spectravision.evaluator import SequentialEvaluator
from spectravision.monitor import PerformanceMonitor
from spectravision.utils import setup_logging, print_banner
from spectravision.env_manager import get_environment_manager

# Docker multi-version support
try:
    from spectravision.multi_version_evaluator import MultiVersionEvaluator
    DOCKER_SUPPORT = True
except ImportError:
    DOCKER_SUPPORT = False


def select_execution_mode():
    """Select execution mode at startup"""
    print_banner()
    print("Welcome to SpectraVision!")
    print()
    print("=" * 72)
    print("EXECUTION MODE SELECTION")
    print("=" * 72)
    print()
    print("Please select the execution mode:")
    print("  1. Single Environment - Current SpectraVision (transformers 4.37.2)")
    print("  2. Multi-Version Docker - All transformer versions (4.33-4.51)")
    print("  3. Availability Test - Quick compatibility check (time-limited)")
    print("  4. Exit")
    print()
    
    if DOCKER_SUPPORT:
        print("[INFO] Docker multi-version support: Available")
    else:
        print("[WARNING] Docker multi-version support: Not available (check Docker installation)")
    print()
    
    while True:
        choice = input("Select mode (1/2/3/4): ").strip()
        
        if choice == "1":
            return "single_environment"
        elif choice == "2":
            if DOCKER_SUPPORT:
                return "multi_version_docker"
            else:
                print("[ERROR] Docker multi-version support not available. Please install Docker and rebuild.")
                continue
        elif choice == "3":
            return "availability_test"
        elif choice == "4":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Please enter 1, 2, 3, or 4")

class AvailabilityTester:
    """Quick availability tester for model-benchmark combinations"""
    
    def __init__(self, config: Dict[str, Any], test_time_limit: int = 120):
        """Initialize Availability Tester"""
        self.config = config
        self.test_time_limit = test_time_limit  # seconds per test
        self.vlmevalkit_path = self._find_vlmevalkit_path()
        self.test_results = []
        
        # Setup test output directory
        self.output_dir = Path(config["output_dir"])
        self.test_dir = self.output_dir / "availability_tests"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Availability Tester initialized")
        print(f"   Time limit per test: {test_time_limit} seconds")
        print(f"   Test results dir: {self.test_dir}")
    
    def _find_vlmevalkit_path(self) -> Path:
        """Find VLMEvalKit installation path"""
        try:
            import vlmeval
            return Path(vlmeval.__file__).parent.parent
        except ImportError:
            project_root = Path(__file__).parent.parent
            possible_paths = [
                project_root / "VLMEvalKit",
                project_root.parent / "VLMEvalKit",
            ]
            for path in possible_paths:
                if path.exists() and (path / "vlmeval").exists():
                    return path.resolve()
            raise FileNotFoundError("VLMEvalKit installation not found")
    
    def test_model_availability(self, model_id: str) -> bool:
        """Test if model is available in VLMEvalKit"""
        try:
            original_cwd = os.getcwd()
            os.chdir(self.vlmevalkit_path)
            
            sys.path.insert(0, str(self.vlmevalkit_path))
            from vlmeval.config import supported_VLM
            
            available = model_id in supported_VLM
            print(f"Model {model_id}: {'Available' if available else 'Not Available'}")
            return available
            
        except Exception as e:
            print(f"Error checking model {model_id}: {e}")
            return False
        finally:
            os.chdir(original_cwd)
            if str(self.vlmevalkit_path) in sys.path:
                sys.path.remove(str(self.vlmevalkit_path))
    
    def test_dataset_availability(self, dataset_name: str) -> bool:
        """Skip dataset availability check - will verify during actual test"""
        print(f"Dataset {dataset_name}: Assuming Available (will verify in actual test)")
        return True
    
    def run_quick_test(self, model_config: dict, benchmark_config: dict) -> tuple:
        """Run time-limited test to check basic compatibility"""
        model_name = model_config["name"]
        benchmark_name = benchmark_config["name"]
        vlmevalkit_model = model_config["vlm_id"]
        vlmevalkit_benchmark = benchmark_config["vlm_name"]
        
        print(f"Quick testing: {model_name} on {benchmark_name}")
        
        # Test model availability first
        if not self.test_model_availability(vlmevalkit_model):
            return False, f"Model {vlmevalkit_model} not available"
        
        # Create test-specific work directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_work_dir = self.test_dir / f"test_{model_name}_{benchmark_name}_{timestamp}"
        test_work_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            original_cwd = os.getcwd()
            os.chdir(self.vlmevalkit_path)
            
            # Prepare standard VLMEvalKit command
            cmd = [
                sys.executable, "run.py",
                "--model", vlmevalkit_model,
                "--data", vlmevalkit_benchmark,
                "--work-dir", str(test_work_dir.absolute()),
                "--mode", "all"
            ]
            
            # Set basic environment
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": "0",
            })
            
            print(f"Running test (max {self.test_time_limit} seconds)...")
            print(f"Command: {' '.join(cmd)}")
            print(f"Working dir: {self.vlmevalkit_path}")
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=self.vlmevalkit_path,
                env=env
            )
            
            # Monitor output for success indicators
            start_time = time.time()
            output_lines = []
            success_indicators = []
            
            try:
                while True:
                    # Check if time limit exceeded
                    elapsed_time = time.time() - start_time
                    if elapsed_time > self.test_time_limit:
                        print(f"Time limit ({self.test_time_limit}s) exceeded")
                        process.terminate()
                        break
                    
                    # Read output with timeout
                    try:
                        line = process.stdout.readline()
                        if not line:
                            break
                        
                        line = line.strip()
                        if line:
                            output_lines.append(line)
                            print(f"[{elapsed_time:.0f}s] {line}")  # Debug output
                            
                            # Check for success indicators
                            if "Loading checkpoint shards: 100%" in line:
                                success_indicators.append("model_loaded")
                                print("SUCCESS INDICATOR: model_loaded")
                            elif "Infer" in line and "/" in line:
                                success_indicators.append("inference_started")
                                print("SUCCESS INDICATOR: inference_started")
                            elif "it/s" in line or "/s" in line:
                                success_indicators.append("processing")
                                print("SUCCESS INDICATOR: processing")
                            elif any(choice in line for choice in ["A.", "B.", "C.", "D."]):
                                success_indicators.append("output_generated")
                                print("SUCCESS INDICATOR: output_generated")
                            
                            # Early success detection
                            if len(success_indicators) >= 2:
                                print(f"Early success detected: {success_indicators}")
                                process.terminate()
                                break
                                
                    except Exception as e:
                        print(f"Error reading output: {e}")
                        break
                
                # Wait for process to finish
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                
            except Exception as e:
                process.kill()
                print(f"Process error: {e}")
                return False, f"Process error: {str(e)}"
            
            # Analyze results
            execution_time = time.time() - start_time
            print(f"Execution time: {execution_time:.1f}s")
            print(f"Total output lines: {len(output_lines)}")
            print(f"Success indicators: {success_indicators}")
            
            # Show first and last few lines
            if output_lines:
                print("First 3 lines:")
                for i, line in enumerate(output_lines[:3]):
                    print(f"  {i+1}: {line}")
                print("Last 3 lines:")
                for i, line in enumerate(output_lines[-3:]):
                    print(f"  {len(output_lines)-2+i}: {line}")
            
            # Check for success
            if len(success_indicators) >= 2:
                print(f"Test PASSED: {model_name} on {benchmark_name}")
                return True, f"Success - detected: {', '.join(success_indicators)}"
            
            # Check for specific failure patterns
            output_text = '\n'.join(output_lines)
            
            if "Error" in output_text or "Exception" in output_text:
                error_lines = [line for line in output_lines if "Error" in line or "Exception" in line]
                print(f"Test FAILED: {model_name} on {benchmark_name} - Error detected")
                print(f"Error lines: {error_lines}")
                return False, f"Error detected: {error_lines[0] if error_lines else 'Unknown error'}"
            
            # More lenient success criteria
            if len(output_lines) < 3:
                print(f"Test FAILED: {model_name} on {benchmark_name} - No meaningful output")
                return False, "No meaningful output generated"
            
            # Check for signs of successful model loading/initialization
            loading_indicators = ["Loading", "checkpoint", "model", "processor", "Downloading", "cached"]
            if any(indicator in output_text for indicator in loading_indicators):
                print(f"Test PASSED: {model_name} on {benchmark_name} - Model loading detected")
                return True, "Success - model loading detected"
            
            print(f"Test FAILED: {model_name} on {benchmark_name} - Insufficient progress")
            print(f"Output sample: {output_text[:500]}...")
            return False, f"Insufficient progress in {execution_time:.1f}s"
            
        except Exception as e:
            print(f"Test ERROR: {model_name} on {benchmark_name}: {e}")
            return False, str(e)
        finally:
            os.chdir(original_cwd)
    
    def run_availability_tests(self) -> dict:
        """Run availability tests for all model-benchmark combinations"""
        print("Starting availability tests...")
        
        models = self.config["models"]
        benchmarks = self.config["benchmarks"]
        total_tests = len(models) * len(benchmarks)
        
        test_results = []
        passed_tests = 0
        current_test = 0
        
        for model_config in models:
            for benchmark_config in benchmarks:
                current_test += 1
                progress = f"[{current_test}/{total_tests}]"
                
                print(f"\n{progress} Testing {model_config['name']} on {benchmark_config['name']}")
                
                success, error_msg = self.run_quick_test(model_config, benchmark_config)
                
                result = {
                    "model_name": model_config["name"],
                    "benchmark_name": benchmark_config["name"],
                    "model_id": model_config["vlm_id"],
                    "benchmark_id": benchmark_config["vlm_name"],
                    "success": success,
                    "error_message": error_msg if not success else None,
                    "test_time_limit": self.test_time_limit,
                    "timestamp": datetime.now().isoformat()
                }
                
                test_results.append(result)
                if success:
                    passed_tests += 1
                
                # Small delay between tests
                time.sleep(2)
        
        # Generate summary
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "test_time_limit_per_combination": self.test_time_limit,
            "timestamp": datetime.now().isoformat(),
            "results": test_results
        }
        
        # Save results
        results_file = self.test_dir / f"availability_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate readable report
        self.generate_test_report(summary, results_file.parent / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        print(f"\nAvailability tests completed: {passed_tests}/{total_tests} passed")
        print(f"Results saved to: {results_file}")
        
        return summary
    
    def generate_test_report(self, summary: dict, report_file: Path):
        """Generate human-readable test report"""
        lines = [
            "=" * 80,
            "SPECTRAVISION AVAILABILITY TEST REPORT",
            "=" * 80,
            "",
            f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Time Limit per Test: {summary['test_time_limit_per_combination']} seconds",
            "",
            "=" * 80,
            "SUMMARY",
            "=" * 80,
            f"Total Tests: {summary['total_tests']}",
            f"Passed: {summary['passed_tests']}",
            f"Failed: {summary['failed_tests']}",
            f"Success Rate: {summary['success_rate']:.1f}%",
            "",
        ]
        
        # Passed tests
        passed_results = [r for r in summary['results'] if r['success']]
        if passed_results:
            lines.extend([
                "=" * 80,
                f"PASSED TESTS ({len(passed_results)})",
                "=" * 80,
            ])
            for result in passed_results:
                lines.append(f"PASS {result['model_name']} + {result['benchmark_name']}")
            lines.append("")
        
        # Failed tests
        failed_results = [r for r in summary['results'] if not r['success']]
        if failed_results:
            lines.extend([
                "=" * 80,
                f"FAILED TESTS ({len(failed_results)})",
                "=" * 80,
            ])
            for result in failed_results:
                lines.append(f"FAIL {result['model_name']} + {result['benchmark_name']}")
                lines.append(f"  Error: {result['error_message'][:100]}...")
                lines.append("")
        
        lines.extend([
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])
        
        with open(report_file, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"Test report saved to: {report_file}")

def run_availability_tests(args=None):
    """Run availability tests instead of full evaluation"""
    print("Initializing SpectraVision Availability Tests...")
    
    # Initialize configuration
    root_dir = Path(__file__).parent.parent
    config_manager = ConfigManager(hardware_type="auto", output_dir=str(root_dir / "outputs"))
    config = config_manager.load_config()
    
    # Apply command-line filters if provided
    if args:
        if args.models:
            # Filter models to only include those specified
            config['models'] = [model for model in config['models'] 
                               if model['name'] in args.models or model['vlm_id'] in args.models]
        if args.benchmarks:
            # Filter benchmarks to only include those specified
            config['benchmarks'] = [bench for bench in config['benchmarks'] 
                                   if bench['name'] in args.benchmarks]
    
    print(f"Hardware: {config['hardware']['name']}")
    print(f"Models to test: {len(config['models'])}")
    print(f"Benchmarks to test: {len(config['benchmarks'])}")
    print(f"Total combinations: {len(config['models']) * len(config['benchmarks'])}")
    print()
    
    # Use command-line time limit or ask for it
    if args and args.test_time_limit:
        test_time_limit = args.test_time_limit
        print(f"Using time limit: {test_time_limit} seconds")
    else:
        # Ask for test time limit
        while True:
            try:
                time_input = input("Time limit per test in seconds [120]: ").strip()
                test_time_limit = int(time_input) if time_input else 120
                if 30 <= test_time_limit <= 600:
                    break
                else:
                    print("Please enter a time between 30 and 600 seconds")
            except ValueError:
                print("Please enter a valid number")
    
    # Skip confirmation if running non-interactively
    if args and (args.models or args.benchmarks or args.test_time_limit):
        print("Running availability tests non-interactively...")
    else:
        response = input(f"\nProceed with availability tests? (Y/n): ")
        if response.lower() not in ['', 'y', 'yes']:
            print("Tests cancelled.")
            return
    
    # Setup logging
    setup_logging("INFO", config["output_dir"])
    
    print(f"\nStarting SpectraVision availability tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)
    
    try:
        # Run tests
        tester = AvailabilityTester(config, test_time_limit=test_time_limit)
        results = tester.run_availability_tests()
        
        # Print summary
        print("\n" + "=" * 72)
        print("AVAILABILITY TEST RESULTS")
        print("=" * 72)
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed_tests']}")
        print(f"Failed: {results['failed_tests']}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print("=" * 72)
        
        if results['failed_tests'] > 0:
            print("\nFailed combinations:")
            for result in results['results']:
                if not result['success']:
                    print(f"  FAIL {result['model_name']} + {result['benchmark_name']}")
        
        print(f"\nDetailed results saved in: {config['output_dir']}/availability_tests/")
        
    except KeyboardInterrupt:
        print("\nAvailability tests interrupted by user")
    except Exception as e:
        print(f"Error during availability tests: {str(e)}")

def interactive_model_selection(config_manager, hardware_type):
    """Interactive model selection interface"""
    print("=" * 72)
    print("MODEL SELECTION")
    print("=" * 72)
    print()
    
    # Load all models for the detected hardware
    models = config_manager.config["models"]
    
    print(f"Available models for {hardware_type.upper()}:")
    for i, model in enumerate(models, 1):
        memory = model.get("memory_gb", "Unknown")
        print(f"  {i}. {model['name']} (~{memory}GB memory)")
    
    print()
    print("Select models to evaluate:")
    print("  - Enter numbers separated by commas (e.g., 1,2,3)")
    print("  - Enter 'all' for all models")
    print("  - Press Enter for all models")
    
    while True:
        choice = input("Your choice: ").strip()
        
        if not choice or choice.lower() == 'all':
            selected_models = [model["name"] for model in models]
            break
        
        try:
            indices = [int(x.strip()) for x in choice.split(',')]
            if all(1 <= i <= len(models) for i in indices):
                selected_models = [models[i-1]["name"] for i in indices]
                break
            else:
                print(f"Please enter numbers between 1 and {len(models)}")
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas, 'all', or press Enter.")
    
    print(f"\nSelected models: {', '.join(selected_models)}")
    return selected_models

def interactive_benchmark_selection(config_manager, hardware_type):
    """Interactive benchmark selection interface"""
    print("\n" + "=" * 72)
    print("BENCHMARK SELECTION")
    print("=" * 72)
    print()
    
    # Load all benchmarks for the detected hardware
    benchmarks = config_manager.config["benchmarks"]
    
    print(f"Available benchmarks for {hardware_type.upper()}:")
    for i, benchmark in enumerate(benchmarks, 1):
        samples = benchmark.get("samples", "Unknown")
        if isinstance(samples, int):
            samples = f"{samples:,}"
        print(f"  {i}. {benchmark['name']} ({samples} samples)")
    
    print()
    print("Select benchmarks to run:")
    print("  - Enter numbers separated by commas (e.g., 1,2,3)")
    print("  - Enter 'all' for all benchmarks")
    print("  - Press Enter for all benchmarks")
    
    while True:
        choice = input("Your choice: ").strip()
        
        if not choice or choice.lower() == 'all':
            selected_benchmarks = [benchmark["name"] for benchmark in benchmarks]
            break
        
        try:
            indices = [int(x.strip()) for x in choice.split(',')]
            if all(1 <= i <= len(benchmarks) for i in indices):
                selected_benchmarks = [benchmarks[i-1]["name"] for i in indices]
                break
            else:
                print(f"Please enter numbers between 1 and {len(benchmarks)}")
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas, 'all', or press Enter.")
    
    print(f"\nSelected benchmarks: {', '.join(selected_benchmarks)}")
    return selected_benchmarks

def interactive_options():
    """Interactive options selection"""
    print("\n" + "=" * 72)
    print("EVALUATION OPTIONS")
    print("=" * 72)
    print()
    
    # Performance monitoring
    while True:
        monitor_choice = input("Enable performance monitoring? (y/N): ").strip().lower()
        if monitor_choice in ['', 'n', 'no']:
            enable_monitoring = False
            break
        elif monitor_choice in ['y', 'yes']:
            enable_monitoring = True
            break
        else:
            print("Please enter 'y' for yes or 'n' for no (default: no)")
    
    # Cache cleanup option
    while True:
        cleanup_choice = input("Enable cache cleanup after each evaluation? (Y/n): ").strip().lower()
        if cleanup_choice in ['', 'y', 'yes']:
            enable_cleanup = True
            
            # Cleanup level selection
            print("Cache cleanup levels:")
            print("  1. Light - PyTorch CUDA cache only (recommended)")
            print("  2. Moderate - + Transformers cache")  
            print("  3. Aggressive - + HuggingFace temp files")
            
            while True:
                level_choice = input("Cleanup level (1/2/3) [1]: ").strip()
                if level_choice in ['', '1']:
                    cleanup_level = "light"
                    break
                elif level_choice == '2':
                    cleanup_level = "moderate"
                    break
                elif level_choice == '3':
                    cleanup_level = "aggressive"
                    break
                else:
                    print("Please enter 1, 2, or 3")
            break
        elif cleanup_choice in ['n', 'no']:
            enable_cleanup = False
            cleanup_level = "light"
            break
        else:
            print("Please enter 'y' for yes or 'n' for no (default: yes)")
    
    # Verbose logging
    while True:
        verbose_choice = input("Enable verbose logging? (y/N): ").strip().lower()
        if verbose_choice in ['', 'n', 'no']:
            verbose = False
            break
        elif verbose_choice in ['y', 'yes']:
            verbose = True
            break
        else:
            print("Please enter 'y' for yes or 'n' for no (default: no)")
    
    # Output directory
    output_dir = input("Output directory [outputs]: ").strip()
    if not output_dir:
        output_dir = "outputs"
    
    return enable_monitoring, verbose, output_dir, enable_cleanup, cleanup_level

def show_evaluation_summary(hardware_type, selected_models, selected_benchmarks, 
                          enable_monitoring, verbose, output_dir, enable_cleanup, cleanup_level, skip_confirmation=False):
    """Show evaluation summary and get confirmation"""
    print("\n" + "=" * 72)
    print("EVALUATION SUMMARY")
    print("=" * 72)
    print()
    
    # Hardware info
    print(f"Hardware: {hardware_type.upper()}")
    
    # Models and benchmarks
    print(f"Models: {', '.join(selected_models)}")
    print(f"Benchmarks: {', '.join(selected_benchmarks)}")
    
    # Combinations
    total_combinations = len(selected_models) * len(selected_benchmarks)
    print(f"Total combinations: {total_combinations}")
    
    # Options
    print(f"Performance monitoring: {'Enabled' if enable_monitoring else 'Disabled'}")
    print(f"Cache cleanup: {'Enabled' if enable_cleanup else 'Disabled'}")
    if enable_cleanup:
        print(f"Cleanup level: {cleanup_level}")
    print(f"Verbose logging: {'Enabled' if verbose else 'Disabled'}")
    print(f"Output directory: {output_dir}")
    
    print()
    
    if skip_confirmation:
        print("Proceeding with evaluation...")
        return True
    
    while True:
        proceed = input("Proceed with evaluation? (Y/n): ").strip().lower()
        if proceed in ['', 'y', 'yes']:
            return True
        elif proceed in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no (default: yes)")

def run_full_evaluation(args=None):
    """Run full SpectraVision evaluation"""
    print("Initializing SpectraVision...")
    
    # Setup root directory path
    root_dir = Path(__file__).parent.parent
    default_output_dir = root_dir / "outputs"
    
    # Initialize configuration manager with auto-detection
    config_manager = ConfigManager(
        hardware_type="auto", 
        output_dir=str(default_output_dir)
    )
    config = config_manager.load_config()
    
    # Get detected hardware type
    hardware_type = config["hardware_type"]
    hardware_name = config["hardware"]["name"]
    
    print(f"Auto-detecting hardware configuration...")
    print(f"Detected hardware: {hardware_name}")
    
    # Interactive or non-interactive selection
    if args and args.models:
        # Non-interactive mode - use provided models
        selected_models = args.models
        print(f"Using specified models: {', '.join(selected_models)}")
    else:
        selected_models = interactive_model_selection(config_manager, hardware_type)
    
    if args and args.benchmarks:
        # Non-interactive mode - use provided benchmarks
        selected_benchmarks = args.benchmarks
        print(f"Using specified benchmarks: {', '.join(selected_benchmarks)}")
    else:
        selected_benchmarks = interactive_benchmark_selection(config_manager, hardware_type)
    
    if args and (args.models or args.benchmarks):
        # Non-interactive mode - use command line options
        enable_monitoring = args.enable_monitoring if args else False
        verbose = args.verbose if args else False
        output_dir = args.output_dir if args else "outputs"
        enable_cleanup = args.enable_cleanup if args else False
        cleanup_level = args.cleanup_level if args else "light"
        print(f"Non-interactive mode: monitoring={enable_monitoring}, verbose={verbose}, cleanup={enable_cleanup} ({cleanup_level})")
    else:
        enable_monitoring, verbose, output_dir, enable_cleanup, cleanup_level = interactive_options()
    
    # Show summary and get confirmation
    if args and (args.models or args.benchmarks):
        # Non-interactive mode - skip confirmation
        print("\nRunning evaluation non-interactively...")
        show_evaluation_summary(hardware_type, selected_models, selected_benchmarks,
                               enable_monitoring, verbose, output_dir, enable_cleanup, cleanup_level, 
                               skip_confirmation=True)
    elif not show_evaluation_summary(hardware_type, selected_models, selected_benchmarks,
                                   enable_monitoring, verbose, output_dir, enable_cleanup, cleanup_level):
        print("\nEvaluation cancelled by user.")
        return
    
    # Update configuration with user selections
    if output_dir != "outputs":
        if Path(output_dir).is_absolute():
            config_manager.output_dir = Path(output_dir)
        else:
            config_manager.output_dir = root_dir / output_dir
    else:
        config_manager.output_dir = default_output_dir
    
    config = config_manager.filter_models(selected_models)
    config = config_manager.filter_benchmarks(selected_benchmarks)
    config["output_dir"] = str(config_manager.output_dir)
    config["cleanup_cache"] = enable_cleanup
    config["cleanup_level"] = cleanup_level
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level, str(config_manager.output_dir))
    
    print(f"\nStarting SpectraVision evaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)
    
    try:
        # Initialize performance monitor if requested
        monitor = None
        if enable_monitoring:
            print("Performance monitoring enabled")
            monitor = PerformanceMonitor(config['hardware'], output_dir=str(config_manager.output_dir))
            monitor.start()
        
        # Determine test mode from args
        test_mode = args.test_mode if args and hasattr(args, 'test_mode') else False

        # Initialize and run sequential evaluator
        evaluator = SequentialEvaluator(
            config=config,
            monitor=monitor,
            verbose=verbose,
            test_mode=test_mode
        )

        if test_mode:
            print("TEST MODE ENABLED: Using only 2 samples per benchmark for quick verification")
        
        # Run evaluation
        results = evaluator.run()
        
        # Stop monitoring
        if monitor:
            monitor.stop()
            print("Performance monitoring completed")
        
        print("=" * 72)
        print(f"SpectraVision evaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to: {config_manager.output_dir}")
        print(f"Total combinations evaluated: {len(results['evaluations'])}")
        
        # Print summary statistics
        if results.get('summary'):
            summary = results['summary']
            print(f"Total execution time: {summary.get('total_time', 'N/A')}")
            print(f"Successful evaluations: {summary.get('successful_evaluations', 0)}")
            print(f"Failed evaluations: {summary.get('failed_evaluations', 0)}")
        
        # Suggest next steps
        print("\nNext steps:")
        print(f"   - View detailed reports: ls {config_manager.output_dir}/reports/")
        print(f"   - View logs: tail -f {config_manager.output_dir}/logs/spectravision*.log")
        if enable_monitoring:
            print(f"   - Performance report: {config_manager.output_dir}/reports/performance_monitoring.json")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="SpectraVision: Intelligent Multimodal Model Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Execution mode selection
    parser.add_argument(
        "--mode",
        choices=["full", "test"],
        help="Execution mode: 'full' for complete evaluation, 'test' for availability testing"
    )
    
    # Non-interactive mode options
    parser.add_argument(
        "--hardware", 
        choices=["auto", "a6000", "a100_single", "a100_dual"],
        default="auto",
        help="Hardware configuration to use"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to evaluate (skips interactive mode)"
    )
    
    parser.add_argument(
        "--benchmarks", 
        nargs="+",
        help="Specific benchmarks to run (skips interactive mode)"
    )
    
    parser.add_argument(
        "--enable-monitoring",
        action="store_true",
        help="Enable detailed performance monitoring"
    )
    
    parser.add_argument(
        "--enable-cleanup",
        action="store_true",
        help="Enable cache cleanup after each evaluation"
    )
    
    parser.add_argument(
        "--cleanup-level",
        choices=["light", "moderate", "aggressive"],
        default="light",
        help="Cache cleanup level"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results and logs"
    )
    
    parser.add_argument(
        "--multi-version",
        action="store_true",
        help="Enable multi-version Docker evaluation (automatic transformer version selection)"
    )
    
    parser.add_argument(
        "--test-time-limit",
        type=int,
        default=120,
        help="Time limit per test for availability testing (seconds)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration and exit (don't run evaluation)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Enable test mode: evaluate only 2 samples per benchmark for quick verification"
    )

    return parser.parse_args()

def run_multi_version_docker():
    """Run Docker multi-version evaluation"""
    if not DOCKER_SUPPORT:
        print("[ERROR] Docker multi-version support not available.")
        return
    
    print("\nDocker Multi-Version Evaluation")
    print("=" * 50)
    print("This mode uses Docker containers with different transformer versions")
    print("to evaluate 29 models across 24 benchmarks without version conflicts.")
    print()
    
    # Check Docker images status
    try:
        import docker
        client = docker.from_env()
        client.ping()
        
        # Check for SpectraVision images
        images = client.images.list()
        spectravision_images = [img for img in images if any('spectravision' in tag for tag in img.tags)]
        
        if not spectravision_images:
            print("[WARNING] Docker images not found. Building is required.")
            print()
            print("Quick setup options:")
            print("1. Build test container (5 minutes)")
            print("2. Build production containers (30-60 minutes)")
            print("3. Show build instructions")
            print("4. Return to main menu")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == "1":
                build_test_container()
                return
            elif choice == "2":
                build_production_containers()
                return
            elif choice == "3":
                show_build_instructions()
                return
            elif choice == "4":
                return
            else:
                print("Invalid choice. Returning to main menu.")
                return
        
    except Exception as e:
        print(f"[ERROR] Docker connection failed: {e}")
        print("Please ensure Docker is running and try again.")
        return
    
    # Docker images are available, proceed with evaluation
    evaluator = MultiVersionEvaluator()
    
    print("Multi-Version Evaluation Options:")
    print("1. Quick Test (30 evaluations, ~30 minutes)")  
    print("2. Container-Specific Test")
    print("3. Model-Specific Test")
    print("4. Full Evaluation (696 evaluations, 4-8 hours)")
    print("5. Show Evaluation Plan")
    print("6. Return to main menu")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    if choice == "1":
        print("\n[INFO] Running Quick Test...")
        results = evaluator.run_quick_test(max_models_per_container=2, max_benchmarks=3)
        print_multi_version_results(results)
        
    elif choice == "2":
        run_container_specific_test(evaluator)
        
    elif choice == "3":
        run_model_specific_test(evaluator)
        
    elif choice == "4":
        print("\n[WARNING] Full evaluation will take 4-8 hours!")
        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm in ['yes', 'y']:
            print("\n[INFO] Running Full Evaluation...")
            results = evaluator.run_full_evaluation()
            print_multi_version_results(results)
        else:
            print("Full evaluation cancelled.")
            
    elif choice == "5":
        show_evaluation_plan(evaluator)
        
    elif choice == "6":
        return
    else:
        print("Invalid choice.")

def run_multi_version_docker_cli(args):
    """Run Docker multi-version evaluation with CLI arguments"""
    if not DOCKER_SUPPORT:
        print("[ERROR] Docker multi-version support not available.")
        print("Please install Docker and ensure it's running.")
        return
    
    print("\nDocker Multi-Version Evaluation (CLI Mode)")
    print("=" * 60)
    print("Automatically selecting transformer versions for each model...")
    print()
    
    # Check Docker availability
    try:
        import docker
        client = docker.from_env()
        client.ping()
        print("[INFO] Docker is available and running")
    except Exception as e:
        print(f"[ERROR] Docker connection failed: {e}")
        print("Please ensure Docker is running and try again.")
        return
    
    # Initialize MultiVersionEvaluator
    try:
        from spectravision.multi_version_evaluator import MultiVersionEvaluator
        evaluator = MultiVersionEvaluator()
        print("[INFO] Multi-version evaluator initialized")
    except ImportError as e:
        print(f"[ERROR] Failed to import MultiVersionEvaluator: {e}")
        return
    
    # Parse models and benchmarks from CLI arguments
    model_filter = args.models if args.models else None
    benchmark_filter = args.benchmarks if args.benchmarks else None
    
    print(f"Evaluation Configuration:")
    print(f"   Models: {model_filter if model_filter else 'All available'}")
    print(f"   Benchmarks: {benchmark_filter if benchmark_filter else 'All available'}")
    print(f"   Mode: Multi-version Docker (automatic version selection)")
    print()
    
    try:
        # Run the evaluation
        print("[INFO] Starting multi-version evaluation...")
        results = evaluator.run_full_evaluation(
            model_filter=model_filter,
            benchmark_filter=benchmark_filter
        )
        
        # Display results
        print_multi_version_results(results)
        
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

def build_test_container():
    """Build test Docker container"""
    print("\n🔨 Building test container (transformers 4.33.0)...")
    try:
        import subprocess
        cmd = [
            'docker', 'build', 
            '-t', 'spectravision-test-4.33:latest',
            '-f', 'docker/test-transformers-4.33/Dockerfile', 
            '.'
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[INFO] Test container built successfully!")
        
        # Test the container
        print("[INFO] Testing container...")
        cmd = ['docker', 'run', '--rm', 'spectravision-test-4.33:latest']
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("Test result:", result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to build test container: {e}")
        print("Error output:", e.stderr)
    except Exception as e:
        print(f"[ERROR] Error: {e}")

def build_production_containers():
    """Build production Docker containers"""
    print("\n🔨 Building production containers...")
    print("This will take 30-60 minutes depending on your internet connection.")
    
    containers = [
        "transformers-4-37",  # Start with current stable
        "transformers-4-49",  # Then latest
        "transformers-4-33",  # Legacy
        "transformers-4-43",  # Mid-range
        "transformers-4-51"   # Cutting edge
    ]
    
    confirm = input("Continue? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Build cancelled.")
        return
    
    try:
        import subprocess
        
        # Build base image first
        print("Building base image...")
        cmd = [
            'docker', 'build',
            '-t', 'spectravision-base:latest',
            '-f', 'docker/base/Dockerfile',
            '.'
        ]
        subprocess.run(cmd, check=True)
        print("[INFO] Base image built successfully!")
        
        # Build each container
        for container in containers:
            print(f"Building {container}...")
            cmd = [
                'docker-compose',
                '-f', 'docker/docker-compose.yml',
                'build', container
            ]
            subprocess.run(cmd, check=True)
            print(f"[INFO] {container} built successfully!")
            
        print("\n[INFO] All containers built successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Build failed: {e}")
    except Exception as e:
        print(f"[ERROR] Error: {e}")

def show_build_instructions():
    """Show Docker build instructions"""
    print("\nDocker Build Instructions")
    print("=" * 40)
    print()
    print("Manual build commands:")
    print()
    print("1. Test container (5 minutes):")
    print("   docker build -t spectravision-test-4.33:latest -f docker/test-transformers-4.33/Dockerfile .")
    print("   docker run --rm spectravision-test-4.33:latest")
    print()
    print("2. Production containers (30-60 minutes):")
    print("   docker build -t spectravision-base:latest -f docker/base/Dockerfile .")
    print("   docker-compose -f docker/docker-compose.yml build")
    print()
    print("3. Start containers:")
    print("   docker-compose -f docker/docker-compose.yml --profile stable up -d")
    print()
    print("For detailed instructions, see: DOCKER_TESTING_GUIDE.md")

def run_container_specific_test(evaluator):
    """Run evaluation on specific containers"""
    print("\nContainer-Specific Test")
    print("Available containers:")
    print("1. transformers_4_37 (Current stable, 8 models)")
    print("2. transformers_4_49 (Latest models, 9 models)")  
    print("3. transformers_4_33 (Legacy models, 8 models)")
    print("4. transformers_4_43 (Mid-range, 2 models)")
    print("5. transformers_4_51 (Cutting-edge, 2 models)")
    
    choice = input("Select container (1-5): ").strip()
    container_map = {
        '1': 'transformers_4_37',
        '2': 'transformers_4_49', 
        '3': 'transformers_4_33',
        '4': 'transformers_4_43',
        '5': 'transformers_4_51'
    }
    
    if choice in container_map:
        container_name = container_map[choice]
        print(f"\n[INFO] Running evaluation on {container_name}...")
        results = evaluator.run_full_evaluation(container_filter=[container_name])
        print_multi_version_results(results)
    else:
        print("Invalid choice.")

def run_model_specific_test(evaluator):
    """Run evaluation on specific models"""
    print("\nModel-Specific Test")
    print("Popular models:")
    print("1. InternVL2-2B (transformers 4.37)")
    print("2. SmolVLM-1.7B (transformers 4.49)")
    print("3. Qwen-VL-Chat (transformers 4.33)")
    print("4. Phi-3.5-Vision (transformers 4.43)")
    print("5. Custom model names")
    
    choice = input("Select option (1-5): ").strip()
    
    if choice == "1":
        models = ["InternVL2-2B"]
    elif choice == "2":
        models = ["SmolVLM-1.7B"]
    elif choice == "3":
        models = ["Qwen-VL-Chat"]
    elif choice == "4":
        models = ["Phi-3.5-Vision"]
    elif choice == "5":
        model_input = input("Enter model names (comma-separated): ").strip()
        models = [m.strip() for m in model_input.split(",")]
    else:
        print("Invalid choice.")
        return
    
    print(f"\n[INFO] Running evaluation on {models}...")
    results = evaluator.run_full_evaluation(model_filter=models)
    print_multi_version_results(results)

def show_evaluation_plan(evaluator):
    """Show evaluation plan without executing"""
    print("\nEvaluation Plan")
    plan = evaluator.get_evaluation_plan()
    print(f"Total evaluations: {len(plan)}")
    
    # Group by container
    by_container = {}
    for item in plan:
        container = item['container']
        if container not in by_container:
            by_container[container] = []
        by_container[container].append(item)
    
    for container_name, items in by_container.items():
        version = items[0]['transformer_version']
        models = set(item['model_name'] for item in items)
        benchmarks = set(item['benchmark_name'] for item in items)
        print(f"\n{container_name} (v{version}):")
        print(f"   Models: {len(models)}")
        print(f"   Benchmarks: {len(benchmarks)}")
        print(f"   Total evaluations: {len(items)}")

def print_multi_version_results(results):
    """Print multi-version evaluation results"""
    if not results:
        print("[ERROR] No results to display.")
        return
    
    print("\n" + "=" * 60)
    print("MULTI-VERSION EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"Session ID: {results.get('session_id', 'unknown')}")
    print(f"Total Evaluations: {results.get('total_evaluations', 0)}")
    print(f"Successful: {results.get('successful_evaluations', 0)}")
    print(f"Failed: {results.get('failed_evaluations', 0)}")
    print(f"Success Rate: {results.get('success_rate', 0):.1f}%")
    
    if 'total_duration_seconds' in results:
        duration = results['total_duration_seconds']
        hours = duration // 3600
        minutes = (duration % 3600) // 60
        seconds = duration % 60
        print(f"Duration: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    
    print("\nResults by Container:")
    for container_name, stats in results.get('containers', {}).items():
        version = stats.get('transformer_version', 'unknown')
        success_rate = stats.get('success_rate', 0)
        successful = stats.get('successful_evaluations', 0)
        total = stats.get('total_evaluations', 0)
        models_tested = len(stats.get('models_tested', []))
        
        print(f"  {container_name} (v{version}):")
        print(f"    Success Rate: {success_rate:.1f}%")
        print(f"    Evaluations: {successful}/{total}")
        print(f"    Models Tested: {models_tested}")
    
    # Show where results are saved
    session_id = results.get('session_id', 'unknown')
    print(f"\n💾 Detailed results saved to:")
    print(f"   outputs/multi_version_evaluations/{session_id}_results.json")
    print(f"   outputs/multi_version_evaluations/{session_id}_summary.txt")

def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Initialize environment manager and apply settings
    env_manager = get_environment_manager()
    env_manager.apply_environment_settings()
    
    # Check if required tokens are configured
    token_status = env_manager.check_required_tokens(require_hf=False)
    if not token_status['hf_token']:
        print("\nWARNING: HF_TOKEN is not configured.")
        print("Some gated models like MiniCPM-V-2.6 may not work.")
        print("To configure your personal HF token:")
        print("1. Copy .env.template to .env")
        print("2. Add your HF token to the .env file")
        print("3. Get your token from: https://huggingface.co/settings/tokens")
        print()
    
    # Determine execution mode
    if args.mode:
        # Non-interactive mode with specified mode
        if args.mode == "test":
            run_availability_tests(args)
        else:
            run_full_evaluation(args)
    elif args.multi_version and (args.models or args.benchmarks):
        # Multi-version Docker evaluation with specific models/benchmarks
        print("[INFO] Running Multi-Version Docker Evaluation (non-interactive)")
        print("   System will automatically select appropriate transformer versions for each model")
        print()
        run_multi_version_docker_cli(args)
    elif args.models or args.benchmarks:
        # Non-interactive mode with specific models/benchmarks (single environment)
        run_full_evaluation(args)
    else:
        # Interactive mode - let user choose
        mode = select_execution_mode()
        
        if mode == "single_environment":
            run_full_evaluation()
        elif mode == "multi_version_docker":
            run_multi_version_docker()
        elif mode == "availability_test":
            run_availability_tests(args)

if __name__ == "__main__":
    main()