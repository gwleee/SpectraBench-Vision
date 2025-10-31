"""
SpectraVision Multi-Version Sequential Evaluator
Automatically runs evaluation across all transformer versions and models
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from .docker_orchestrator import DockerOrchestrator
from .config import ConfigManager
from .utils import setup_logger

class MultiVersionEvaluator:
    """Sequential evaluator across all transformer versions"""
    
    def __init__(self, config_path: str = "configs/models.yaml",
                 benchmarks_path: str = "configs/benchmarks.yaml"):
        self.logger = setup_logger("multi_version_evaluator")
        
        # Initialize orchestrator and config
        self.orchestrator = DockerOrchestrator(config_path)
        self.config_manager = ConfigManager()
        
        # Load benchmark configurations
        self.benchmarks = self.load_benchmarks(benchmarks_path)
        
        # Results storage
        self.results = {
            'session_id': f"multi_version_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': None,
            'end_time': None,
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'containers': {},
            'detailed_results': []
        }
        
        self.logger.info("Multi-version evaluator initialized")
    
    def load_benchmarks(self, benchmarks_path: str) -> List[Dict]:
        """Load benchmark configurations"""
        try:
            import yaml
            with open(benchmarks_path, 'r') as f:
                config = yaml.safe_load(f)
            
            benchmarks = config.get('benchmarks', [])
            self.logger.info(f"Loaded {len(benchmarks)} benchmarks")
            return benchmarks
        
        except Exception as e:
            self.logger.error(f"Failed to load benchmarks: {e}")
            return []
    
    def get_evaluation_plan(self, container_filter: List[str] = None,
                          model_filter: List[str] = None,
                          benchmark_filter: List[str] = None) -> List[Dict]:
        """Generate evaluation plan across all versions"""
        
        # Get all available models by container
        all_models = self.orchestrator.get_available_models()
        
        # Filter containers if specified
        if container_filter:
            all_models = {k: v for k, v in all_models.items() if k in container_filter}
        
        evaluation_plan = []
        
        for container_name, models in all_models.items():
            container_config = self.orchestrator.container_configs[container_name]
            version = container_config.get('version', 'unknown')
            
            # Filter models if specified
            if model_filter:
                models = [m for m in models if m['name'] in model_filter]
            
            for model in models:
                # Filter benchmarks if specified  
                benchmarks_to_test = self.benchmarks
                if benchmark_filter:
                    benchmarks_to_test = [b for b in self.benchmarks if b['name'] in benchmark_filter]
                
                for benchmark in benchmarks_to_test:
                    evaluation_plan.append({
                        'container': container_name,
                        'transformer_version': version,
                        'model_name': model['name'],
                        'model_vlm_id': model['vlm_id'],
                        'model_memory_gb': model['memory_gb'],
                        'benchmark_name': benchmark['name'],
                        'benchmark_vlm_name': benchmark['vlm_name'],
                        'benchmark_samples': benchmark['samples'],
                        'benchmark_purpose': benchmark['purpose']
                    })
        
        self.logger.info(f"Generated evaluation plan: {len(evaluation_plan)} total evaluations")
        return evaluation_plan
    
    def run_full_evaluation(self, 
                           container_filter: List[str] = None,
                           model_filter: List[str] = None, 
                           benchmark_filter: List[str] = None,
                           max_concurrent_containers: int = 2,
                           gpu_allocation: str = "round_robin") -> Dict:
        """
        Run full evaluation across all transformer versions
        
        Args:
            container_filter: Only run these containers (e.g., ['transformers_4_37', 'transformers_4_49'])
            model_filter: Only run these models (e.g., ['InternVL2-2B', 'SmolVLM-1.7B'])
            benchmark_filter: Only run these benchmarks (e.g., ['MMBench', 'TextVQA'])
            max_concurrent_containers: Maximum containers running simultaneously
            gpu_allocation: 'round_robin', 'dedicated', or 'shared'
        """
        
        self.results['start_time'] = datetime.now().isoformat()
        
        # Generate evaluation plan
        evaluation_plan = self.get_evaluation_plan(
            container_filter, model_filter, benchmark_filter
        )
        
        if not evaluation_plan:
            self.logger.error("No evaluations to run. Check your filters.")
            return self.results
        
        self.results['total_evaluations'] = len(evaluation_plan)
        
        self.logger.info(f"Starting full multi-version evaluation")
        self.logger.info(f"   Total evaluations: {len(evaluation_plan)}")
        self.logger.info(f"   Max concurrent containers: {max_concurrent_containers}")
        self.logger.info(f"   GPU allocation: {gpu_allocation}")
        
        # Group evaluations by container for sequential processing
        by_container = {}
        for eval_item in evaluation_plan:
            container = eval_item['container']
            if container not in by_container:
                by_container[container] = []
            by_container[container].append(eval_item)
        
        # Run evaluations container by container
        gpu_id = 0
        for container_name, container_evaluations in by_container.items():
            
            self.logger.info(f"\nProcessing container: {container_name}")
            self.logger.info(f"   Evaluations in this container: {len(container_evaluations)}")
            
            # Initialize container results
            self.results['containers'][container_name] = {
                'transformer_version': container_evaluations[0]['transformer_version'],
                'total_evaluations': len(container_evaluations),
                'successful_evaluations': 0,
                'failed_evaluations': 0,
                'start_time': datetime.now().isoformat(),
                'end_time': None,
                'models_tested': set(),
                'benchmarks_tested': set()
            }
            
            try:
                # Start container
                self.logger.info(f"   Starting container on GPU {gpu_id}")
                self.orchestrator.start_container(container_name, gpu_id)
                
                # Run all evaluations in this container
                for i, eval_item in enumerate(container_evaluations, 1):
                    
                    self.logger.info(f"\n   Evaluation {i}/{len(container_evaluations)}")
                    self.logger.info(f"      Model: {eval_item['model_name']}")
                    self.logger.info(f"      Benchmark: {eval_item['benchmark_name']}")
                    
                    # Run single evaluation
                    result = self.run_single_evaluation(eval_item, gpu_id)
                    
                    # Store detailed result
                    self.results['detailed_results'].append(result)
                    
                    # Update counters
                    if result['status'] == 'success':
                        self.results['successful_evaluations'] += 1
                        self.results['containers'][container_name]['successful_evaluations'] += 1
                    else:
                        self.results['failed_evaluations'] += 1
                        self.results['containers'][container_name]['failed_evaluations'] += 1
                    
                    # Track tested models and benchmarks
                    self.results['containers'][container_name]['models_tested'].add(eval_item['model_name'])
                    self.results['containers'][container_name]['benchmarks_tested'].add(eval_item['benchmark_name'])
                    
                    # Progress update
                    total_progress = (self.results['successful_evaluations'] + self.results['failed_evaluations'])
                    progress_pct = (total_progress / self.results['total_evaluations']) * 100
                    self.logger.info(f"      Overall progress: {total_progress}/{self.results['total_evaluations']} ({progress_pct:.1f}%)")
                
                # Container completed
                self.results['containers'][container_name]['end_time'] = datetime.now().isoformat()
                self.results['containers'][container_name]['models_tested'] = list(self.results['containers'][container_name]['models_tested'])
                self.results['containers'][container_name]['benchmarks_tested'] = list(self.results['containers'][container_name]['benchmarks_tested'])
                
                # Stop container
                self.orchestrator.stop_container(container_name)
                self.logger.info(f"   Container {container_name} completed successfully")
                
            except Exception as e:
                self.logger.error(f"   Container {container_name} failed: {e}")
                self.results['containers'][container_name]['error'] = str(e)
                self.results['containers'][container_name]['end_time'] = datetime.now().isoformat()
                
                # Try to stop container
                try:
                    self.orchestrator.stop_container(container_name)
                except:
                    pass
            
            # Move to next GPU (round robin)
            if gpu_allocation == "round_robin":
                gpu_id = (gpu_id + 1) % max_concurrent_containers
        
        # Finalize results
        self.results['end_time'] = datetime.now().isoformat()
        
        # Calculate summary statistics
        self.calculate_summary_stats()
        
        # Save results
        self.save_results()
        
        return self.results
    
    def run_single_evaluation(self, eval_item: Dict, gpu_id: int) -> Dict:
        """Run a single model-benchmark evaluation"""
        
        start_time = time.time()
        
        try:
            result = self.orchestrator.run_evaluation_in_container(
                container_name=eval_item['container'],
                model_name=eval_item['model_name'],
                benchmark_name=eval_item['benchmark_vlm_name'],  # Use VLM name for actual evaluation
                gpu_id=gpu_id
            )
            
            # Enhance result with evaluation details
            result.update({
                'evaluation_id': f"{eval_item['model_name']}_{eval_item['benchmark_name']}_{int(start_time)}",
                'container': eval_item['container'],
                'transformer_version': eval_item['transformer_version'],
                'model_memory_gb': eval_item['model_memory_gb'],
                'benchmark_samples': eval_item['benchmark_samples'],
                'benchmark_purpose': eval_item['benchmark_purpose'],
                'duration_seconds': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {eval_item['model_name']} on {eval_item['benchmark_name']}: {e}")
            
            return {
                'status': 'error',
                'model': eval_item['model_name'],
                'benchmark': eval_item['benchmark_name'], 
                'container': eval_item['container'],
                'transformer_version': eval_item['transformer_version'],
                'error': str(e),
                'duration_seconds': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def calculate_summary_stats(self):
        """Calculate summary statistics"""
        
        # Overall success rate
        if self.results['total_evaluations'] > 0:
            success_rate = (self.results['successful_evaluations'] / self.results['total_evaluations']) * 100
        else:
            success_rate = 0
        
        self.results['success_rate'] = success_rate
        
        # Time analysis
        if self.results['start_time'] and self.results['end_time']:
            start = datetime.fromisoformat(self.results['start_time'])
            end = datetime.fromisoformat(self.results['end_time'])
            duration = (end - start).total_seconds()
            self.results['total_duration_seconds'] = duration
            self.results['average_evaluation_seconds'] = duration / max(self.results['total_evaluations'], 1)
        
        # Container statistics
        for container_name, container_stats in self.results['containers'].items():
            if container_stats['total_evaluations'] > 0:
                container_stats['success_rate'] = (container_stats['successful_evaluations'] / container_stats['total_evaluations']) * 100
            else:
                container_stats['success_rate'] = 0
        
        self.logger.info(f"Summary statistics calculated")
        self.logger.info(f"   Overall success rate: {success_rate:.1f}%")
        self.logger.info(f"   Total duration: {self.results.get('total_duration_seconds', 0):.1f} seconds")
    
    def save_results(self):
        """Save evaluation results"""
        
        # Create output directory
        output_dir = Path("outputs/multi_version_evaluations")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / f"{self.results['session_id']}_results.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to: {results_file}")
            
            # Also save a summary report
            self.generate_summary_report(output_dir)
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def generate_summary_report(self, output_dir: Path):
        """Generate human-readable summary report"""
        
        report_file = output_dir / f"{self.results['session_id']}_summary.txt"
        
        try:
            with open(report_file, 'w') as f:
                f.write("SpectraVision Multi-Version Evaluation Summary\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Session ID: {self.results['session_id']}\n")
                f.write(f"Start Time: {self.results['start_time']}\n")
                f.write(f"End Time: {self.results['end_time']}\n")
                f.write(f"Total Duration: {self.results.get('total_duration_seconds', 0):.1f} seconds\n\n")
                
                f.write("Overall Results:\n")
                f.write(f"  Total Evaluations: {self.results['total_evaluations']}\n")
                f.write(f"  Successful: {self.results['successful_evaluations']}\n")
                f.write(f"  Failed: {self.results['failed_evaluations']}\n")
                f.write(f"  Success Rate: {self.results.get('success_rate', 0):.1f}%\n\n")
                
                f.write("Results by Container:\n")
                for container_name, stats in self.results['containers'].items():
                    f.write(f"\n  {container_name} (v{stats['transformer_version']}):\n")
                    f.write(f"    Evaluations: {stats['total_evaluations']}\n")
                    f.write(f"    Success: {stats['successful_evaluations']}\n")
                    f.write(f"    Failed: {stats['failed_evaluations']}\n")
                    f.write(f"    Success Rate: {stats.get('success_rate', 0):.1f}%\n")
                    f.write(f"    Models Tested: {len(stats.get('models_tested', []))}\n")
                    f.write(f"    Benchmarks Tested: {len(stats.get('benchmarks_tested', []))}\n")
                
                f.write(f"\nDetailed results available in: {self.results['session_id']}_results.json\n")
            
            self.logger.info(f"Summary report saved to: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")
    
    def run_quick_test(self, max_models_per_container: int = 2, 
                      max_benchmarks: int = 3) -> Dict:
        """Run a quick test with limited models and benchmarks"""
        
        self.logger.info(f"Running quick test")
        self.logger.info(f"   Max models per container: {max_models_per_container}")
        self.logger.info(f"   Max benchmarks: {max_benchmarks}")
        
        # Get limited set of models and benchmarks
        all_models = self.orchestrator.get_available_models()
        
        # Select representative models from each container
        test_models = []
        for container_name, models in all_models.items():
            # Sort by memory (test smaller models first)
            sorted_models = sorted(models, key=lambda x: x['memory_gb'])
            test_models.extend([m['name'] for m in sorted_models[:max_models_per_container]])
        
        # Select representative benchmarks
        test_benchmarks = [b['name'] for b in self.benchmarks[:max_benchmarks]]
        
        self.logger.info(f"   Selected {len(test_models)} models: {test_models}")
        self.logger.info(f"   Selected {len(test_benchmarks)} benchmarks: {test_benchmarks}")
        
        return self.run_full_evaluation(
            model_filter=test_models,
            benchmark_filter=test_benchmarks,
            max_concurrent_containers=1
        )