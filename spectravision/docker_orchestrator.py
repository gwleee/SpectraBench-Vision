"""
SpectraVision Docker Orchestrator
Manages multiple Docker containers with different transformer versions
for comprehensive VLM evaluation without dependency conflicts
"""

import os
import sys
import yaml
import docker
import logging
import subprocess
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .config import ConfigManager
from .utils import setup_logger

class DockerOrchestrator:
    """Manages Docker containers for multi-version transformer evaluation with GPU support"""
    
    def __init__(self, config_path: str = "configs/models.yaml", gpu_count: int = None):
        self.logger = setup_logger("docker_orchestrator")
        self.config_path = Path(config_path)
        self.docker_client = None
        self.containers = {}
        self.container_configs = {}
        
        # Detect GPU configuration
        self.gpu_count = gpu_count or self._detect_gpu_count()
        self.gpu_config = self._configure_gpu_settings()
        
        # Load Docker model configurations
        self.load_docker_configs()
        
        # Initialize Docker client
        self.init_docker_client()
        
    def _detect_gpu_count(self) -> int:
        """Detect the number of available GPUs"""
        try:
            cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            gpu_count = len([line for line in result.stdout.strip().split('\n') if line.strip()])
            self.logger.info(f"Detected {gpu_count} GPU(s)")
            return gpu_count
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("Could not detect GPU count, defaulting to 1")
            return 1
    
    def _configure_gpu_settings(self) -> Dict[str, str]:
        """Configure GPU settings based on detected hardware (supports unlimited GPUs)"""
        if self.gpu_count <= 0:
            self.logger.warning("No GPUs detected, using default single GPU config")
            return {
                'NVIDIA_VISIBLE_DEVICES': '0',
                'CUDA_VISIBLE_DEVICES': '0'
            }
        
        # Generate device list for any number of GPUs
        devices = ','.join(str(i) for i in range(self.gpu_count))
        
        self.logger.info(f"Configuring for {self.gpu_count} GPU(s): {devices}")
        
        return {
            'NVIDIA_VISIBLE_DEVICES': devices,
            'CUDA_VISIBLE_DEVICES': devices
        }
        
    def load_docker_configs(self):
        """Load Docker container and model configurations"""
        try:
            with open(self.config_path, 'r') as f:
                self.container_configs = yaml.safe_load(f)
            
            self.logger.info(f"Loaded Docker configurations from {self.config_path}")
            
            # Log available containers
            containers = list(self.container_configs.keys())
            if 'hardware_tiers' in containers:
                containers.remove('hardware_tiers')
            if 'deployment_strategy' in containers:
                containers.remove('deployment_strategy')
                
            self.logger.info(f"Available containers: {containers}")
            
        except Exception as e:
            self.logger.error(f"Failed to load Docker configurations: {e}")
            raise
    
    def init_docker_client(self):
        """Initialize Docker client and check connectivity"""
        try:
            self.docker_client = docker.from_env()
            
            # Test Docker connection
            self.docker_client.ping()
            
            # Check NVIDIA runtime availability
            info = self.docker_client.info()
            runtimes = info.get('Runtimes', {})
            
            if 'nvidia' not in runtimes:
                self.logger.warning("NVIDIA Docker runtime not found. GPU support may not work.")
            else:
                self.logger.info("NVIDIA Docker runtime detected and available")
                
            self.logger.info("Docker client initialized successfully")
            
        except docker.errors.DockerException as e:
            self.logger.error(f"Failed to initialize Docker client: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error initializing Docker: {e}")
            raise
    
    def ensure_image_available(self, container_name: str) -> bool:
        """Ensure Docker image is available, pulling from registry if needed"""
        # Extract version from container name (e.g., transformers_4_37 -> 4.37)
        parts = container_name.split('_')
        if len(parts) >= 3:
            version = f"{parts[1]}.{parts[2]}"  # transformers_4_37 -> 4.37
        else:
            version = parts[-1]  # fallback
        image_name = f"spectravision-{version}"
        registry_image = f"ghcr.io/gwleee/{image_name}:latest"
        
        try:
            # Check if image exists locally
            self.docker_client.images.get(f"{image_name}:latest")
            self.logger.info(f"Image {image_name}:latest found locally")
            return True
            
        except docker.errors.ImageNotFound:
            self.logger.info(f"Image {image_name}:latest not found locally")
            
            # Try to pull from registry
            try:
                self.logger.info(f"📥 Pulling {registry_image}...")
                self.docker_client.images.pull(registry_image)
                
                # Tag it locally for convenience
                image = self.docker_client.images.get(registry_image)
                image.tag(image_name, "latest")
                
                self.logger.info(f"Successfully pulled and tagged {image_name}:latest")
                return True
                
            except Exception as pull_error:
                self.logger.error(f"Failed to pull {registry_image}: {pull_error}")
                self.logger.info("You may need to build the image locally:")
                self.logger.info(f"   docker build -t {image_name}:latest -f docker/{container_name}/Dockerfile .")
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking image availability: {e}")
            return False
    
    def find_container_for_model(self, model_name: str) -> Optional[str]:
        """Find which container supports the given model"""
        for container_name, config in self.container_configs.items():
            if container_name in ['hardware_tiers', 'deployment_strategy']:
                continue
                
            if 'models' in config:
                for model in config['models']:
                    if model['name'] == model_name or model['vlm_id'] == model_name:
                        return container_name
        
        self.logger.warning(f"No container found for model: {model_name}")
        return None
    
    def get_available_models(self, container_name: str = None) -> Dict[str, List[Dict]]:
        """Get available models by container or all models"""
        if container_name:
            if container_name in self.container_configs and 'models' in self.container_configs[container_name]:
                return {container_name: self.container_configs[container_name]['models']}
            else:
                return {}
        
        # Return all models grouped by container
        all_models = {}
        for container_name, config in self.container_configs.items():
            if container_name in ['hardware_tiers', 'deployment_strategy']:
                continue
            if 'models' in config:
                all_models[container_name] = config['models']
        
        return all_models
    
    def build_containers(self, containers: List[str] = None, force_rebuild: bool = False):
        """Build Docker containers"""
        if containers is None:
            containers = [name for name in self.container_configs.keys() 
                         if name not in ['hardware_tiers', 'deployment_strategy']]
        
        # Build base image first
        self.logger.info("Building base Docker image...")
        try:
            subprocess.run([
                'docker', 'build', 
                '-t', 'spectravision-base:latest',
                '-f', 'docker/base/Dockerfile',
                '.'
            ], cwd=Path.cwd(), check=True, capture_output=True, text=True)
            
            self.logger.info("Base image built successfully")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to build base image: {e.stderr}")
            raise
        
        # Build specific container images
        for container_name in containers:
            if container_name not in self.container_configs:
                self.logger.warning(f"Unknown container: {container_name}")
                continue
                
            self.logger.info(f"Building container: {container_name}")
            
            try:
                # Build using docker-compose for proper context
                cmd = [
                    'docker-compose', 
                    '-f', 'docker/docker-compose.yml',
                    'build'
                ]
                
                if force_rebuild:
                    cmd.append('--no-cache')
                
                # Build specific service based on container name
                service_name = container_name.replace('_', '-')
                cmd.append(service_name)
                
                subprocess.run(cmd, cwd=Path.cwd(), check=True, capture_output=True, text=True)
                
                self.logger.info(f"Container {container_name} built successfully")
                
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to build container {container_name}: {e.stderr}")
                raise
    
    def start_container(self, container_name: str, gpu_id: int = 0) -> str:
        """Start a specific container"""
        if container_name not in self.container_configs:
            raise ValueError(f"Unknown container: {container_name}")
        
        # Ensure image is available before starting container
        if not self.ensure_image_available(container_name):
            raise RuntimeError(f"Cannot start container {container_name}: Docker image not available")
        
        service_name = container_name.replace('_', '-')
        
        try:
            # Start using docker-compose
            env = os.environ.copy()
            env['NVIDIA_VISIBLE_DEVICES'] = str(gpu_id)
            
            subprocess.run([
                'docker-compose',
                '-f', 'docker/docker-compose.yml',
                'up', '-d', service_name
            ], cwd=Path.cwd(), check=True, env=env)
            
            self.logger.info(f"Started container {container_name} on GPU {gpu_id}")
            
            # Get container ID
            result = subprocess.run([
                'docker', 'ps', '--filter', f'name=spectravision-{service_name}',
                '--format', '{{.ID}}'
            ], capture_output=True, text=True, check=True)
            
            container_id = result.stdout.strip()
            self.containers[container_name] = container_id
            
            return container_id
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start container {container_name}: {e}")
            raise
    
    def stop_container(self, container_name: str):
        """Stop a specific container"""
        service_name = container_name.replace('_', '-')
        
        try:
            subprocess.run([
                'docker-compose',
                '-f', 'docker/docker-compose.yml',
                'stop', service_name
            ], cwd=Path.cwd(), check=True)
            
            self.logger.info(f"Stopped container {container_name}")
            
            if container_name in self.containers:
                del self.containers[container_name]
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to stop container {container_name}: {e}")
            raise
    
    def run_evaluation_in_container(self, container_name: str, model_name: str, 
                                   benchmark_name: str, gpu_id: int = 0) -> Dict:
        """Run evaluation inside a specific container"""
        
        # Ensure Docker image is available first
        if not self.ensure_image_available(container_name):
            return {
                'status': 'error',
                'message': f'Docker image not available for container: {container_name}',
                'model': model_name,
                'benchmark': benchmark_name
            }
        
        # Ensure container is running
        if container_name not in self.containers:
            self.start_container(container_name, gpu_id)
        
        container_id = self.containers[container_name]
        
        # Find model configuration
        model_config = None
        for model in self.container_configs[container_name]['models']:
            if model['name'] == model_name:
                model_config = model
                break
        
        if not model_config:
            raise ValueError(f"Model {model_name} not found in container {container_name}")
        
        vlm_id = model_config['vlm_id']
        
        # Construct VLMEvalKit command
        command = [
            'python', '/workspace/VLMEvalKit/run.py',
            '--model', vlm_id,
            '--data', benchmark_name,
            '--mode', 'all'
        ]
        
        try:
            self.logger.info(f"Running evaluation: {model_name} on {benchmark_name} in {container_name}")
            self.logger.debug(f"Command: {' '.join(command)}")
            
            # Execute command in container
            container_obj = self.docker_client.containers.get(container_id)
            
            # Use configured GPU settings
            environment = {
                'CUDA_VISIBLE_DEVICES': self.gpu_config['CUDA_VISIBLE_DEVICES'],
                'PYTHONPATH': '/workspace:/workspace/VLMEvalKit'
            }
            
            exec_result = container_obj.exec_run(
                command,
                environment=environment,
                workdir='/workspace'
            )
            
            # Parse results
            if exec_result.exit_code == 0:
                self.logger.info(f"Evaluation completed successfully for {model_name}")
                return {
                    'status': 'success',
                    'model': model_name,
                    'benchmark': benchmark_name,
                    'container': container_name,
                    'output': exec_result.output.decode('utf-8'),
                    'exit_code': exec_result.exit_code
                }
            else:
                self.logger.error(f"Evaluation failed for {model_name}: {exec_result.output.decode('utf-8')}")
                return {
                    'status': 'error',
                    'model': model_name,
                    'benchmark': benchmark_name,
                    'container': container_name,
                    'error': exec_result.output.decode('utf-8'),
                    'exit_code': exec_result.exit_code
                }
                
        except Exception as e:
            self.logger.error(f"Failed to run evaluation in container {container_name}: {e}")
            return {
                'status': 'error',
                'model': model_name,
                'benchmark': benchmark_name,
                'container': container_name,
                'error': str(e),
                'exit_code': -1
            }
    
    def evaluate_model_benchmark(self, model_name: str, benchmark_name: str, 
                                gpu_id: int = 0) -> Dict:
        """Evaluate a model on a benchmark using appropriate container"""
        # Find container for model
        container_name = self.find_container_for_model(model_name)
        
        if not container_name:
            raise ValueError(f"No container found for model: {model_name}")
        
        return self.run_evaluation_in_container(
            container_name, model_name, benchmark_name, gpu_id
        )
    
    def cleanup(self):
        """Stop all containers and cleanup"""
        self.logger.info("Cleaning up Docker containers...")
        
        try:
            # Stop all profiles to ensure all containers are cleaned up
            subprocess.run([
                'docker-compose',
                '-f', 'docker/docker-compose.yml',
                '--profile', 'all',
                'down'
            ], cwd=Path.cwd(), check=True)
            
            self.logger.info("All containers stopped successfully")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to cleanup containers with docker-compose: {e}")
            
            # Fallback: try to stop containers individually
            self.logger.info("Attempting individual container cleanup...")
            try:
                running_containers = self.docker_client.containers.list()
                for container in running_containers:
                    if container.name.startswith('spectravision-'):
                        self.logger.info(f"Stopping container: {container.name}")
                        container.stop()
                        container.remove()
                        
                self.logger.info("Individual container cleanup completed")
                        
            except Exception as fallback_error:
                self.logger.error(f"Fallback cleanup also failed: {fallback_error}")
        
        self.containers.clear()
    
    def get_container_status(self) -> Dict[str, str]:
        """Get status of all containers"""
        status = {}
        
        for container_name in self.container_configs.keys():
            if container_name in ['hardware_tiers', 'deployment_strategy']:
                continue
            
            service_name = container_name.replace('_', '-')
            container_full_name = f"spectravision-{service_name}"
            
            try:
                result = subprocess.run([
                    'docker', 'ps', '--filter', f'name={container_full_name}',
                    '--format', '{{.Status}}'
                ], capture_output=True, text=True, check=True)
                
                if result.stdout.strip():
                    status[container_name] = 'running'
                else:
                    status[container_name] = 'stopped'
                    
            except subprocess.CalledProcessError:
                status[container_name] = 'unknown'
        
        return status
    
    def run_batch_evaluation(self, models: List[str], benchmarks: List[str], 
                           gpu_ids: List[int] = [0]) -> List[Dict]:
        """Run batch evaluation of multiple models and benchmarks"""
        self.logger.info(f"Starting batch evaluation: {len(models)} models × {len(benchmarks)} benchmarks")
        
        results = []
        gpu_index = 0
        
        try:
            for model in models:
                # Find appropriate container for model
                container_name = self.find_container_for_model(model)
                if not container_name:
                    self.logger.warning(f"No container found for model: {model}")
                    for benchmark in benchmarks:
                        results.append({
                            'status': 'error',
                            'model': model,
                            'benchmark': benchmark,
                            'container': 'none',
                            'error': f'No container found for model: {model}',
                            'exit_code': -1
                        })
                    continue
                
                # Ensure container is available and running
                current_gpu = gpu_ids[gpu_index % len(gpu_ids)]
                if container_name not in self.containers:
                    self.logger.info(f"Starting container {container_name} on GPU {current_gpu}")
                    try:
                        self.start_container(container_name, current_gpu)
                    except Exception as e:
                        self.logger.error(f"Failed to start container {container_name}: {e}")
                        for benchmark in benchmarks:
                            results.append({
                                'status': 'error',
                                'model': model,
                                'benchmark': benchmark,
                                'container': container_name,
                                'error': f'Failed to start container: {str(e)}',
                                'exit_code': -1
                            })
                        continue
                
                # Run evaluations for all benchmarks with this model
                for benchmark in benchmarks:
                    self.logger.info(f"Evaluating {model} on {benchmark} (GPU {current_gpu})")
                    result = self.run_evaluation_in_container(
                        container_name, model, benchmark, current_gpu
                    )
                    results.append(result)
                    
                    # Brief pause between evaluations for stability
                    time.sleep(2)
                
                gpu_index += 1
        
        except KeyboardInterrupt:
            self.logger.warning("Batch evaluation interrupted by user")
        except Exception as e:
            self.logger.error(f"Batch evaluation failed: {e}")
            
        self.logger.info(f"Batch evaluation completed: {len(results)} results")
        return results
    
    def interactive_mode(self):
        """Interactive mode for model and benchmark selection"""
        print("\nSpectraBench-Vision Docker Interactive Mode")
        print("=" * 60)
        
        try:
            # Show available models by container
            available_models = self.get_available_models()
            if not available_models:
                print("ERROR: No models available in configuration")
                return
            
            # Interactive model selection
            selected_models = self._interactive_model_selection(available_models)
            if not selected_models:
                print("No models selected. Exiting.")
                return
            
            # Interactive benchmark selection
            selected_benchmarks = self._interactive_benchmark_selection()
            if not selected_benchmarks:
                print("No benchmarks selected. Exiting.")
                return
            
            # GPU selection
            gpu_ids = self._interactive_gpu_selection()
            
            # Confirm and run
            total_evaluations = len(selected_models) * len(selected_benchmarks)
            print(f"\nEvaluation Summary:")
            print(f"   Models: {len(selected_models)} ({', '.join(selected_models[:3])}{'...' if len(selected_models) > 3 else ''})")
            print(f"   Benchmarks: {len(selected_benchmarks)} ({', '.join(selected_benchmarks[:3])}{'...' if len(selected_benchmarks) > 3 else ''})")
            print(f"   Total evaluations: {total_evaluations}")
            print(f"   GPUs: {gpu_ids}")
            
            confirm = input("\nProceed with evaluation? (y/N): ").lower().strip()
            if confirm != 'y':
                print("Evaluation cancelled.")
                return
            
            # Run evaluation
            print(f"\nStarting evaluation of {total_evaluations} combinations...")
            results = self.run_batch_evaluation(selected_models, selected_benchmarks, gpu_ids)
            
            # Display results summary
            self._display_results_summary(results)
            
        except KeyboardInterrupt:
            print("\nInteractive mode cancelled by user")
        except Exception as e:
            self.logger.error(f"Interactive mode failed: {e}")
            print(f"ERROR: {e}")
    
    def test_system(self):
        """Test system functionality and container availability"""
        print("\nSpectraBench-Vision Docker System Test")
        print("=" * 50)
        
        test_results = {}
        
        try:
            # Test Docker connectivity
            print("Testing Docker connectivity...")
            self.init_docker_client()
            print("SUCCESS: Docker client initialized successfully")
            
            # Test GPU configuration
            print(f"GPU Configuration: {self.gpu_count} GPU(s) detected")
            print(f"   GPU settings: {self.gpu_config}")
            
            # Test all container images
            print(f"\nTesting {len([k for k in self.container_configs.keys() if k not in ['hardware_tiers', 'deployment_strategy']])} container images...")
            
            for container_name in self.container_configs.keys():
                if container_name in ['hardware_tiers', 'deployment_strategy']:
                    continue
                    
                print(f"\nTesting {container_name}...")
                
                # Test image availability
                if not self.ensure_image_available(container_name):
                    test_results[container_name] = 'Image not available'
                    print(f"   ERROR: Image not available")
                    continue
                
                print(f"   SUCCESS: Image available")
                
                # Test container startup
                try:
                    print(f"   Starting container...")
                    container_id = self.start_container(container_name, 0)
                    print(f"   SUCCESS: Container started (ID: {container_id[:12]})")
                    
                    # Test basic functionality in container
                    container_obj = self.docker_client.containers.get(container_id)
                    
                    # Test Python and transformers import
                    test_cmd = ['python', '-c', 'import transformers; print(f"Transformers {transformers.__version__} OK")']
                    exec_result = container_obj.exec_run(test_cmd, workdir='/workspace')
                    
                    if exec_result.exit_code == 0:
                        print(f"   SUCCESS: Basic functionality test passed")
                        
                        # Test GPU access if available
                        gpu_test_cmd = ['python', '-c', 'import torch; print(f"CUDA available: {torch.cuda.is_available()}")']
                        gpu_result = container_obj.exec_run(gpu_test_cmd, workdir='/workspace')
                        
                        if gpu_result.exit_code == 0:
                            print(f"   SUCCESS: GPU test: {gpu_result.output.decode('utf-8').strip()}")
                        
                        # Test sample model if available
                        models = self.get_available_models(container_name)
                        if models and container_name in models and models[container_name]:
                            sample_model = models[container_name][0]['name']
                            print(f"   Testing sample model: {sample_model}")
                            
                            # Quick availability test (not full evaluation)
                            model_test_cmd = [
                                'python', '-c',
                                f'from vlmeval.config import supported_VLM; print("{sample_model}:" + str("{sample_model}" in supported_VLM or any("{sample_model}" in v for v in supported_VLM.keys())))'
                            ]
                            model_result = container_obj.exec_run(model_test_cmd, workdir='/workspace')
                            
                            if model_result.exit_code == 0:
                                print(f"   SUCCESS: Model test: {model_result.output.decode('utf-8').strip()}")
                        
                        test_results[container_name] = 'OK'
                        
                    else:
                        test_results[container_name] = f'Basic test failed: {exec_result.output.decode("utf-8")}'
                        print(f"   ERROR: Basic functionality test failed")
                    
                    # Stop container
                    print(f"   Stopping container...")
                    self.stop_container(container_name)
                    print(f"   SUCCESS: Container stopped")
                    
                except Exception as e:
                    test_results[container_name] = f'Container test error: {str(e)}'
                    print(f"   ERROR: Container test failed: {e}")
                    
                    # Try to stop container if it exists
                    try:
                        if container_name in self.containers:
                            self.stop_container(container_name)
                    except:
                        pass
                        
                # Brief pause between container tests
                time.sleep(1)
            
            # Display test results summary
            print(f"\nSystem Test Results:")
            print("=" * 50)
            
            success_count = sum(1 for status in test_results.values() if status == 'OK')
            total_count = len(test_results)
            
            for container_name, status in test_results.items():
                status_icon = "SUCCESS" if status == "OK" else "ERROR"
                print(f"{status_icon}: {container_name}: {status}")
            
            print(f"\nSummary: {success_count}/{total_count} containers passed tests")
            
            if success_count == total_count:
                print("SUCCESS: All system tests passed! System is ready for evaluation.")
            else:
                print("WARNING: Some containers failed tests. Check logs for details.")
                
        except Exception as e:
            self.logger.error(f"System test failed: {e}")
            print(f"ERROR: System test error: {e}")
    
    def _interactive_model_selection(self, available_models: Dict[str, List[Dict]]) -> List[str]:
        """Interactive model selection with container grouping"""
        print(f"\nAvailable Models by Container:")
        
        all_models = []
        container_models = {}
        
        for container_name, models in available_models.items():
            print(f"\n{container_name}:")
            container_models[container_name] = []
            for i, model in enumerate(models):
                model_name = model['name']
                memory_gb = model.get('memory_gb', 'Unknown')
                print(f"   {len(all_models)+1:2d}. {model_name} ({memory_gb}GB)")
                all_models.append(model_name)
                container_models[container_name].append(model_name)
        
        print(f"\nTotal: {len(all_models)} models available")
        print(f"Options:")
        print(f"  - Enter numbers (e.g., '1,3,5' or '1-5')")
        print(f"  - Enter 'all' for all models")
        print(f"  - Enter container name for all models in that container")
        
        while True:
            try:
                selection = input("\nSelect models: ").strip()
                
                if not selection:
                    return []
                
                if selection.lower() == 'all':
                    return all_models
                
                # Check if it's a container name
                if selection in container_models:
                    return container_models[selection]
                
                # Parse number ranges and individual numbers
                selected_models = []
                parts = selection.split(',')
                
                for part in parts:
                    part = part.strip()
                    if '-' in part:
                        # Range
                        start, end = map(int, part.split('-'))
                        for i in range(start-1, min(end, len(all_models))):
                            if 0 <= i < len(all_models):
                                selected_models.append(all_models[i])
                    else:
                        # Individual number
                        idx = int(part) - 1
                        if 0 <= idx < len(all_models):
                            selected_models.append(all_models[idx])
                
                if selected_models:
                    # Remove duplicates while preserving order
                    unique_models = []
                    for model in selected_models:
                        if model not in unique_models:
                            unique_models.append(model)
                    return unique_models
                else:
                    print("Invalid selection. Please try again.")
                    
            except (ValueError, IndexError):
                print("Invalid format. Please use numbers, ranges, or 'all'.")
            except KeyboardInterrupt:
                return []
    
    def _interactive_benchmark_selection(self) -> List[str]:
        """Interactive benchmark selection"""
        # Load benchmark configurations
        benchmarks_config_path = Path('configs/benchmarks.yaml')
        if not benchmarks_config_path.exists():
            print("ERROR: Benchmark configuration not found")
            return []
        
        try:
            import yaml
            with open(benchmarks_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            benchmarks = config.get('benchmarks', [])
            
            print(f"\nAvailable Benchmarks ({len(benchmarks)}):")
            for i, benchmark in enumerate(benchmarks):
                name = benchmark['name']
                purpose = benchmark.get('purpose', 'No description')
                samples = benchmark.get('samples', 'Unknown')
                print(f"   {i+1:2d}. {name} ({samples} samples) - {purpose}")
            
            print(f"\nOptions:")
            print(f"  - Enter numbers (e.g., '1,3,5' or '1-5')")
            print(f"  - Enter 'all' for all benchmarks")
            print(f"  - Enter 'korean' for Korean benchmarks only")
            print(f"  - Enter 'basic' for basic benchmarks (MMBench, TextVQA, GQA)")
            
            while True:
                try:
                    selection = input("\nSelect benchmarks: ").strip()
                    
                    if not selection:
                        return []
                    
                    if selection.lower() == 'all':
                        return [b['name'] for b in benchmarks]
                    
                    if selection.lower() == 'korean':
                        return [b['name'] for b in benchmarks if b['name'].startswith('K-') or 'Korean' in b['name']]
                    
                    if selection.lower() == 'basic':
                        return ['MMBench', 'TextVQA', 'GQA']
                    
                    # Parse number ranges and individual numbers
                    selected_benchmarks = []
                    parts = selection.split(',')
                    
                    for part in parts:
                        part = part.strip()
                        if '-' in part:
                            # Range
                            start, end = map(int, part.split('-'))
                            for i in range(start-1, min(end, len(benchmarks))):
                                if 0 <= i < len(benchmarks):
                                    selected_benchmarks.append(benchmarks[i]['name'])
                        else:
                            # Individual number
                            idx = int(part) - 1
                            if 0 <= idx < len(benchmarks):
                                selected_benchmarks.append(benchmarks[idx]['name'])
                    
                    if selected_benchmarks:
                        # Remove duplicates while preserving order
                        unique_benchmarks = []
                        for benchmark in selected_benchmarks:
                            if benchmark not in unique_benchmarks:
                                unique_benchmarks.append(benchmark)
                        return unique_benchmarks
                    else:
                        print("Invalid selection. Please try again.")
                        
                except (ValueError, IndexError):
                    print("Invalid format. Please use numbers, ranges, or preset options.")
                except KeyboardInterrupt:
                    return []
                    
        except Exception as e:
            self.logger.error(f"Failed to load benchmark configuration: {e}")
            print(f"ERROR: Error loading benchmarks: {e}")
            return []
    
    def _interactive_gpu_selection(self) -> List[int]:
        """Interactive GPU selection"""
        print(f"\nGPU Configuration:")
        print(f"   Detected GPUs: {self.gpu_count}")
        print(f"   GPU settings: {self.gpu_config}")
        
        if self.gpu_count <= 1:
            print("   Using single GPU (GPU 0)")
            return [0]
        
        print(f"\nOptions:")
        print(f"  - Enter GPU IDs (e.g., '0,1,2')")
        print(f"  - Enter 'all' for all available GPUs")
        print(f"  - Press Enter for GPU 0 only")
        
        while True:
            try:
                selection = input(f"\nSelect GPUs [0]: ").strip()
                
                if not selection:
                    return [0]
                
                if selection.lower() == 'all':
                    return list(range(self.gpu_count))
                
                # Parse GPU IDs
                gpu_ids = []
                parts = selection.split(',')
                
                for part in parts:
                    gpu_id = int(part.strip())
                    if 0 <= gpu_id < self.gpu_count:
                        if gpu_id not in gpu_ids:
                            gpu_ids.append(gpu_id)
                    else:
                        print(f"Invalid GPU ID: {gpu_id} (available: 0-{self.gpu_count-1})")
                        gpu_ids = []
                        break
                
                if gpu_ids:
                    return sorted(gpu_ids)
                else:
                    print("Invalid GPU selection. Please try again.")
                    
            except (ValueError, IndexError):
                print("Invalid format. Please use GPU numbers separated by commas.")
            except KeyboardInterrupt:
                return [0]
    
    def _display_results_summary(self, results: List[Dict]):
        """Display evaluation results summary"""
        print(f"\nEvaluation Results Summary")
        print("=" * 60)
        
        if not results:
            print("No results to display.")
            return
        
        # Group results by status
        success_results = [r for r in results if r['status'] == 'success']
        error_results = [r for r in results if r['status'] == 'error']
        
        print(f"SUCCESS: Successful evaluations: {len(success_results)}")
        print(f"ERROR: Failed evaluations: {len(error_results)}")
        print(f"Success rate: {len(success_results)/len(results)*100:.1f}%")
        
        if error_results:
            print(f"\nFailed Evaluations:")
            for result in error_results[:10]:  # Show first 10 errors
                model = result['model']
                benchmark = result['benchmark']
                error = result.get('error', 'Unknown error')[:100]
                print(f"   {model} x {benchmark}: {error}")
            
            if len(error_results) > 10:
                print(f"   ... and {len(error_results) - 10} more errors")
        
        if success_results:
            print(f"\nRecent Successful Evaluations:")
            for result in success_results[-5:]:  # Show last 5 successes
                model = result['model']
                benchmark = result['benchmark']
                container = result.get('container', 'unknown')
                print(f"   SUCCESS: {model} x {benchmark} (in {container})")
        
        print(f"\nFull results saved to outputs/ directory")
        print(f"Check logs for detailed information")