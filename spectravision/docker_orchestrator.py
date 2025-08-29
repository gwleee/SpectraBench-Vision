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
            subprocess.run([
                'docker-compose',
                '-f', 'docker/docker-compose.yml',
                'down'
            ], cwd=Path.cwd(), check=True)
            
            self.logger.info("All containers stopped successfully")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to cleanup containers: {e}")
        
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