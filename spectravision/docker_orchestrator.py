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
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from .config import ConfigManager
from .utils import setup_logger

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

# Load environment variables at module import
load_env_file()

class DockerOrchestrator:
    """Manages Docker containers for multi-version transformer evaluation with GPU support"""

    def __init__(self, config_path: str = "configs/models.yaml", gpu_count: int = None, monitor=None):
        self.logger = setup_logger("docker_orchestrator")
        self.config_path = Path(config_path)
        self.docker_client = None
        self.containers = {}
        self.container_configs = {}
        self.monitor = monitor  # Performance monitor (optional)
        
        # Initialize evaluation session with counter-based naming
        base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_counter = 1

        # Check if directory already exists and increment counter if needed
        while True:
            session_dir_name = f"{base_timestamp}_{session_counter:03d}"
            session_output_path = f"outputs/{session_dir_name}"
            if not os.path.exists(session_output_path):
                break
            session_counter += 1

        self.session_timestamp = session_dir_name
        self.session_output_dir = session_output_path
        
        # Get host user ID for proper file permissions (auto-detect if not set)
        self.host_uid = os.environ.get('HOST_UID', str(os.getuid()))
        self.host_gid = os.environ.get('HOST_GID', str(os.getgid()))
        self.logger.info(f"Using host UID:GID = {self.host_uid}:{self.host_gid}")
        
        # Detect GPU configuration
        self.gpu_count = gpu_count or self._detect_gpu_count()
        self.gpu_config = self._configure_gpu_settings()
        
        # Load Docker model configurations
        self.load_docker_configs()
        
        # Initialize Docker client
        self.init_docker_client()

        # Ensure proper permissions for output directories
        self._ensure_output_directories()
        
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

    def _ensure_output_directories(self):
        """Ensure output directories exist with proper permissions"""
        import stat

        # List of directories to create
        directories = [
            'outputs',
            'outputs/logs',
            'outputs/reports',
            'outputs/results',
            'outputs/vlmevalkit_results',
            'data'
        ]

        for dir_path in directories:
            try:
                os.makedirs(dir_path, exist_ok=True)
                # Set permissions to allow read/write for user and group
                os.chmod(dir_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
                self.logger.debug(f"Created directory with proper permissions: {dir_path}")
            except Exception as e:
                self.logger.warning(f"Could not set permissions for {dir_path}: {e}")

        self.logger.info("Output directories initialized with proper permissions")
        
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
            import docker
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

        except ImportError:
            self.logger.error("Docker Python library not found. Please install with: pip install docker")
            raise
        except Exception as e:
            # Handle all Docker-related errors more generically
            self.logger.error(f"Failed to initialize Docker client: {e}")
            raise
    
    def ensure_image_available(self, container_name: str) -> bool:
        """Ensure Docker image is available with strict validation"""
        # Extract version from container name (e.g., transformers_4_37 -> 4.37)
        parts = container_name.split('_')
        if len(parts) >= 3:
            version = f"{parts[1]}.{parts[2]}"  # transformers_4_37 -> 4.37
        else:
            version = parts[-1]  # fallback

        # CRITICAL FIX: Use standardized naming convention only
        standard_image_name = f"ghcr.io/gwleee/spectravision:{version}"

        try:
            # Check if standardized image exists
            self.docker_client.images.get(standard_image_name)
            self.logger.info(f"✅ Docker image available: {standard_image_name}")
            return True

        except docker.errors.ImageNotFound:
            self.logger.error(f"❌ CRITICAL: Docker image not found: {standard_image_name}")
            self.logger.error(f"Container {container_name} cannot be started without proper image")
            self.logger.error(f"Build the image first: docker build -t {standard_image_name} -f docker/transformers/Dockerfile --build-arg TRANSFORMERS_VERSION={version} .")

            # CRITICAL FIX: No fallback - fail fast for safety
            return False

        except Exception as e:
            self.logger.error(f"Error checking image availability: {e}")
            return False


    def find_container_for_model(self, model_name: str) -> Optional[str]:
        """Find which container supports the given model with flexible matching"""
        # First pass: exact matches
        for container_name, config in self.container_configs.items():
            if container_name in ['hardware_tiers', 'deployment_strategy']:
                continue
                
            if 'models' in config:
                for model in config['models']:
                    if model['name'] == model_name or model['vlm_id'] == model_name:
                        return container_name
        
        # Second pass: partial/fuzzy matches for common aliases
        model_aliases = {
            'SmolVLM': 'SmolVLM',  # SmolVLM-1.7B has vlm_id "SmolVLM"
            'InternVL2-2B': 'InternVL2-2B',
            'LLaVA-1.5-7B': 'llava_v1.5_7b',
            'CogVLM-7B': 'cogvlm-chat',
            'Qwen-VL-Chat': 'qwen_chat',
            'VisualGLM-6B': 'VisualGLM_6b'
        }
        
        # Check if input model name has a known alias
        if model_name in model_aliases:
            target_vlm_id = model_aliases[model_name]
            for container_name, config in self.container_configs.items():
                if container_name in ['hardware_tiers', 'deployment_strategy']:
                    continue
                    
                if 'models' in config:
                    for model in config['models']:
                        if model['vlm_id'] == target_vlm_id:
                            self.logger.info(f"Found model {model_name} -> {target_vlm_id} in {container_name}")
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
        """Build Docker containers with standardized naming"""
        if containers is None:
            containers = [name for name in self.container_configs.keys()
                         if name not in ['hardware_tiers', 'deployment_strategy']]

        # CRITICAL FIX: Build base image with standardized tag
        self.logger.info("Building base Docker image...")
        try:
            subprocess.run([
                'docker', 'build',
                '-t', 'ghcr.io/gwleee/spectravision:base',
                '-f', 'docker/base/Dockerfile',
                '.'
            ], cwd=Path.cwd(), check=True, capture_output=True, text=True)

            self.logger.info("Base image built successfully: ghcr.io/gwleee/spectravision:base")
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
            # CRITICAL FIX: Use standardized naming convention only
            version = f"{container_name.split('_')[1]}.{container_name.split('_')[2]}"  # e.g., "4.33"
            image_name = f"ghcr.io/gwleee/spectravision:{version}"

            # Verify standardized image exists (fail fast if not)
            self.docker_client.images.get(image_name)
            self.logger.info(f"Using standardized image: {image_name}")

            # Configure GPU access
            device_requests = [
                docker.types.DeviceRequest(
                    device_ids=[str(gpu_id)],
                    capabilities=[['gpu']]
                )
            ] if gpu_id is not None else []

            # Prepare environment variables including tokens from .env
            container_env = {
                'NVIDIA_VISIBLE_DEVICES': str(gpu_id),
                'CUDA_VISIBLE_DEVICES': str(gpu_id),
                'PYTHONPATH': '/workspace:/workspace/VLMEvalKit'
            }

            # Add important environment variables from host if they exist
            important_env_vars = [
                'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HF_HUB_TOKEN',
                'OPENAI_API_KEY', 'GITHUB_TOKEN',
                'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'
            ]

            for env_var in important_env_vars:
                if env_var in os.environ:
                    container_env[env_var] = os.environ[env_var]
                    self.logger.info(f"Passing {env_var} to container")

            # Container configuration with long-running command and proper user permissions
            container_config = {
                'image': image_name,
                'name': f"spectravision-{container_name}-{gpu_id}",
                'command': ['tail', '-f', '/dev/null'],  # Keep container running
                'detach': True,
                'runtime': 'nvidia',
                'user': f"{self.host_uid}:{self.host_gid}",  # Use host user permissions
                'environment': container_env,
                'volumes': {
                    '/workspace/outputs': {'bind': '/workspace/outputs', 'mode': 'rw'},
                    '/workspace/data': {'bind': '/workspace/data', 'mode': 'rw'},
                    # Mount current project directory to use latest code changes
                    str(Path.cwd()): {'bind': '/workspace/spectrabench-host', 'mode': 'ro'}
                },
                'working_dir': '/workspace',
                'device_requests': device_requests,
                'shm_size': '16G'  # Shared memory for large models
            }

            # CRITICAL FIX: Conditionally mount .env file only if it exists
            env_file_path = Path.cwd() / '.env'
            if env_file_path.exists():
                container_config['volumes'][str(env_file_path)] = {'bind': '/workspace/.env', 'mode': 'ro'}
                self.logger.info(f"Mounting .env file: {env_file_path}")
            else:
                self.logger.warning(f"No .env file found at {env_file_path}, skipping mount")
            
            # Start container
            container = self.docker_client.containers.run(**container_config)
            container_id = container.id
            
            self.logger.info(f"Started container {container_name} on GPU {gpu_id} (ID: {container_id[:12]})")
            self.containers[container_name] = container_id

            # CRITICAL: Run GPU smoke test immediately after container start
            smoke_test_success = self._run_container_smoke_test(container_name, container_id)
            if not smoke_test_success:
                self.logger.error(f"Container {container_name} failed smoke test - stopping immediately")
                # Stop and remove the failing container
                try:
                    container.stop(timeout=5)
                    container.remove()
                    del self.containers[container_name]
                except Exception as cleanup_error:
                    self.logger.error(f"Error cleaning up failed container: {cleanup_error}")
                raise RuntimeError(f"Container {container_name} failed GPU smoke test")

            return container_id
            
        except Exception as e:
            self.logger.error(f"Failed to start container {container_name}: {e}")
            raise

    def _run_container_smoke_test(self, container_name: str, container_id: str) -> bool:
        """Run GPU smoke test inside container immediately after startup"""
        try:
            self.logger.info(f"Running GPU smoke test for container {container_name}...")
            container = self.docker_client.containers.get(container_id)

            # Copy smoke test script to container
            smoke_test_script = "/workspace/spectrabench-host/scripts/runtime_gpu_smoke_test.py"

            # Run smoke test with timeout
            smoke_test_cmd = ['python3', smoke_test_script, '--json']
            exec_result = container.exec_run(
                smoke_test_cmd,
                workdir='/workspace',
                timeout=60  # 1 minute timeout for smoke test
            )

            success = exec_result.exit_code == 0
            output = exec_result.output.decode('utf-8', errors='ignore')

            if success:
                self.logger.info(f"✅ Container {container_name} passed GPU smoke test")
                # Log key hardware info from smoke test
                try:
                    import json
                    test_results = json.loads(output)
                    hw_info = test_results.get('hardware_info', {})
                    if hw_info:
                        gpu_name = hw_info.get('gpu_name', 'Unknown')
                        gpu_memory = hw_info.get('gpu_memory_gb', 0)
                        self.logger.info(f"GPU Hardware: {gpu_name} ({gpu_memory}GB)")
                except:
                    pass  # JSON parsing failed, but test passed
            else:
                self.logger.error(f"❌ Container {container_name} FAILED GPU smoke test")
                self.logger.error(f"Smoke test output:\n{output}")

                # Save failed smoke test results for debugging
                smoke_test_log = f"{self.session_output_dir}/smoke_test_failures.log"
                try:
                    with open(smoke_test_log, 'a') as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"Container: {container_name} ({container_id[:12]})\n")
                        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Exit Code: {exec_result.exit_code}\n")
                        f.write(f"Output:\n{output}\n")
                except Exception as log_error:
                    self.logger.warning(f"Could not save smoke test failure log: {log_error}")

            return success

        except Exception as e:
            self.logger.error(f"Error running smoke test for container {container_name}: {e}")
            return False  # Treat smoke test errors as failure
    
    def stop_container(self, container_name: str):
        """Stop a specific container using Docker Python API"""
        try:
            if container_name in self.containers:
                container_id = self.containers[container_name]
                
                # Get container object and stop it
                container = self.docker_client.containers.get(container_id)
                container.stop(timeout=10)
                container.remove()
                
                self.logger.info(f"Stopped container {container_name} (ID: {container_id[:12]})")
                del self.containers[container_name]
            else:
                self.logger.warning(f"Container {container_name} not found in active containers")
                
        except Exception as e:
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
        
        # Find model configuration with flexible matching
        model_config = None
        for model in self.container_configs[container_name]['models']:
            if model['name'] == model_name or model['vlm_id'] == model_name:
                model_config = model
                break
        
        # If not found directly, try alias mapping
        if not model_config:
            model_aliases = {
                'SmolVLM': 'SmolVLM',
                'InternVL2-2B': 'InternVL2-2B',
                'LLaVA-1.5-7B': 'llava_v1.5_7b',
                'CogVLM-7B': 'cogvlm-chat',
                'Qwen-VL-Chat': 'qwen_chat',
                'VisualGLM-6B': 'VisualGLM_6b'
            }
            
            if model_name in model_aliases:
                target_vlm_id = model_aliases[model_name]
                for model in self.container_configs[container_name]['models']:
                    if model['vlm_id'] == target_vlm_id:
                        model_config = model
                        self.logger.info(f"Found model {model_name} -> {target_vlm_id} in {container_name}")
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

            # Start performance monitoring if available
            if self.monitor:
                self.monitor.start_evaluation(model_name, benchmark_name)

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

            # End performance monitoring if available
            if self.monitor:
                peak_memory = self.monitor.end_evaluation()
                self.logger.info(f"Peak GPU memory usage: {peak_memory:.1f}GB")
            
            # Parse results
            if exec_result.exit_code == 0:
                # Copy results to timestamp-based directory structure and copy VLMEvalKit outputs
                try:
                    container_obj = self.docker_client.containers.get(container_id)

                    # Create model/benchmark directory structure
                    model_benchmark_dir = f"/workspace/{self.session_output_dir}/{model_name}/{benchmark_name}"
                    mkdir_cmd = ['sh', '-c', f'mkdir -p {model_benchmark_dir}']
                    container_obj.exec_run(mkdir_cmd, workdir='/workspace')

                    # Also create a legacy result directory for compatibility
                    result_dir = f"/workspace/{self.session_output_dir}/{model_name}"
                    mkdir_legacy_cmd = ['sh', '-c', f'mkdir -p {result_dir}']
                    container_obj.exec_run(mkdir_legacy_cmd, workdir='/workspace')

                    # Copy VLMEvalKit results to structured directory
                    copy_cmd = ['sh', '-c', f'cp -r /root/LMUData/* {result_dir}/ 2>/dev/null || true']
                    copy_result = container_obj.exec_run(copy_cmd, workdir='/workspace')

                    # Copy VLMEvalKit outputs directory to SpectraBench-Vision outputs
                    self._copy_vlmevalkit_outputs_from_container(container_obj, model_name, benchmark_name)

                    # Generate individual model-benchmark summary
                    summarize_cmd = ['python', '/workspace/VLMEvalKit/scripts/summarize.py', '--model', vlm_id, '--data', benchmark_name]
                    summary_result = container_obj.exec_run(summarize_cmd, workdir='/workspace')

                    if summary_result.exit_code == 0:
                        # Save individual summary
                        individual_summary_path = f"{result_dir}/{model_name}_{benchmark_name}_summary.csv"
                        copy_summary_cmd = ['sh', '-c', f'cp summ.csv {individual_summary_path} 2>/dev/null || true']
                        container_obj.exec_run(copy_summary_cmd, workdir='/workspace')
                        self.logger.info(f"Generated individual summary: {individual_summary_path}")
                    else:
                        self.logger.warning(f"Failed to generate individual summary: {summary_result.output.decode('utf-8')}")

                    self.logger.info(f"Results saved to structured directory: {result_dir}")

                    # Fix file permissions for host access
                    self._fix_output_permissions(result_dir)

                    # Also fix permissions for VLMEvalKit results
                    vlmevalkit_results_path = f"/workspace/outputs/vlmevalkit_results"
                    self._fix_output_permissions(vlmevalkit_results_path)

                except Exception as e:
                    self.logger.warning(f"Error processing results: {e}")
                
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
            # Stop containers tracked by this orchestrator
            for container_name in list(self.containers.keys()):
                try:
                    self.stop_container(container_name)
                except Exception as e:
                    self.logger.warning(f"Failed to stop tracked container {container_name}: {e}")
            
            # Stop any remaining SpectraVision containers
            running_containers = self.docker_client.containers.list()
            for container in running_containers:
                if 'spectravision' in container.name:
                    try:
                        self.logger.info(f"Stopping untracked container: {container.name}")
                        container.stop(timeout=10)
                        container.remove()
                    except Exception as e:
                        self.logger.warning(f"Failed to stop container {container.name}: {e}")
            
            self.logger.info("Container cleanup completed")
                        
        except Exception as e:
            self.logger.error(f"Container cleanup failed: {e}")
            raise
        
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
    
    def _get_all_model_names(self) -> List[str]:
        """Get all available model names from all containers"""
        all_models = []
        for container_name, config in self.container_configs.items():
            if container_name in ['hardware_tiers', 'deployment_strategy']:
                continue
            if 'models' in config:
                for model in config['models']:
                    # Use vlm_id as primary identifier (matches VLMEvalKit)
                    all_models.append(model['vlm_id'])
        return all_models
    
    def _get_all_benchmark_names(self) -> List[str]:
        """Get all available benchmark names"""
        # Standard VLMEvalKit benchmarks
        return [
            "MMBench_DEV_EN", "MMBench_TEST_EN", "MMBench_DEV_CN", "MMBench_TEST_CN",
            "CCBench", "MMStar", "RealWorldQA", "MLLMGuard_DS", "BLINK",
            "TextVQA_VAL", "ChartQA_VAL", "GQA_VAL", "VizWiz_VAL",
            "DocVQA_VAL", "InfoVQA_VAL", "OCRBench", "AI2D_TEST",
            "MathVista_MINI", "HallusionBench", "LLaVABench", "MMVet",
            "SEED_IMG", "Pope", "ScienceQA_VAL", "MMT-Bench_VAL", "Q-Bench_VAL"
        ]
    
    def run_batch_evaluation(self, models: List[str], benchmarks: List[str], 
                           gpu_ids: List[int] = [0]) -> List[Dict]:
        """Run batch evaluation of multiple models and benchmarks"""
        
        # Handle "all" parameters
        if models == ["all"] or (len(models) == 1 and models[0] == "all"):
            models = self._get_all_model_names()
            self.logger.info(f"Expanded 'all' to {len(models)} models: {models[:5]}{'...' if len(models) > 5 else ''}")
        
        if benchmarks == ["all"] or (len(benchmarks) == 1 and benchmarks[0] == "all"):
            benchmarks = self._get_all_benchmark_names()
            self.logger.info(f"Expanded 'all' to {len(benchmarks)} benchmarks: {benchmarks[:5]}{'...' if len(benchmarks) > 5 else ''}")
        
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
        
        # Generate comprehensive summary after all evaluations
        try:
            self.logger.info("Generating comprehensive evaluation summary...")
            self.generate_final_summary(results)

            # Generate final report with accuracy tables
            self.generate_final_report_with_tables(results)
        except Exception as e:
            self.logger.warning(f"Failed to generate final summary: {e}")
            
        self.logger.info(f"Batch evaluation completed: {len(results)} results")
        return results
    
    def generate_final_summary(self, results: List[Dict]):
        """Generate comprehensive final report with detailed analysis"""
        try:
            import pandas as pd
            
            # Extract successful evaluations
            successful_results = [r for r in results if r.get('status') == 'success']
            failed_results = [r for r in results if r.get('status') == 'error']
            
            if not successful_results and not failed_results:
                self.logger.warning("No evaluation results to summarize")
                return
            
            # Create comprehensive report directory
            report_dir = f"{self.session_output_dir}/reports"
            os.makedirs(report_dir, exist_ok=True)
            
            # 1. Create evaluation summary DataFrame
            summary_data = []
            model_benchmark_map = {}
            
            for result in successful_results:
                model = result.get('model', '')
                benchmark = result.get('benchmark', '')
                container = result.get('container', 'unknown')
                duration = result.get('duration_seconds', 0.0)
                peak_memory = result.get('peak_gpu_memory_gb', 0.0)

                if model and benchmark:
                    if model not in model_benchmark_map:
                        model_benchmark_map[model] = {'benchmarks': [], 'container': container}
                    model_benchmark_map[model]['benchmarks'].append(benchmark)

                    summary_data.append({
                        'Model': model,
                        'Benchmark': benchmark,
                        'Container': container,
                        'Status': 'Success',
                        'Duration_Seconds': f"{duration:.1f}" if duration > 0 else 'N/A',
                        'Peak_GPU_Memory_GB': f"{peak_memory:.1f}" if peak_memory > 0 else 'N/A',
                        'Session': self.session_timestamp
                    })

            # Add failed evaluations with detailed error information
            for result in failed_results:
                error_msg = result.get('error', 'Unknown error')
                error_type = result.get('error_type', 'RuntimeError')

                summary_data.append({
                    'Model': result.get('model', 'unknown'),
                    'Benchmark': result.get('benchmark', 'unknown'),
                    'Container': result.get('container', 'unknown'),
                    'Status': 'Failed',
                    'Duration_Seconds': 'N/A',
                    'Peak_GPU_Memory_GB': 'N/A',
                    'Error_Type': error_type,
                    'Error_Message': error_msg[:200],  # Limit error message length
                    'Session': self.session_timestamp
                })
            
            # 2. Save detailed evaluation log
            detailed_df = pd.DataFrame(summary_data)
            detailed_path = f"{report_dir}/detailed_evaluation_log.csv"
            detailed_df.to_csv(detailed_path, index=False)
            
            # 3. Create model-wise summary
            model_summary = []
            for model, info in model_benchmark_map.items():
                model_summary.append({
                    'Model': model,
                    'Container': info['container'],
                    'Successful_Benchmarks': len(info['benchmarks']),
                    'Benchmark_List': ', '.join(info['benchmarks']),
                    'Session_Timestamp': self.session_timestamp
                })
            
            model_df = pd.DataFrame(model_summary)
            model_summary_path = f"{report_dir}/model_summary.csv"
            model_df.to_csv(model_summary_path, index=False)
            
            # 4. Create comprehensive report metadata
            report_metadata = {
                'session_timestamp': self.session_timestamp,
                'total_evaluations': len(results),
                'successful_evaluations': len(successful_results),
                'failed_evaluations': len(failed_results),
                'unique_models': len(model_benchmark_map),
                'total_model_benchmark_pairs': len([b for info in model_benchmark_map.values() for b in info['benchmarks']]),
                'gpu_configuration': {
                    'gpu_count': self.gpu_count,
                    'cuda_visible_devices': self.gpu_config.get('CUDA_VISIBLE_DEVICES', 'unknown')
                },
                'container_usage': list(set([info['container'] for info in model_benchmark_map.values()])),
                'output_directory': self.session_output_dir
            }
            
            metadata_path = f"{report_dir}/evaluation_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(report_metadata, f, indent=2)
            
            # 5. Generate final report
            self._generate_final_report_file(report_dir, report_metadata, model_df, detailed_df)
            
            # 6. Fix permissions for all generated files
            self._fix_output_permissions(self.session_output_dir)
            
            # Log success
            self.logger.info(f"📊 Comprehensive evaluation report generated:")
            self.logger.info(f"   📁 Report Directory: {report_dir}")
            self.logger.info(f"   📝 Model Summary: {model_summary_path}")
            self.logger.info(f"   📋 Detailed Log: {detailed_path}")
            self.logger.info(f"   🔧 Metadata: {metadata_path}")
            self.logger.info(f"✅ Session {self.session_timestamp}: {len(successful_results)}/{len(results)} evaluations successful")
            
            # Print summary table
            print("\n" + "="*100)
            print(f"🎯 SPECTRAVISION EVALUATION REPORT - SESSION {self.session_timestamp}")
            print("="*100)
            if not model_df.empty:
                print("📊 MODEL PERFORMANCE SUMMARY:")
                print(model_df.to_string(index=False))
                print("\n📋 EVALUATION DETAILS:")
                print(f"   Total Evaluations: {len(results)}")
                print(f"   Successful: {len(successful_results)} ✅")
                print(f"   Failed: {len(failed_results)} ❌")
                print(f"   Unique Models: {len(model_benchmark_map)}")
                print(f"   GPU Configuration: {self.gpu_count} GPU(s)")
            print(f"\n📁 Results saved to: {self.session_output_dir}")
            print("="*100)
            
        except Exception as e:
            self.logger.error(f"Failed to generate final summary: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _generate_final_report_file(self, report_dir: str, metadata: dict, model_df, detailed_df):
        """Generate a human-readable final report file"""
        try:
            import pandas as pd
            report_path = f"{report_dir}/EVALUATION_REPORT.md"
            
            with open(report_path, 'w') as f:
                f.write(f"# SpectraVision Evaluation Report\n\n")
                f.write(f"**Session:** {metadata['session_timestamp']}\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"## Summary\n\n")
                f.write(f"- **Total Evaluations:** {metadata['total_evaluations']}\n")
                f.write(f"- **Successful:** {metadata['successful_evaluations']} ✅\n")
                f.write(f"- **Failed:** {metadata['failed_evaluations']} ❌\n")
                f.write(f"- **Success Rate:** {metadata['successful_evaluations']/metadata['total_evaluations']*100:.1f}%\n")
                f.write(f"- **Unique Models:** {metadata['unique_models']}\n")
                f.write(f"- **GPU Count:** {metadata['gpu_configuration']['gpu_count']}\n\n")
                
                f.write(f"## Model Performance\n\n")
                if not model_df.empty:
                    f.write(model_df.to_markdown(index=False))
                f.write(f"\n\n")
                
                f.write(f"## Container Usage\n\n")
                for container in metadata['container_usage']:
                    f.write(f"- `{container}`\n")
                f.write(f"\n")
                
                f.write(f"## Output Structure\n\n")
                f.write(f"```\n")
                f.write(f"{metadata['output_directory']}/\n")
                f.write(f"├── reports/\n")
                f.write(f"│   ├── EVALUATION_REPORT.md\n")
                f.write(f"│   ├── model_summary.csv\n")
                f.write(f"│   ├── detailed_evaluation_log.csv\n")
                f.write(f"│   └── evaluation_metadata.json\n")
                for model in model_df['Model'].unique():
                    f.write(f"├── {model}/\n")
                    f.write(f"│   └── [VLMEvalKit results]\n")
                f.write(f"```\n\n")
                
                f.write(f"## System Information\n\n")
                f.write(f"- **Session Timestamp:** {metadata['session_timestamp']}\n")
                f.write(f"- **Output Directory:** `{metadata['output_directory']}`\n")
                f.write(f"- **CUDA Visible Devices:** {metadata['gpu_configuration']['cuda_visible_devices']}\n")
            
            self.logger.info(f"📄 Final report saved: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate final report file: {e}")
    
    def _copy_vlmevalkit_outputs_from_container(self, container_obj, model_name: str, benchmark_name: str):
        """Copy VLMEvalKit outputs from Docker container to host SpectraBench-Vision outputs"""
        try:
            # CRITICAL FIX: Ensure consistent host/container path mapping
            host_model_dir = f"{self.session_output_dir}/{model_name}"
            host_model_benchmark_dir = f"{self.session_output_dir}/{model_name}/{benchmark_name}"

            # Container paths (absolute within container)
            # CRITICAL FIX: Container paths must match mounted volume paths
            container_model_dir = f"/workspace/{self.session_output_dir}/{model_name}"
            container_model_benchmark_dir = f"/workspace/{self.session_output_dir}/{model_name}/{benchmark_name}"

            # Create directories in container first
            mkdir_cmd = ['sh', '-c', f'mkdir -p {container_model_benchmark_dir}']
            container_obj.exec_run(mkdir_cmd, workdir='/workspace')

            # Also create on host (mounted volume)
            os.makedirs(host_model_benchmark_dir, exist_ok=True)

            # Copy VLMEvalKit outputs to target directory
            vlmevalkit_outputs = "/workspace/VLMEvalKit/outputs"

            # Check if VLMEvalKit outputs directory exists in container
            check_cmd = ['sh', '-c', f'ls -la {vlmevalkit_outputs} 2>/dev/null || echo "NO_OUTPUTS_DIR"']
            check_result = container_obj.exec_run(check_cmd, workdir='/workspace')

            if "NO_OUTPUTS_DIR" in check_result.output.decode('utf-8'):
                self.logger.warning(f"VLMEvalKit outputs directory not found in container: {vlmevalkit_outputs}")
                return

            self.logger.info(f"Copying VLMEvalKit outputs from container to host outputs directory...")

            # Look for model-specific directories and files
            model_patterns = [
                model_name,
                model_name.replace('-', '_'),
                model_name.replace('_', '-'),
                model_name.lower(),
                model_name.upper()
            ]

            copied_files_count = 0

            # Search and copy model directories
            for model_pattern in model_patterns:
                # Find and copy model directories
                find_model_dir_cmd = ['find', vlmevalkit_outputs, '-type', 'd', '-name', f'*{model_pattern}*', '-print0']
                find_result = container_obj.exec_run(find_model_dir_cmd, workdir='/workspace')

                if find_result.exit_code == 0 and find_result.output:
                    model_dirs = find_result.output.decode('utf-8').strip('\0').split('\0')

                    for model_dir in model_dirs:
                        if model_dir.strip():
                            # CRITICAL FIX: Copy to correct container path (will be visible on host via volume mount)
                            target_model_dir = f"{container_model_benchmark_dir}/{os.path.basename(model_dir)}"
                            copy_dir_cmd = ['sh', '-c', f'mkdir -p "{os.path.dirname(target_model_dir)}" && cp -r "{model_dir}" "{target_model_dir}" 2>/dev/null && echo "COPIED_DIR: {model_dir}" || echo "FAILED_DIR: {model_dir}"']
                            copy_result = container_obj.exec_run(copy_dir_cmd, workdir='/workspace')

                            if "COPIED_DIR:" in copy_result.output.decode('utf-8'):
                                copied_files_count += 1
                                self.logger.debug(f"Copied directory: {model_dir}")

            # Search and copy individual result files
            benchmark_patterns = [
                benchmark_name,
                benchmark_name.replace('-', '_'),
                benchmark_name.replace('_', '-'),
                benchmark_name.lower(),
                benchmark_name.upper()
            ]

            for model_pattern in model_patterns:
                for bench_pattern in benchmark_patterns:
                    # Search for result files with both model and benchmark names
                    patterns_to_search = [
                        f'*{model_pattern}*{bench_pattern}*',
                        f'*{bench_pattern}*{model_pattern}*',
                        f'{model_pattern}_{bench_pattern}*',
                        f'{bench_pattern}_{model_pattern}*'
                    ]

                    for pattern in patterns_to_search:
                        # Find files matching pattern
                        find_files_cmd = ['find', vlmevalkit_outputs, '-type', 'f', '-name', pattern, '-print0']
                        find_result = container_obj.exec_run(find_files_cmd, workdir='/workspace')

                        if find_result.exit_code == 0 and find_result.output:
                            result_files = find_result.output.decode('utf-8').strip('\0').split('\0')

                            for file_path in result_files:
                                # CRITICAL: Format priority - CSV > XLSX > JSON > TXT (matches evaluator.py:866)
                                if file_path.strip() and any(file_path.endswith(ext) for ext in ['.csv', '.xlsx', '.json', '.txt']):
                                    # CRITICAL FIX: Copy to correct container path (will be visible on host via volume mount)
                                    filename = os.path.basename(file_path)
                                    copy_file_cmd = ['sh', '-c', f'mkdir -p "{container_model_benchmark_dir}" && cp "{file_path}" "{container_model_benchmark_dir}/{filename}" 2>/dev/null && echo "COPIED_FILE: {file_path}" || echo "FAILED_FILE: {file_path}"']
                                    copy_result = container_obj.exec_run(copy_file_cmd, workdir='/workspace')

                                    if "COPIED_FILE:" in copy_result.output.decode('utf-8'):
                                        copied_files_count += 1
                                        self.logger.debug(f"Copied result file: {file_path}")

            if copied_files_count > 0:
                self.logger.info(f"Successfully copied {copied_files_count} files/directories from VLMEvalKit container outputs")
            else:
                self.logger.warning(f"No VLMEvalKit output files found for model '{model_name}' and benchmark '{benchmark_name}' in container")

                # CRITICAL FIX: Copy any recent files as fallback with correct path mapping
                fallback_target_dir = f"{container_model_benchmark_dir}/recent_outputs"
                host_fallback_dir = f"{host_model_benchmark_dir}/recent_outputs"
                mkdir_fallback_cmd = ['sh', '-c', f'mkdir -p "{fallback_target_dir}"']
                container_obj.exec_run(mkdir_fallback_cmd, workdir='/workspace')

                # Also create on host
                os.makedirs(host_fallback_dir, exist_ok=True)

                # Find recent files (modified in last hour)
                recent_files_cmd = ['find', vlmevalkit_outputs, '-type', 'f',
                                   '(', '-name', '*.csv', '-o', '-name', '*.xlsx', '-o', '-name', '*.json', ')',
                                   '-mmin', '-60', '-print0']
                recent_result = container_obj.exec_run(recent_files_cmd, workdir='/workspace')

                if recent_result.exit_code == 0 and recent_result.output:
                    recent_files = recent_result.output.decode('utf-8').strip('\0').split('\0')

                    for file_path in recent_files[:3]:  # Copy up to 3 recent files
                        if file_path.strip():
                            filename = os.path.basename(file_path)
                            copy_recent_cmd = ['sh', '-c', f'cp "{file_path}" "{fallback_target_dir}/{filename}" 2>/dev/null && echo "COPIED_RECENT: {filename}" || echo "FAILED_RECENT: {filename}"']
                            copy_result = container_obj.exec_run(copy_recent_cmd, workdir='/workspace')

                            if "COPIED_RECENT:" in copy_result.output.decode('utf-8', errors='ignore'):
                                self.logger.info(f"Copied recent file as fallback: {filename}")
                            else:
                                self.logger.warning(f"Failed to copy recent file: {filename}")

        except Exception as e:
            self.logger.error(f"Error copying VLMEvalKit outputs from container: {e}")

    def generate_final_report_with_tables(self, results: List[Dict]):
        """Generate final report with accuracy tables for Docker orchestrator results"""
        try:
            import pandas as pd

            self.logger.info("Generating final report with accuracy tables...")

            # Create final_report directory
            final_report_dir = f"{self.session_output_dir}/final_report"
            mkdir_cmd = f"mkdir -p {final_report_dir}"
            os.system(mkdir_cmd)

            # Parse results to extract accuracy information
            accuracy_data = []
            successful_results = []

            for result in results:
                if result.get('status') == 'success':
                    model_name = result.get('model', 'Unknown')
                    benchmark_name = result.get('benchmark', 'Unknown')

                    # Try to read accuracy from result files
                    accuracy_score = self._extract_accuracy_from_docker_result(model_name, benchmark_name)

                    if accuracy_score is not None:
                        accuracy_data.append({
                            'Model': model_name,
                            'Benchmark': benchmark_name,
                            'Accuracy': f"{accuracy_score:.1%}" if 0 <= accuracy_score <= 1 else f"{accuracy_score:.2f}",
                            'Status': 'Success'
                        })
                        successful_results.append(result)
                    else:
                        accuracy_data.append({
                            'Model': model_name,
                            'Benchmark': benchmark_name,
                            'Accuracy': 'N/A',
                            'Status': 'No Score'
                        })
                else:
                    accuracy_data.append({
                        'Model': result.get('model', 'Unknown'),
                        'Benchmark': result.get('benchmark', 'Unknown'),
                        'Accuracy': 'Failed',
                        'Status': 'Error'
                    })

            if not accuracy_data:
                self.logger.warning("No results to generate final report")
                return

            # Create accuracy summary CSV
            accuracy_df = pd.DataFrame(accuracy_data)
            accuracy_summary_file = f"{final_report_dir}/accuracy_summary.csv"
            accuracy_df.to_csv(accuracy_summary_file, index=False)
            self.logger.info(f"Accuracy summary saved to: {accuracy_summary_file}")

            # Create model vs benchmark matrix
            successful_df = accuracy_df[accuracy_df['Status'] == 'Success']
            if not successful_df.empty:
                pivot_df = successful_df.pivot(index='Model', columns='Benchmark', values='Accuracy')
                pivot_df = pivot_df.fillna('N/A')
                pivot_table_file = f"{final_report_dir}/accuracy_matrix.csv"
                pivot_df.to_csv(pivot_table_file)
                self.logger.info(f"Accuracy matrix saved to: {pivot_table_file}")

            # Generate HTML report
            self._generate_docker_html_report(accuracy_df, final_report_dir, results)

            # Generate summary statistics
            self._generate_docker_summary_statistics(results, final_report_dir)

            self.logger.info(f"Final report with accuracy tables generated in: {final_report_dir}")

        except Exception as e:
            self.logger.error(f"Error generating final report with tables: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _extract_accuracy_from_docker_result(self, model_name: str, benchmark_name: str) -> float:
        """Extract accuracy score from result files"""
        try:
            import pandas as pd
            import numpy as np

            # Look for result files in model/benchmark directory
            result_dir = f"{self.session_output_dir}/{model_name}/{benchmark_name}"

            if not os.path.exists(result_dir):
                return None

            # Look for CSV files with accuracy information
            import glob
            csv_files = glob.glob(f"{result_dir}/*.csv")

            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)

                    # Look for common accuracy column names
                    accuracy_columns = ['Overall', 'Accuracy', 'accuracy', 'Score', 'score']
                    for col in accuracy_columns:
                        if col in df.columns:
                            # Get the last row's value
                            value = df[col].iloc[-1]
                            if pd.notna(value) and isinstance(value, (int, float)):
                                return float(value)

                    # If no direct accuracy column, look for numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        # Use the last numeric column's last value
                        value = df[numeric_cols[-1]].iloc[-1]
                        if pd.notna(value):
                            return float(value)

                except Exception:
                    continue

            return None

        except Exception:
            return None

    def _generate_docker_html_report(self, accuracy_df, final_report_dir: str, results: List[Dict]):
        """Generate HTML report for Docker orchestrator results"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            session_name = self.session_timestamp

            successful_count = len([r for r in results if r.get('status') == 'success'])
            failed_count = len(results) - successful_count

            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SpectraBench-Vision Docker Report - {session_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #34495e; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .success {{ color: #27ae60; }}
        .error {{ color: #e74c3c; }}
        .summary-stats {{ background: #ecf0f1; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>SpectraBench-Vision Docker Evaluation Report</h1>
        <p>Session: {session_name} | Generated: {timestamp}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div class="summary-stats">
            <p><strong>Total Evaluations:</strong> {len(results)}</p>
            <p><strong>Successful:</strong> <span class="success">{successful_count}</span></p>
            <p><strong>Failed:</strong> <span class="error">{failed_count}</span></p>
            <p><strong>Success Rate:</strong> {successful_count/len(results)*100:.1f}%</p>
            <p><strong>Models Tested:</strong> {len(accuracy_df['Model'].unique())}</p>
            <p><strong>Benchmarks Tested:</strong> {len(accuracy_df['Benchmark'].unique())}</p>
        </div>
    </div>

    <div class="section">
        <h2>Accuracy Results</h2>
        {accuracy_df.to_html(index=False, escape=False)}
    </div>

</body>
</html>
"""

            html_file = f"{final_report_dir}/docker_performance_report.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            self.logger.info(f"Docker HTML report saved to: {html_file}")

        except Exception as e:
            self.logger.error(f"Error generating Docker HTML report: {e}")

    def _generate_docker_summary_statistics(self, results: List[Dict], final_report_dir: str):
        """Generate summary statistics for Docker orchestrator results"""
        try:
            successful_results = [r for r in results if r.get('status') == 'success']
            failed_results = [r for r in results if r.get('status') != 'success']

            stats = {
                'session_info': {
                    'session_name': self.session_timestamp,
                    'timestamp': datetime.now().isoformat(),
                    'total_evaluations': len(results),
                    'successful_evaluations': len(successful_results),
                    'failed_evaluations': len(failed_results),
                    'success_rate': len(successful_results) / len(results) * 100 if results else 0
                },
                'docker_stats': {
                    'gpu_count': self.gpu_count,
                    'gpu_config': self.gpu_config,
                    'session_output_dir': self.session_output_dir
                },
                'model_stats': {
                    'total_models': len(set(r.get('model', '') for r in results)),
                    'models_list': sorted(list(set(r.get('model', '') for r in results if r.get('model'))))
                },
                'benchmark_stats': {
                    'total_benchmarks': len(set(r.get('benchmark', '') for r in results)),
                    'benchmarks_list': sorted(list(set(r.get('benchmark', '') for r in results if r.get('benchmark'))))
                }
            }

            stats_file = f"{final_report_dir}/docker_summary_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)

            self.logger.info(f"Docker summary statistics saved to: {stats_file}")

        except Exception as e:
            self.logger.error(f"Error generating Docker summary statistics: {e}")

    def _fix_output_permissions(self, output_path: str):
        """Fix file permissions for host user access"""
        try:
            # Change ownership to host user
            chown_cmd = f"chown -R {self.host_uid}:{self.host_gid} {output_path}"
            os.system(chown_cmd)
            self.logger.debug(f"Fixed permissions for {output_path} to {self.host_uid}:{self.host_gid}")
        except Exception as e:
            self.logger.warning(f"Failed to fix permissions for {output_path}: {e}")
    
    def get_session_info(self) -> dict:
        """Get current session information"""
        return {
            'timestamp': self.session_timestamp,
            'output_dir': self.session_output_dir,
            'gpu_count': self.gpu_count,
            'gpu_config': self.gpu_config,
            'host_uid': self.host_uid,
            'host_gid': self.host_gid
        }
    
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

    def run_evaluation(self, eval_config: Dict) -> bool:
        """Run full evaluation using Docker containers"""
        try:
            self.logger.info("Starting Docker-based multi-container evaluation...")

            models_config = eval_config.get('models', {})
            benchmarks_config = eval_config.get('benchmarks', {})

            if not models_config:
                self.logger.error("No models configuration found")
                return False

            if not benchmarks_config:
                self.logger.error("No benchmarks configuration found")
                return False

            # Count total evaluations
            total_evaluations = 0
            for container_name, container_config in models_config.items():
                if container_name in ['hardware_tiers', 'deployment_strategy']:
                    continue
                models = container_config.get('models', []) if isinstance(container_config, dict) else container_config
                total_evaluations += len(models) * len(benchmarks_config)

            self.logger.info(f"Planning {total_evaluations} total evaluations")

            # Execute evaluations container by container
            results = []
            gpu_idx = 0

            for container_name, container_config in models_config.items():
                if container_name in ['hardware_tiers', 'deployment_strategy']:
                    continue

                self.logger.info(f"Processing container: {container_name}")

                # Get models for this container
                models = container_config.get('models', []) if isinstance(container_config, dict) else container_config

                if not models:
                    self.logger.warning(f"No models found for container {container_name}")
                    continue

                # Start container for this transformer version
                current_gpu = gpu_idx % len(self.gpu_ids)
                try:
                    self.start_container(container_name, current_gpu)

                    # Run evaluations for all models in this container
                    for model in models:
                        model_name = model.get('name') if isinstance(model, dict) else str(model)
                        self.logger.info(f"Evaluating model: {model_name}")

                        for benchmark_name in benchmarks_config.keys():
                            self.logger.info(f"  Running {model_name} on {benchmark_name}")

                            try:
                                result = self.run_evaluation_in_container(
                                    container_name, model_name, benchmark_name, current_gpu
                                )
                                results.append(result)

                                if result['status'] == 'success':
                                    self.logger.info(f"  ✅ {model_name} × {benchmark_name} completed")
                                else:
                                    self.logger.error(f"  ❌ {model_name} × {benchmark_name} failed: {result.get('error', 'Unknown error')}")

                            except Exception as e:
                                self.logger.error(f"  💥 {model_name} × {benchmark_name} crashed: {str(e)}")
                                results.append({
                                    'status': 'error',
                                    'model': model_name,
                                    'benchmark': benchmark_name,
                                    'container': container_name,
                                    'error': str(e)
                                })

                    # Stop container after processing
                    self.stop_container(container_name)

                except Exception as e:
                    self.logger.error(f"Failed to process container {container_name}: {e}")
                    continue

                gpu_idx += 1

            # Display results summary
            self._display_results_summary(results)

            success_count = len([r for r in results if r['status'] == 'success'])
            total_count = len(results)

            self.logger.info(f"Evaluation completed: {success_count}/{total_count} successful")

            return success_count > 0

        except Exception as e:
            self.logger.error(f"Evaluation failed with error: {e}")
            return False