"""
Configuration Manager for SpectraVision
Handles YAML config loading, hardware detection, and configuration validation
"""

import os
import yaml
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration loading and hardware detection"""
    
    def __init__(self, hardware_type: str = "auto", output_dir: str = "outputs"):
        """
        Initialize ConfigManager
        
        Args:
            hardware_type: Hardware configuration type or 'auto' for detection
            output_dir: Output directory for results and logs
        """
        self.hardware_type = hardware_type
        
        # Convert to absolute path
        if Path(output_dir).is_absolute():
            self.output_dir = Path(output_dir)
        else:
            # Find project root based on config.py location and interpret relative path
            project_root = Path(__file__).parent.parent
            self.output_dir = project_root / output_dir
        
        self.output_dir = self.output_dir.resolve()
        
        # Find configs directory from project root
        self.config_dir = Path(__file__).parent.parent / "configs"
        self.config = {}
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ConfigManager initialized:")
        logger.info(f"   Config dir: {self.config_dir}")
        logger.info(f"   Output dir: {self.output_dir}")
        
    def detect_hardware(self) -> str:
        """
        Auto-detect hardware configuration based on available GPUs
        
        Returns:
            Hardware type string (a6000, a100_single, a100_dual, a100_quad, h100_dual, etc.)
        """
        logger.info("Auto-detecting hardware configuration...")
        
        try:
            # Get GPU memory information
            cmd = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse memory values (in MB)
            memory_values = [int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
            
            if not memory_values:
                logger.warning("No GPUs detected, falling back to a6000 config")
                return "a6000"
            
            gpu_count = len(memory_values)
            max_memory_gb = max(memory_values) / 1024  # Convert to GB
            
            logger.info(f"Detected {gpu_count} GPU(s) with max {max_memory_gb:.1f}GB memory")
            
            # Load hardware thresholds
            hardware_config = self._load_yaml("hardware.yaml")
            thresholds = hardware_config.get("auto_detection", {})
            
            # Determine hardware type based on memory and count
            # Check for cluster configurations first (10+ GPUs)
            if gpu_count >= thresholds.get("cluster_gpu_min_count", 10):
                detected = "gpu_cluster"
                logger.info(f"Detected GPU cluster configuration with {gpu_count} GPUs")
            elif max_memory_gb >= thresholds.get("h200_min_memory", 135):
                if gpu_count >= 4:
                    detected = "h200_quad" if "h200_quad" in hardware_config else "h200_dual"
                elif gpu_count >= 2:
                    detected = "h200_dual"
                else:
                    detected = "h200_141gb"
            elif max_memory_gb >= thresholds.get("h100_min_memory", 75):
                if gpu_count >= 4:
                    detected = "h100_quad"
                elif gpu_count >= 2:
                    detected = "h100_dual"
                else:
                    detected = "h100_80gb"
            elif max_memory_gb >= thresholds.get("a100_min_memory", 75):
                if gpu_count >= thresholds.get("octo_gpu_min_count", 8):
                    detected = "a100_octo"
                elif gpu_count >= 4:
                    detected = "a100_quad"
                elif gpu_count >= 2:
                    detected = "a100_dual"
                else:
                    detected = "a100_single"
            elif max_memory_gb >= thresholds.get("a6000_min_memory", 45):
                detected = "a6000"
            elif max_memory_gb >= thresholds.get("rtx4090_min_memory", 22):
                if gpu_count >= 2:
                    detected = "rtx4090_dual"
                else:
                    detected = "rtx4090"
            elif max_memory_gb >= thresholds.get("rtx_mid_min_memory", 14):
                if gpu_count >= 2:
                    detected = "rtx4080_dual"
                else:
                    detected = "rtx4080"
            elif max_memory_gb >= thresholds.get("rtx_low_min_memory", 10):
                if gpu_count >= 4:
                    detected = "v100_quad"
                else:
                    detected = "v100"
            else:
                logger.warning(f"GPU memory ({max_memory_gb:.1f}GB) below recommended minimums")
                detected = thresholds.get("fallback", "rtx4080")
            
            logger.info(f"Detected hardware configuration: {detected}")
            return detected
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run nvidia-smi: {e}")
            logger.warning("Non-NVIDIA GPU or nvidia-smi not available")
            logger.info("Attempting generic GPU detection...")
            return self._detect_generic_gpu()
        except Exception as e:
            logger.error(f"Hardware detection error: {e}")
            logger.warning("Falling back to safe configuration")
            return self._detect_generic_gpu()
    
    def _detect_generic_gpu(self) -> str:
        """Fallback GPU detection for non-NVIDIA or unknown GPUs"""
        try:
            # Try PyTorch CUDA detection
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                total_memory = 0
                
                for i in range(gpu_count):
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory
                    gpu_memory_gb = gpu_memory / (1024**3)
                    total_memory = max(total_memory, gpu_memory_gb)
                    
                logger.info(f"Generic GPU detection: {gpu_count} GPU(s), max {total_memory:.1f}GB")
                
                # Use memory-based classification for unknown GPUs
                if gpu_count >= 10:
                    return "gpu_cluster"
                elif total_memory >= 75:
                    return "a100_dual" if gpu_count >= 2 else "a100_single"
                elif total_memory >= 45:
                    return "a6000"
                elif total_memory >= 20:
                    return "rtx4090_dual" if gpu_count >= 2 else "rtx4090"
                else:
                    return "rtx4080_dual" if gpu_count >= 2 else "rtx4080"
                    
        except ImportError:
            logger.warning("PyTorch not available for generic GPU detection")
        except Exception as e:
            logger.warning(f"Generic GPU detection failed: {e}")
            
        # Final fallback
        logger.info("Using conservative fallback configuration")
        return "rtx4080"
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {filename}: {e}")
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load complete configuration based on hardware type
        
        Returns:
            Complete configuration dictionary
        """
        # Determine actual hardware type
        if self.hardware_type == "auto":
            actual_hardware = self.detect_hardware()
        else:
            actual_hardware = self.hardware_type
            logger.info(f"Using user-specified hardware: {actual_hardware}")
        
        # Load hardware config to get all valid types
        hardware_config = self._load_yaml("hardware.yaml")
        valid_types = [key for key in hardware_config.keys() if key != "auto_detection"]
        
        if actual_hardware not in valid_types:
            logger.warning(f"Hardware type {actual_hardware} not in config, falling back to a6000")
            actual_hardware = "a6000"
        
        logger.info(f"Loading configuration for: {actual_hardware}")
        
        # Load all configuration files
        models_config = self._load_yaml("models.yaml")
        benchmarks_config = self._load_yaml("benchmarks.yaml")
        # hardware_config already loaded above for validation
        
        # Check if models.yaml is in Docker multi-version format
        if self._is_docker_format(models_config):
            # Extract all models from all transformer versions for backward compatibility
            all_models = []
            for version_key, version_data in models_config.items():
                if isinstance(version_data, dict) and "models" in version_data:
                    all_models.extend(version_data["models"])
            
            self.config = {
                "hardware_type": actual_hardware,
                "hardware": hardware_config.get(actual_hardware, {}),
                "models": all_models,
                "benchmarks": benchmarks_config.get("benchmarks", []),  # Use unified benchmark list
                "output_dir": str(self.output_dir),
                "docker_format": True,
                "transformer_versions": models_config
            }
        else:
            # Legacy hardware-specific format
            self.config = {
                "hardware_type": actual_hardware,
                "hardware": hardware_config.get(actual_hardware, {}),
                "models": models_config.get(actual_hardware, {}).get("models", []),
                "benchmarks": benchmarks_config.get("benchmarks", []),  # Use unified benchmark list
                "output_dir": str(self.output_dir),
                "docker_format": False
            }
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"Configuration loaded successfully")
        logger.info(f"   Models: {len(self.config['models'])}")
        logger.info(f"   Benchmarks: {len(self.config['benchmarks'])}")
        
        return self.config
    
    def _is_docker_format(self, models_config: Dict) -> bool:
        """Check if models.yaml is in Docker multi-version format"""
        # Docker format has transformer version keys like 'transformers_4_37', 'transformers_4_33', etc.
        transformer_keys = [k for k in models_config.keys() if k.startswith('transformers_')]
        return len(transformer_keys) > 0
    
    def _validate_config(self):
        """Validate loaded configuration"""
        if not self.config.get("models"):
            raise ValueError("No models configured for selected hardware")
        
        if not self.config.get("benchmarks"):
            raise ValueError("No benchmarks configured for selected hardware")
        
        if not self.config.get("hardware"):
            raise ValueError("Hardware configuration missing")
        
        # Validate model memory requirements (simplified)
        hardware = self.config["hardware"]
        max_model_memory = hardware.get("max_model_memory", 0)
        
        for model in self.config["models"]:
            model_memory = model.get("memory_gb", 0)  # Changed from 'estimated_memory'
            if model_memory > max_model_memory:
                logger.warning(
                    f"Model {model['name']} ({model_memory}GB) exceeds "
                    f"hardware limit ({max_model_memory}GB)"
                )
    
    def filter_models(self, model_names: List[str]) -> Dict[str, Any]:
        """
        Filter configuration to include only specified models
        
        Args:
            model_names: List of model names to include
            
        Returns:
            Updated configuration with filtered models
        """
        available_models = {m["name"]: m for m in self.config["models"]}
        filtered_models = []
        
        for name in model_names:
            if name in available_models:
                filtered_models.append(available_models[name])
            else:
                logger.warning(f"Model '{name}' not found in configuration")
        
        if not filtered_models:
            raise ValueError("No valid models found after filtering")
        
        self.config["models"] = filtered_models
        return self.config
    
    def filter_benchmarks(self, benchmark_names: List[str]) -> Dict[str, Any]:
        """
        Filter configuration to include only specified benchmarks
        
        Args:
            benchmark_names: List of benchmark names to include
            
        Returns:
            Updated configuration with filtered benchmarks
        """
        available_benchmarks = {b["name"]: b for b in self.config["benchmarks"]}
        filtered_benchmarks = []
        
        for name in benchmark_names:
            if name in available_benchmarks:
                filtered_benchmarks.append(available_benchmarks[name])
            else:
                logger.warning(f"Benchmark '{name}' not found in configuration")
        
        if not filtered_benchmarks:
            raise ValueError("No valid benchmarks found after filtering")
        
        self.config["benchmarks"] = filtered_benchmarks
        return self.config
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*72)
        print("SPECTRAVISION CONFIGURATION SUMMARY")
        print("="*72)
        
        # Hardware info
        hw = self.config["hardware"]
        print(f"Hardware: {hw.get('name', 'Unknown')}")
        print(f"   Memory: {hw.get('memory_gb', 0)}GB")
        if hw.get('gpu_count'):
            print(f"   GPUs: {hw.get('gpu_count', 1)}")
        print(f"   Max Model Memory: {hw.get('max_model_memory', 0)}GB")
        
        # Models
        print(f"\nModels ({len(self.config['models'])}):")
        for model in self.config["models"]:
            memory = model.get('memory_gb', 'Unknown')
            print(f"   - {model['name']} - {memory}GB")
        
        # Benchmarks  
        print(f"\nBenchmarks ({len(self.config['benchmarks'])}):")
        for benchmark in self.config["benchmarks"]:
            samples = benchmark.get('samples', 'Unknown')
            if isinstance(samples, int):
                samples = f"{samples:,}"
            print(f"   - {benchmark['name']} - {samples} samples")
        
        # Totals
        total_combinations = len(self.config["models"]) * len(self.config["benchmarks"])
        print(f"\nTotal Combinations: {total_combinations}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Config Directory: {self.config_dir}")
        print("="*72)


# Convenience functions for backward compatibility
def load_config(hardware_type: str = "auto", output_dir: str = "outputs") -> Dict[str, Any]:
    """
    Convenience function to load configuration
    
    Args:
        hardware_type: Hardware configuration type or 'auto' for detection
        output_dir: Output directory for results and logs
        
    Returns:
        Complete configuration dictionary
    """
    config_manager = ConfigManager(hardware_type=hardware_type, output_dir=output_dir)
    return config_manager.load_config()