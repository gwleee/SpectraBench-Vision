"""
Utility functions for SpectraVision
Logging setup, banner printing, file operations, etc.
"""

import os
import sys
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with consistent formatting"""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def print_banner():
    """Print SpectraVision ASCII banner"""
    banner = """
    ================================================================
    ║                                                            ║
    ║   SPECTRAVISION                                            ║
    ║   Intelligent Multimodal Model Evaluation                 ║
    ║   KISTI AI Research Center                                 ║
    ║   Version 0.1.0                                           ║
    ║                                                            ║
    ================================================================
    """
    print(banner)

def setup_logging(log_level: str = "INFO", output_dir: str = "outputs"):
    """
    Setup comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        output_dir: Directory to save log files
    """
    # Create logs directory
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"spectravision_{timestamp}.log"
    
    # Configure logging format
    log_format = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Setup root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            # File handler - all logs
            logging.FileHandler(log_file, encoding='utf-8'),
            # Console handler - info and above
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Configure console handler with different level if needed
    console_handler = logging.getLogger().handlers[-1]
    if log_level.upper() == "DEBUG":
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    # Log setup completion
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}")
    logger.info(f"Log file: {log_file}")

def ensure_directory(path: str) -> Path:
    """
    Ensure directory exists, create if necessary
    
    Args:
        path: Directory path string
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def get_file_size(file_path: str) -> str:
    """
    Get human-readable file size
    
    Args:
        file_path: Path to file
        
    Returns:
        Formatted file size string
    """
    try:
        size_bytes = Path(file_path).stat().st_size
        
        # Convert to human readable format
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
        
    except (OSError, FileNotFoundError):
        return "Unknown"

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

def safe_filename(name: str) -> str:
    """
    Convert string to safe filename by removing/replacing problematic characters
    
    Args:
        name: Original name string
        
    Returns:
        Safe filename string
    """
    if not name:
        return "unknown"
    
    # Convert to string if not already
    safe_name = str(name)
    
    # Replace problematic characters
    replacements = {
        '%': 'percent',
        '/': '_',
        '\\': '_',
        ':': '_',
        '*': '_',
        '?': '_',
        '"': '_',
        '<': '_',
        '>': '_',
        '|': '_',
        ' ': '_',
        '\t': '_',
        '\n': '_',
        '\r': '_'
    }
    
    for old_char, new_char in replacements.items():
        safe_name = safe_name.replace(old_char, new_char)
    
    # Remove any remaining non-alphanumeric characters except dots, hyphens, and underscores
    safe_name = re.sub(r'[^\w\-_\.]', '_', safe_name)
    
    # Remove consecutive underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    
    # Remove leading/trailing underscores and dots
    safe_name = safe_name.strip('_.')
    
    # Ensure we have something left
    if not safe_name:
        return "unknown"
    
    # Limit length to reasonable size
    if len(safe_name) > 100:
        safe_name = safe_name[:100]
    
    return safe_name

def get_gpu_info() -> dict:
    """
    Get basic GPU information if available
    
    Returns:
        Dictionary with GPU information
    """
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        
        if not gpus:
            return {"gpu_count": 0, "gpus": []}
        
        gpu_info = []
        for i, gpu in enumerate(gpus):
            gpu_info.append({
                "id": i,
                "name": gpu.name,
                "memory_total": f"{gpu.memoryTotal/1024:.1f}GB",
                "memory_free": f"{gpu.memoryFree/1024:.1f}GB",
                "temperature": f"{gpu.temperature}C",
                "utilization": f"{gpu.load*100:.1f}%"
            })
        
        return {
            "gpu_count": len(gpus),
            "gpus": gpu_info
        }
        
    except ImportError:
        return {"gpu_count": 0, "error": "GPUtil not available"}
    except Exception as e:
        return {"gpu_count": 0, "error": str(e)}

def validate_yaml_config(config_path: str) -> bool:
    """
    Validate YAML configuration file
    
    Args:
        config_path: Path to YAML file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        import yaml
        
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)
        return True
        
    except (FileNotFoundError, yaml.YAMLError, UnicodeDecodeError):
        return False

def create_results_structure(output_dir: str) -> dict:
    """
    Create standard results directory structure
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Dictionary mapping structure names to paths
    """
    base_dir = Path(output_dir)
    
    structure = {
        "base": base_dir,
        "logs": base_dir / "logs",
        "results": base_dir / "results", 
        "reports": base_dir / "reports",
        "cache": base_dir / "cache",
        "vlmevalkit": base_dir / "vlmevalkit_results",
        "analysis": base_dir / "analysis"
    }
    
    # Create all directories
    for path in structure.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return {name: str(path) for name, path in structure.items()}

def print_system_info():
    """Print basic system information"""
    import platform
    
    logger = logging.getLogger(__name__)
    
    logger.info("System Information:")
    logger.info(f"   Platform: {platform.platform()}")
    logger.info(f"   Python: {platform.python_version()}")
    logger.info(f"   Architecture: {platform.architecture()[0]}")
    
    # GPU info
    gpu_info = get_gpu_info()
    if gpu_info.get("gpu_count", 0) > 0:
        logger.info(f"   GPUs: {gpu_info['gpu_count']}")
        for gpu in gpu_info["gpus"]:
            logger.info(f"     - {gpu['name']}: {gpu['memory_total']}")
    else:
        logger.warning("   No GPUs detected or GPUtil not available")
    
    # Memory info
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"   System RAM: {memory.total / (1024**3):.1f}GB")
    except ImportError:
        logger.info("   System RAM: Unknown (psutil not available)")

def sanitize_string(text: str) -> str:
    """
    Sanitize string for safe usage in formatting and file operations
    
    Args:
        text: Input string to sanitize
        
    Returns:
        Sanitized string
    """
    if not text:
        return "unknown"
    
    # Convert to string
    sanitized = str(text)
    
    # Replace percent signs to avoid format specifier issues
    sanitized = sanitized.replace('%', 'percent')
    
    # Replace other problematic characters
    sanitized = sanitized.replace('{', '(').replace('}', ')')
    
    # Handle potential unicode issues
    try:
        sanitized = sanitized.encode('utf-8', errors='replace').decode('utf-8')
    except Exception:
        sanitized = repr(sanitized)
    
    return sanitized

def safe_format(template: str, **kwargs) -> str:
    """
    Safely format string template with error handling
    
    Args:
        template: String template
        **kwargs: Format arguments
        
    Returns:
        Formatted string or error message
    """
    try:
        # Sanitize all string arguments
        safe_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, str):
                safe_kwargs[key] = sanitize_string(value)
            else:
                safe_kwargs[key] = value
        
        return template.format(**safe_kwargs)
    except (KeyError, ValueError, TypeError) as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"String formatting error: {e}")
        return f"Format error: {template} (Error: {e})"

class ProgressTracker:
    """Simple progress tracking utility"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
        
    def update(self, increment: int = 1):
        """Update progress"""
        self.current = min(self.current + increment, self.total)
        self._print_progress()
        
    def _print_progress(self):
        """Print current progress"""
        if self.total == 0:
            return
            
        percentage = (self.current / self.total) * 100
        elapsed = datetime.now() - self.start_time
        
        if self.current > 0:
            eta_seconds = (elapsed.total_seconds() / self.current) * (self.total - self.current)
            eta = format_duration(eta_seconds)
        else:
            eta = "Unknown"
        
        progress_bar = "#" * int(percentage // 2) + "-" * (50 - int(percentage // 2))
        
        print(f"\r{self.description}: [{progress_bar}] {percentage:.1f}% ({self.current}/{self.total}) ETA: {eta}", 
              end="", flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete

def handle_exception(func):
    """Decorator for safe function execution with exception handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger = logging.getLogger(func.__module__)
            logger.error(f"Exception in {func.__name__}: {e}")
            return None
    return wrapper

def clean_text_for_logging(text: str) -> str:
    """
    Clean text for safe logging
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text safe for logging
    """
    if not text:
        return "None"
    
    # Convert to string and sanitize
    cleaned = str(text)
    
    # Remove or replace problematic characters
    cleaned = cleaned.replace('%', 'percent')
    cleaned = cleaned.replace('\n', ' ')
    cleaned = cleaned.replace('\r', ' ')
    cleaned = cleaned.replace('\t', ' ')
    
    # Limit length for logging
    if len(cleaned) > 500:
        cleaned = cleaned[:500] + "..."
    
    return cleaned

def validate_numeric_value(value, default=0.0, min_val=None, max_val=None):
    """
    Validate and sanitize numeric values
    
    Args:
        value: Value to validate
        default: Default value if validation fails
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Validated numeric value
    """
    try:
        import numpy as np
        
        # Convert to float
        num_val = float(value)
        
        # Check for NaN or infinity
        if np.isnan(num_val) or np.isinf(num_val):
            return default
        
        # Apply bounds if specified
        if min_val is not None and num_val < min_val:
            return min_val
        if max_val is not None and num_val > max_val:
            return max_val
        
        return num_val
        
    except (ValueError, TypeError, OverflowError):
        return default