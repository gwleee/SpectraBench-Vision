"""
SpectraVision: Intelligent Multimodal Model Evaluation System

A hardware-aware, scalable evaluation framework for vision-language models,
designed for efficient benchmarking across different GPU configurations.
"""

__version__ = "0.1.0"
__author__ = "KISTI AI Research Team"
__description__ = "Intelligent Multimodal Model Evaluation System"

# Core imports
from .config import ConfigManager
from .evaluator import SequentialEvaluator
from .monitor import PerformanceMonitor
from .utils import setup_logging, print_banner

# Package metadata
__all__ = [
    "ConfigManager",
    "SequentialEvaluator", 
    "PerformanceMonitor",
    "setup_logging",
    "print_banner",
]

# Version info
VERSION_INFO = {
    "major": 0,
    "minor": 1,
    "patch": 0,
    "status": "alpha"
}

def get_version_string():
    """Get formatted version string"""
    version = f"{VERSION_INFO['major']}.{VERSION_INFO['minor']}.{VERSION_INFO['patch']}"
    if VERSION_INFO['status'] != 'stable':
        version += f"-{VERSION_INFO['status']}"
    return version

# Phase tracking
CURRENT_PHASE = "Phase 1: Sequential Baseline"
SUPPORTED_MODES = ["sequential"]  # Will expand in Phase 2+

def get_phase_info():
    """Get current development phase information"""
    return {
        "phase": CURRENT_PHASE,
        "supported_modes": SUPPORTED_MODES,
        "hardware_support": ["a6000", "a100_single", "a100_dual"],
        "benchmark_count": {"a6000": 3, "a100_single": 5, "a100_dual": 5},
        "model_count": {"a6000": 3, "a100_single": 5, "a100_dual": 5}
    }