"""
Performance Monitor for SpectraVision
Tracks GPU memory, utilization, temperature and system resources during evaluation
"""

import time
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

try:
    import GPUtil
    import psutil
    from py3nvml import py3nvml
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    GPUtil = None
    psutil = None
    py3nvml = None

logger = logging.getLogger(__name__)

@dataclass
class ResourceSnapshot:
    """Single point-in-time resource measurement"""
    timestamp: float
    gpu_memory_used: float  # GB
    gpu_memory_total: float  # GB
    gpu_utilization: float  # Percentage
    gpu_temperature: float  # Celsius
    cpu_percent: float
    ram_percent: float
    ram_used: float  # GB

@dataclass
class EvaluationMetrics:
    """Resource usage metrics for a single evaluation"""
    model_name: str
    benchmark_name: str
    start_time: float
    end_time: Optional[float]
    duration: Optional[float]  # seconds
    peak_gpu_memory: float  # GB
    avg_gpu_memory: float  # GB
    peak_gpu_utilization: float  # Percentage
    avg_gpu_utilization: float  # Percentage
    max_gpu_temperature: float  # Celsius
    snapshots: List[ResourceSnapshot]

class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self, hardware_config: Dict[str, Any], 
                 monitoring_interval: float = 1.0,
                 output_dir: Optional[str] = None):
        """
        Initialize Performance Monitor
        
        Args:
            hardware_config: Hardware configuration dictionary
            monitoring_interval: Seconds between measurements
            output_dir: Directory to save monitoring results
        """
        if not MONITORING_AVAILABLE:
            logger.warning("Monitoring libraries not available. Install GPUtil, psutil, py3nvml")
            self.enabled = False
            return
            
        self.hardware_config = hardware_config
        self.monitoring_interval = monitoring_interval
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")
        self.enabled = True
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.current_evaluation = None
        self.all_snapshots = []
        self.evaluation_metrics = []
        
        # Initialize NVML
        try:
            py3nvml.nvmlInit()
            self.gpu_count = py3nvml.nvmlDeviceGetCount()
            logger.info(f"Performance monitor initialized ({self.gpu_count} GPUs)")
        except Exception as e:
            logger.warning(f"Failed to initialize NVML: {e}")
            self.enabled = False
    
    def _take_snapshot(self) -> Optional[ResourceSnapshot]:
        """Take a single resource measurement snapshot"""
        if not self.enabled:
            return None
            
        try:
            # GPU metrics (using first GPU)
            gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
            
            if gpu:
                gpu_memory_used = gpu.memoryUsed / 1024  # Convert MB to GB
                gpu_memory_total = gpu.memoryTotal / 1024  # Convert MB to GB
                gpu_utilization = gpu.load * 100  # Convert to percentage
                gpu_temperature = gpu.temperature
            else:
                gpu_memory_used = gpu_memory_total = gpu_utilization = gpu_temperature = 0.0
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            ram_percent = ram.percent
            ram_used = ram.used / (1024**3)  # Convert to GB
            
            return ResourceSnapshot(
                timestamp=time.time(),
                gpu_memory_used=gpu_memory_used,
                gpu_memory_total=gpu_memory_total,
                gpu_utilization=gpu_utilization,
                gpu_temperature=gpu_temperature,
                cpu_percent=cpu_percent,
                ram_percent=ram_percent,
                ram_used=ram_used
            )
            
        except Exception as e:
            logger.warning(f"Failed to take resource snapshot: {e}")
            return None
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread"""
        logger.debug("Starting monitoring loop")
        
        while self.is_monitoring:
            snapshot = self._take_snapshot()
            if snapshot:
                self.all_snapshots.append(snapshot)
                
                # Check for potential issues
                self._check_resource_warnings(snapshot)
            
            time.sleep(self.monitoring_interval)
        
        logger.debug("Monitoring loop stopped")
    
    def _check_resource_warnings(self, snapshot: ResourceSnapshot):
        """Check resource usage and log warnings if needed"""
        hw_limits = self.hardware_config.get("monitoring", {})
        
        # Memory warning
        memory_usage_percent = (snapshot.gpu_memory_used / snapshot.gpu_memory_total) * 100
        if memory_usage_percent > hw_limits.get("memory_threshold", 90):
            logger.warning(f"High GPU memory usage: {memory_usage_percent:.1f}%")
        
        # Temperature warning
        if snapshot.gpu_temperature > hw_limits.get("temperature_threshold", 85):
            logger.warning(f"High GPU temperature: {snapshot.gpu_temperature}°C")
        
        # Utilization warning
        if snapshot.gpu_utilization > hw_limits.get("utilization_threshold", 95):
            logger.warning(f"High GPU utilization: {snapshot.gpu_utilization:.1f}%")
    
    def start(self):
        """Start background monitoring"""
        if not self.enabled:
            logger.info("Performance monitoring disabled (libraries not available)")
            return
            
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return
        
        logger.info("Starting performance monitoring")
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop(self):
        """Stop background monitoring"""
        if not self.enabled or not self.is_monitoring:
            return
        
        logger.info("Stopping performance monitoring")
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        # Save all monitoring data
        self._save_monitoring_results()
    
    def start_evaluation(self, model_name: str, benchmark_name: str):
        """Mark start of individual evaluation"""
        if not self.enabled:
            return
            
        self.current_evaluation = EvaluationMetrics(
            model_name=model_name,
            benchmark_name=benchmark_name,
            start_time=time.time(),
            end_time=None,
            duration=None,
            peak_gpu_memory=0.0,
            avg_gpu_memory=0.0,
            peak_gpu_utilization=0.0,
            avg_gpu_utilization=0.0,
            max_gpu_temperature=0.0,
            snapshots=[]
        )
        
        logger.debug(f"Started monitoring: {model_name} on {benchmark_name}")
    
    def end_evaluation(self) -> float:
        """
        Mark end of individual evaluation and calculate metrics
        
        Returns:
            Peak GPU memory usage during evaluation (GB)
        """
        if not self.enabled or not self.current_evaluation:
            return 0.0
        
        end_time = time.time()
        self.current_evaluation.end_time = end_time
        self.current_evaluation.duration = end_time - self.current_evaluation.start_time
        
        # Filter snapshots for this evaluation period
        eval_snapshots = [
            snapshot for snapshot in self.all_snapshots
            if self.current_evaluation.start_time <= snapshot.timestamp <= end_time
        ]
        
        self.current_evaluation.snapshots = eval_snapshots
        
        if eval_snapshots:
            # Calculate metrics
            gpu_memories = [s.gpu_memory_used for s in eval_snapshots]
            gpu_utilizations = [s.gpu_utilization for s in eval_snapshots]
            gpu_temperatures = [s.gpu_temperature for s in eval_snapshots]
            
            self.current_evaluation.peak_gpu_memory = max(gpu_memories)
            self.current_evaluation.avg_gpu_memory = sum(gpu_memories) / len(gpu_memories)
            self.current_evaluation.peak_gpu_utilization = max(gpu_utilizations)
            self.current_evaluation.avg_gpu_utilization = sum(gpu_utilizations) / len(gpu_utilizations)
            self.current_evaluation.max_gpu_temperature = max(gpu_temperatures)
            
            logger.debug(
                f"Evaluation metrics: Peak GPU memory: {self.current_evaluation.peak_gpu_memory:.1f}GB, "
                f"Max temp: {self.current_evaluation.max_gpu_temperature}°C"
            )
        
        # Store completed evaluation
        self.evaluation_metrics.append(self.current_evaluation)
        peak_memory = self.current_evaluation.peak_gpu_memory
        
        self.current_evaluation = None
        return peak_memory
    
    def get_results(self) -> Dict[str, Any]:
        """Get complete monitoring results"""
        if not self.enabled:
            return {"monitoring_enabled": False}
        
        return {
            "monitoring_enabled": True,
            "hardware_config": self.hardware_config,
            "monitoring_interval": self.monitoring_interval,
            "total_snapshots": len(self.all_snapshots),
            "total_evaluations": len(self.evaluation_metrics),
            "evaluation_metrics": [asdict(metric) for metric in self.evaluation_metrics],
            "overall_stats": self._calculate_overall_stats()
        }
    
    def _calculate_overall_stats(self) -> Dict[str, Any]:
        """Calculate overall monitoring statistics"""
        if not self.all_snapshots:
            return {}
        
        gpu_memories = [s.gpu_memory_used for s in self.all_snapshots]
        gpu_utilizations = [s.gpu_utilization for s in self.all_snapshots]
        gpu_temperatures = [s.gpu_temperature for s in self.all_snapshots]
        
        return {
            "peak_gpu_memory": max(gpu_memories),
            "avg_gpu_memory": sum(gpu_memories) / len(gpu_memories),
            "peak_gpu_utilization": max(gpu_utilizations),
            "avg_gpu_utilization": sum(gpu_utilizations) / len(gpu_utilizations),
            "max_gpu_temperature": max(gpu_temperatures),
            "avg_gpu_temperature": sum(gpu_temperatures) / len(gpu_temperatures)
        }
    
    def _save_monitoring_results(self):
        """Save monitoring results to file"""
        if not self.enabled:
            return
        
        # Create monitoring reports directory
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = reports_dir / "performance_monitoring.json"
        results = self.get_results()
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Monitoring results saved to: {results_file}")
        
        # Save CSV summary for easy analysis
        self._save_monitoring_csv(reports_dir)
    
    def _save_monitoring_csv(self, reports_dir: Path):
        """Save monitoring data as CSV for easy analysis"""
        try:
            import pandas as pd
            
            if self.evaluation_metrics:
                # Create evaluation summary CSV
                eval_data = []
                for metric in self.evaluation_metrics:
                    eval_data.append({
                        'model_name': metric.model_name,
                        'benchmark_name': metric.benchmark_name,
                        'duration_seconds': metric.duration,
                        'peak_gpu_memory_gb': metric.peak_gpu_memory,
                        'avg_gpu_memory_gb': metric.avg_gpu_memory,
                        'peak_gpu_utilization': metric.peak_gpu_utilization,
                        'avg_gpu_utilization': metric.avg_gpu_utilization,
                        'max_gpu_temperature': metric.max_gpu_temperature
                    })
                
                df = pd.DataFrame(eval_data)
                csv_file = reports_dir / "evaluation_metrics.csv"
                df.to_csv(csv_file, index=False)
                logger.info(f"Evaluation metrics CSV saved to: {csv_file}")
            
        except ImportError:
            logger.debug("pandas not available, skipping CSV export")