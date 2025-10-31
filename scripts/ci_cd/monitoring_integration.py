#!/usr/bin/env python3
"""
Enhanced Error Logging and Monitoring Integration
Provides comprehensive monitoring capabilities for SpectraBench-Vision system
"""

import os
import sys
import json
import time
import logging
import psutil
import threading
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import signal

@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    gpu_info: Dict[str, Any]
    docker_stats: Dict[str, Any]
    process_count: int
    load_average: List[float]

@dataclass
class ErrorEvent:
    """Error event structure"""
    timestamp: float
    level: str
    source: str
    message: str
    details: Dict[str, Any]
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class GPUMonitor:
    """GPU monitoring utilities"""

    @staticmethod
    def get_gpu_stats() -> Dict[str, Any]:
        """Get GPU statistics using nvidia-ml-py or fallback to nvidia-smi"""
        gpu_stats = {
            "available": False,
            "devices": [],
            "total_memory_gb": 0,
            "used_memory_gb": 0,
            "utilization_percent": 0
        }

        try:
            # Try nvidia-ml-py first (more efficient)
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            gpu_stats["available"] = True
            total_memory = 0
            used_memory = 0

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                device_stats = {
                    "id": i,
                    "name": name,
                    "memory_total_mb": mem_info.total // (1024 ** 2),
                    "memory_used_mb": mem_info.used // (1024 ** 2),
                    "memory_free_mb": mem_info.free // (1024 ** 2),
                    "utilization_gpu": util.gpu,
                    "utilization_memory": util.memory
                }

                gpu_stats["devices"].append(device_stats)
                total_memory += mem_info.total
                used_memory += mem_info.used

            gpu_stats["total_memory_gb"] = round(total_memory / (1024 ** 3), 2)
            gpu_stats["used_memory_gb"] = round(used_memory / (1024 ** 3), 2)
            gpu_stats["utilization_percent"] = sum(d["utilization_gpu"] for d in gpu_stats["devices"]) / len(gpu_stats["devices"])

        except ImportError:
            # Fallback to nvidia-smi
            try:
                result = subprocess.run([
                    "nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits"
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    gpu_stats["available"] = True
                    lines = result.stdout.strip().split('\n')

                    for i, line in enumerate(lines):
                        parts = line.split(', ')
                        if len(parts) >= 4:
                            name, total_mem, used_mem, utilization = parts[:4]

                            device_stats = {
                                "id": i,
                                "name": name.strip(),
                                "memory_total_mb": int(total_mem),
                                "memory_used_mb": int(used_mem),
                                "memory_free_mb": int(total_mem) - int(used_mem),
                                "utilization_gpu": int(utilization),
                                "utilization_memory": 0  # Not available via nvidia-smi query
                            }

                            gpu_stats["devices"].append(device_stats)

                    if gpu_stats["devices"]:
                        gpu_stats["total_memory_gb"] = sum(d["memory_total_mb"] for d in gpu_stats["devices"]) / 1024
                        gpu_stats["used_memory_gb"] = sum(d["memory_used_mb"] for d in gpu_stats["devices"]) / 1024
                        gpu_stats["utilization_percent"] = sum(d["utilization_gpu"] for d in gpu_stats["devices"]) / len(gpu_stats["devices"])

            except Exception:
                pass  # GPU stats not available

        except Exception:
            pass  # GPU stats not available

        return gpu_stats

class DockerMonitor:
    """Docker container monitoring"""

    @staticmethod
    def get_docker_stats() -> Dict[str, Any]:
        """Get Docker container statistics"""
        docker_stats = {
            "available": False,
            "containers": [],
            "images": [],
            "total_containers": 0,
            "running_containers": 0,
            "spectravision_containers": []
        }

        try:
            # Get running containers
            result = subprocess.run([
                "docker", "ps", "--format", "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                docker_stats["available"] = True
                lines = result.stdout.strip().split('\n')[1:]  # Skip header

                for line in lines:
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            name, image, status = parts[:3]
                            container_info = {
                                "name": name.strip(),
                                "image": image.strip(),
                                "status": status.strip()
                            }
                            docker_stats["containers"].append(container_info)

                            if "spectravision" in image.lower():
                                docker_stats["spectravision_containers"].append(container_info)

                docker_stats["running_containers"] = len(docker_stats["containers"])

            # Get all containers (including stopped)
            result_all = subprocess.run([
                "docker", "ps", "-a", "--format", "{{.Names}}"
            ], capture_output=True, text=True)

            if result_all.returncode == 0:
                all_containers = result_all.stdout.strip().split('\n')
                docker_stats["total_containers"] = len([c for c in all_containers if c.strip()])

            # Get images
            result_images = subprocess.run([
                "docker", "images", "--format", "{{.Repository}}:{{.Tag}}\t{{.Size}}"
            ], capture_output=True, text=True)

            if result_images.returncode == 0:
                lines = result_images.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            image, size = parts[:2]
                            docker_stats["images"].append({
                                "name": image.strip(),
                                "size": size.strip()
                            })

        except Exception:
            pass  # Docker not available

        return docker_stats

class MonitoringIntegration:
    """Main monitoring integration class"""

    def __init__(self, output_dir: str = "outputs/monitoring", log_level: str = "INFO"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_level = getattr(logging, log_level.upper())
        self.setup_logging()

        self.metrics_history = []
        self.error_history = []
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 30  # seconds

        # Performance thresholds
        self.thresholds = {
            "cpu_critical": 90,
            "memory_critical": 85,
            "disk_critical": 90,
            "gpu_memory_critical": 90
        }

        # Initialize components
        self.gpu_monitor = GPUMonitor()
        self.docker_monitor = DockerMonitor()

        self.logger.info(f"Monitoring integration initialized - output: {self.output_dir}")

    def setup_logging(self):
        """Setup enhanced logging with multiple handlers"""

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Setup logger
        self.logger = logging.getLogger('spectravision_monitoring')
        self.logger.setLevel(self.log_level)
        self.logger.handlers = []  # Clear existing handlers

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)

        # File handler - detailed logs
        log_file = self.output_dir / "monitoring.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)

        # Error file handler - errors only
        error_file = self.output_dir / "errors.log"
        error_handler = logging.FileHandler(error_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(error_handler)

        # JSON handler for structured logs
        json_file = self.output_dir / "monitoring.jsonl"
        self.json_handler = logging.FileHandler(json_file)
        self.json_handler.setLevel(logging.INFO)
        # JSON handler uses custom formatting in log_event method

    def collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""

        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # Disk usage for current directory
        disk = psutil.disk_usage(Path.cwd())

        # Load average (Unix-like systems)
        try:
            load_avg = list(os.getloadavg())
        except (OSError, AttributeError):
            load_avg = [0.0, 0.0, 0.0]  # Windows fallback

        # GPU stats
        gpu_info = self.gpu_monitor.get_gpu_stats()

        # Docker stats
        docker_stats = self.docker_monitor.get_docker_stats()

        # Process count
        process_count = len(psutil.pids())

        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=round(memory.used / (1024 ** 3), 2),
            memory_total_gb=round(memory.total / (1024 ** 3), 2),
            disk_usage_percent=disk.percent,
            gpu_info=gpu_info,
            docker_stats=docker_stats,
            process_count=process_count,
            load_average=load_avg
        )

    def check_thresholds(self, metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """Check metrics against thresholds and generate alerts"""
        alerts = []

        # CPU threshold
        if metrics.cpu_percent > self.thresholds["cpu_critical"]:
            alerts.append({
                "type": "cpu_critical",
                "value": metrics.cpu_percent,
                "threshold": self.thresholds["cpu_critical"],
                "message": f"CPU usage critical: {metrics.cpu_percent}%"
            })

        # Memory threshold
        if metrics.memory_percent > self.thresholds["memory_critical"]:
            alerts.append({
                "type": "memory_critical",
                "value": metrics.memory_percent,
                "threshold": self.thresholds["memory_critical"],
                "message": f"Memory usage critical: {metrics.memory_percent}%"
            })

        # Disk threshold
        if metrics.disk_usage_percent > self.thresholds["disk_critical"]:
            alerts.append({
                "type": "disk_critical",
                "value": metrics.disk_usage_percent,
                "threshold": self.thresholds["disk_critical"],
                "message": f"Disk usage critical: {metrics.disk_usage_percent}%"
            })

        # GPU memory thresholds
        if metrics.gpu_info["available"]:
            for device in metrics.gpu_info["devices"]:
                if device["memory_total_mb"] > 0:
                    memory_percent = (device["memory_used_mb"] / device["memory_total_mb"]) * 100
                    if memory_percent > self.thresholds["gpu_memory_critical"]:
                        alerts.append({
                            "type": "gpu_memory_critical",
                            "device": device["name"],
                            "value": memory_percent,
                            "threshold": self.thresholds["gpu_memory_critical"],
                            "message": f"GPU {device['name']} memory critical: {memory_percent:.1f}%"
                        })

        return alerts

    def log_event(self, level: str, source: str, message: str, details: Dict[str, Any] = None, context: Dict[str, Any] = None):
        """Log event with structured format"""

        event = ErrorEvent(
            timestamp=time.time(),
            level=level,
            source=source,
            message=message,
            details=details or {},
            context=context
        )

        # Add to error history if it's an error
        if level in ["ERROR", "CRITICAL"]:
            self.error_history.append(event)
            # Keep only last 1000 errors
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-1000:]

        # Log to standard logger
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(log_level, f"[{source}] {message} - {details}")

        # Log structured JSON
        json_record = asdict(event)
        self.json_handler.stream.write(json.dumps(json_record) + '\n')
        self.json_handler.flush()

    def start_monitoring(self):
        """Start background monitoring thread"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"Started monitoring with {self.monitor_interval}s interval")

    def stop_monitoring(self):
        """Stop background monitoring"""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Stopped monitoring")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self.collect_system_metrics()
                self.metrics_history.append(metrics)

                # Keep only last 1000 metrics (about 8 hours at 30s interval)
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]

                # Check thresholds
                alerts = self.check_thresholds(metrics)
                for alert in alerts:
                    self.log_event("WARNING", "threshold_monitor", alert["message"], alert)

                # Save metrics periodically
                if len(self.metrics_history) % 10 == 0:  # Every 10 intervals
                    self.save_metrics_snapshot()

            except Exception as e:
                self.log_event("ERROR", "monitor_loop", f"Monitoring error: {e}")

            time.sleep(self.monitor_interval)

    def save_metrics_snapshot(self):
        """Save current metrics snapshot to file"""
        snapshot_file = self.output_dir / f"metrics_snapshot_{int(time.time())}.json"

        snapshot_data = {
            "timestamp": time.time(),
            "metrics_count": len(self.metrics_history),
            "recent_metrics": [asdict(m) for m in self.metrics_history[-10:]],  # Last 10 metrics
            "error_count": len(self.error_history),
            "recent_errors": [asdict(e) for e in self.error_history[-10:]]  # Last 10 errors
        }

        try:
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot_data, f, indent=2)
            self.logger.debug(f"Saved metrics snapshot: {snapshot_file}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics snapshot: {e}")

    @contextmanager
    def monitor_operation(self, operation_name: str, context: Dict[str, Any] = None):
        """Context manager for monitoring specific operations"""
        start_time = time.time()
        self.log_event("INFO", "operation_start", f"Started operation: {operation_name}", context=context)

        try:
            yield
            duration = time.time() - start_time
            self.log_event("INFO", "operation_success",
                         f"Completed operation: {operation_name}",
                         {"duration_seconds": round(duration, 2)}, context)
        except Exception as e:
            duration = time.time() - start_time
            self.log_event("ERROR", "operation_failure",
                         f"Failed operation: {operation_name} - {e}",
                         {"duration_seconds": round(duration, 2), "error": str(e)}, context)
            raise

    def generate_report(self) -> Dict[str, Any]:
        """Generate monitoring report"""
        if not self.metrics_history:
            return {"error": "No metrics collected yet"}

        latest_metrics = self.metrics_history[-1]

        # Calculate averages over last hour (120 data points at 30s interval)
        recent_metrics = self.metrics_history[-120:] if len(self.metrics_history) >= 120 else self.metrics_history

        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_gpu_util = 0
        if latest_metrics.gpu_info["available"] and latest_metrics.gpu_info["devices"]:
            gpu_utils = []
            for m in recent_metrics:
                if m.gpu_info["available"] and m.gpu_info["devices"]:
                    avg_util = sum(d["utilization_gpu"] for d in m.gpu_info["devices"]) / len(m.gpu_info["devices"])
                    gpu_utils.append(avg_util)
            avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0

        # Error summary
        error_count_1h = len([e for e in self.error_history if time.time() - e.timestamp < 3600])
        error_count_24h = len([e for e in self.error_history if time.time() - e.timestamp < 86400])

        report = {
            "timestamp": time.time(),
            "monitoring_duration_hours": round((time.time() - self.metrics_history[0].timestamp) / 3600, 2),
            "current_status": {
                "cpu_percent": latest_metrics.cpu_percent,
                "memory_percent": latest_metrics.memory_percent,
                "memory_used_gb": latest_metrics.memory_used_gb,
                "disk_usage_percent": latest_metrics.disk_usage_percent,
                "gpu_available": latest_metrics.gpu_info["available"],
                "gpu_memory_used_gb": latest_metrics.gpu_info["used_memory_gb"],
                "docker_containers_running": latest_metrics.docker_stats["running_containers"],
                "spectravision_containers": len(latest_metrics.docker_stats["spectravision_containers"])
            },
            "averages_last_hour": {
                "cpu_percent": round(avg_cpu, 2),
                "memory_percent": round(avg_memory, 2),
                "gpu_utilization_percent": round(avg_gpu_util, 2)
            },
            "error_summary": {
                "total_errors": len(self.error_history),
                "errors_last_1h": error_count_1h,
                "errors_last_24h": error_count_24h
            },
            "system_info": {
                "memory_total_gb": latest_metrics.memory_total_gb,
                "process_count": latest_metrics.process_count,
                "load_average": latest_metrics.load_average
            }
        }

        if latest_metrics.gpu_info["available"]:
            report["gpu_info"] = {
                "device_count": len(latest_metrics.gpu_info["devices"]),
                "total_memory_gb": latest_metrics.gpu_info["total_memory_gb"],
                "devices": latest_metrics.gpu_info["devices"]
            }

        return report

def main():
    """Main function for standalone usage"""
    import argparse

    parser = argparse.ArgumentParser(description="SpectraBench-Vision Monitoring Integration")
    parser.add_argument("--output-dir", default="outputs/monitoring", help="Output directory")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--duration", type=int, help="Monitoring duration in seconds (default: run forever)")
    parser.add_argument("--report", action="store_true", help="Generate and print report")

    args = parser.parse_args()

    # Initialize monitoring
    monitor = MonitoringIntegration(args.output_dir, args.log_level)
    monitor.monitor_interval = args.interval

    if args.report:
        # Generate report from existing data
        report = monitor.generate_report()
        print(json.dumps(report, indent=2))
        return

    # Start monitoring
    monitor.start_monitoring()

    try:
        if args.duration:
            print(f"Running monitoring for {args.duration} seconds...")
            time.sleep(args.duration)
        else:
            print("Running monitoring indefinitely (Ctrl+C to stop)...")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
    finally:
        monitor.stop_monitoring()

        # Generate final report
        print("\nFinal Report:")
        report = monitor.generate_report()
        print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()