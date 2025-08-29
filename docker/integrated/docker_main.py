#!/usr/bin/env python3
"""
SpectraBench-Vision Docker Orchestrator Main Entry Point
Manages multi-container evaluation system automatically
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectravision.docker_orchestrator import DockerOrchestrator
from spectravision.config import ConfigManager

def main():
    parser = argparse.ArgumentParser(description="SpectraBench-Vision Docker Orchestrator")
    parser.add_argument("--mode", choices=["interactive", "batch", "test"], 
                       default="interactive", help="Execution mode")
    parser.add_argument("--models", nargs="+", help="Specific models to evaluate")
    parser.add_argument("--benchmarks", nargs="+", help="Specific benchmarks to run")
    parser.add_argument("--hardware", default="auto", help="Hardware configuration")
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0], help="GPU IDs to use")
    
    args = parser.parse_args()
    
    print("🐳 SpectraBench-Vision Docker Orchestrator")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = DockerOrchestrator()
    
    if args.mode == "test":
        print("🧪 Running system tests...")
        orchestrator.test_system()
    elif args.mode == "interactive":
        print("🎮 Starting interactive mode...")
        orchestrator.interactive_mode()
    elif args.mode == "batch":
        if not args.models or not args.benchmarks:
            print("❌ Batch mode requires --models and --benchmarks")
            sys.exit(1)
        
        print(f"🚀 Running batch evaluation:")
        print(f"   Models: {args.models}")
        print(f"   Benchmarks: {args.benchmarks}")
        
        results = orchestrator.run_batch_evaluation(
            models=args.models,
            benchmarks=args.benchmarks,
            gpu_ids=args.gpu_ids
        )
        
        print(f"✅ Completed {len(results)} evaluations")

if __name__ == "__main__":
    main()