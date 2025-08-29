# SpectraBench-Vision 🔍

> **English**: README.en.md | **한국어**: [README.md](README.md)

**Complete integrated system for evaluating 30 VLM models with a single download**

---

## 📋 Overview

SpectraBench-Vision is a **Docker-in-Docker based integrated VLM evaluation system** developed by **KISTI Large-scale AI Research Center**. It's designed to help researchers easily evaluate 30 cutting-edge Vision-Language models without complex environment setup.

## ✨ Key Features

- 🚀 **1-minute setup**: Instant access to entire system with single Docker command
- 🤖 **Full automation**: Automatic selection of optimal transformer version containers per model
- 📦 **Complete reproducibility**: Guaranteed identical evaluation environment anywhere  
- 🎯 **30 model support**: From SmolVLM to Qwen2.5-VL-72B, all latest models included
- 📊 **24 benchmarks**: Support for all standard benchmarks including MMBench, TextVQA, DocVQA
- 🔧 **GPU optimization**: Automatic optimization from single GPU to multi-GPU clusters

## 🚀 Quick Start

### Basic Usage (Recommended)

```bash
# 1. Start integrated system (automatic support for all 30 models)
docker run -it --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectrabench-vision:latest

# 2. Interactive mode inside container
python3 scripts/docker_main.py --mode interactive
```

### Direct Evaluation

```bash
# Benchmark evaluation with specific model (one-line command)
docker run --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectrabench-vision:latest \
  python3 scripts/docker_main.py --mode batch \
  --models "SmolVLM" --benchmarks "MMBench"
```

> 📖 **More usage**: [Docker Usage Guide (EN)](DOCKER_USAGE_GUIDE_EN.md) | [Docker 사용 가이드](DOCKER_USAGE_GUIDE.md)

## 🏛️ Development Background

**SpectraBench-Vision** was developed by the **AI Platform Team at KISTI Large-scale AI Research Center**, providing intelligent model-benchmark combinations based on available GPU resources and comprehensive performance monitoring and analysis capabilities.

The Large-scale AI Research Center officially launched in March 2024, building upon KISTI's generative large language model 'KONI (KISTI Open Natural Intelligence)' released in December 2023. **The AI Platform Team is responsible for developing AI model and agent service technologies**, and SpectraBench-Vision demonstrates their commitment to building sophisticated evaluation frameworks for the research community.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA Required](https://img.shields.io/badge/CUDA-Required-green.svg)](https://developer.nvidia.com/cuda-downloads)

## 📊 Supported Models and Benchmarks

### 🤖 Supported Models (30)

| Transformer Version | Key Models | Memory Range | Purpose |
|---------------------|------------|--------------|---------|
| **4.33.0** | Qwen-VL, VisualGLM | 8GB - 48GB | Legacy models |
| **4.37.2** | InternVL2, LLaVA | 8GB - 45GB | Stable models |
| **4.43.0** | Phi-3.5-Vision | 8GB - 18GB | Mid-range models |
| **4.49.0** | SmolVLM, Qwen2.5-VL | 3GB - 300GB | Latest models |
| **4.51.0** | Phi-4-Vision | 45GB - 200GB | Experimental models |

**Popular Models:**
- **SmolVLM**: 256M, 500M, 1.7B (ultra-lightweight, 3-8GB memory)
- **Qwen2.5-VL**: 3B, 7B, 32B, 72B (latest generation, 12GB-300GB memory)  
- **InternVL2**: 2B, 8B (high-performance, 8GB-30GB memory)
- **LLaVA**: 7B, 13B (stable, 25GB-45GB memory)

### 📊 Supported Benchmarks (24)

| Category | Key Benchmarks | Description |
|----------|----------------|-------------|
| **Basic VQA** | MMBench, TextVQA, GQA | Multimodal reasoning, text understanding, compositional reasoning |
| **Document Understanding** | DocVQA, ChartQA, InfoVQA | Document QA, chart/graph understanding |
| **Science/Professional** | ScienceQA, AI2D, MMMU | Scientific problem solving, diagram understanding |
| **Advanced Evaluation** | HallusionBench, MMStar, RealWorldQA | Hallucination detection, real-world reasoning |
| **Korean** | K-MMBench, K-SEED, Korean-OCR | Korean multimodal evaluation |

**Total evaluation combinations: 720** (30 models × 24 benchmarks)

## 🔧 Additional Usage

### Interactive Mode
```bash
# Menu-based model selection inside container
python3 scripts/docker_main.py --mode interactive
```

### Multi-Model Comparison
```bash
# Performance comparison across multiple models
docker run --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectrabench-vision:latest \
  python3 scripts/docker_main.py --mode batch \
  --models "SmolVLM" "InternVL2-8B" "Qwen2.5-VL-3B" \
  --benchmarks "MMBench" "TextVQA"
```

### Memory/GPU Optimization

```bash
# Use only latest models (memory saving)
docker run -it --gpus all -e PULL_IMAGES=minimal \
  -v /var/run/docker.sock:/var/run/docker.sock \
  ghcr.io/gwleee/spectrabench-vision:latest

# Multi-GPU utilization
docker run -it --gpus all -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  ghcr.io/gwleee/spectrabench-vision:latest
```

### System Testing
```bash
# Installation verification and GPU testing
docker run --rm --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  ghcr.io/gwleee/spectrabench-vision:latest \
  python3 scripts/docker_main.py --mode test
```

### Individual Container Direct Usage (Advanced)
```bash
# Direct use of specific transformer version (for development/debugging)
docker run --gpus all -it ghcr.io/gwleee/spectravision-4.49:latest

# Direct VLMEvalKit usage
cd /workspace/VLMEvalKit
python run.py --model SmolVLM-Instruct --data MMBench_DEV_EN
```

## 📁 Project Structure

```
SpectraBench-Vision/
├── .env.template              # Environment variables template
├── requirements.txt           # Python dependencies
├── quick_start.sh             # One-command setup script
│
├── configs/                   # Configuration files
│   ├── hardware.yaml          # GPU memory limits and detection
│   ├── models.yaml            # Model definitions (unified)
│   └── benchmarks.yaml        # Unified benchmark list (24 benchmarks)
│
├── scripts/                   # Main execution scripts
│   ├── main.py                # Main entry point
│   └── setup_dependencies.py  # Automated dependency setup
│
├── spectravision/             # Core evaluation system
│   ├── config.py              # Configuration management
│   ├── docker_orchestrator.py # Docker container management
│   ├── env_manager.py         # Environment variable management
│   ├── evaluator.py           # Sequential evaluation engine
│   ├── monitor.py             # Performance monitoring
│   ├── multi_version_evaluator.py # Multi-version orchestration
│   └── utils.py               # Logging and utility functions
│
├── analysis/                  # Performance analysis tools
│   ├── analyzer.py            # Performance analysis engine
│   └── visualizer.py          # Results visualization
│
├── VLMEvalKit/               # Auto-downloaded evaluation framework
├── outputs/                  # Results, logs, and reports
│
└── .env                      # Your personal environment config (not in git)
```

## 🛠️ Configuration

### Environment Configuration

**Personal API Keys**: Copy `.env.template` to `.env` and add your keys:
```bash
cp .env.template .env
nano .env  # Add your HF_TOKEN and other keys
```

**GPU Configuration**: Set which GPU(s) to use in `.env`:
```bash
CUDA_VISIBLE_DEVICES=0,1  # Use first two GPUs
```

See `.env.template` file for detailed environment configuration.

### Adding New Models

1. Check if the model is supported in VLMEvalKit
2. Add to appropriate hardware tier in `configs/models.yaml`:

```yaml
# Under appropriate hardware section (e.g., a6000_models)
- name: "New-Model-7B"
  vlm_id: "exact_vlmevalkit_id"  # Must match VLMEvalKit
  memory_gb: 28
```

### Adding New Benchmarks

1. Verify benchmark exists in VLMEvalKit
2. Add to the unified benchmark list in `configs/benchmarks.yaml`:

```yaml
benchmarks:
  - name: "NewBench"
    vlm_name: "NewBench_DEV"  # Must match VLMEvalKit dataset name
    samples: 100
    purpose: "Description of benchmark purpose"
```

## 🔍 Troubleshooting

### Common Issues

**"HF_TOKEN is required" or "You are trying to access a gated repo"**
- Configure your personal HF token in `.env` file
- Get your token from https://huggingface.co/settings/tokens
- Add `HF_TOKEN=your_token` to your `.env` file

**"No module named 'llava'"**
```bash
python scripts/setup_dependencies.py  # Re-run setup
```

**"TypeError: expected str, bytes or os.PathLike object, not NoneType" (Yi-VL)**
- Ensure Yi repository was cloned properly
- Run setup script again: `python scripts/setup_dependencies.py`
- Check if Yi_ROOT is set correctly in VLMEvalKit/vlmeval/config.py

**".env file not found" warning**
```bash
cp .env.template .env  # Create from template
nano .env              # Add your API keys
```

**GPU Memory Errors**
- Check available GPU memory with `nvidia-smi`
- Use smaller models or reduce batch size
- Enable memory cleanup with `--enable-cleanup`

### Getting Help

1. **Environment Setup**: Use `.env.template` to configure API keys
2. **Commands**: Run `python scripts/main.py --help` for command options
3. **Quick Start**: Use `./quick_start.sh` for automated setup

## 📊 Results

Results are saved to `outputs/` with timestamps:
- **Availability tests**: Quick compatibility checks
- **Full evaluations**: Complete benchmark results
- **Performance reports**: Resource utilization analysis
- **Logs**: Detailed execution logs

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
- Supports models from Hugging Face, LLaVA, and other frameworks
- Developed for hardware-aware multimodal evaluation

## 🏛️ Citation

```bibtex
@software{spectrabench-vision2025,
  title={SpectraBench-Vision},
  author={KISTI Large-scale AI Research Center},
  year={2025},
  url={https://github.com/gwleee/SpectraBench-Vision/},
  license={Apache-2.0},
}
```

---

*Developed with ❤️ by Gunwoo Lee from the AI Platform Team (Leader: Ryong Lee) at KISTI Large-scale AI Research Center (Director: Kyong-Ha Lee)*

*Supporting the Korean AI ecosystem with intelligent benchmarking tools through automated Docker multi-version evaluation*