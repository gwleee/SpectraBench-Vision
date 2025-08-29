# 🐳 SpectraBench-Vision Docker Usage Guide

## 🎯 System Overview

SpectraBench-Vision is a **Docker-in-Docker integrated system** designed for **reproducibility and ease of use**:

- **Integrated Image**: Complete 30-model VLM evaluation system with single download
- **Automatic Container Management**: Pulls/starts/stops individual transformer version containers as needed
- **Docker-in-Docker**: Complete isolation with multi-version support

## 📦 System Architecture

### 🎯 Integrated System (Recommended)

| Image | Description | Models | Usage |
|--------|-------------|--------|-------|
| **spectrabench-vision** | **Complete Integrated System** | **30** | **All Users** |

**Key Features:**
- ✅ Docker-in-Docker architecture for complete automation
- ✅ Automatic container pull/start/stop as needed
- ✅ Built-in DockerOrchestrator for intelligent model management
- ✅ Single command access to entire system
- ✅ Automatic GPU detection and optimization

### 🔧 Individual Containers (Advanced Users/Developers)

Managed automatically by integrated system, but can be used directly:

| Image | Transformers | Main Models | Purpose |
|--------|-------------|-------------|---------|
| **spectravision-4.33** | 4.33.0 | Qwen-VL, VisualGLM | Legacy models |
| **spectravision-4.37** | 4.37.2 | InternVL2, LLaVA | Stable version |
| **spectravision-4.43** | 4.43.0 | Phi-3.5-Vision | Mid-range models |
| **spectravision-4.49** | 4.49.0 | SmolVLM, Qwen2.5-VL | Latest models |
| **spectravision-4.51** | 4.51.0 | Phi-4-Vision | Experimental |

## 📥 Image Downloads

### Recommended Image

| Image | Description | Models | Use Case | Download |
|-------|-------------|--------|----------|----------|
| **spectrabench-vision** | Integrated System | **30 models** | **General Users** | `docker pull ghcr.io/gwleee/ghcr.io/gwleee/spectrabench-vision:latest` |

### Developer - Multi-Version Images

| Image | Transformers | Key Models | Use Case | Download |
|-------|-------------|------------|----------|----------|
| **spectravision-4.33** | 4.33.0 | Qwen-VL, VisualGLM | Legacy models | `docker pull ghcr.io/gwleee/spectravision-4.33:latest` |
| **spectravision-4.37** | 4.37.2 | InternVL2, LLaVA | Stable version | `docker pull ghcr.io/gwleee/spectravision-4.37:latest` |
| **spectravision-4.43** | 4.43.0 | Phi-3.5-Vision | Mid-range models | `docker pull ghcr.io/gwleee/spectravision-4.43:latest` |
| **spectravision-4.49** | 4.49.0 | SmolVLM, Qwen2.5-VL | Latest models | `docker pull ghcr.io/gwleee/spectravision-4.49:latest` |
| **spectravision-4.51** | 4.51.0 | Phi-4-Vision | Experimental | `docker pull ghcr.io/gwleee/spectravision-4.51:latest` |

### Quick Download Methods (examples)
 
```bash
# Integrated system (recommended)
docker pull ghcr.io/gwleee/ghcr.io/gwleee/spectrabench-vision:latest

# Latest models only
docker pull ghcr.io/gwleee/spectravision-4.49:latest

# Stable versions only  
docker pull ghcr.io/gwleee/spectravision-4.37:latest
```

## 🚀 Quick Start (1 minute)

### Basic Usage
```bash
# 1. Start integrated system (automatic support for all 30 models)
docker run -it --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/ghcr.io/gwleee/spectrabench-vision:latest

# 2. Interactive mode inside container
python3 scripts/main.py --mode interactive
```

### Direct Evaluation
```bash
# Benchmark evaluation with specific model (one-line command)
docker run --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/ghcr.io/gwleee/spectrabench-vision:latest \
  python3 scripts/main.py --mode docker \
  --models "SmolVLM" --benchmarks "MMBench"
```

### Memory Optimization Options
```bash
# Latest models only (SmolVLM, Qwen2.5-VL)
docker run -it --gpus all \
  -e PULL_IMAGES=minimal \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/ghcr.io/gwleee/spectrabench-vision:latest

# Stable versions (InternVL2, LLaVA)
docker run -it --gpus all \
  -e PULL_IMAGES=stable \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/ghcr.io/gwleee/spectrabench-vision:latest
```

## ⚡ Multi-GPU Usage

```bash
# Multi-GPU utilization
docker run -it --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/ghcr.io/gwleee/spectrabench-vision:latest \
  python3 scripts/main.py --mode docker \
  --models "Qwen2.5-VL-32B" "Qwen2.5-VL-72B" \
  --benchmarks "MMBench" --gpu-ids 0 1 2 3
```

## 🛠️ Advanced Configuration

### Docker Compose Usage (For Developers)
```bash
# Start specific versions only
docker-compose -f docker/docker-compose.prod.yml --profile latest up -d

# Direct container usage
docker run --gpus all -it ghcr.io/gwleee/spectravision-4.49:latest
```

## 🛠️ Troubleshooting

### Common Issues

**Docker daemon connection error**
```bash
sudo systemctl restart docker
docker version
```

**GPU not recognized**
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8-runtime-ubuntu22.04 nvidia-smi
```

**Memory insufficient**
```bash
docker system prune -f
docker volume prune -f
```

## 🎉 System Verification

### Success Test
```bash
# 1. Basic system test
docker run --rm --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  ghcr.io/gwleee/spectrabench-vision:latest \
  python3 scripts/main.py --mode test

# 2. GPU and Docker connection check
docker run --rm --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  ghcr.io/gwleee/spectrabench-vision:latest \
  python3 -c "import torch; import docker; print(f'✅ CUDA: {torch.cuda.is_available()}'); print('✅ Docker: Connected'); print('🎯 SpectraBench-Vision ready!')"
```

### Expected Results
```
✅ CUDA: True
✅ GPU Count: 4
✅ Docker: Connected
🎯 SpectraBench-Vision integrated system ready!

📦 Pulling transformer version images...
✅ spectravision-4.49:latest ready
🎯 30 models, 24 benchmarks, 720 evaluation combinations available!
```

---

## 🎊 Success! Complete Integrated System Ready

**🎯 Congratulations!** SpectraBench-Vision integrated Docker system has been successfully set up!

### ✨ What you can do now:
- 🚀 **1-minute start**: Complete 30-model system with single command
- 🤖 **Full automation**: Optimal container auto-selection and management
- 📈 **Scalability**: Easy addition of new models/versions
- 🔧 **Reproducibility**: Identical evaluation environment anywhere

### 🎯 Next Steps:
```bash
# Start powerful VLM evaluation with integrated system!
docker run -it --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectrabench-vision:latest
```

**Happy Evaluating! 🚀✨**

