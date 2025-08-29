# SpectraBench-Vision Docker Usage Guide

## Quick Start (2 minutes) - Integrated Container

```bash
# Option 1: Direct Run (Fastest)
docker run -it --gpus all -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectrabench-vision:latest

# Option 2: With Environment Files
# 1. Clone repository (optional)
git clone https://github.com/gwleee/SpectraBench-Vision.git
cd SpectraBench-Vision

# 2. Environment setup
cp .env.template .env
# Open .env file and add your HF_TOKEN

# 3. Run integrated container
docker run -it --gpus all -v $(pwd):/workspace \
  -v $(pwd)/outputs:/workspace/outputs \
  --env-file .env \
  ghcr.io/gwleee/spectrabench-vision:latest
```

## Advanced Usage - Multi-Version Containers

```bash
# 1. Clone repository
git clone https://github.com/gwleee/SpectraBench-Vision.git
cd SpectraBench-Vision

# 2. Environment setup
cp .env.template .env
# Open .env file and add your HF_TOKEN

# 3. Download individual images (optional)
docker pull ghcr.io/gwleee/spectravision-4.49:latest  # Latest models
docker pull ghcr.io/gwleee/spectravision-4.37:latest  # Stable version

# 4. Run multi-version mode
python scripts/main.py
# Select "2. Multi-Version Docker"
```

## Available Images

### Recommended Image

| Image | Description | Models | Use Case | Download |
|-------|-------------|---------|----------|----------|
| **spectrabench-vision** | Integrated system | **30 models** | **General users** | `docker pull ghcr.io/gwleee/spectrabench-vision:latest` |

### Developer - Multi-Version Images

| Image | Transformers | Key Models | Use Case | Download |
|-------|-------------|------------|----------|----------|
| **spectravision-4.33** | 4.33.0 | Qwen-VL, VisualGLM | Legacy models | `docker pull ghcr.io/gwleee/spectravision-4.33:latest` |
| **spectravision-4.37** | 4.37.2 | InternVL2, LLaVA | Stable version | `docker pull ghcr.io/gwleee/spectravision-4.37:latest` |
| **spectravision-4.43** | 4.43.0 | Phi-3.5-Vision | Mid-range models | `docker pull ghcr.io/gwleee/spectravision-4.43:latest` |
| **spectravision-4.49** | 4.49.0 | SmolVLM, Qwen2.5-VL | Latest models | `docker pull ghcr.io/gwleee/spectravision-4.49:latest` |
| **spectravision-4.51** | 4.51.0 | Cutting-edge models | Experimental | `docker pull ghcr.io/gwleee/spectravision-4.51:latest` |

### Base Environment
| Image | Purpose | Download |
|-------|---------|----------|
| **spectravision-base** | CUDA + common dependencies | `docker pull ghcr.io/gwleee/spectravision-base:latest` |

## Model-Specific Recommendations

### Quick Start (Recommended)
```bash
# Integrated container - includes all models
docker run -it --gpus all -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectrabench-vision:latest
```

### By Model Type (Advanced users)
```bash  
# Latest models only (SmolVLM, Qwen2.5-VL)
docker run -it --gpus all ghcr.io/gwleee/spectravision-4.49:latest

# Stable models (LLaVA, InternVL2)
docker run -it --gpus all ghcr.io/gwleee/spectravision-4.37:latest

# Legacy models (Qwen-VL, VisualGLM)
docker run -it --gpus all ghcr.io/gwleee/spectravision-4.33:latest
```

## Integrated vs Multi-Version Comparison

### Integrated Container (Recommended)
**Advantages:**
- Instant use: All functionality with a single command
- Complete compatibility: Supports all 30 models
- Simple setup: No complex environment configuration
- Quick start: Begin evaluation within 2 minutes

**Use Cases:**
- General users
- Quick prototyping
- Demos and education

### Multi-Version Containers (Advanced)
**Advantages:**
- Perfect isolation: Independent environment per transformer version
- Memory efficient: Use only required versions
- Development friendly: Easy debugging per version
- Fine control: Detailed configuration per environment

**Use Cases:**
- Research developers
- Performance comparison across transformer versions
- Memory-constrained environments

## Usage Methods

### Method 1: Integrated execution via main.py (Recommended)
```bash
python scripts/main.py
# Select "2. Multi-Version Docker" from menu
# Automatically selects and runs appropriate container
```

### Method 2: Using Docker Compose
```bash
# Start specific version only
docker-compose -f docker/docker-compose.prod.yml --profile latest up -d

# Start all versions (requires significant resources)  
docker-compose -f docker/docker-compose.prod.yml --profile all up -d
```

### Method 3: Direct container execution
```bash
# Interactive mode container entry
docker run --gpus all -it \
  -v $(pwd)/VLMEvalKit:/workspace/VLMEvalKit \
  -v $(pwd)/outputs:/workspace/outputs \
  -e HF_TOKEN=$HF_TOKEN \
  ghcr.io/gwleee/spectravision-4.49:latest

# Run evaluation inside container
cd /workspace/VLMEvalKit
python run.py --model SmolVLM-Instruct --data MMBench_DEV_EN
```

## Environment Configuration

### .env File Setup
```bash
# Add the following to .env file
HF_TOKEN=your_huggingface_token_here
CUDA_VISIBLE_DEVICES=0
```

### GPU Configuration Check
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8-runtime-ubuntu22.04 nvidia-smi

# Test GPU with SpectraVision image
docker run --rm --gpus all ghcr.io/gwleee/spectravision-minimal:latest python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Performance Comparison

### Setup Time Comparison
| Method | Time | Success Rate | Supported Models |
|--------|------|--------------|------------------|
| Traditional local setup | 30-60 min | Low | 8 models |
| **Docker method** | **5-10 min** | **High** | **30 models** |

### Memory Usage
| Image | Size | Recommended GPU Memory |
|-------|------|-----------------------|
| minimal | 12.2GB | 16GB+ |
| 4.33 | 27.9GB | 32GB+ |  
| 4.43/4.49/4.51 | 12.2GB | 16GB+ |

## Troubleshooting

### Image Download Failed
```bash
# Docker login (if needed)
docker login ghcr.io

# Retry on network issues
docker pull ghcr.io/gwleee/spectravision-minimal:latest --retry 3
```

### GPU Not Recognized
```bash
# Check NVIDIA Docker installation
nvidia-docker version

# Restart Docker daemon
sudo systemctl restart docker
```

### Memory Shortage
```bash  
# Clean Docker
docker system prune -f
docker volume prune -f

# Remove unused images
docker image prune -a
```

### Container Internal Debugging
```bash
# Enter container
docker exec -it <container_name> /bin/bash

# Check logs
docker logs <container_name>
```

## Post-Setup Verification

### Success Tests
```bash
# 1. Verify image works normally
docker run --rm ghcr.io/gwleee/spectravision-minimal:latest python -c "import transformers; print(f'Transformers {transformers.__version__} ready!')"

# 2. Verify GPU availability  
docker run --rm --gpus all ghcr.io/gwleee/spectravision-minimal:latest python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 3. Full system test
python scripts/main.py --mode test
```

### Expected Results
```
Transformers 4.37.2 ready!
CUDA: True
30 models, 24 benchmarks, 720 evaluation combinations available!
```

---

**Congratulations!** SpectraVision Docker multi-version system has been successfully set up. You can now freely evaluate 30 VLM models in just 5 minutes!

**Next Steps**: Run `python scripts/main.py` and select "2. Multi-Version Docker" to start powerful multi-model evaluation!