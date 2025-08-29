# SpectraVision Troubleshooting Guide

## 🚨 Common Issues and Solutions

### Docker and Container Issues

#### "Docker daemon not running"
```bash
# Ubuntu/Debian
sudo systemctl start docker
sudo systemctl enable docker

# macOS
open -a Docker

# Windows
Start Docker Desktop application
```

#### "Permission denied" accessing Docker
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
# Log out and back in for changes to take effect

# Or run with sudo (not recommended for production)
sudo docker-compose -f docker/docker-compose.prod.yml up -d
```

#### "NVIDIA Docker runtime not found"
```bash
# Install nvidia-container-toolkit
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### GPU and Memory Issues

#### "CUDA out of memory"
**Solution 1: Reduce batch size**
```bash
# Set smaller batch size in .env
VLMEVAL_BATCH_SIZE=1
```

**Solution 2: Use smaller models**
```bash
# Switch to lighter models
python scripts/main.py --models "SmolVLM-256M" --benchmarks "MMBench"
```

**Solution 3: Enable multi-GPU**
```bash
# Use multiple GPUs for large models
GPU_COUNT=2 NVIDIA_VISIBLE_DEVICES=0,1 docker-compose -f docker/docker-compose.prod.yml --profile latest up -d
```

#### "No CUDA device detected"
```bash
# Check NVIDIA driver
nvidia-smi

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Check GPU visibility in container
docker run --rm --gpus all -e NVIDIA_VISIBLE_DEVICES=all ubuntu:22.04 nvidia-smi
```

### Authentication and Token Issues

#### "Access denied to gated model"
**HuggingFace Token Issues:**
1. Get token from: https://huggingface.co/settings/tokens
2. Add to `.env` file:
```bash
HF_TOKEN=hf_your_token_here
```

3. Verify token works:
```bash
# Test token
docker run --rm -e HF_TOKEN=$HF_TOKEN \
  ghcr.io/gwleee/spectravision-4.49:latest \
  python -c "from huggingface_hub import whoami; print(whoami())"
```

#### "Repository not found" for private models
- Ensure your HF account has access to the model
- For private models, contact model authors for access
- Use alternative public models with similar capabilities

### Model Download and Loading Issues

#### "Connection timeout" during model download
```bash
# Set longer timeout
export HF_HUB_DOWNLOAD_TIMEOUT=1200

# Use proxy if behind corporate firewall
export https_proxy=http://your-proxy:port
export http_proxy=http://your-proxy:port
```

#### "Disk space full" during model download
```bash
# Check disk usage
df -h

# Clean Docker system
docker system prune -a -f

# Clean model cache
rm -rf ~/.cache/huggingface/hub/

# Use external storage for cache
mkdir -p /external/storage/cache
export HF_HOME=/external/storage/cache
```

### VLMEvalKit Specific Issues

#### "ModuleNotFoundError: No module named 'vlmeval'"
**Cause**: VLMEvalKit not properly installed or PYTHONPATH incorrect

**Solution**:
```bash
# Verify PYTHONPATH in container
docker exec -it container_name python -c "import sys; print(sys.path)"

# Check VLMEvalKit directory exists
docker exec -it container_name ls -la /workspace/VLMEvalKit/

# Reinstall VLMEvalKit in container
docker exec -it container_name pip install -e /workspace/VLMEvalKit/
```

#### "Model not supported" error
**Check supported models:**
```bash
# List all supported models
docker exec -it container_name python -c "from vlmeval.config import supported_VLM; print(list(supported_VLM.keys()))"

# Verify model ID spelling
python scripts/main.py --list-models
```

### Performance Issues

#### "Evaluation taking too long"
**Optimization strategies:**
1. **Use faster models**: SmolVLM series instead of large models
2. **Enable caching**: Mount persistent cache volume
3. **Reduce dataset size**: Use subset for testing
4. **Multiple GPUs**: Parallelize evaluation

```bash
# Quick test with small dataset
python scripts/main.py --models "SmolVLM-256M" --benchmarks "SEED" --mode test --test-time-limit 60
```

#### "Container startup slow"
**Solutions:**
1. **Pre-pull images**:
```bash
docker-compose -f docker/docker-compose.prod.yml pull
```

2. **Use SSD storage**: Move Docker data to SSD
```bash
# Check Docker data location
docker info | grep "Docker Root Dir"

# Move to SSD (advanced)
sudo systemctl stop docker
sudo mv /var/lib/docker /ssd/path/docker
sudo ln -s /ssd/path/docker /var/lib/docker
sudo systemctl start docker
```

### Network and Connectivity Issues

#### "Image pull failed"
```bash
# Check network connectivity
ping ghcr.io

# Use different registry mirror
docker pull --platform linux/amd64 ghcr.io/gwleee/spectravision-4.49:latest

# Build locally instead
./scripts/build_local_images.sh
```

#### "DNS resolution failed"
```bash
# Configure DNS in Docker
# Add to /etc/docker/daemon.json
{
  "dns": ["8.8.8.8", "8.8.4.4"]
}

sudo systemctl restart docker
```

## 🔧 Advanced Debugging

### Enable Debug Logging
```bash
# Set debug environment
export SPECTRAVISION_DEBUG=1
export VLMEVAL_DEBUG=1

# Run with verbose output
python scripts/main.py --verbose --models "SmolVLM" --benchmarks "MMBench"
```

### Container Inspection
```bash
# Enter running container for debugging
docker exec -it container_name /bin/bash

# Check container logs
docker logs container_name

# Inspect container configuration
docker inspect container_name
```

### Model Loading Debugging
```python
# Test model loading manually in container
python -c "
from vlmeval.config import supported_VLM
model_name = 'SmolVLM'
if model_name in supported_VLM:
    model_class = supported_VLM[model_name]
    print(f'Model class: {model_class}')
    model = model_class()
    print('Model loaded successfully')
else:
    print(f'Model {model_name} not supported')
"
```

## 📞 Getting Help

If you encounter issues not covered in this guide:

1. **Check GitHub Issues**: https://github.com/gwleee/SpectraBench-Vision/issues
2. **VLMEvalKit Documentation**: https://github.com/open-compass/VLMEvalKit
3. **Docker Documentation**: https://docs.docker.com/

### Reporting Bugs

When reporting issues, please include:
- System information (OS, GPU, Docker version)
- Full error message and stack trace
- Steps to reproduce the issue
- Docker logs: `docker-compose logs > logs.txt`

### Performance Optimization Consultation

For production deployments requiring optimization:
- Hardware recommendations
- Model selection guidance  
- Batch processing strategies
- Multi-GPU setup assistance