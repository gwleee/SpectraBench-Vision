#!/bin/bash
set -e

echo "🚀 Starting SpectraBench-Vision Integrated System..."

# Start Docker daemon in background
dockerd &
DOCKERD_PID=$!

# Wait for Docker daemon to be ready
echo "⏳ Waiting for Docker daemon..."
while ! docker info > /dev/null 2>&1; do
    sleep 1
done
echo "✅ Docker daemon ready"

# Log in to GitHub Container Registry if credentials provided
if [ -n "$GITHUB_TOKEN" ]; then
    echo "🔐 Logging in to GitHub Container Registry..."
    # Use default username if GITHUB_USERNAME not set
    GITHUB_USER="${GITHUB_USERNAME:-gwleee}"
    echo "$GITHUB_TOKEN" | docker login ghcr.io -u "$GITHUB_USER" --password-stdin
fi

# Check GPU availability
if command -v nvidia-smi > /dev/null 2>&1; then
    echo "🎮 GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️ No NVIDIA GPU detected"
fi

# Pull necessary images based on user preference or all by default
if [ "$PULL_IMAGES" = "minimal" ]; then
    echo "📦 Pulling minimal image set..."
    docker pull ghcr.io/gwleee/spectravision-4.49:latest
elif [ "$PULL_IMAGES" = "stable" ]; then
    echo "📦 Pulling stable image set..."
    docker pull ghcr.io/gwleee/spectravision-4.37:latest
    docker pull ghcr.io/gwleee/spectravision-4.49:latest
elif [ "$PULL_IMAGES" != "none" ]; then
    echo "📦 Pulling all transformer version images..."
    docker pull ghcr.io/gwleee/spectravision-4.33:latest
    docker pull ghcr.io/gwleee/spectravision-4.37:latest
    docker pull ghcr.io/gwleee/spectravision-4.43:latest
    docker pull ghcr.io/gwleee/spectravision-4.49:latest
    docker pull ghcr.io/gwleee/spectravision-4.51:latest
fi

echo "🎯 SpectraBench-Vision is ready!"
echo ""
echo "Usage Options:"
echo "1. Interactive Mode: python3 scripts/docker_main.py --mode interactive"
echo "2. Batch Mode: python3 scripts/docker_main.py --mode batch --models MODEL --benchmarks BENCHMARK"
echo "3. Test Mode: python3 scripts/docker_main.py --mode test"
echo ""

# If no command specified, start interactive shell
if [ $# -eq 0 ]; then
    echo "🐚 Starting interactive shell..."
    exec bash
else
    echo "🚀 Executing: $@"
    exec "$@"
fi