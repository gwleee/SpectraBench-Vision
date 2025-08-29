#!/bin/bash

# SpectraVision Production Docker Images Build Script
# Builds optimized production images with .dockerignore

set -e

VERSIONS=("4.33" "4.37" "4.43" "4.49" "4.51")
REGISTRY="ghcr.io/gwleee"  # GitHub Container Registry
BUILD_DATE=$(date -u +'%Y%m%d-%H%M%S')

echo "🏗️ SpectraVision Production Image Builder"
echo "=========================================="
echo "Build Date: $BUILD_DATE"
echo "Registry: $REGISTRY"
echo "Versions to build: ${VERSIONS[*]}"
echo

# Build optimized base image
echo "📦 Step 1: Building optimized base image..."
docker build --no-cache \
  -t spectravision-base:production \
  -t $REGISTRY/spectravision-base:production \
  -t $REGISTRY/spectravision-base:latest \
  -f docker/base/Dockerfile .

if [ $? -eq 0 ]; then
    echo "✅ Base image build successful"
else
    echo "❌ Base image build failed"
    exit 1
fi

# Build all transformer versions
echo
echo "🔧 Step 2: Building transformer-specific production images..."

for version in "${VERSIONS[@]}"; do
    echo "Building spectravision-${version}:production..."
    
    docker build --no-cache \
      -t spectravision-${version}:production \
      -t $REGISTRY/spectravision-${version}:production \
      -t $REGISTRY/spectravision-${version}:latest \
      -f docker/transformers-${version}/Dockerfile .
    
    if [ $? -eq 0 ]; then
        echo "✅ spectravision-${version} build successful"
    else
        echo "❌ spectravision-${version} build failed"
        exit 1
    fi
    echo
done

echo "🎉 All production images built successfully!"
echo
echo "📋 Built images:"
docker images | grep spectravision | grep production

echo
echo "🚀 Ready for deployment!"
echo "To push to registry: ./scripts/push_to_registry.sh"