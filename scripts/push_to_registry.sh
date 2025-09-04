#!/bin/bash

# SpectraVision Docker Registry Push Script
# Pushes production images to GitHub Container Registry

set -e

VERSIONS=("4.33" "4.37" "4.43" "4.49" "4.51")
REGISTRY="ghcr.io/gwleee"  # GitHub Container Registry

echo "SpectraVision Registry Push"
echo "=============================="
echo "Registry: $REGISTRY"
echo "Versions: ${VERSIONS[*]}"
echo

# Check if user is logged in to registry
if ! docker info | grep -q "Username"; then
    echo "Please login to Docker registry first:"
    echo "docker login ghcr.io"
    exit 1
fi

# Push base image
echo "Pushing base image..."
docker push $REGISTRY/spectravision-base:production
docker push $REGISTRY/spectravision-base:latest

if [ $? -eq 0 ]; then
    echo "Base image push successful"
else
    echo "Base image push failed"
    exit 1
fi

# Push all transformer versions
echo
echo "Pushing transformer-specific images..."

for version in "${VERSIONS[@]}"; do
    echo "Pushing spectravision-${version}..."
    
    docker push $REGISTRY/spectravision-${version}:production
    docker push $REGISTRY/spectravision-${version}:latest
    
    if [ $? -eq 0 ]; then
        echo "spectravision-${version} push successful"
    else
        echo "spectravision-${version} push failed"
        exit 1
    fi
done

echo
echo "All images pushed successfully!"
echo
echo "Available images on registry:"
for version in "${VERSIONS[@]}"; do
    echo "- $REGISTRY/spectravision-${version}:production"
    echo "- $REGISTRY/spectravision-${version}:latest"
done
echo "- $REGISTRY/spectravision-base:production"
echo "- $REGISTRY/spectravision-base:latest"

echo
echo "🏁 Deployment ready!"
echo "Users can now pull images with:"
echo "docker pull $REGISTRY/spectravision-4.37:latest"