#!/bin/bash
# SpectraVision Docker Images Build and Push Script
# Builds all Docker images and pushes them to registry

set -e

# Get GitHub username from parameter or prompt
if [ -z "$1" ]; then
    read -p "Enter your GitHub username: " GITHUB_USERNAME
else
    GITHUB_USERNAME="$1"
fi

# Configuration
DOCKER_REGISTRY="ghcr.io"
DOCKER_USERNAME="$GITHUB_USERNAME"
PROJECT_NAME="spectravision"

echo "Build Configuration:"
echo "  Registry: $DOCKER_REGISTRY"
echo "  Username: $DOCKER_USERNAME"
echo "  Project: $PROJECT_NAME"
echo

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}SpectraVision Docker Images Build and Push${NC}"
echo "=================================================="

# Check if logged in to registry
echo -e "${YELLOW}Checking registry login...${NC}"
if ! docker info | grep -q "Registry:"; then
    echo -e "${RED}Please login to Docker registry first:${NC}"
    echo "docker login $DOCKER_REGISTRY"
    exit 1
fi

# Define all images to build
declare -A IMAGES=(
    ["integrated"]="All 30 models integrated (recommended)"
    ["transformers-4.33"]="Legacy models (Qwen, VisualGLM)"
    ["transformers-4.37"]="Current stable models (InternVL2, LLaVA)"
    ["transformers-4.43"]="Mid-range models (Phi-3.5-Vision)"
    ["transformers-4.49"]="Latest models (SmolVLM, Qwen2.5-VL)"
    ["transformers-4.51"]="Cutting-edge models (Phi-4-Vision)"
)

# Build base image first
echo -e "${YELLOW}Building base image...${NC}"
docker build -t spectravision-base:latest -f docker/base/Dockerfile .

# Tag and push base image
BASE_TAG="$DOCKER_REGISTRY/$DOCKER_USERNAME/$PROJECT_NAME-base:latest"
docker tag spectravision-base:latest $BASE_TAG
echo -e "${YELLOW}Pushing base image...${NC}"
docker push $BASE_TAG
echo -e "${GREEN}Base image pushed: $BASE_TAG${NC}"

# Build and push each version
for image_name in "${!IMAGES[@]}"; do
    description="${IMAGES[$image_name]}"
    
    echo -e "${YELLOW}Building $image_name...${NC}"
    echo "   Description: $description"
    
    if [ "$image_name" = "integrated" ]; then
        # Build integrated container with all models
        docker build -t spectrabench-vision:latest -f docker/integrated/Dockerfile .
        local_tag="spectrabench-vision:latest"
        registry_tag="$DOCKER_REGISTRY/$DOCKER_USERNAME/spectrabench-vision:latest"
    else
        # Build using docker-compose for multi-version containers
        docker-compose -f docker/docker-compose.yml build $image_name
        
        # Tag for registry
        local_tag="spectravision-${image_name##*-}:latest"
        registry_tag="$DOCKER_REGISTRY/$DOCKER_USERNAME/$PROJECT_NAME-${image_name##*-}:latest"
    fi
    
    docker tag $local_tag $registry_tag
    
    echo -e "${YELLOW}Pushing $image_name...${NC}"
    docker push $registry_tag
    
    echo -e "${GREEN}$image_name pushed: $registry_tag${NC}"
    echo
done

# Create summary
echo -e "${GREEN}All images built and pushed successfully!${NC}"
echo
echo "Summary of pushed images:"
echo "  Integrated System: $DOCKER_REGISTRY/$DOCKER_USERNAME/spectrabench-vision:latest"
echo "  Base: $DOCKER_REGISTRY/$DOCKER_USERNAME/$PROJECT_NAME-base:latest"
for image_name in "${!IMAGES[@]}"; do
    if [ "$image_name" != "integrated" ]; then
        version="${image_name##*-}"
        echo "  $image_name: $DOCKER_REGISTRY/$DOCKER_USERNAME/$PROJECT_NAME-$version:latest"
    fi
done

echo
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Update docker-compose.yml with registry paths"
echo "2. Update README.md with pull instructions"
echo "3. Test pulling and running images"
echo
echo -e "${GREEN}Users can now use:${NC}"
echo "# Integrated Container (Recommended):"
echo "docker run -it --gpus all $DOCKER_REGISTRY/$DOCKER_USERNAME/spectrabench-vision:latest"
echo ""
echo "# Multi-Version System:"
echo "docker-compose -f docker/docker-compose.yml pull"