#!/bin/bash
# SpectraVision Local Docker Images Build Script
# Builds all Docker images locally for testing (without pushing to registry)

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🐳 SpectraVision Local Docker Images Build${NC}"
echo "==============================================="
echo "Building all images locally for testing..."
echo

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Define all images to build (essential versions only)
declare -A IMAGES=(
    ["transformers-4.33"]="Legacy models (Qwen, VisualGLM)"
    ["transformers-4.37"]="Current stable models (InternVL2, LLaVA)" 
    ["transformers-4.43"]="Mid-range models (Phi-3.5-Vision)"
    ["transformers-4.49"]="Latest models (SmolVLM, Qwen2.5-VL)"
    ["transformers-4.51"]="Cutting-edge models (Phi-4-Vision)"
)

# Build base image first
echo -e "${YELLOW}📦 Building base image...${NC}"
if docker build -t spectravision-base:latest -f docker/base/Dockerfile .; then
    echo -e "${GREEN}✅ Base image built successfully${NC}"
else
    echo -e "${RED}❌ Base image build failed${NC}"
    exit 1
fi
echo

# Build each transformer version
SUCCESS_COUNT=0
TOTAL_COUNT=${#IMAGES[@]}

for image_name in "${!IMAGES[@]}"; do
    description="${IMAGES[$image_name]}"
    version="${image_name##*-}"
    local_tag="spectravision-$version:latest"
    
    echo -e "${YELLOW}📦 Building $image_name...${NC}"
    echo "   Description: $description"
    echo "   Tag: $local_tag"
    
    # Build the image
    if docker build -t $local_tag -f docker/$image_name/Dockerfile .; then
        echo -e "${GREEN}✅ $image_name built successfully${NC}"
        ((SUCCESS_COUNT++))
    else
        echo -e "${RED}❌ $image_name build failed${NC}"
    fi
    echo
done

# Create summary
echo "========================================================"
echo -e "${GREEN}🎉 Build Summary${NC}"
echo "Successfully built: $SUCCESS_COUNT/$TOTAL_COUNT images"
echo
echo "📋 Built images:"
echo "  spectravision-base:latest"
for image_name in "${!IMAGES[@]}"; do
    version="${image_name##*-}"
    if docker images | grep -q "spectravision-$version"; then
        echo -e "  ${GREEN}✅${NC} spectravision-$version:latest"
    else
        echo -e "  ${RED}❌${NC} spectravision-$version:latest"
    fi
done

echo
echo -e "${YELLOW}📝 Next steps:${NC}"
echo "1. Test images with: docker run -it --gpus all spectravision-4.37:latest bash"
echo "2. Run evaluation test: python test_docker_simple.py"  
echo "3. If tests pass, push to registry with: ./scripts/build_and_push_images.sh"

if [ $SUCCESS_COUNT -eq $TOTAL_COUNT ]; then
    echo -e "${GREEN}🚀 All images built successfully! Ready for testing.${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some images failed to build. Check the logs above.${NC}"
    exit 1
fi