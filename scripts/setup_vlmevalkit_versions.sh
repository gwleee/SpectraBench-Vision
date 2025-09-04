#!/bin/bash
# setup_vlmevalkit_versions.sh - Create version-specific VLMEvalKit directories with patches

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VERSIONS=("4.33" "4.37" "4.43" "4.49" "4.51")

echo "Setting up version-specific VLMEvalKit directories..."

# Check if base VLMEvalKit exists
if [ ! -d "$PROJECT_ROOT/VLMEvalKit" ]; then
    echo "Base VLMEvalKit directory not found. Cloning..."
    cd "$PROJECT_ROOT"
    git clone https://github.com/open-compass/VLMEvalKit.git
    cd VLMEvalKit
    git submodule update --init --recursive
    cd ..
fi

# Create version-specific directories
for version in "${VERSIONS[@]}"; do
    vlm_dir="$PROJECT_ROOT/VLMEvalKit-$version"
    
    if [ -d "$vlm_dir" ]; then
        echo "VLMEvalKit-$version already exists. Skipping..."
        continue
    fi
    
    echo "Creating VLMEvalKit-$version..."
    
    # Copy base VLMEvalKit
    cp -r "$PROJECT_ROOT/VLMEvalKit" "$vlm_dir"
    
    # Remove git history to save space
    rm -rf "$vlm_dir/.git"
    
    # Apply patches for this version
    echo "Applying patches for transformer version $version..."
    cd "$vlm_dir"
    
    # Apply all patches using existing script
    if [ -f "$PROJECT_ROOT/scripts/apply_patches.sh" ]; then
        VLMEVALKIT_DIR="$vlm_dir" "$PROJECT_ROOT/scripts/apply_patches.sh"
    fi
    
    echo "VLMEvalKit-$version setup complete"
done

echo ""
echo "All VLMEvalKit versions setup complete!"
echo "Note: Version-specific directories are not committed to Git (automatically generated)"