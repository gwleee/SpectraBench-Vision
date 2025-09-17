#!/bin/bash
# apply_patches.sh - Automatically apply VLMEvalKit token fixes
# VLMEvalKit 토큰 인증 패치 적용
# Docker 빌드 중 자동 실행됨
# 10개 모델별 패치 관리

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PATCHES_DIR="$PROJECT_ROOT/patches"
VLMEVALKIT_DIR="$PROJECT_ROOT/VLMEvalKit"

echo "Applying VLMEvalKit token authentication patches..."

if [ ! -d "$VLMEVALKIT_DIR" ]; then
    echo "Error: VLMEvalKit directory not found: $VLMEVALKIT_DIR"
    exit 1
fi

if [ ! -d "$PATCHES_DIR" ]; then
    echo "Error: Patches directory not found: $PATCHES_DIR"
    exit 1
fi

cd "$VLMEVALKIT_DIR"

# Check if patches have already been applied
PATCH_APPLIED_MARKER=".spectravision_patches_applied"
if [ -f "$PATCH_APPLIED_MARKER" ]; then
    echo "Patches already applied. Skipping..."
    exit 0
fi

# Apply each patch file
PATCHES_APPLIED=0
for patch_file in "$PATCHES_DIR"/*.patch; do
    if [ -f "$patch_file" ]; then
        patch_name=$(basename "$patch_file")
        echo "Applying patch: $patch_name"
        
        # Try to apply patch (check if it can be applied first)
        if git apply --check "$patch_file" 2>/dev/null; then
            git apply "$patch_file"
            echo "Successfully applied: $patch_name"
            PATCHES_APPLIED=$((PATCHES_APPLIED + 1))
        elif patch --dry-run -p1 < "$patch_file" >/dev/null 2>&1; then
            # Fallback to patch command if git apply fails
            patch -p1 < "$patch_file"
            echo "Successfully applied (using patch): $patch_name"
            PATCHES_APPLIED=$((PATCHES_APPLIED + 1))
        else
            echo "Warning: Could not apply patch: $patch_name (may already be applied or incompatible)"
        fi
    fi
done

if [ $PATCHES_APPLIED -gt 0 ]; then
    echo "Applied $PATCHES_APPLIED patches successfully"
    # Mark patches as applied
    echo "$(date): Applied $PATCHES_APPLIED patches" > "$PATCH_APPLIED_MARKER"
    echo "VLMEvalKit now supports HuggingFace token authentication!"
    echo ""
    echo "Make sure your .env file contains:"
    echo "   HUGGING_FACE_HUB_TOKEN=hf_your_token_here"
    echo ""
else
    echo "No new patches applied"
fi