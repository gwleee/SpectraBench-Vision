#!/bin/bash

# SpectraVision Environment Loader
# Usage: source ./load_env.sh

set -a  # Automatically export all variables
source .env
set +a  # Stop auto-export

echo "🚀 SpectraVision Environment Loaded"
echo "📍 Current directory: $(pwd)"
echo ""
echo "🔑 API Keys Status:"
echo "   HF_TOKEN: $([ -n "$HF_TOKEN" ] && echo "✅ Set" || echo "❌ Not set")"
echo "   ANTHROPIC_API_KEY: $([ -n "$ANTHROPIC_API_KEY" ] && echo "✅ Set" || echo "❌ Not set")"  
echo "   OPENAI_API_KEY: $([ -n "$OPENAI_API_KEY" ] && echo "✅ Set" || echo "❌ Not set")"
echo ""
echo "🖥️  Hardware Config:"
echo "   CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not set}"
echo "   LOG_LEVEL: ${LOG_LEVEL:-Not set}"
echo ""
echo "💡 Usage:"
echo "   - Run: source ./load_env.sh"
echo "   - Or: source load_env.sh"
echo "   - Your ANTHROPIC_API_KEY is now available in this session"