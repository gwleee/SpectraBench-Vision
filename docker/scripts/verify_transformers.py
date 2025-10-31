#!/usr/bin/env python3
"""Transformers version verification script"""
import sys
import os

# Get expected version from environment
EXPECTED_VERSION = os.environ.get('TRANSFORMERS_VERSION', '')
if not EXPECTED_VERSION:
    print("❌ TRANSFORMERS_VERSION environment variable not set")
    sys.exit(1)

import torch
import transformers as tr

print(f'=== Transformers {EXPECTED_VERSION} Verification ===')
print(f'Transformers version: {tr.__version__}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

# Verify transformer version (allow patch versions, e.g., 4.33.0 accepts 4.33.x)
expected_major_minor = '.'.join(EXPECTED_VERSION.split('.')[:2])  # e.g., "4.33"
actual_major_minor = '.'.join(tr.__version__.split('.')[:2])

if actual_major_minor != expected_major_minor:
    print(f'❌ Version mismatch: Expected {expected_major_minor}.x, got {tr.__version__}')
    sys.exit(1)

print(f'✅ Transformers version verified: {tr.__version__} (expected {expected_major_minor}.x)')

# Test critical imports (no downloads)
try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForCausalLM,
        AutoProcessor, AutoImageProcessor
    )
    print('✅ Core components imported successfully')
except ImportError as e:
    print(f'❌ Critical import failed: {e}')
    sys.exit(1)

# Test additional components
try:
    import timm
    import accelerate
    import peft
    print('✅ All components available')
except ImportError as e:
    print(f'⚠️ Optional component missing: {e}')

print(f'=== Transformers {EXPECTED_VERSION} Verification Complete ===')
