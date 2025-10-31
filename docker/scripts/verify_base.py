#!/usr/bin/env python3
"""Base dependencies verification script"""
import sys
import pkg_resources

print('=== Base Dependencies Verification ===')
print(f'Python version: {sys.version}')

# Verify core packages
required_packages = ['numpy', 'pillow', 'requests', 'tqdm']
for package in required_packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f'✅ {package}: {version}')
    except pkg_resources.DistributionNotFound:
        print(f'❌ {package}: NOT FOUND')
        sys.exit(1)

# Test imports (minimal for speed)
try:
    import numpy as np
    import PIL
    import requests
    print('✅ Critical base imports successful')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)

print('=== Base Verification Complete ===')
