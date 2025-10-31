# Reproducible Dependency Management

This directory contains the dependency reproducibility system for SpectraBench-Vision Docker images.

## Overview
The system uses pip-tools to generate locked requirements files with cryptographic hashes, ensuring reproducible builds across different environments and time periods.

## Files Generated
- `*.in` files: Human-readable dependency templates
- `*.lock` files: Auto-generated locked requirements with exact versions and SHA256 hashes
- `generate_locked_requirements.py`: Script to generate locked requirements

## Usage

### Generate Locked Requirements
```bash
# Generate all locked requirements from .in templates
python3 scripts/generate_locked_requirements.py --all

# Generate only specific templates
python3 scripts/generate_locked_requirements.py --generate-locks
```

### Docker Integration
In Dockerfiles, use the locked requirements:

```dockerfile
COPY docker/requirements/transformers-4.33.lock /tmp/requirements.lock
RUN pip install --require-hashes -r /tmp/requirements.lock
```

The `--require-hashes` flag ensures that pip verifies the SHA256 hash of every downloaded package, preventing supply chain attacks and ensuring reproducible builds.

## Transformer Version Dependencies

### 4.33 (Legacy Models)
- InstructBLIP, Qwen-VL-Chat, mPLUG-Owl2, Monkey
- Python 3.9 compatibility required
- LAVIS excluded due to open3d compatibility issues

### 4.37 (Stable Models)
- LLaVA, InternVL2, CogVLM series
- Modern dependency stack with DeepSpeed support

### 4.49 (Latest Models)
- SmolVLM, Qwen2.5-VL, Phi-3.5-Vision
- Flash-attention 2.4.2+ for optimal performance

### 4.51 (Cutting-edge)
- Phi-4-Vision, latest experimental models
- Bleeding-edge dependency versions

## Benefits

1. **Reproducible Builds**: Same exact package versions every time
2. **Supply Chain Security**: Cryptographic hash verification
3. **Version Conflicts Prevention**: Pre-resolved dependency trees
4. **Build Speed**: No dependency resolution during Docker build
5. **Audit Trail**: Complete record of all package versions and sources

## Maintenance

Lock files should be regenerated when:
- Base dependencies change in .in files
- Security updates are needed
- New transformer versions are added
- Monthly maintenance cycles

The system ensures that all users building Docker images get identical dependency versions, preventing the "works on my machine" problem.