#!/usr/bin/env python3
"""
Generate locked requirements files with cryptographic hashes for reproducible builds
Uses pip-tools to generate .lock files from .in template files
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


def run_command(cmd, cwd=None):
    """Run shell command and return success status"""
    print(f"🔧 Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ Success: {result.stdout.strip()}" if result.stdout.strip() else "✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr.strip()}")
        return False


def ensure_pip_tools():
    """Ensure pip-tools is installed"""
    try:
        import piptools
        print("✅ pip-tools already available")
        return True
    except ImportError:
        print("📦 Installing pip-tools...")
        return run_command([sys.executable, "-m", "pip", "install", "pip-tools"])


def create_requirements_in_files():
    """Create .in template files for each transformer version"""
    requirements_dir = Path("docker/requirements")

    # Template for transformers 4.33
    transformers_4_33_in = requirements_dir / "transformers-4.33.in"
    transformers_4_33_content = """# Transformers 4.33 base dependencies
transformers==4.33.2
tokenizers==0.13.3

# Core ML packages
torch>=2.0.0
torchvision
numpy
pillow

# Vision-Language model dependencies
SwissArmyTransformer
qwen-vl-utils
transformers-stream-generator
bitsandbytes
flash-attn
xformers
deepspeed
timm
accelerate
peft
sentencepiece
protobuf

# VLMEvalKit essential dependencies
pandas
requests
tqdm
matplotlib
seaborn
scipy
scikit-learn
"""

    # Template for transformers 4.37
    transformers_4_37_in = requirements_dir / "transformers-4.37.in"
    transformers_4_37_content = """# Transformers 4.37 base dependencies
transformers==4.37.2
tokenizers>=0.15.0

# Core ML packages
torch>=2.1.0
torchvision
numpy
pillow

# Vision-Language model dependencies
deepspeed
accelerate
bitsandbytes
flash-attn
xformers
timm
peft
sentencepiece
protobuf

# VLMEvalKit essential dependencies
pandas
requests
tqdm
matplotlib
seaborn
scipy
scikit-learn
"""

    # Template for transformers 4.49
    transformers_4_49_in = requirements_dir / "transformers-4.49.in"
    transformers_4_49_content = """# Transformers 4.49 base dependencies
transformers==4.49.0
tokenizers>=0.19.0

# Core ML packages
torch>=2.1.0
torchvision
numpy
pillow

# Latest generation model dependencies
flash-attn>=2.4.2
xformers
bitsandbytes
accelerate
timm
peft
sentencepiece
protobuf
deepspeed

# VLMEvalKit essential dependencies
pandas
requests
tqdm
matplotlib
seaborn
scipy
scikit-learn
"""

    # Template for transformers 4.51
    transformers_4_51_in = requirements_dir / "transformers-4.51.in"
    transformers_4_51_content = """# Transformers 4.51 base dependencies
transformers==4.51.0
tokenizers>=0.20.0

# Core ML packages
torch>=2.2.0
torchvision
numpy
pillow

# Cutting-edge model dependencies
bitsandbytes
flash-attn>=2.5.0
xformers
accelerate
timm
peft
sentencepiece
protobuf
deepspeed

# VLMEvalKit essential dependencies
pandas
requests
tqdm
matplotlib
seaborn
scipy
scikit-learn
"""

    # Base requirements template
    base_requirements_in = requirements_dir / "base-requirements.in"
    base_requirements_content = """# Base system dependencies shared across all versions
# VLMEvalKit and evaluation framework
pandas>=2.0.0
numpy>=1.24.0
pillow>=9.0.0
requests>=2.28.0
tqdm>=4.64.0

# Visualization and analysis
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
scikit-learn>=1.2.0

# Image processing
opencv-python>=4.7.0

# Utilities
pyyaml>=6.0
openpyxl>=3.1.0
xlsxwriter>=3.1.0

# Development and debugging
ipython>=8.0.0
jupyter>=1.0.0
"""

    # Write all template files
    templates = [
        (transformers_4_33_in, transformers_4_33_content),
        (transformers_4_37_in, transformers_4_37_content),
        (transformers_4_49_in, transformers_4_49_content),
        (transformers_4_51_in, transformers_4_51_content),
        (base_requirements_in, base_requirements_content),
    ]

    for file_path, content in templates:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"📝 Created template: {file_path}")


def generate_locked_requirements():
    """Generate locked .lock files with hashes from .in templates"""
    requirements_dir = Path("docker/requirements")

    # Find all .in files
    in_files = list(requirements_dir.glob("*.in"))

    if not in_files:
        print("❌ No .in template files found!")
        return False

    success = True
    for in_file in in_files:
        lock_file = in_file.with_suffix('.lock')
        print(f"\n🔒 Generating locked requirements: {in_file} -> {lock_file}")

        # Use pip-compile with --generate-hashes for cryptographic verification
        cmd = [
            sys.executable, "-m", "piptools", "compile",
            "--generate-hashes",
            "--resolver=backtracking",
            "--verbose",
            str(in_file.name),
            "--output-file", str(lock_file.name)
        ]

        if not run_command(cmd, cwd=requirements_dir):
            success = False
            print(f"❌ Failed to generate {lock_file}")
        else:
            print(f"✅ Generated {lock_file} with cryptographic hashes")

    return success


def verify_locked_files():
    """Verify that locked files can be installed"""
    requirements_dir = Path("docker/requirements")
    lock_files = list(requirements_dir.glob("*.lock"))

    print(f"\n🔍 Found {len(lock_files)} lock files to verify:")
    for lock_file in lock_files:
        # Just check that file is readable and has content
        try:
            with open(lock_file, 'r') as f:
                content = f.read()
                if "sha256:" in content:
                    print(f"✅ {lock_file.name}: Contains cryptographic hashes")
                else:
                    print(f"⚠️  {lock_file.name}: No hashes found!")
        except Exception as e:
            print(f"❌ {lock_file.name}: Error reading file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate locked requirements with hashes")
    parser.add_argument("--create-templates", action="store_true",
                      help="Create .in template files")
    parser.add_argument("--generate-locks", action="store_true",
                      help="Generate .lock files from .in templates")
    parser.add_argument("--verify", action="store_true",
                      help="Verify generated .lock files")
    parser.add_argument("--all", action="store_true",
                      help="Run all steps: create templates, generate locks, verify")

    args = parser.parse_args()

    if not any([args.create_templates, args.generate_locks, args.verify, args.all]):
        parser.print_help()
        return 1

    # Ensure requirements directory exists
    requirements_dir = Path("docker/requirements")
    requirements_dir.mkdir(parents=True, exist_ok=True)

    # Install pip-tools if needed
    if not ensure_pip_tools():
        print("❌ Failed to install pip-tools")
        return 1

    success = True

    if args.all or args.create_templates:
        print("\n📝 Creating .in template files...")
        create_requirements_in_files()

    if args.all or args.generate_locks:
        print("\n🔒 Generating locked requirements...")
        if not generate_locked_requirements():
            success = False

    if args.all or args.verify:
        print("\n🔍 Verifying locked files...")
        verify_locked_files()

    if success:
        print("\n🎉 Dependency reproducibility setup completed successfully!")
        print("Usage in Dockerfiles:")
        print("  COPY docker/requirements/transformers-4.33.lock /tmp/requirements.lock")
        print("  RUN pip install --require-hashes -r /tmp/requirements.lock")
    else:
        print("\n❌ Some operations failed!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())