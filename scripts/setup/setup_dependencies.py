#!/usr/bin/env python3
"""
SpectraVision Setup Script
Automatically sets up dependencies and configurations for running on any machine.
"""

import os
import sys
import subprocess
import warnings
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from spectravision.env_manager import EnvironmentManager
except ImportError:
    # Fallback if env_manager is not available yet
    EnvironmentManager = None


def run_command(cmd, cwd=None, check=True):
    """Run shell command with error handling."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error output: {e.stderr}")
        if check:
            raise
        return e


def setup_vlmevalkit():
    """Clone and setup VLMEvalKit repository."""
    spectravision_root = Path(__file__).parent.parent
    vlm_path = spectravision_root / "VLMEvalKit"
    
    if not vlm_path.exists():
        print("Cloning VLMEvalKit repository...")
        run_command("git clone https://github.com/open-compass/VLMEvalKit.git", cwd=spectravision_root)
    
    # Install VLMEvalKit
    if vlm_path.exists():
        print("Installing VLMEvalKit...")
        run_command("pip install -e .", cwd=vlm_path)




def setup_environment_config():
    """Setup environment configuration and personal API keys."""
    spectravision_root = Path(__file__).parent.parent
    
    print("\n" + "="*60)
    print("ENVIRONMENT CONFIGURATION SETUP")
    print("="*60)
    
    # Initialize environment manager
    if EnvironmentManager:
        env_manager = EnvironmentManager()
        
        # Create .env file if it doesn't exist
        if env_manager.create_env_file_if_missing():
            print("\nA new .env file has been created from the template.")
            print("Please edit the .env file and add your personal API keys:")
            print(f"   - File location: {env_manager.env_file}")
            print("   - Required: HF_TOKEN (for gated models like MiniCPM-V-2.6)")
            print("   - Optional: OPENAI_API_KEY")
            print("")
            
            # Ask if user wants to set HF token now
            response = input("Would you like to set your HF token now? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                setup_hf_token_interactive(env_manager.env_file)
            else:
                print("You can set your HF token later by editing the .env file.")
        else:
            print(f"Using existing .env file: {env_manager.env_file}")
            
            # Check token status
            token_status = env_manager.check_required_tokens()
            if token_status['hf_token']:
                print("HF_TOKEN is configured")
            else:
                print("WARNING: HF_TOKEN is not configured")
                response = input("Would you like to set your HF token now? (y/N): ").strip().lower()
                if response in ['y', 'yes']:
                    setup_hf_token_interactive(env_manager.env_file)
        
        # Apply environment settings
        env_manager.apply_environment_settings()
        
        # Also create .env in VLMEvalKit for backward compatibility
        vlm_env_path = spectravision_root / "VLMEvalKit" / ".env"
        if env_manager.config.hf_token and not vlm_env_path.exists():
            vlm_env_path.parent.mkdir(exist_ok=True)
            with open(vlm_env_path, 'w') as f:
                f.write(f'HF_TOKEN={env_manager.config.hf_token}\n')
            print(f"HF_TOKEN also saved to VLMEvalKit .env file")
    else:
        # Fallback to old method if env_manager is not available
        setup_hf_token_legacy()

def setup_hf_token_interactive(env_file_path):
    """Interactive HF token setup."""
    print("\nHugging Face Token Setup:")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'Read' permissions")
    print("3. Enter it below")
    
    token = input("Enter your HF token: ").strip()
    
    if token:
        try:
            # Read existing .env content
            existing_lines = []
            if env_file_path.exists():
                with open(env_file_path, 'r') as f:
                    existing_lines = f.readlines()
            
            # Update or add HF_TOKEN
            token_updated = False
            for i, line in enumerate(existing_lines):
                if line.startswith('HF_TOKEN='):
                    existing_lines[i] = f'HF_TOKEN={token}\n'
                    token_updated = True
                    break
            
            if not token_updated:
                existing_lines.append(f'HF_TOKEN={token}\n')
            
            # Write back to file
            with open(env_file_path, 'w') as f:
                f.writelines(existing_lines)
            
            print(f"HF_TOKEN saved to {env_file_path}")
        except Exception as e:
            print(f"Error saving HF token: {e}")
    else:
        print("No token provided. You can set it later in the .env file.")

def setup_hf_token_legacy():
    """Legacy HF token setup for backward compatibility."""
    spectravision_root = Path(__file__).parent.parent
    env_path = spectravision_root / "VLMEvalKit" / ".env"
    
    # Check if .env exists and has HF_TOKEN
    if env_path.exists():
        with open(env_path, 'r') as f:
            content = f.read()
        if 'HF_TOKEN=' in content:
            print("HF_TOKEN already configured in VLMEvalKit .env")
            return
    
    print("\n" + "="*60)
    print("HUGGING FACE TOKEN SETUP (Legacy)")
    print("="*60)
    print("For MiniCPM-V-2_6 access, you need a Hugging Face token.")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'Read' permissions")
    print("3. Enter it below (or press Enter to skip)")
    
    token = input("Enter your HF token (or press Enter to skip): ").strip()
    
    if token:
        # Ensure .env file exists
        env_path.parent.mkdir(exist_ok=True)
        
        # Add token to VLMEvalKit .env file
        with open(env_path, 'w') as f:
            f.write(f'HF_TOKEN={token}\n')
        
        print(f"HF_TOKEN saved to {env_path}")
    else:
        print("Skipping HF_TOKEN setup. MiniCPM-V-2_6 may not work.")


def install_core_dependencies():
    """Install core dependencies with correct versions."""
    print("Installing core dependencies...")
    
    # Core packages with specific versions for compatibility
    core_packages = [
        "transformers==4.37.2",
        "tokenizers==0.15.1", 
        "timm==0.6.13",
        "torch==2.1.2",
        "torchvision==0.16.2",
        "numpy==1.26.4"
    ]
    
    for package in core_packages:
        print(f"Installing {package}...")
        run_command(f"pip install {package}")


def install_llava():
    """Install LLaVA from source."""
    print("Installing LLaVA from source...")
    run_command("pip uninstall -y llava", check=False)  # Remove if exists
    run_command("pip install git+https://github.com/haotian-liu/LLaVA.git")


def update_models_config():
    """Ensure models.yaml has correct configuration."""
    spectravision_root = Path(__file__).parent.parent
    models_config = spectravision_root / "configs" / "models.yaml"
    
    if models_config.exists():
        with open(models_config, 'r') as f:
            content = f.read()
        
        # Ensure DeepSeek-VL is not in the config
        if 'DeepSeek-VL' in content:
            print("Warning: DeepSeek-VL found in models.yaml.")
            print("DeepSeek-VL requires different transformer versions and may cause conflicts.")
            print("Consider removing it from models.yaml for better compatibility.")


def main():
    """Main setup function."""
    print("SpectraVision Dependency Setup")
    print("="*50)
    
    try:
        # 1. Install core dependencies
        install_core_dependencies()
        
        # 2. Setup VLMEvalKit
        setup_vlmevalkit()
        
        # 3. Install LLaVA
        install_llava()
        
        # 4. Setup environment configuration and API keys
        setup_environment_config()
        
        # 5. Check models config
        update_models_config()
        
        print("\n" + "="*50)
        print("SETUP COMPLETE!")
        print("="*50)
        print("Working models:")
        print("- InternVL2-2B")
        print("- MiniCPM-V-2_6 (requires HF token)")
        print("- LLaVA-1.5-7B")
        print("- CogVLM-7B")
        print("- InternVL2-8B")
        print("\nRemoved models:")
        print("- DeepSeek-VL-7B (transformer version conflict)")
        print("\nYou can now run:")
        print("python scripts/main.py --mode test")
        
    except Exception as e:
        print(f"\nSetup failed with error: {e}")
        print("Please check the error messages above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()