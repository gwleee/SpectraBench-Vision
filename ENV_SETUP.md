# Environment Configuration Guide

This guide explains how to configure personal API keys and environment variables for SpectraVision.

## Quick Setup

1. **Copy the environment template:**
   ```bash
   cp .env.template .env
   ```

2. **Edit the .env file and add your personal API keys:**
   ```bash
   nano .env  # or use your preferred editor
   ```

3. **Set your Hugging Face token (required for gated models):**
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with "Read" permissions
   - Add it to your .env file:
   ```
   HF_TOKEN=your_actual_token_here
   ```

## Required API Keys

### Hugging Face Token (Required)
**Purpose**: Access gated models like MiniCPM-V-2.6

**How to get**:
1. Visit https://huggingface.co/settings/tokens
2. Click "New token"
3. Choose "Read" permissions
4. Copy the generated token

**Add to .env file**:
```
HF_TOKEN=hf_your_actual_token_here
```

### OpenAI API Key (Optional)
**Purpose**: Use OpenAI GPT models for evaluation

**How to get**:
1. Visit https://platform.openai.com/api-keys
2. Create a new API key
3. Copy the generated key

**Add to .env file**:
```
OPENAI_API_KEY=sk-your_actual_key_here
```

## Environment Variables Reference

### Essential Settings
```bash
# Required for gated models
HF_TOKEN=your_huggingface_token

# GPU configuration
CUDA_VISIBLE_DEVICES=0  # Use first GPU, or "0,1" for multiple GPUs

# Evaluation settings
VLMEVAL_BATCH_SIZE=1
LOG_LEVEL=INFO
```

### Advanced Settings
```bash
# Cache management
CACHE_CLEANUP_LEVEL=light  # light, moderate, aggressive
ENABLE_CACHE_CLEANUP=true

# Performance monitoring
ENABLE_MONITORING=false

# Output configuration
OUTPUT_DIR=outputs
MAX_LOG_SIZE=100
LOG_FILE_COUNT=5
```

### Offline Mode (for restricted networks)
```bash
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

## Security Best Practices

### ✅ DO:
- Keep your .env file private and never commit it to version control
- Use specific tokens with minimal required permissions
- Rotate your API keys periodically
- Use different tokens for different projects/environments

### ❌ DON'T:
- Share your .env file or API keys with others
- Commit .env files to GitHub or other version control systems
- Use overly broad token permissions
- Hardcode API keys in your source code

## Troubleshooting

### Common Issues

**1. "HF_TOKEN is required" error**
- Make sure you've set HF_TOKEN in your .env file
- Verify the token is correct and has "Read" permissions
- Check that the .env file is in the correct location

**2. "Permission denied" for gated models**
- Request access to the specific model on Hugging Face
- Wait for access approval (can take a few minutes to hours)
- Ensure your token has the necessary permissions

**3. ".env file not found" warning**
- Run `cp .env.template .env` to create the file
- Make sure you're in the SpectraVision project root directory

**4. Models downloading repeatedly**
- Check your HF_TOKEN is correctly set
- Verify internet connection for initial downloads
- Consider setting up HuggingFace cache directory

### Validation Commands

**Check if your environment is properly configured:**
```bash
python -c "from spectravision.env_manager import get_environment_manager; print(get_environment_manager().get_config_summary())"
```

**Test HF token access:**
```bash
python -c "from huggingface_hub import HfApi; api = HfApi(); print('Token valid:', api.whoami())"
```

## Multiple Environment Support

You can use different .env files for different purposes:

**Development environment:**
```bash
export SPECTRAVISION_ENV_FILE=.env.dev
python scripts/main.py
```

**Production environment:**
```bash
export SPECTRAVISION_ENV_FILE=.env.prod
python scripts/main.py
```

**Global user configuration:**
```bash
mkdir -p ~/.spectravision
cp .env.template ~/.spectravision/.env
# Edit ~/.spectravision/.env with your settings
```

## Environment File Locations

SpectraVision looks for .env files in this order:
1. `./env` (project root)
2. Current working directory
3. `~/.spectravision/.env` (user home directory)

## Integration with Setup Script

The setup script (`scripts/setup_dependencies.py`) will:
1. Create a .env file from the template if it doesn't exist
2. Prompt you to enter your HF token interactively
3. Validate your configuration
4. Apply environment settings automatically

## Support

If you encounter issues with environment configuration:

1. Check this guide and the troubleshooting section
2. Review the .env.template file for all available options
3. Check the [main README](README.md) for additional setup information
4. Open an issue on GitHub with your configuration details (without sharing actual API keys)