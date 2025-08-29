# SpectraBench-Vision 🔍

**Hardware-aware multimodal model evaluation system with automated Docker container management**

## 🐳 Automated Multi-Version Support

SpectraBench-Vision automatically manages **5 Docker containers** with different transformer versions, enabling seamless evaluation of **30 models** without dependency conflicts.

**How it works:**
1. **Model Selection**: Choose any model from the comprehensive list
2. **Automatic Container**: System selects the right transformer version container
3. **GPU Allocation**: Automatically configures GPU access for the container
4. **Seamless Execution**: Run evaluation without knowing technical details
5. **Result Collection**: All results saved in unified format for analysis

**User Experience:**
```bash
python scripts/main.py
# Select model: "Qwen2.5-VL-7B"
# ✅ System automatically uses transformers 4.49.0 container
# ✅ Downloads pre-built image from GitHub registry
# ✅ Configures GPU access and runs evaluation
# ✅ Results saved in standard format
```

**Supported Transformer Versions:**
- **4.33.0**: Legacy models - VisualGLM, Qwen series (9 models) - `spectravision-4.33`
- **4.37.2**: Current stable - LLaVA, InternVL series (8 models) - `spectravision-4.37`
- **4.43.0**: Mid-range modern models (2 models) - `spectravision-4.43` 
- **4.49.0**: Latest generation - SmolVLM, Qwen2.5-VL series (9 models) - `spectravision-4.49`
- **4.51.0**: Cutting-edge models (2 models) - `spectravision-4.51`

**Total Coverage: 30 models × 24 benchmarks = 720 evaluation combinations**

**AI Platform Team** at **KISTI Large-scale AI Research Center** ([Korea Institute of Science and Technology Information](https://www.kisti.re.kr/)) developed SpectraBench-Vision to provide intelligent model-benchmark combinations based on available GPU resources with comprehensive performance monitoring and analysis capabilities. 

The Large-scale AI Research Center was officially launched in March 2024, building upon KISTI's generative large language model 'KONI (KISTI Open Natural Intelligence)' unveiled in December 2023. As the **AI Platform Team is responsible for developing AI model and Agent service technologies**, SpectraBench-Vision represents their commitment to advancing AI infrastructure and creating sophisticated evaluation frameworks for the research community.


[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA Required](https://img.shields.io/badge/CUDA-Required-green.svg)](https://developer.nvidia.com/cuda-downloads)

## 🚀 Quick Start

### Ultra-Fast Docker Setup (Recommended) 🐳

**Complete setup in 2 minutes with integrated system container:**

```bash
# Option 1: All-in-One Container (Fastest)
docker run -it --gpus all -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectrabench-vision:latest

# Option 2: Clone and Run
git clone https://github.com/gwleee/SpectraBench-Vision.git
cd SpectraBench-Vision
cp .env.template .env  # Add your HF_TOKEN

# Run integrated container with .env file
docker run -it --gpus all -v $(pwd):/workspace \
  -v $(pwd)/outputs:/workspace/outputs \
  --env-file .env \
  ghcr.io/gwleee/spectrabench-vision:latest
```

### Advanced Multi-Version Docker Setup 🔧

**Option 1: Automatic Multi-Version CLI (Recommended) 🚀**
```bash
# Clone and setup
git clone https://github.com/gwleee/SpectraBench-Vision.git
cd SpectraBench-Vision
cp .env.template .env  # Add your HF_TOKEN

# Zero-configuration multi-version evaluation
python scripts/main.py --multi-version --models "SmolVLM-1.7B,Qwen2.5-VL-7B" --benchmarks "MMBench,TextVQA"
# ✅ System automatically selects transformer versions for each model
# ✅ Downloads appropriate Docker images from GitHub Container Registry  
# ✅ Handles GPU allocation and container orchestration
# ✅ Integrates results across all versions
```

**Option 2: Interactive Multi-Version Setup**
```bash
# Clone and setup with interactive mode
git clone https://github.com/gwleee/SpectraBench-Vision.git
cd SpectraBench-Vision
cp .env.template .env  # Add your HF_TOKEN

# Start evaluation (interactive selection)
python scripts/main.py
```

**Advanced Setup Benefits:**
- **Container Selection**: System picks the right transformer version for your model
- **Image Download**: Pre-built images pulled from GitHub Container Registry
- **GPU Configuration**: Automatic GPU allocation and Docker GPU access setup  
- **Dependency Isolation**: Each transformer version runs in isolated container
- **Result Integration**: All results saved in unified format for analysis

**Total Coverage**: 30 models, 5 transformer versions, 720 evaluations, zero conflicts!

📖 **For detailed usage instructions:**
- **Korean**: [Docker Usage Guide](DOCKER_USAGE_GUIDE.md) 
- **English**: [Docker Usage Guide (EN)](DOCKER_USAGE_GUIDE_EN.md)

### Alternative: Local Installation (Not Recommended)

```bash
# Clone and setup everything automatically
git clone https://github.com/gwleee/SpectraBench-Vision.git
cd SpectraBench-Vision
./quick_start.sh
```

The quick start script will:
- Check all prerequisites (Python, pip, git, NVIDIA GPU)
- Set up virtual environment (optional)
- Install all dependencies with correct versions
- Download and configure VLMEvalKit
- Clone Yi repository for Yi-VL support
- Install LLaVA from source
- **Set up personal API keys and environment configuration**
- Run quick availability tests
- Show usage examples

**Script Options:**
```bash
./quick_start.sh --help              # Show all options
./quick_start.sh --quick             # Skip tests, faster setup
./quick_start.sh --skip-prerequisites # Skip system checks
```

### Manual Local Setup (Advanced Users)

#### Step 1: Clone and Install Dependencies
```bash
# Clone the repository
git clone https://github.com/gwleee/SpectraBench-Vision.git
cd SpectraBench-Vision

# Install dependencies and setup
pip install -r requirements.txt
python scripts/setup_dependencies.py
```

#### Step 2: Configure Personal API Keys
```bash
# Copy environment template
cp .env.template .env

# Edit .env file and add your personal keys
nano .env  # Add your HF_TOKEN and other keys
```

**Required**: Get your Hugging Face token from https://huggingface.co/settings/tokens and add it to the .env file.

See the Environment Configuration section below for detailed setup.

#### Step 3: Run Tests
```bash
# Run availability tests
python scripts/main.py --mode test --test-time-limit 90
```

## 📋 Supported Models (30+ Models)

### Multi-Version Docker Architecture

| Transformer Version | Models | Memory Range | Container |
|---------------------|---------|--------------|-----------|
| **4.33.0** | VisualGLM-6B, Qwen2-VL-2B, Qwen-VL-Chat, mPLUG-Owl2-7B, Monkey-7B, InternLM-XComposer2-7B, IDEFICS-9B, InstructBLIP-13B, PandaGPT-13B | 8GB - 48GB | `spectravision-4.33` |
| **4.37.2** | InternVL2-2B, MiniCPM-V-2_6, LLaVA-1.5-7B, CogVLM-7B, InternVL2-8B, InternLM-XComposer2-VL, mPLUG-Owl2, LLaVA-1.5-13B | 8GB - 45GB | `spectravision-4.37` |
| **4.43.0** | Phi-3.5-Vision, Moondream2 | 8GB - 18GB | `spectravision-4.43` |
| **4.49.0** | SmolVLM-256M, SmolVLM-500M, SmolVLM-1.7B, Qwen2.5-VL-3B, Aria-3.9B, Qwen2.5-VL-7B, Pixtral-12B, Qwen2.5-VL-32B, Qwen2.5-VL-72B | 3GB - 300GB | `spectravision-4.49` |
| **4.51.0** | Phi-4-Vision, Llama-4-Scout-17B | 45GB - 200GB | `spectravision-4.51` |

**Key Features:**
- **Automatic Selection**: System picks the right container for each model
- **Pre-built Images**: Download from GitHub Container Registry (ghcr.io)
- **GPU Scaling**: From single GPU (8GB) to multi-GPU clusters (1TB+)
- **Zero Conflicts**: Each transformer version isolated in separate container

**Popular Models:**
- **SmolVLM Series**: 256M, 500M, 1.7B (Ultra-efficient, 3-8GB) - `4.49.0`
- **Qwen2.5-VL**: 3B, 7B, 32B, 72B (Latest generation, 12GB-300GB) - `4.49.0`
- **InternVL2**: 2B, 8B (High performance, 8GB-30GB) - `4.37.2`
- **LLaVA Series**: 7B, 13B (Stable and reliable, 25GB-45GB) - `4.37.2`

## 🎯 Supported Benchmarks (24 Benchmarks)

| Category | Benchmark | Purpose | Samples |
|----------|-----------|---------|---------|
| **Vision-Language** | MMBench | Multi-modal reasoning | 2,974 |
| | TextVQA | Reading text in images | 5,000 |
| | GQA | Compositional reasoning | 12,578 |
| | MMMU | College-level multi-discipline | 900 |
| **Document** | DocVQA | Document question answering | 5,349 |
| | ChartQA | Chart and graph understanding | 1,250 |
| | InfoVQA | Infographic understanding | 2,118 |
| **Scene/Object** | OCRBench | OCR capabilities | 1,000 |
| | AI2D | Science diagram understanding | 3,088 |
| **Scientific** | ScienceQA | Science question answering | 4,241 |
| **Spatial** | POPE | Object existence evaluation | 3,000 |
| **Instruction** | MME | Comprehensive evaluation | 2,374 |
| **Advanced** | HallusionBench | Hallucination detection | 346 |
| | MMStar | Multi-modal understanding | 1,500 |
| | RealWorldQA | Real-world reasoning | 700 |
| | NaturalBench | Natural imagery VQA | 3,000 |
| | VisOnlyQA | Visual perception | 2,000 |
| | VizWiz | Visual QA for visually impaired | 4,000 |
| | SEED | Spatial reasoning | 19,000 |
| | BLINK | Multimodal reasoning | 3,807 |
| **Korean** | K-MMBench | Multi-modal reasoning (Korean) | 3,000 |
| | K-SEED | Spatial reasoning (Korean) | 19,000 |
| | K-MMStar | Multi-modal understanding (Korean) | 1,500 |
| | Korean-OCR | Korean OCR capabilities | 150 |

**Total: 720 evaluation combinations** (30 models × 24 benchmarks)

## 🔧 Usage Examples

### Integrated Docker Container (Recommended)

```bash
# Start integrated container
docker run -it --gpus all -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectrabench-vision:latest

# Inside container - Interactive mode
python scripts/main.py
# ✅ All models and versions pre-installed
# ✅ Select from 30 models across 5 transformer versions

# Direct evaluation from host
docker run --gpus all -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectrabench-vision:latest \
  python scripts/main.py --models "Qwen2.5-VL-7B" --benchmarks "MMBench"

# Multi-model evaluation
docker run --gpus all -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectrabench-vision:latest \
  python scripts/main.py --models "SmolVLM-1.7B" "InternVL2-8B" --benchmarks "TextVQA"
```

💡 **Need more Docker options?** Check the Docker Usage Guide ([Korean](DOCKER_USAGE_GUIDE.md) / [English](DOCKER_USAGE_GUIDE_EN.md)) for advanced configuration, manual image management, and troubleshooting.

### Advanced Multi-Version Docker Usage

```bash
# Clone repository for multi-version setup
git clone https://github.com/gwleee/SpectraBench-Vision.git
cd SpectraBench-Vision

# Interactive mode with automatic container management
python scripts/main.py
# ✅ Automatically downloads individual transformer version containers
# ✅ Perfect isolation between transformer versions

# Test specific transformer version
python scripts/main.py --models "Qwen2.5-VL-7B" --benchmarks "MMBench"
# ✅ Uses transformers 4.49.0 container automatically

# Multi-version evaluation  
python scripts/main.py --models "SmolVLM-1.7B" "InternVL2-8B" --benchmarks "TextVQA"
# ✅ Uses different containers for each model automatically
```

### Advanced Configuration

```bash
# Automatic multi-version evaluation (recommended)
python scripts/main.py --multi-version --models "SmolVLM-1.7B,Qwen2.5-VL-7B" --benchmarks "MMBench,TextVQA"

# Production deployment with Docker Compose  
docker-compose -f docker/docker-compose.prod.yml --profile stable up -d

# Custom hardware detection
python scripts/main.py --hardware a100_single

# Extended time limit for complex benchmarks
python scripts/main.py --mode test --test-time-limit 180

# Non-interactive single-environment mode (legacy)
python scripts/main.py --models "LLaVA-1.5-7B" "CogVLM-7B" --benchmarks "TextVQA" "GQA" --mode full

# Build local images (for offline environments)
./scripts/build_local_images.sh
```

## 📁 Project Structure

```
SpectraBench-Vision/
├── .env.template              # Environment variables template
├── ENV_SETUP.md               # Personal API key setup guide
├── requirements.txt           # Python dependencies
├── quick_start.sh             # One-command setup script
│
├── configs/                   # Configuration files
│   ├── hardware.yaml          # GPU memory limits and detection
│   ├── models.yaml            # Model definitions by hardware tier (single-env mode)
│   ├── models_docker.yaml     # Docker multi-version model definitions
│   └── benchmarks.yaml        # Unified benchmark list (24 benchmarks)
│
├── scripts/                   # Main execution scripts
│   ├── main.py                # Main entry point
│   └── setup_dependencies.py  # Automated dependency setup
│
├── spectravision/             # Core evaluation system
│   ├── config.py              # Configuration management
│   ├── docker_orchestrator.py # Docker container management
│   ├── env_manager.py         # Environment variable management
│   ├── evaluator.py           # Sequential evaluation engine
│   ├── monitor.py             # Performance monitoring
│   ├── multi_version_evaluator.py # Multi-version orchestration
│   └── utils.py               # Logging and utility functions
│
├── analysis/                  # Performance analysis tools
│   ├── analyzer.py            # Performance analysis engine
│   └── visualizer.py          # Results visualization
│
├── VLMEvalKit/               # Auto-downloaded evaluation framework
├── outputs/                  # Results, logs, and reports
│
└── .env                      # Your personal environment config (not in git)
```

## 🛠️ Configuration

### Environment Configuration

**Personal API Keys**: Copy `.env.template` to `.env` and add your keys:
```bash
cp .env.template .env
nano .env  # Add your HF_TOKEN and other keys
```

**GPU Configuration**: Set which GPU(s) to use in `.env`:
```bash
CUDA_VISIBLE_DEVICES=0,1  # Use first two GPUs
```

See [ENV_SETUP.md](ENV_SETUP.md) for detailed environment configuration guide.

### Adding New Models

1. Check if the model is supported in VLMEvalKit
2. Add to appropriate hardware tier in `configs/models.yaml`:

```yaml
# Under appropriate hardware section (e.g., a6000_models)
- name: "New-Model-7B"
  vlm_id: "exact_vlmevalkit_id"  # Must match VLMEvalKit
  memory_gb: 28
```

### Adding New Benchmarks

1. Verify benchmark exists in VLMEvalKit
2. Add to the unified benchmark list in `configs/benchmarks.yaml`:

```yaml
benchmarks:
  - name: "NewBench"
    vlm_name: "NewBench_DEV"  # Must match VLMEvalKit dataset name
    samples: 100
    purpose: "Description of benchmark purpose"
```

## 🔍 Troubleshooting

### Common Issues

**"HF_TOKEN is required" or "You are trying to access a gated repo"**
- Configure your personal HF token in `.env` file
- Get your token from https://huggingface.co/settings/tokens
- Add `HF_TOKEN=your_token` to your `.env` file

**"No module named 'llava'"**
```bash
python scripts/setup_dependencies.py  # Re-run setup
```

**"TypeError: expected str, bytes or os.PathLike object, not NoneType" (Yi-VL)**
- Ensure Yi repository was cloned properly
- Run setup script again: `python scripts/setup_dependencies.py`
- Check if Yi_ROOT is set correctly in VLMEvalKit/vlmeval/config.py

**".env file not found" warning**
```bash
cp .env.template .env  # Create from template
nano .env              # Add your API keys
```

**GPU Memory Errors**
- Check available GPU memory with `nvidia-smi`
- Use smaller models or reduce batch size
- Enable memory cleanup with `--enable-cleanup`

### Getting Help

1. **Environment Setup**: Use `.env.template` to configure API keys
2. **Commands**: Run `python scripts/main.py --help` for command options
3. **Quick Start**: Use `./quick_start.sh` for automated setup

## 📊 Results

Results are saved to `outputs/` with timestamps:
- **Availability tests**: Quick compatibility checks
- **Full evaluations**: Complete benchmark results
- **Performance reports**: Resource utilization analysis
- **Logs**: Detailed execution logs

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
- Supports models from Hugging Face, LLaVA, and other frameworks
- Developed for hardware-aware multimodal evaluation

## 🏛️ Citation

```bibtex
@software{spectrabench-vision2025,
  title={SpectraBench-Vision},
  author={KISTI Large-scale AI Research Center},
  year={2025},
  url={https://github.com/gwleee/SpectraBench-Vision/},
  license={Apache-2.0},
}
```

---

*Developed with ❤️ by Gunwoo Lee from the AI Platform Team (Leader: Ryong Lee) at KISTI Large-scale AI Research Center (Director: Kyong-Ha Lee)*

*Supporting the Korean AI ecosystem with intelligent benchmarking tools through automated Docker multi-version evaluation*