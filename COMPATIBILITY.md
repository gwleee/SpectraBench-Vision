# SpectraVision Model-Benchmark Compatibility Matrix

## 🎯 Overview

This document provides detailed compatibility information for all supported models and benchmarks in SpectraVision. Use this guide to plan your evaluations and avoid known issues.

## 📊 Compatibility Legend

- ✅ **Fully Compatible**: Tested and verified to work correctly
- ⚠️ **Partial Support**: Works with known limitations or requirements
- ❌ **Not Compatible**: Known issues or incompatibilities
- 🔄 **Testing**: Currently being validated
- 💾 **High Memory**: Requires >32GB GPU memory
- 🔥 **Multi-GPU**: Requires multiple GPUs (GPU_COUNT>=2)

## 🤖 Model Compatibility Matrix

### Transformer 4.49.0 Models (Latest - Recommended)

| Model | MMBench | TextVQA | VQAv2 | GQA | SEED | OCRBench | MathVista | Status |
|-------|---------|---------|-------|-----|------|----------|-----------|--------|
| **SmolVLM-256M** | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️¹ | ✅ | Stable |
| **SmolVLM-500M** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Stable |
| **SmolVLM-1.7B** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Stable |
| **Qwen2.5-VL-3B** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Stable |
| **Qwen2.5-VL-7B** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Stable |
| **Aria-3.9B** | ✅ | ✅ | ✅ | ⚠️² | ✅ | ✅ | ⚠️² | Testing |
| **Pixtral-12B** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Stable |
| **Qwen2.5-VL-32B** | 💾 | 💾 | 💾 | 💾 | 💾 | 💾 | 💾 | High Memory |
| **Qwen2.5-VL-72B** | 🔥 | 🔥 | 🔥 | 🔥 | 🔥 | 🔥 | 🔥 | Multi-GPU Required |

### Transformer 4.37.2 Models (Stable)

| Model | MMBench | TextVQA | VQAv2 | GQA | SEED | OCRBench | MathVista | Status |
|-------|---------|---------|-------|-----|------|----------|-----------|--------|
| **InternVL2-2B** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Stable |
| **InternVL2-8B** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Stable |
| **LLaVA-1.5-7B** | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️³ | ✅ | Stable |
| **LLaVA-1.5-13B** | 💾 | 💾 | 💾 | 💾 | 💾 | 💾 | 💾 | High Memory |
| **CogVLM-7B** | ✅ | ✅ | ✅ | ✅ | ⚠️⁴ | ✅ | ✅ | Stable |
| **MiniCPM-V-2_6** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Stable |
| **mPLUG-Owl2** | ✅ | ✅ | ✅ | ⚠️⁵ | ✅ | ✅ | ⚠️⁵ | Partial |
| **XComposer2** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Stable |

### Transformer 4.51.0 Models (Cutting-Edge)

| Model | MMBench | TextVQA | VQAv2 | GQA | SEED | OCRBench | MathVista | Status |
|-------|---------|---------|-------|-----|------|----------|-----------|--------|
| **Phi-4-Vision** | 💾 | 💾 | 💾 | 💾 | ⚠️⁶ | 💾 | 💾 | High Memory |
| **Llama-4-Scout** | 🔥 | 🔥 | 🔥 | 🔥 | 🔥 | 🔥 | 🔥 | Multi-GPU Required |

### Transformer 4.33.0 Models (Legacy)

| Model | MMBench | TextVQA | VQAv2 | GQA | SEED | OCRBench | MathVista | Status |
|-------|---------|---------|-------|-----|------|----------|-----------|--------|
| **VisualGLM-6B** | ✅ | ✅ | ✅ | ⚠️⁷ | ✅ | ⚠️⁷ | ⚠️⁷ | Legacy |
| **Qwen2-VL-2B** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Stable |
| **PandaGPT-13B** | 💾 | 💾 | 💾 | 💾 | 💾 | 💾 | 💾 | High Memory |

### Transformer 4.43.0 Models (Modern)

| Model | MMBench | TextVQA | VQAv2 | GQA | SEED | OCRBench | MathVista | Status |
|-------|---------|---------|-------|-----|------|----------|-----------|--------|
| **Phi-3.5-Vision** | ✅ | ✅ | ✅ | ✅ | ⚠️⁸ | ✅ | ✅ | Stable |
| **Moondream2** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️⁹ | Stable |

## 🏆 Recommended Model-Benchmark Combinations

### For Quick Testing (< 30 minutes)
```bash
# Ultra-fast evaluation
python scripts/main.py --models "SmolVLM-256M" --benchmarks "SEED"

# Balanced speed/quality  
python scripts/main.py --models "SmolVLM-1.7B" --benchmarks "MMBench"
```

### For Production Evaluation (1-3 hours)
```bash
# High-quality results
python scripts/main.py --models "Qwen2.5-VL-7B,InternVL2-8B" --benchmarks "MMBench,TextVQA,VQAv2"

# Comprehensive assessment
python scripts/main.py --models "SmolVLM-1.7B,Qwen2.5-VL-7B,InternVL2-8B" --benchmarks "MMBench,TextVQA,VQAv2,GQA,SEED"
```

### For Research/Benchmarking (6+ hours)
```bash
# Full evaluation suite
python scripts/main.py --models "SmolVLM-1.7B,Qwen2.5-VL-7B,InternVL2-8B,LLaVA-1.5-7B" --benchmarks "MMBench,TextVQA,VQAv2,GQA,SEED,OCRBench,MathVista"
```

## ⚠️ Known Issues and Limitations

### Footnotes:

¹ **SmolVLM-256M + OCRBench**: May struggle with complex text recognition due to model size limitations

² **Aria-3.9B**: Occasional timeout issues on GQA and MathVista with large images

³ **LLaVA-1.5-7B + OCRBench**: Suboptimal performance on dense text images

⁴ **CogVLM-7B + SEED**: Some image-text alignment issues with specific SEED prompts

⁵ **mPLUG-Owl2**: Inconsistent performance on GQA and MathVista due to prompt format sensitivity

⁶ **Phi-4-Vision + SEED**: Beta model may have stability issues

⁷ **VisualGLM-6B**: Legacy model with limited support for newer benchmark formats

⁸ **Phi-3.5-Vision + SEED**: Occasional memory spikes during evaluation

⁹ **Moondream2 + MathVista**: Mathematical reasoning limitations

## 🔧 Hardware Requirements by Model Category

### Lightweight Models (≤2B parameters)
- **Models**: SmolVLM series, InternVL2-2B, Qwen2-VL-2B
- **GPU**: RTX 4080 16GB or better
- **Memory**: 8-16GB GPU memory
- **Eval Time**: 15-60 minutes per benchmark

### Standard Models (3-8B parameters)  
- **Models**: Qwen2.5-VL-7B, InternVL2-8B, LLaVA-1.5-7B
- **GPU**: RTX 4090 24GB or A6000 48GB
- **Memory**: 20-40GB GPU memory
- **Eval Time**: 30-120 minutes per benchmark

### Large Models (13-32B parameters)
- **Models**: LLaVA-1.5-13B, Qwen2.5-VL-32B, PandaGPT-13B
- **GPU**: A100 80GB or better
- **Memory**: 45-140GB GPU memory
- **Eval Time**: 60-300 minutes per benchmark

### Ultra-Large Models (>32B parameters)
- **Models**: Qwen2.5-VL-72B, Llama-4-Scout
- **GPU**: 2+ A100 80GB or H100
- **Memory**: 200GB+ GPU memory (multi-GPU)
- **Eval Time**: 120-600 minutes per benchmark

## 🚀 Performance Optimization Tips

### Speed Optimization
1. **Use SmolVLM models** for rapid prototyping
2. **Enable model caching** with persistent volumes
3. **Reduce batch size** if memory constrained
4. **Use SEED benchmark** for quick model comparison

### Quality Optimization  
1. **Use Qwen2.5-VL-7B or InternVL2-8B** for best balance
2. **Run multiple benchmarks** for comprehensive assessment
3. **Use larger models** for critical evaluations
4. **Validate results** across multiple runs

### Resource Optimization
1. **Start with lightweight models** to test setup
2. **Scale up gradually** based on requirements  
3. **Use multi-GPU** for large models only when necessary
4. **Monitor GPU utilization** to optimize batch sizes

## 📈 Benchmark Characteristics

| Benchmark | Type | Avg Questions | Eval Time | GPU Memory | Best Models |
|-----------|------|---------------|-----------|------------|-------------|
| **MMBench** | Multi-modal Reasoning | 2,974 | 20-60 min | Moderate | Qwen2.5-VL, InternVL2 |
| **TextVQA** | Text Reading | 5,000 | 30-90 min | Moderate | SmolVLM, Qwen2.5-VL |
| **VQAv2** | Visual QA | 40,504 | 60-180 min | High | InternVL2, LLaVA-1.5 |
| **GQA** | Scene Graph QA | 12,578 | 45-120 min | High | Qwen2.5-VL, CogVLM |
| **SEED** | Comprehensive | 19,242 | 35-90 min | Moderate | SmolVLM, InternVL2 |
| **OCRBench** | OCR Tasks | 1,000 | 15-45 min | Low | Qwen2.5-VL, Pixtral |
| **MathVista** | Math Reasoning | 6,141 | 30-75 min | Moderate | Qwen2.5-VL, InternVL2 |

## 🔄 Update Schedule

This compatibility matrix is updated regularly based on:
- Community testing and feedback
- New model releases and updates
- Benchmark format changes
- Performance optimizations

**Last Updated**: August 2025  
**Next Review**: September 2025

For the most current information, check our [GitHub repository](https://github.com/gwleee/SpectraBench-Vision) and [issue tracker](https://github.com/gwleee/SpectraBench-Vision/issues).