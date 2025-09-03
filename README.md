# SpectraBench-Vision 🔍

> **English**: [README.en.md](README.en.md) | **한국어**: README.md

**한 번의 다운로드로 30개 VLM 모델을 평가하는 완전 통합 시스템**

---

## 📋 개요

SpectraBench-Vision은 **KISTI 초거대AI연구센터**에서 개발한 **Docker-in-Docker 기반 통합 VLM 평가 시스템**입니다. 이 시스템은 연구자들이 복잡한 환경 설정이나 모델이 사용하는 transfomer 버전을 신경 쓸 필요없이 30개의 최신 Vision-Language 모델을 쉽게 평가할 수 있도록 설계되었습니다.

## ✨ 주요 특징

- 🚀 **1분 설치**: 단일 Docker 명령으로 전체 시스템 즉시 사용
- 🤖 **완전 자동화**: 모델별 최적 transformer 버전 컨테이너 자동 선택
- 📦 **완전한 재현성**: 어디서든 동일한 평가 환경 보장  
- 🎯 **30개 모델 지원**: SmolVLM부터 Qwen2.5-VL-72B까지 모든 최신 모델
- 📊 **24개 벤치마크**: MMBench, TextVQA, DocVQA 등 표준 벤치마크 모두 지원
- 🔧 **GPU 최적화**: 단일 GPU부터 다중 GPU 클러스터까지 자동 최적화

## 🚀 사용법

> 📖 [Docker 사용 가이드](DOCKER_USAGE_GUIDE.md) | [Docker Usage Guide (EN)](DOCKER_USAGE_GUIDE_EN.md)

## 🏛️ 개발 배경

**KISTI 초거대AI연구센터 AI플랫폼팀**에서 개발한 SpectraBench-Vision은 GPU 자원에 따른 모델-벤치마크 조합 제공과 종합적인 성능 모니터링 및 분석 기능을 제공합니다.

초거대AI연구센터는 2024년 3월 공식 출범하였으며, 2023년 12월 공개된 KISTI의 생성형 대규모 언어모델 'KONI(KISTI Open Natural Intelligence)'를 기반으로 합니다. **AI 플랫폼팀은 AI 모델 및 에이전트 서비스 기술 개발을 담당**하며, SpectraBench-Vision은 연구 커뮤니티를 위한 정교한 평가 프레임워크 구축에 대한 연구 센터의 노력을 보여줍니다.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA Required](https://img.shields.io/badge/CUDA-Required-green.svg)](https://developer.nvidia.com/cuda-downloads)

## 📊 지원 모델 및 벤치마크

### 🤖 지원 모델 (30개)

| Transformer 버전 | 주요 모델 | 메모리 범위 | 용도 |
|-----------------|----------|------------|-----|
| **4.33.0** | Qwen-VL, VisualGLM | 8GB - 48GB | 레거시 모델 |
| **4.37.2** | InternVL2, LLaVA | 8GB - 45GB | 안정적인 모델 |
| **4.43.0** | Phi-3.5-Vision | 8GB - 18GB | 중급 모델 |
| **4.49.0** | SmolVLM, Qwen2.5-VL | 3GB - 300GB | 최신 모델 |
| **4.51.0** | Phi-4-Vision | 45GB - 200GB | 실험적 모델 |

**인기 모델:**
- **SmolVLM**: 256M, 500M, 1.7B (초경량, 3-8GB 메모리)
- **Qwen2.5-VL**: 3B, 7B, 32B, 72B (최신세대, 12GB-300GB 메모리)  
- **InternVL2**: 2B, 8B (고성능, 8GB-30GB 메모리)
- **LLaVA**: 7B, 13B (안정적, 25GB-45GB 메모리)

### 📊 지원 벤치마크 (24개)

| 분야 | 주요 벤치마크 | 설명 |
|------|-------------|------|
| **기본 VQA** | MMBench, TextVQA, GQA | 멀티모달 추론, 텍스트 이해, 구성적 추론 |
| **문서 이해** | DocVQA, ChartQA, InfoVQA | 문서 질의응답, 차트/그래프 이해 |
| **과학/전문** | ScienceQA, AI2D, MMMU | 과학 문제해결, 다이어그램 이해 |
| **고급 평가** | HallusionBench, MMStar, RealWorldQA | 환각 감지, 실세계 추론 |
| **한국어** | K-MMBench, K-SEED, Korean-OCR | 한국어 멀티모달 평가 |

**총 평가 조합: 720개** (30개 모델 × 24개 벤치마크)

## 🔧 추가 사용법

### 🐳 Docker 통합 시스템 사용법 (권장)

> 🏗️ **완전한 Docker 아키텍처**: Base Image → 개별 Transformer 버전 컨테이너 → 통합 시스템 → 프로덕션 환경
> 
> 📖 **자세한 내용**: [Docker 사용 가이드](DOCKER_USAGE_GUIDE.md) | [Docker Usage Guide (EN)](DOCKER_USAGE_GUIDE_EN.md)

#### 대화형 모드
```bash
# Docker 컨테이너 실행하고 대화형 모드로 진입
docker run -it --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectrabench-vision:latest \
  python3 scripts/docker_main.py --mode interactive
```

#### 다중 모델 비교 평가
```bash
# 여러 모델로 성능 비교
docker run --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectrabench-vision:latest \
  python3 scripts/docker_main.py --mode batch \
  --models "SmolVLM" "InternVL2-8B" "Qwen2.5-VL-3B" \
  --benchmarks "MMBench" "TextVQA"
```

### 메모리/GPU 최적화

```bash
# 최신 모델만 사용 (메모리 절약)
docker run -it --gpus all -e PULL_IMAGES=minimal \
  -v /var/run/docker.sock:/var/run/docker.sock \
  ghcr.io/gwleee/spectrabench-vision:latest

# 다중 GPU 활용
docker run -it --gpus all -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  ghcr.io/gwleee/spectrabench-vision:latest
```

### 시스템 테스트
```bash
# 설치 확인 및 GPU 테스트  
docker run --rm --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  ghcr.io/gwleee/spectrabench-vision:latest \
  python3 scripts/docker_main.py --mode test
```

### 💻 로컬 설치 사용법 (개발/연구용)

```bash
# 로컬 환경에서 직접 실행 (현재 설치된 transformer 버전으로 제한)
python scripts/main.py --models "InternVL2-2B" --benchmarks "MMBench"
python scripts/main.py --mode interactive  # 실행 모드 선택 메뉴
python scripts/main.py --mode test         # 빠른 호환성 테스트

# 특정 모드로 직접 실행
python scripts/main.py --mode single      # 단일 환경 평가 (transformers 4.37.2)
python scripts/main.py --mode docker      # Docker 다중 버전 평가 (Docker 지원시)
```

### 개별 컨테이너 직접 사용 (고급)
```bash
# 특정 transformer 버전 직접 사용 (개발/디버깅용)
docker run --gpus all -it ghcr.io/gwleee/spectravision-4.49:latest

# VLMEvalKit 직접 사용
cd /workspace/VLMEvalKit
python run.py --model SmolVLM-Instruct --data MMBench_DEV_EN
```

## 📁 프로젝트 구조

```
SpectraBench-Vision/
├── .env.template              # 환경 변수 템플릿
├── LICENSE                    # 라이선스 파일
├── README.md                  # 프로젝트 문서 (한국어)
├── README.en.md               # 프로젝트 문서 (영어)
├── DOCKER_USAGE_GUIDE.md      # Docker 사용 가이드 (한국어)
├── DOCKER_USAGE_GUIDE_EN.md   # Docker 사용 가이드 (영어)
├── requirements.txt           # Python 의존성
├── quick_start.sh             # 원클릭 설정 스크립트
│
├── .github/                   # GitHub Actions 워크플로우
│   └── workflows/
│       └── build-and-push-docker.yml  # Docker 자동 빌드 및 푸시
│
├── configs/                   # 설정 파일들
│   ├── hardware.yaml          # GPU 메모리 제한 및 감지
│   ├── models.yaml            # 모델 정의 (transformer 버전별)
│   └── benchmarks.yaml        # 통합 벤치마크 목록 (24개)
│
├── scripts/                   # 실행 및 빌드 스크립트
│   ├── main.py                # 로컬 평가 메인 진입점
│   ├── setup_dependencies.py  # 자동화된 의존성 설정
│   ├── build_local_images.sh  # 로컬 Docker 이미지 빌드
│   ├── build_production_images.sh # 프로덕션 이미지 빌드
│   ├── build_and_push_images.sh   # 이미지 빌드 및 푸시
│   └── push_to_registry.sh    # Docker 레지스트리 푸시
│
├── docker/                    # Docker 인프라
│   ├── docker-compose.yml     # 개발용 컨테이너 오케스트레이션
│   ├── docker-compose.prod.yml # 프로덕션용 컨테이너 오케스트레이션
│   ├── base/
│   │   └── Dockerfile         # 기본 베이스 이미지
│   ├── integrated/            # 통합 시스템 Docker-in-Docker
│   │   ├── Dockerfile         # 통합 시스템 이미지
│   │   ├── docker_main.py     # 통합 시스템 메인 스크립트
│   │   └── start_spectrabench.sh # 통합 시스템 시작 스크립트
│   ├── transformers-4.33/     # Transformer 4.33.0 컨테이너
│   │   └── Dockerfile
│   ├── transformers-4.37/     # Transformer 4.37.2 컨테이너
│   │   └── Dockerfile
│   ├── transformers-4.43/     # Transformer 4.43.0 컨테이너
│   │   └── Dockerfile
│   ├── transformers-4.49/     # Transformer 4.49.0 컨테이너
│   │   └── Dockerfile
│   └── transformers-4.51/     # Transformer 4.51.0 컨테이너
│       └── Dockerfile
│
├── spectravision/             # 핵심 평가 시스템
│   ├── config.py              # 설정 관리 및 하드웨어 감지
│   ├── docker_orchestrator.py # Docker 컨테이너 자동 관리
│   ├── env_manager.py         # 환경 변수 관리
│   ├── evaluator.py           # 순차 평가 엔진
│   ├── monitor.py             # 성능 모니터링 및 리소스 추적
│   ├── multi_version_evaluator.py # 다중 버전 오케스트레이션
│   └── utils.py               # 로깅 및 유틸리티 함수
│
├── analysis/                  # 성능 분석 도구
│   ├── analyzer.py            # 성능 분석 엔진
│   └── visualizer.py          # 결과 시각화
│
├── VLMEvalKit/               # VLMEvalKit 서브모듈 (자동 다운로드)
└── outputs/                  # 결과, 로그 및 리포트 (로컬 생성)
```

## 🛠️ 설정

### 환경 설정

**개인 API 키**: `.env.template`을 `.env`로 복사하고 키를 추가하세요:
```bash
cp .env.template .env
nano .env  # HF_TOKEN 등 키 추가
```

**GPU 설정**: `.env`에서 사용할 GPU 설정:
```bash
CUDA_VISIBLE_DEVICES=0,1  # 첫 번째와 두 번째 GPU 사용
```

자세한 환경 설정은 `.env.template` 파일을 참고하세요.

### 새 모델 추가

1. VLMEvalKit에서 모델이 지원되는지 확인
2. `configs/models.yaml`의 적절한 하드웨어 계층에 추가:

```yaml
# 적절한 하드웨어 섹션에 (예: a6000_models)
- name: "New-Model-7B"
  vlm_id: "exact_vlmevalkit_id"  # VLMEvalKit과 정확히 일치해야 함
  memory_gb: 28
```

### 새 벤치마크 추가

1. VLMEvalKit에 벤치마크가 존재하는지 확인
2. `configs/benchmarks.yaml`의 통합 벤치마크 목록에 추가:

```yaml
benchmarks:
  - name: "NewBench"
    vlm_name: "NewBench_DEV"  # VLMEvalKit 데이터셋 이름과 일치해야 함
    samples: 100
    purpose: "벤치마크 목적 설명"
```

## 🔍 문제 해결

### 일반적인 문제

**"HF_TOKEN is required" 또는 "You are trying to access a gated repo"**
- `.env` 파일에 개인 HF 토큰 설정
- https://huggingface.co/settings/tokens 에서 토큰 발급
- `.env` 파일에 `HF_TOKEN=your_token` 추가

**"No module named 'llava'"**
```bash
python scripts/setup_dependencies.py  # 설정 재실행
```

**".env file not found" 경고**
```bash
cp .env.template .env  # 템플릿에서 생성
nano .env              # API 키 추가
```

**GPU 메모리 오류**
- `nvidia-smi`로 사용 가능한 GPU 메모리 확인
- 더 작은 모델 사용 또는 배치 크기 감소
- `--enable-cleanup`으로 메모리 정리 활성화

### 도움말

1. **환경 설정**: `.env.template`을 사용하여 API 키 구성
2. **명령어**: `python scripts/main.py --help`로 명령어 옵션 확인
3. **빠른 시작**: `./quick_start.sh`로 자동 설정 사용

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