# 🐳 SpectraBench-Vision Docker 사용 가이드

## 🎯 시스템 개요

SpectraBench-Vision은 **재현성과 편의성**을 위해 설계된 Docker 기반 VLM 평가 시스템입니다:

- **다중 컨테이너 시스템**: 각 transformer 버전별 독립 컨테이너
- **DockerOrchestrator**: 지능형 컨테이너 자동 관리 시스템
- **Docker-in-Docker 지원**: 완전히 격리된 환경에서 다중 버전 지원

## 📦 시스템 아키텍처

### 🔧 완전한 Docker 아키텍처

SpectraBench-Vision은 계층화된 Docker 아키텍처로 구성되어 있습니다:

#### 🎯 1. Base Image (공통 기반)
- **`spectravision-base:latest`** - 모든 컨테이너의 기본 이미지
- CUDA 11.8, Python 3, VLMEvalKit 기본 설치
- 공통 종속성 및 SpectraVision 코드베이스 포함

#### 🚀 2. 통합 시스템 (권장)
- **`spectrabench-vision:latest`** - Docker-in-Docker 통합 시스템
- DockerOrchestrator 포함하여 개별 컨테이너 자동 관리
- 사용자 친화적 단일 진입점

#### 🔧 3. 개별 Transformer 버전 컨테이너들

| 이미지 | Transformers | 주요 모델 | 용도 |
|--------|-------------|----------|------|
| **spectravision-4.33** | 4.33.0 | Qwen-VL, VisualGLM | 레거시 모델 |
| **spectravision-4.37** | 4.37.2 | InternVL2, LLaVA | 안정 버전 |
| **spectravision-4.43** | 4.43.0 | Phi-3.5-Vision | 중급 모델 |
| **spectravision-4.49** | 4.49.0 | SmolVLM, Qwen2.5-VL | 최신 모델 |
| **spectravision-4.51** | 4.51.0 | Phi-4-Vision | 실험용 |

#### 🏭 4. 프로덕션 환경
- **`docker-compose.prod.yml`** - 프로덕션 환경용 오케스트레이션
- GitHub Container Registry 이미지 사용 (`ghcr.io/gwleee/spectravision-*`)
- 스케일링 지원 및 고급 리소스 관리

## 📥 이미지 다운로드

### 사용 가능한 이미지들

| 이미지 | Transformers | 주요 모델 | 사용 사례 | 다운로드 |
|-------|-------------|----------|----------|----------|
| **spectravision-base** | Base | 공통 기반 | 베이스 이미지 | `docker build -t spectravision-base:latest -f docker/base/Dockerfile .` |
| **spectrabench-vision** | Latest | 통합 시스템 | 권장 사용 | `docker build -t spectrabench-vision:latest -f docker/integrated/Dockerfile .` |
| **spectravision-4.33** | 4.33.0 | Qwen-VL, VisualGLM | 레거시 모델 | `docker pull ghcr.io/gwleee/spectravision-4.33:latest` |
| **spectravision-4.37** | 4.37.2 | InternVL2, LLaVA | 안정 버전 | `docker pull ghcr.io/gwleee/spectravision-4.37:latest` |
| **spectravision-4.43** | 4.43.0 | Phi-3.5-Vision | 중급 모델 | `docker pull ghcr.io/gwleee/spectravision-4.43:latest` |
| **spectravision-4.49** | 4.49.0 | SmolVLM, Qwen2.5-VL | 최신 모델 | `docker pull ghcr.io/gwleee/spectravision-4.49:latest` |
| **spectravision-4.51** | 4.51.0 | Phi-4-Vision | 실험용 | `docker pull ghcr.io/gwleee/spectravision-4.51:latest` |

### 빠른 다운로드 방법 (예시)

```bash
# 통합 시스템 빌드 (권장)
docker build -t spectrabench-vision:latest -f docker/integrated/Dockerfile .

# 최신 모델 (SmolVLM, Qwen2.5-VL)
docker pull ghcr.io/gwleee/spectravision-4.49:latest

# 안정 버전 (InternVL2, LLaVA)
docker pull ghcr.io/gwleee/spectravision-4.37:latest

# 베이스 이미지부터 전체 빌드
docker build -t spectravision-base:latest -f docker/base/Dockerfile .

# 모든 이미지 다운로드/빌드
./scripts/build_local_images.sh
```

## 🚀 빠른 시작하기

### 🎯 방법 1: DockerOrchestrator 사용 (권장)

DockerOrchestrator는 **지능형 컨테이너 자동 관리 시스템**으로, 모델별로 최적의 transformer 버전 컨테이너를 자동으로 선택하고 관리합니다.

#### 📋 사용 가능한 모드들
- **`--mode interactive`**: 대화형 모델/벤치마크 선택
- **`--mode batch`**: 명령줄로 직접 지정하여 실행
- **`--mode test`**: 모든 컨테이너 상태 및 연결 테스트

#### 🎮 대화형 모드 (추천)
```bash
# 1. 대화형 모드로 시작 - 모든 옵션을 GUI로 선택
python3 docker/integrated/docker_main.py --mode interactive
```

**대화형 모드에서 할 수 있는 것들:**
- 📦 컨테이너별로 그룹화된 모델 목록 확인
- 🎯 개별 모델 선택 또는 전체/컨테이너별 일괄 선택
- 📊 24개 벤치마크 중 원하는 것만 선택 (all, korean, basic 옵션)
- 🎮 다중 GPU 설정 및 자동 배분

#### ⚡ 배치 모드 (고급 사용자)
```bash
# 2. 배치 모드 - 명령줄로 직접 지정
python3 docker/integrated/docker_main.py --mode batch \
  --models "SmolVLM-256M" "InternVL2-2B" "Qwen2.5-VL-3B" \
  --benchmarks "MMBench" "TextVQA" "DocVQA" \
  --gpu-ids 0

# 3. 다중 GPU로 대용량 모델 평가
python3 docker/integrated/docker_main.py --mode batch \
  --models "Qwen2.5-VL-32B" "Qwen2.5-VL-72B" \
  --benchmarks "MMBench" "MMMU" \
  --gpu-ids 0 1 2 3
```

#### 🧪 시스템 테스트
```bash
# 4. 전체 시스템 상태 확인
python3 docker/integrated/docker_main.py --mode test
```

**테스트 내용:**
- 🐳 Docker 연결 상태 확인
- 🎮 GPU 감지 및 설정 확인
- 📦 모든 컨테이너 이미지 가용성 테스트
- 🚀 컨테이너 시작/중지 기능 테스트
- ⚙️ 각 컨테이너 내부 기본 동작 확인

### 방법 2: 개별 컨테이너 직접 사용
```bash
# 최신 모델 컨테이너 실행
docker run --gpus all -it \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectravision-4.49:latest

# 컨테이너 내부에서 평가 실행
cd /workspace/VLMEvalKit
python run.py --model SmolVLM-Instruct --data MMBench_DEV_EN --mode all
```

### 방법 3: Docker Compose 사용 (개발자용)
```bash
# 특정 버전 컨테이너 시작
docker-compose -f docker/docker-compose.yml up -d transformers-4-49

# 컨테이너에 접속하여 작업
docker exec -it spectravision-transformers-4-49 /bin/bash
```

## ⚡ 다중 GPU 사용법

```bash
# DockerOrchestrator를 통한 다중 GPU 사용
python3 docker/integrated/docker_main.py --mode batch \
  --models "Qwen2.5-VL-32B" "Qwen2.5-VL-72B" \
  --benchmarks "MMBench" "TextVQA" \
  --gpu-ids 0 1 2 3

# 개별 컨테이너에서 특정 GPU 지정
docker run --gpus "device=0,1" -it \
  -e NVIDIA_VISIBLE_DEVICES=0,1 \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectravision-4.49:latest
```

## 🛠️ 고급 설정 옵션

### 🏭 프로덕션 Docker Compose 사용
```bash
# 프로덕션 환경용 - 모든 컨테이너 시작
docker-compose -f docker/docker-compose.prod.yml --profile all up -d

# 특정 프로필만 시작 (최신 모델만)
docker-compose -f docker/docker-compose.prod.yml --profile latest up -d

# 멀티 GPU 설정으로 시작
GPU_COUNT=4 NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  docker-compose -f docker/docker-compose.prod.yml --profile all up -d

# 스케일링 지원 (transformers-4-37을 3개 인스턴스로)
docker-compose -f docker/docker-compose.prod.yml --profile stable up -d --scale transformers-4-37=3

# 모든 프로덕션 이미지 다운로드
docker-compose -f docker/docker-compose.prod.yml pull
```

### 🔧 개발용 Docker Compose 사용
```bash
# 개발환경용 - 특정 버전만 시작
docker-compose -f docker/docker-compose.yml up -d transformers-4-49

# 개별 컨테이너 직접 사용
docker run --gpus all -it ghcr.io/gwleee/spectravision-4.49:latest
```

### 🛠️ 베이스 이미지부터 빌드
```bash
# 베이스 이미지 빌드
docker build -t spectravision-base:latest -f docker/base/Dockerfile .

# 특정 transformer 버전 빌드 (베이스 이미지 필요)
docker build -t spectravision-4.49:latest -f docker/transformers-4.49/Dockerfile .

# 통합 시스템 빌드
docker build -t spectrabench-vision:latest -f docker/integrated/Dockerfile .
```

## 🎯 DockerOrchestrator 핵심 기능

### 🧠 지능형 자동 관리
DockerOrchestrator가 자동으로 처리하는 것들:

#### 📦 **컨테이너 자동 매핑**
```
SmolVLM-256M → spectravision-4.49 컨테이너 자동 선택
InternVL2-2B → spectravision-4.37 컨테이너 자동 선택  
Qwen-VL-Chat → spectravision-4.33 컨테이너 자동 선택
```

#### 🚀 **이미지 자동 관리**
- 필요한 이미지가 없으면 **자동으로 Registry에서 pull**
- 로컬 이미지 우선 사용으로 **네트워크 대역폭 절약**
- 이미지 태그 자동 매핑 (`spectravision-4.49:latest`)

#### ⚙️ **컨테이너 생명주기 관리**
- 평가 시작 전 **자동으로 컨테이너 시작**
- GPU 할당 및 환경변수 자동 설정
- 평가 완료 후 **리소스 정리 옵션**

#### 🎮 **GPU 자동 최적화**
- GPU 수 자동 감지 (`nvidia-smi` 기반)
- 다중 GPU 환경에서 **자동 로드 밸런싱**
- 컨테이너별 GPU 할당 최적화

### 📊 실행 예시 흐름

```bash
# 명령 실행
python3 docker/integrated/docker_main.py --mode batch \
  --models "SmolVLM-256M" "InternVL2-2B" --benchmarks "MMBench"

# DockerOrchestrator 자동 처리 과정:
# 1. 모델 → 컨테이너 매핑 확인
#    SmolVLM-256M → transformers_4_49
#    InternVL2-2B → transformers_4_37
#
# 2. 필요한 이미지 확인 및 다운로드
#    spectravision-4.49:latest ✓ 로컬에 존재
#    spectravision-4.37:latest ✗ Registry에서 pull 시작
#
# 3. 컨테이너 시작 및 평가 실행
#    transformers_4_49 시작 → SmolVLM-256M 평가
#    transformers_4_37 시작 → InternVL2-2B 평가
#
# 4. 결과 수집 및 정리
#    평가 완료 → 컨테이너 중지 → 결과 보고서 생성
```

## 🔧 환경 설정

### .env 파일 설정
```bash
# .env 파일에 다음 내용 추가
HF_TOKEN=your_huggingface_token_here
CUDA_VISIBLE_DEVICES=0
```

### GPU 설정 확인
```bash
# NVIDIA Docker 런타임 확인
docker run --rm --gpus all nvidia/cuda:11.8-runtime-ubuntu22.04 nvidia-smi

# SpectraVision 이미지로 GPU 테스트
docker run --rm --gpus all ghcr.io/gwleee/spectravision-4.49:latest python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🛠️ 문제 해결

### 이미지 다운로드 실패
```bash
# Docker 로그인 (필요시)
docker login ghcr.io

# 네트워크 문제시 재시도
docker pull ghcr.io/gwleee/spectravision-4.49:latest
```

### GPU 인식 안됨
```bash
# NVIDIA Docker 설치 확인
nvidia-docker version

# Docker 데몬 재시작
sudo systemctl restart docker
```

### 메모리 부족
```bash  
# Docker 정리
docker system prune -f
docker volume prune -f

# 사용하지 않는 이미지 삭제
docker image prune -a
```

### 컨테이너 내부 디버깅
```bash
# 컨테이너 내부 진입
docker exec -it <container_name> /bin/bash

# 로그 확인
docker logs <container_name>
```

## 🎉 시스템 검증 및 테스트

### 🧪 시스템 테스트

#### DockerOrchestrator 테스트
```bash
# 1. DockerOrchestrator 시스템 테스트
python3 docker/integrated/docker_main.py --mode test

# 2. 개별 컨테이너 기본 동작 확인
docker run --rm --gpus all \
  ghcr.io/gwleee/spectravision-4.49:latest \
  python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU Count: {torch.cuda.device_count()}')
print('SpectraVision container ready!')
"
```

#### 실제 평가 테스트
```bash
# 3. 빠른 평가 테스트 (SmolVLM + MMBench)
python3 docker/integrated/docker_main.py --mode batch \
  --models "SmolVLM-256M" \
  --benchmarks "MMBench" \
  --gpu-ids 0
```

### 📊 예상 결과
```
SpectraBench-Vision Docker System Test
==================================================
Testing Docker connectivity...
SUCCESS: Docker client initialized successfully
GPU Configuration: 1 GPU(s) detected

Testing 5 container images...

Testing transformers_4_49...
SUCCESS: Image available
   Starting container...
   SUCCESS: Container started
   SUCCESS: Basic functionality test passed
   SUCCESS: GPU test: CUDA available: True
   SUCCESS: Container stopped

SUCCESS: All system tests passed! System is ready for evaluation.
```

---

## 🎊 성공! SpectraBench-Vision 시스템 준비 완료

**🎯 축하합니다!** SpectraBench-Vision Docker 시스템이 성공적으로 구축되었습니다!

### ✨ 이제 가능한 것들:
- 🚀 **자동화된 컨테이너 관리**: DockerOrchestrator가 모델별 최적 컨테이너 자동 선택
- 🤖 **30개 모델 지원**: 5개 transformer 버전에 걸쳐 모든 모델 평가 가능
- 📈 **확장성**: 새로운 모델/버전 쉽게 추가 가능
- 🔧 **재현성**: 어디서든 동일한 환경에서 평가 수행

### 🎯 다음 단계:
```bash
# DockerOrchestrator로 강력한 VLM 평가 시작!
python3 docker/integrated/docker_main.py --mode interactive
```

**Happy Evaluating! 🚀✨**