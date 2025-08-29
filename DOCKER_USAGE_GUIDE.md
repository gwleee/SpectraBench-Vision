# 🐳 SpectraBench-Vision Docker 사용 가이드

## 🚀 즉시 시작하기 (2분) - 통합 컨테이너

```bash
# Option 1: 바로 실행 (가장 빠름)
docker run -it --gpus all -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectrabench-vision:latest

# Option 2: 환경 파일과 함께
# 1. 저장소 클론 (선택)
git clone https://github.com/gwleee/SpectraBench-Vision.git
cd SpectraBench-Vision

# 2. 환경 설정
cp .env.template .env
# .env 파일을 열어 HF_TOKEN을 추가하세요

# 3. 통합 컨테이너 실행
docker run -it --gpus all -v $(pwd):/workspace \
  -v $(pwd)/outputs:/workspace/outputs \
  --env-file .env \
  ghcr.io/gwleee/spectrabench-vision:latest
```

## 🔧 고급 사용법 - 다중 버전 컨테이너

```bash
# 1. 저장소 클론
git clone https://github.com/gwleee/SpectraBench-Vision.git
cd SpectraBench-Vision

# 2. 환경 설정
cp .env.template .env
# .env 파일을 열어 HF_TOKEN을 추가하세요

# 3. 개별 이미지 다운로드 (선택)
docker pull ghcr.io/gwleee/spectravision-4.49:latest  # 최신 모델용
docker pull ghcr.io/gwleee/spectravision-4.37:latest  # 안정 버전

# 4. 다중 버전 모드 실행
python scripts/main.py
# → "2. Multi-Version Docker" 선택
```

## 📦 사용 가능한 이미지들

### 추천 이미지

| 이미지 | 설명 | 모델 수 | 용도 | 다운로드 |
|--------|------|---------|------|----------|
| **spectrabench-vision** | 통합 시스템 | **30개** | **일반 사용자** | `docker pull ghcr.io/gwleee/spectrabench-vision:latest` |

### 개발자용 - 다중 버전 이미지들

| 이미지 | Transformers | 주요 모델 | 용도 | 다운로드 |
|--------|-------------|----------|------|----------|
| **spectravision-4.33** | 4.33.0 | Qwen-VL, VisualGLM | 레거시 모델 | `docker pull ghcr.io/gwleee/spectravision-4.33:latest` |
| **spectravision-4.37** | 4.37.2 | InternVL2, LLaVA | 안정 버전 | `docker pull ghcr.io/gwleee/spectravision-4.37:latest` |
| **spectravision-4.43** | 4.43.0 | Phi-3.5-Vision | 중급 모델 | `docker pull ghcr.io/gwleee/spectravision-4.43:latest` |
| **spectravision-4.49** | 4.49.0 | SmolVLM, Qwen2.5-VL | 최신 모델 | `docker pull ghcr.io/gwleee/spectravision-4.49:latest` |
| **spectravision-4.51** | 4.51.0 | 최첨단 모델 | 실험용 | `docker pull ghcr.io/gwleee/spectravision-4.51:latest` |

### 기본 환경
| 이미지 | 용도 | 다운로드 |
|--------|------|----------|
| **spectravision-base** | CUDA + 공통 의존성 | `docker pull ghcr.io/gwleee/spectravision-base:latest` |

## 🎯 모델별 추천 사용법

### 빠른 시작 (추천)
```bash
# 통합 컨테이너 - 모든 모델 포함
docker run -it --gpus all -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectrabench-vision:latest
```

### 특정 모델 타입별 (고급 사용자)
```bash  
# 최신 모델만 필요한 경우 (SmolVLM, Qwen2.5-VL)
docker run -it --gpus all ghcr.io/gwleee/spectravision-4.49:latest

# 안정적인 모델들 (LLaVA, InternVL2)
docker run -it --gpus all ghcr.io/gwleee/spectravision-4.37:latest

# 레거시 모델들 (Qwen-VL, VisualGLM)
docker run -it --gpus all ghcr.io/gwleee/spectravision-4.33:latest
```

## 🚀 통합 컨테이너 vs 다중 버전 비교

### 통합 컨테이너 (권장)
**장점:**
- ✅ **즉시 사용**: 한 번의 명령으로 모든 기능 사용
- ✅ **완전한 호환성**: 모든 30개 모델 지원
- ✅ **간단한 설정**: 복잡한 환경 설정 불필요
- ✅ **빠른 시작**: 2분 내 평가 시작

**사용 케이스:**
- 일반 사용자
- 빠른 프로토타이핑
- 데모 및 교육

### 다중 버전 컨테이너 (고급)
**장점:**
- ✅ **완벽한 격리**: 각 transformer 버전별 독립 환경
- ✅ **메모리 효율**: 필요한 버전만 사용
- ✅ **개발 친화적**: 특정 버전 디버깅 용이
- ✅ **세밀한 제어**: 환경별 세부 설정 가능

**사용 케이스:**
- 연구 개발자
- transformer 버전별 성능 비교
- 메모리 제약이 있는 환경

## 💻 사용법

### 방법 1: main.py를 통한 통합 실행 (권장)
```bash
python scripts/main.py
# 메뉴에서 "2. Multi-Version Docker" 선택
# 자동으로 적절한 컨테이너 선택 및 실행
```

### 방법 2: Docker Compose 사용
```bash
# 특정 버전만 시작
docker-compose -f docker/docker-compose.prod.yml --profile latest up -d

# 모든 버전 시작 (리소스 많이 필요)  
docker-compose -f docker/docker-compose.prod.yml --profile all up -d
```

### 방법 3: 직접 컨테이너 실행
```bash
# 대화형 모드로 컨테이너 진입
docker run --gpus all -it \
  -v $(pwd)/VLMEvalKit:/workspace/VLMEvalKit \
  -v $(pwd)/outputs:/workspace/outputs \
  -e HF_TOKEN=$HF_TOKEN \
  ghcr.io/gwleee/spectravision-4.49:latest

# 컨테이너 안에서 평가 실행
cd /workspace/VLMEvalKit
python run.py --model SmolVLM-Instruct --data MMBench_DEV_EN
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
docker run --rm --gpus all ghcr.io/gwleee/spectravision-minimal:latest python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📊 성능 비교

### 설정 시간 비교
| 방법 | 시간 | 성공률 | 지원 모델 |
|------|------|---------|----------|
| 기존 로컬 설정 | 30-60분 | 낮음 | 8개 |
| **Docker 방식** | **5-10분** | **높음** | **30개** |

### 메모리 사용량
| 이미지 | 크기 | GPU 메모리 권장 |
|--------|------|----------------|
| minimal | 12.2GB | 16GB+ |
| 4.33 | 27.9GB | 32GB+ |  
| 4.43/4.49/4.51 | 12.2GB | 16GB+ |

## 🛠️ 문제 해결

### 이미지 다운로드 실패
```bash
# Docker 로그인 (필요시)
docker login ghcr.io

# 네트워크 문제시 재시도
docker pull ghcr.io/gwleee/spectravision-minimal:latest --retry 3
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

## 🎉 완료 후 확인사항

### 성공 테스트
```bash
# 1. 이미지 정상 작동 확인
docker run --rm ghcr.io/gwleee/spectravision-minimal:latest python -c "import transformers; print(f'✅ Transformers {transformers.__version__} ready!')"

# 2. GPU 사용 가능 확인  
docker run --rm --gpus all ghcr.io/gwleee/spectravision-minimal:latest python -c "import torch; print(f'✅ CUDA: {torch.cuda.is_available()}')"

# 3. 전체 시스템 테스트
python scripts/main.py --mode test
```

### 예상 결과
```
✅ Transformers 4.37.2 ready!
✅ CUDA: True
✅ 30개 모델, 24개 벤치마크, 720개 평가 조합 사용 가능!
```

---

**🎯 축하합니다!** SpectraVision Docker 다중 버전 시스템이 성공적으로 구축되었습니다. 이제 5분만에 30개의 VLM 모델을 자유롭게 평가할 수 있습니다!

**다음 단계**: `python scripts/main.py`를 실행하고 "2. Multi-Version Docker"를 선택하여 강력한 다중 모델 평가를 시작하세요! 🚀