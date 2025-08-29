# 🐳 SpectraBench-Vision Docker 사용 가이드

## 🎯 시스템 개요

SpectraBench-Vision은 **재현성과 편의성**을 위해 설계된 통합 Docker 시스템입니다:

- **통합 이미지**: 한 번의 다운로드로 30개 VLM 모델 평가 시스템 전체 사용
- **자동 컨테이너 관리**: 필요에 따라 개별 transformer 버전 컨테이너 자동 관리
- **Docker-in-Docker**: 완전히 격리된 환경에서 다중 버전 지원

## 📦 시스템 아키텍처

### 🎯 통합 시스템 (권장)

| 이미지 | 설명 | 모델 수 | 용도 |
|--------|------|---------|------|
| **spectrabench-vision** | **완전 통합 시스템** | **30개** | **모든 사용자** |

**핵심 기능:**
- ✅ Docker-in-Docker 아키텍처로 완전 자동화
- ✅ 필요시 개별 컨테이너 자동 pull/start/stop  
- ✅ DockerOrchestrator 내장으로 지능형 모델 관리
- ✅ 한 번의 명령으로 전체 시스템 사용
- ✅ GPU 자동 감지 및 최적 할당

### 🔧 개별 컨테이너 (고급 사용자/개발자용)

통합 시스템이 자동으로 관리하지만, 직접 사용도 가능:

| 이미지 | Transformers | 주요 모델 | 용도 |
|--------|-------------|----------|------|
| **spectravision-4.33** | 4.33.0 | Qwen-VL, VisualGLM | 레거시 모델 |
| **spectravision-4.37** | 4.37.2 | InternVL2, LLaVA | 안정 버전 |
| **spectravision-4.43** | 4.43.0 | Phi-3.5-Vision | 중급 모델 |
| **spectravision-4.49** | 4.49.0 | SmolVLM, Qwen2.5-VL | 최신 모델 |
| **spectravision-4.51** | 4.51.0 | Phi-4-Vision | 실험용 |

## 📥 이미지 다운로드

### 권장 이미지

| 이미지 | 설명 | 모델 수 | 사용 사례 | 다운로드 |
|-------|-----|---------|----------|----------|
| **spectrabench-vision** | 통합 시스템 | **30개 모델** | **일반 사용자** | `docker pull ghcr.io/gwleee/ghcr.io/gwleee/spectrabench-vision:latest` |

### 개발자용 - 다중 버전 이미지

| 이미지 | Transformers | 주요 모델 | 사용 사례 | 다운로드 |
|-------|-------------|----------|----------|----------|
| **spectravision-4.33** | 4.33.0 | Qwen-VL, VisualGLM | 레거시 모델 | `docker pull ghcr.io/gwleee/spectravision-4.33:latest` |
| **spectravision-4.37** | 4.37.2 | InternVL2, LLaVA | 안정 버전 | `docker pull ghcr.io/gwleee/spectravision-4.37:latest` |
| **spectravision-4.43** | 4.43.0 | Phi-3.5-Vision | 중급 모델 | `docker pull ghcr.io/gwleee/spectravision-4.43:latest` |
| **spectravision-4.49** | 4.49.0 | SmolVLM, Qwen2.5-VL | 최신 모델 | `docker pull ghcr.io/gwleee/spectravision-4.49:latest` |
| **spectravision-4.51** | 4.51.0 | Phi-4-Vision | 실험용 | `docker pull ghcr.io/gwleee/spectravision-4.51:latest` |

### 빠른 다운로드 방법 (예시)

```bash
# 통합 시스템 (권장)
docker pull ghcr.io/gwleee/ghcr.io/gwleee/spectrabench-vision:latest

# 최신 모델만 필요한 경우
docker pull ghcr.io/gwleee/spectravision-4.49:latest

# 안정 버전만 필요한 경우  
docker pull ghcr.io/gwleee/spectravision-4.37:latest
```

## 🚀 빠른 시작하기 (1분)

### 기본 사용법
```bash
# 1. 통합 시스템 시작 (모든 30개 모델 자동 지원)
docker run -it --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/ghcr.io/gwleee/spectrabench-vision:latest

# 2. 컨테이너 내부에서 대화형 모드
python3 scripts/main.py --mode interactive
```

### 직접 평가 실행
```bash
# 특정 모델로 벤치마크 평가 (한 줄 명령)
docker run --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/ghcr.io/gwleee/spectrabench-vision:latest \
  python3 scripts/main.py --mode docker \
  --models "SmolVLM" --benchmarks "MMBench"
```

### 메모리 최적화 옵션
```bash
# 최신 모델만 (SmolVLM, Qwen2.5-VL)
docker run -it --gpus all \
  -e PULL_IMAGES=minimal \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/ghcr.io/gwleee/spectrabench-vision:latest

# 안정 버전 (InternVL2, LLaVA)
docker run -it --gpus all \
  -e PULL_IMAGES=stable \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/ghcr.io/gwleee/spectrabench-vision:latest
```

## ⚡ 다중 GPU 사용법

```bash
# 다중 GPU 활용
docker run -it --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/ghcr.io/gwleee/spectrabench-vision:latest \
  python3 scripts/main.py --mode docker \
  --models "Qwen2.5-VL-32B" "Qwen2.5-VL-72B" \
  --benchmarks "MMBench" --gpu-ids 0 1 2 3
```

## 🛠️ 고급 설정 옵션

### Docker Compose 사용 (개발자용)
```bash
# 특정 버전만 시작
docker-compose -f docker/docker-compose.prod.yml --profile latest up -d

# 개별 컨테이너 직접 사용
docker run --gpus all -it ghcr.io/gwleee/spectravision-4.49:latest
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
docker run --rm --gpus all ghcr.io/gwleee/ghcr.io/gwleee/spectrabench-vision:latest python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🛠️ 문제 해결

### 이미지 다운로드 실패
```bash
# Docker 로그인 (필요시)
docker login ghcr.io

# 네트워크 문제시 재시도
docker pull ghcr.io/gwleee/ghcr.io/gwleee/spectrabench-vision:latest
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

### 🧪 통합 시스템 테스트

#### 기본 동작 확인
```bash
# 1. 통합 시스템 기본 테스트
docker run --rm --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  ghcr.io/gwleee/spectrabench-vision:latest \
  python3 scripts/main.py --mode test

# 2. GPU 및 Docker 연결 확인
docker run --rm --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  ghcr.io/gwleee/spectrabench-vision:latest \
  python3 -c "
import torch
import docker
print(f'✅ CUDA: {torch.cuda.is_available()}')
print(f'✅ GPU Count: {torch.cuda.device_count()}')
client = docker.from_env()
print(f'✅ Docker: Connected')
print('🎯 SpectraBench-Vision 통합 시스템 준비 완료!')
"
```

#### 실제 평가 테스트
```bash
# 3. 빠른 평가 테스트 (SmolVLM + MMBench)
docker run --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectrabench-vision:latest \
  python3 scripts/main.py --mode docker \
  --models "SmolVLM" \
  --benchmarks "MMBench_DEV_EN"
```

### 📊 예상 결과
```
🐳 SpectraBench-Vision Docker Orchestrator
==================================================
✅ CUDA: True
✅ GPU Count: 4
✅ Docker: Connected
🎯 SpectraBench-Vision 통합 시스템 준비 완료!

📦 Pulling transformer version images...
✅ spectravision-4.49:latest ready
🎯 30개 모델, 24개 벤치마크, 720개 평가 조합 사용 가능!
```

---

## 🎊 성공! 완전한 통합 시스템 구축 완료

**🎯 축하합니다!** SpectraBench-Vision 통합 Docker 시스템이 성공적으로 구축되었습니다!

### ✨ 이제 가능한 것들:
- 🚀 **1분 만에 시작**: 한 번의 명령으로 전체 30개 모델 시스템 사용
- 🤖 **완전 자동화**: 모델별 최적 컨테이너 자동 선택 및 관리
- 📈 **확장성**: 새로운 모델/버전 쉽게 추가 가능
- 🔧 **재현성**: 어디서든 동일한 환경에서 평가 수행

### 🎯 다음 단계:
```bash
# 통합 시스템으로 강력한 VLM 평가 시작!
docker run -it --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectrabench-vision:latest
```

**Happy Evaluating! 🚀✨**