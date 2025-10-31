# SpectraBench-Vision Docker 사용 가이드

**KISTI 초거대AI연구센터**에서 개발한 VLM 평가 시스템 Docker 활용법

---

## 시스템 개요

SpectraBench-Vision은 서로 다른 `transformers` 버전이 필요한 Vision-Language 모델들을 Docker 컨테이너로 격리하여 평가합니다.

**지원 컨테이너**:
- `ghcr.io/gwleee/spectravision:4.33` - Qwen-VL, mPLUG-Owl2, Monkey-Chat, InternLM-XComposer 시리즈 (5개 모델)
- `ghcr.io/gwleee/spectravision:4.37` - LLaVA 시리즈 (2개 모델)

---

## 사전 준비

### 1. Docker 및 NVIDIA Container Toolkit 설치

```bash
# Docker 설치 확인
docker --version

# NVIDIA Container Toolkit 설치 (Ubuntu)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. GPU 확인

```bash
# GPU 상태 확인
nvidia-smi

# Docker에서 GPU 접근 테스트
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 3. HuggingFace 토큰 설정

```bash
# .env 파일 생성
cat > .env << EOF
HUGGING_FACE_HUB_TOKEN=hf_your_token_here
EOF

# 토큰 발급: https://huggingface.co/settings/tokens
```

---

## 기본 사용법

### Transformers 4.33 컨테이너 (Qwen-VL 시리즈)

```bash
# 단일 모델 평가
docker run --rm --gpus all \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/.env:/workspace/.env \
  ghcr.io/gwleee/spectravision:4.33 \
  python /workspace/VLMEvalKit/run.py \
  --model qwen_chat \
  --data MMBench_DEV_EN

# 여러 벤치마크 평가
docker run --rm --gpus all \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/.env:/workspace/.env \
  ghcr.io/gwleee/spectravision:4.33 \
  python /workspace/VLMEvalKit/run.py \
  --model qwen_chat \
  --data MMBench_DEV_EN TextVQA_VAL GQA_TestDev_Balanced
```

### Transformers 4.37 컨테이너 (LLaVA, InternVL2)

```bash
# LLaVA-1.5-7B 평가 (100% 검증 완료!)
docker run --rm --gpus all \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/.env:/workspace/.env \
  ghcr.io/gwleee/spectravision:4.37 \
  python /workspace/VLMEvalKit/run.py \
  --model llava_v1.5_7b \
  --data MMBench_DEV_EN TextVQA_VAL

# InternVL2-8B 평가
docker run --rm --gpus all \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/.env:/workspace/.env \
  ghcr.io/gwleee/spectravision:4.37 \
  python /workspace/VLMEvalKit/run.py \
  --model InternVL2-8B \
  --data MMBench_DEV_EN
```

---

## 고급 사용법

### 1. 특정 GPU 지정

```bash
# GPU 0번만 사용
docker run --rm --gpus '"device=0"' \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/.env:/workspace/.env \
  ghcr.io/gwleee/spectravision:4.37 \
  python /workspace/VLMEvalKit/run.py \
  --model llava_v1.5_7b \
  --data MMBench_DEV_EN

# GPU 0,1번 사용
docker run --rm --gpus '"device=0,1"' \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/.env:/workspace/.env \
  ghcr.io/gwleee/spectravision:4.37 \
  python /workspace/VLMEvalKit/run.py \
  --model llava_v1.5_13b \
  --data MMBench_DEV_EN
```

### 2. 테스트 모드 (샘플 수 제한)

```bash
# 10개 샘플만 평가 (빠른 테스트)
docker run --rm --gpus all \
  -e VLMEVAL_SAMPLE_LIMIT=10 \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/.env:/workspace/.env \
  ghcr.io/gwleee/spectravision:4.37 \
  python /workspace/VLMEvalKit/run.py \
  --model llava_v1.5_7b \
  --data MMBench_DEV_EN

# 50개 샘플 평가
docker run --rm --gpus all \
  -e VLMEVAL_SAMPLE_LIMIT=50 \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/.env:/workspace/.env \
  ghcr.io/gwleee/spectravision:4.33 \
  python /workspace/VLMEvalKit/run.py \
  --model qwen_chat \
  --data MMBench_DEV_EN
```

### 3. 대화형 모드 (컨테이너 내부 접근)

```bash
# 4.37 컨테이너에 직접 접속
docker run -it --gpus all \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/.env:/workspace/.env \
  ghcr.io/gwleee/spectravision:4.37 \
  /bin/bash

# 컨테이너 내부에서:
cd /workspace/VLMEvalKit
python run.py --model llava_v1.5_7b --data MMBench_DEV_EN
```

### 4. 영구 컨테이너 (반복 평가용)

```bash
# 컨테이너 시작 (백그라운드)
docker run -d --name spectravision_437 \
  --gpus all \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/.env:/workspace/.env \
  ghcr.io/gwleee/spectravision:4.37 \
  tail -f /dev/null

# 평가 실행
docker exec spectravision_437 \
  python /workspace/VLMEvalKit/run.py \
  --model llava_v1.5_7b \
  --data MMBench_DEV_EN

docker exec spectravision_437 \
  python /workspace/VLMEvalKit/run.py \
  --model llava_v1.5_13b \
  --data TextVQA_VAL

# 컨테이너 종료
docker stop spectravision_437
docker rm spectravision_437
```

### 5. 배치 평가 스크립트

```bash
# 제공된 스크립트 사용
chmod +x run_437_llava_only.sh
./run_437_llava_only.sh

# 또는 run_eval_433.sh (Qwen-VL 배치 평가)
chmod +x run_eval_433.sh
./run_eval_433.sh
```

---

## 볼륨 마운트 옵션

### 기본 볼륨 설정

```bash
docker run --rm --gpus all \
  -v $(pwd)/outputs:/workspace/outputs \      # 결과 파일
  -v $(pwd)/.env:/workspace/.env \            # 환경 변수
  ghcr.io/gwleee/spectravision:4.37 \
  python /workspace/VLMEvalKit/run.py --model llava_v1.5_7b --data MMBench
```

### 캐시 디렉토리 재사용 (다운로드 최적화)

```bash
# HuggingFace 캐시와 데이터셋 캐시 마운트
mkdir -p cache/huggingface cache/matplotlib data

docker run --rm --gpus all \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/.env:/workspace/.env \
  -v $(pwd)/data:/workspace/LMUData \
  -v $(pwd)/cache/huggingface:/tmp/.cache/huggingface \
  -v $(pwd)/cache/matplotlib:/tmp/.cache/matplotlib \
  -e LMUData=/workspace/LMUData \
  -e HF_HOME=/tmp/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/tmp/.cache/huggingface \
  -e MPLCONFIGDIR=/tmp/.cache/matplotlib \
  ghcr.io/gwleee/spectravision:4.37 \
  python /workspace/VLMEvalKit/run.py \
  --model llava_v1.5_7b \
  --data MMBench_DEV_EN
```

**장점**:
- 모델 파일 재다운로드 방지
- 데이터셋 재다운로드 방지
- 평가 속도 대폭 향상

---

## 출력 결과

### 디렉토리 구조

```
outputs/
└── [timestamp]/
    ├── logs/
    │   ├── evaluation_summary.log    # 전체 평가 요약
    │   └── [model].log                # 모델별 상세 로그
    ├── results/
    │   ├── [model]_results.csv        # CSV 결과 (자동 생성)
    │   └── [model]_results.xlsx       # Excel 결과 (자동 생성)
    ├── [model]/[benchmark]/           # VLMEvalKit 원본 출력
    └── EVALUATION_REPORT.md           # 최종 평가 리포트
```

### 결과 파일 예시

```bash
# CSV 결과 확인
cat outputs/20251030_001/results/llava_v1.5_7b_results.csv

# Excel 결과 확인 (LibreOffice 또는 Excel로 열기)
libreoffice outputs/20251030_001/results/llava_v1.5_7b_results.xlsx

# 평가 리포트 확인
cat outputs/20251030_001/EVALUATION_REPORT.md
```

---

## 지원 모델

### Transformers 4.33 (5개 모델)

| 모델명 | VLM ID | GPU 메모리 | 상태 |
|--------|--------|------------|------|
| Qwen-VL-Chat | `qwen_chat` | 25GB | 검증 완료 |
| mPLUG-Owl2 | `mPLUG-Owl2` | 26GB | 검증 완료 |
| Monkey-Chat | `monkey-chat` | 28GB | 검증 완료 |
| InternLM-XComposer2 | `XComposer2` | 26GB | 검증 완료 |
| InternLM-XComposer | `XComposer` | 24GB | 검증 완료 |

### Transformers 4.37 (2개 모델)

| 모델명 | VLM ID | GPU 메모리 | 상태 |
|--------|--------|------------|------|
| LLaVA-1.5-7B | `llava_v1.5_7b` | 25GB | 100% 검증 |
| LLaVA-1.5-13B | `llava_v1.5_13b` | 45GB | 100% 검증 |

---

## 지원 벤치마크 (24개)

| 카테고리 | 벤치마크 |
|----------|----------|
| **기본 VQA** | MMBench_DEV_EN, TextVQA_VAL, GQA_TestDev_Balanced |
| **문서 이해** | DocVQA_VAL, ChartQA_TEST, InfoVQA_VAL, OCRBench |
| **과학/전문** | ScienceQA_VAL, AI2D_TEST, MMMU_DEV_VAL |
| **고급 평가** | HallusionBench, MMStar, POPE, RealWorldQA |
| **한국어** | MMBench_DEV_KO, SEEDBench_IMG_KO |
| **기타** | VizWiz, SEEDBench_IMG, BLINK, VisOnlyQA-VLMEvalKit |

---

## 문제 해결

### 1. GPU 메모리 부족

```bash
# 더 작은 모델 사용
docker run --rm --gpus all \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/.env:/workspace/.env \
  ghcr.io/gwleee/spectravision:4.37 \
  python /workspace/VLMEvalKit/run.py \
  --model InternVL2-2B \
  --data MMBench_DEV_EN

# 특정 GPU만 사용
docker run --rm --gpus '"device=1"' \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/.env:/workspace/.env \
  ghcr.io/gwleee/spectravision:4.37 \
  python /workspace/VLMEvalKit/run.py \
  --model llava_v1.5_7b \
  --data MMBench_DEV_EN
```

### 2. HuggingFace 토큰 오류

```bash
# .env 파일 확인
cat .env
# HUGGING_FACE_HUB_TOKEN=hf_... 형태여야 함

# 토큰 재설정
echo "HUGGING_FACE_HUB_TOKEN=hf_your_new_token_here" > .env

# 토큰 발급: https://huggingface.co/settings/tokens
```

### 3. 이미지 다운로드 실패

```bash
# 수동 다운로드
docker pull ghcr.io/gwleee/spectravision:4.33
docker pull ghcr.io/gwleee/spectravision:4.37

# 이미지 확인
docker images | grep spectravision
```

### 4. 권한 오류 (Permission denied)

```bash
# 출력 디렉토리 권한 설정
mkdir -p outputs
chmod -R 777 outputs

# 캐시 디렉토리 권한 설정
mkdir -p cache/huggingface cache/matplotlib
chmod -R 777 cache
```

### 5. 컨테이너 정리

```bash
# 실행 중인 컨테이너 확인
docker ps

# 모든 컨테이너 중지
docker stop $(docker ps -q)

# 사용하지 않는 컨테이너 삭제
docker container prune -f

# 디스크 공간 확보
docker system prune -a -f
```

---

## 성능 최적화

### 1. 영구 컨테이너 사용

반복 평가 시 컨테이너를 유지하여 모델 로딩 시간 절약:

```bash
# 컨테이너 시작
docker run -d --name persistent_437 \
  --gpus all \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/.env:/workspace/.env \
  -v $(pwd)/cache/huggingface:/tmp/.cache/huggingface \
  ghcr.io/gwleee/spectravision:4.37 \
  tail -f /dev/null

# 여러 평가 실행 (모델은 메모리에 유지됨)
for benchmark in MMBench_DEV_EN TextVQA_VAL GQA_TestDev_Balanced; do
  docker exec persistent_437 \
    python /workspace/VLMEvalKit/run.py \
    --model llava_v1.5_7b \
    --data $benchmark
done

# 종료
docker stop persistent_437 && docker rm persistent_437
```

### 2. 멀티 GPU 활용

```bash
# 2개 GPU로 대형 모델 평가
docker run --rm --gpus '"device=0,1"' \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/.env:/workspace/.env \
  ghcr.io/gwleee/spectravision:4.37 \
  python /workspace/VLMEvalKit/run.py \
  --model llava_v1.5_13b \
  --data MMBench_DEV_EN
```

### 3. 병렬 평가

```bash
# 서로 다른 GPU에서 동시 평가
docker run -d --name eval_gpu0 --gpus '"device=0"' \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/.env:/workspace/.env \
  ghcr.io/gwleee/spectravision:4.37 \
  python /workspace/VLMEvalKit/run.py \
  --model llava_v1.5_7b --data MMBench_DEV_EN

docker run -d --name eval_gpu1 --gpus '"device=1"' \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/.env:/workspace/.env \
  ghcr.io/gwleee/spectravision:4.37 \
  python /workspace/VLMEvalKit/run.py \
  --model InternVL2-8B --data TextVQA_VAL

# 로그 모니터링
docker logs -f eval_gpu0
docker logs -f eval_gpu1
```

---

## 추가 자료

- **VLMEvalKit**: https://github.com/open-compass/VLMEvalKit
- **HuggingFace 토큰**: https://huggingface.co/settings/tokens
- **NVIDIA Container Toolkit**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

---

## 문의

**KISTI 초거대AI연구센터**
- 개발자: 이건우 (AI Platform Team)
- 팀장: 이용 (AI Platform Team Leader)
- 센터장: 이경하 (Large-scale AI Research Center Director)

---

*Last Updated: 2025-10-31*
