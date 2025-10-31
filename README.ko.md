# SpectraBench-Vision (Alpha version)

**KISTI 초거대AI연구센터**에서 개발한 Docker 기반 VLM 평가 시스템

---

## 핵심 가치

Vision-Language 모델들은 각기 다른 `transformers` 버전을 요구합니다:
- **Qwen-VL, mPLUG-Owl2, Monkey-Chat, InternLM-XComposer** → transformers 4.33.0
- **LLaVA** → transformers 4.37.2

**문제점:**
- ❌ 모델마다 환경 재설치 필요
- ❌ 의존성 충돌
- ❌ 재현 불가능한 환경

**SpectraBench-Vision 해결책:**
- ✅ 단일 명령으로 모든 모델 사용
- ✅ 자동 의존성 관리
- ✅ 완전한 재현성 (해시 검증 기반)
- ✅ **7개 모델 × 24개 벤치마크 = 168개 조합**

---

## 빠른 시작

### 1. 환경 설정
```bash
# .env 파일 생성 및 HuggingFace 토큰 추가
cp .env.template .env
nano .env  # HUGGING_FACE_HUB_TOKEN=hf_your_token_here
```

### 2. 4.33 모델 평가 (Qwen-VL 시리즈)
```bash
# 직접 컨테이너 사용
docker run --rm --gpus all \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/.env:/workspace/.env \
  ghcr.io/gwleee/spectravision:4.33 \
  python /workspace/VLMEvalKit/run.py \
  --model qwen_chat \
  --data MMBench_DEV_EN
```

### 3. 4.37 모델 평가 (LLaVA, InternVL2)
```bash
# LLaVA 평가 (100% 성공률 검증됨!)
docker run --rm --gpus all \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/.env:/workspace/.env \
  ghcr.io/gwleee/spectravision:4.37 \
  python /workspace/VLMEvalKit/run.py \
  --model llava_v1.5_7b \
  --data MMBench_DEV_EN TextVQA_VAL
```

---

## 지원 모델 (7개)

### Transformers 4.33.0 (5개 모델)
- **Qwen-VL-Chat** (검증 완료)
- **mPLUG-Owl2** (검증 완료)
- **Monkey-Chat** (검증 완료)
- **InternLM-XComposer2** (검증 완료)
- **InternLM-XComposer** (검증 완료)

### Transformers 4.37.2 (2개 모델)
- **LLaVA-1.5-7B** (100% 성공률 검증!)
- **LLaVA-1.5-13B** (100% 성공률 검증!)

---

## 지원 벤치마크 (24개)

| 분야 | 벤치마크 |
|------|----------|
| **기본 VQA** | MMBench, TextVQA, GQA |
| **문서 이해** | DocVQA, ChartQA, InfoVQA, OCRBench |
| **과학/전문** | ScienceQA, AI2D, MMMU |
| **고급 평가** | HallusionBench, MMStar, POPE |
| **한국어** | MMBench_DEV_KO, SEEDBench_IMG_KO |

---

## 최근 테스트 결과 (2025-10-30)

### LLaVA 모델 평가
- **구성**: Transformers 4.37.2 (Flash Attention 비활성화)
- **모델**: LLaVA-1.5-7B, LLaVA-1.5-13B
- **벤치마크**: 20개 × 10 샘플
- **결과**: **40/40 (100% 성공률)**
- **소요 시간**: ~22분

**주요 기능**:
- ✅ CSV/XLSX 자동 생성
- ✅ 모델별 성능 메트릭
- ✅ 지속형 컨테이너 (반복 다운로드 없음)

---

## 출력 결과

```
outputs/[timestamp]/
├── logs/
│   ├── evaluation_summary.log
│   └── [model].log
├── results/
│   ├── [model]_results.csv
│   └── [model]_results.xlsx
├── [model]/[benchmark]/  # VLMEvalKit 원본 출력
└── EVALUATION_REPORT.md
```

---

## 고급 사용법

### 배치 평가
```bash
# run_437_llava_only.sh 스크립트 사용
./run_437_llava_only.sh
```

### 특정 GPU 지정
```bash
docker run --rm --gpus '"device=0,1"' \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectravision:4.37 \
  python /workspace/VLMEvalKit/run.py --model llava_v1.5_7b --data MMBench
```

### 샘플 수 제한 (테스트 모드)
```bash
docker run --rm --gpus all \
  -e VLMEVAL_SAMPLE_LIMIT=10 \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectravision:4.37 \
  python /workspace/VLMEvalKit/run.py --model llava_v1.5_7b --data MMBench
```

---

## 재현 가능한 빌드

모든 의존성은 `pip-tools`로 잠금되어 있습니다:

```
docker/requirements/
├── base-requirements.lock      # 공통 패키지 (SHA256 해시)
├── transformers-4.33.lock      # 4.33.0 전용
└── transformers-4.37.lock      # 4.37.2 전용
```

의존성 업데이트:
```bash
cd docker/requirements
pip-compile --generate-hashes --allow-unsafe \
  --output-file=transformers-4.37.lock \
  transformers-4.37.in base-requirements.in
```

---

## 문제 해결

**GPU 메모리 부족**
```bash
# 더 작은 모델 사용 또는 특정 GPU 지정
docker run --gpus '"device=0"' ...
```

**토큰 오류**
```bash
# .env 파일에 토큰 추가
echo "HUGGING_FACE_HUB_TOKEN=hf_your_token_here" >> .env
```

**이미지 다운로드 실패**
```bash
# 수동 다운로드
docker pull ghcr.io/gwleee/spectravision:4.33
docker pull ghcr.io/gwleee/spectravision:4.37
```

---

## 라이선스

Apache License 2.0

## 감사의 말

- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 기반 구축

## 인용

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

*Developed by Gunwoo Lee, AI Platform Team (Leader: Ryong Lee)*
*KISTI Large-scale AI Research Center (Director: Kyong-Ha Lee)*
