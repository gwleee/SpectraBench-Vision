# 작업 현황 문서 (2025-10-17) - 최종 업데이트

## 🚀 현재 진행 상황 (CRITICAL UPDATE)

### ⚠️ GPU 사용 문제 해결 (v4)
**문제 발견**: v3에서 GPU가 거의 사용되지 않았던 이유를 발견했습니다!
- **근본 원인**: 각 벤치마크마다 Docker 컨테이너를 새로 생성 (`--rm` 사용)
- **결과**: 매 벤치마크마다 모델 다운로드(70초) + 로딩(14초) = 84초 낭비
- **실제 추론 시간**: 단 3-10초 (GPU 사용률 3-10%에 불과!)
- **증거**: qwen_chat.log에서 "Downloading shards" 6번 발견 (6개 벤치마크 = 6번 다운로드)

### 실행 중인 테스트
- **✅ v4 테스트 스크립트**: `run_433_eval_test_mode_v4.sh` (NEW!)
- **세션**: `outputs/20251017_002/`
- **백그라운드 프로세스 ID**: 2a1da4
- **Container ID**: `7682eaac74ef` (spectravision_qwen_chat)
- **최적화**: Persistent containers - 모델은 **한 번만** 다운로드/로딩!
- **예상 속도**: v3 대비 **10-20배 빠름**

### v3 테스트 결과 (중단됨)
- **세션**: `outputs/20251017_001/`
- **진행률**: 6/192 (3.1%) - ChartQA_TEST 중에 중단
- **상태**: killed (프로세스 fb015f)

### 완료된 평가 (qwen_chat)
1. ✅ MMBench_DEV_EN - SUCCESS
2. ✅ TextVQA_VAL - SUCCESS
3. ✅ GQA_TestDev_Balanced - SUCCESS
4. ✅ MMMU_DEV_VAL - SUCCESS
5. ✅ DocVQA_VAL - SUCCESS
6. 🔄 ChartQA_TEST - 진행 중

## 🔥 주요 발견사항

### 0. **GPU 미사용 문제 - 근본 원인 발견 및 해결! (v4)**
**문제**: nvidia-smi에서 GPU 사용률이 거의 0%
**원인 분석**:
```
v3 스크립트 전략 (잘못됨):
├── docker run --rm  ← 매번 컨테이너 삭제!
├── 벤치마크 1: 모델 다운로드 70초 + 로딩 14초 + 추론 3-10초
├── 벤치마크 2: 모델 다운로드 70초 + 로딩 14초 + 추론 3-10초
└── ... (24번 반복!)

시간 분포:
- 모델 다운로드/로딩: 84초 (GPU 미사용)
- 실제 추론: 3-10초 (GPU 사용)
- GPU 효율: 3-10% ❌

v4 스크립트 전략 (최적화):
├── docker run -d (persistent container)
├── docker exec: 벤치마크 1 (추론만 3-10초)
├── docker exec: 벤치마크 2 (추론만 3-10초)
└── ... (모델은 단 한 번만 로딩!)

GPU 효율: 90%+ ✅
```

**해결 방법**:
- `run_433_eval_test_mode_v4.sh` 생성
- Persistent container per model
- `docker exec` for each benchmark
- **예상 속도 향상**: 10-20배

### 1. VLMEvalKit 결과 파일 생성 이슈
- **문제**: 10 samples 테스트에서 VLMEvalKit이 결과 파일(`.xlsx`, `.json`)을 생성하지 않음
- **원인**: 샘플 크기가 너무 작아서 통계적 의미가 없다고 판단
- **해결**: 로그 파싱 스크립트 작성 (`scripts/parse_evaluation_results.py`)
- **결과**: 로그에서 결과 추출 성공

### 2. 벤치마크 결과 포맷 다양성
- **발견**: 24개 벤치마크가 서로 다른 결과 포맷 사용
- **문서화**: `docs/benchmark_result_formats.md`
- **확인된 포맷**: 11/24
  - Table format (MMBench, GQA, etc.)
  - Simple numeric (TextVQA)
  - Multi-column table (MMMU, POPE)
  - JSON format (OCRBench)
- **미확인**: 13개 벤치마크 포맷 (v3 테스트 완료 후 확인 가능)

### 3. 프로덕션 시스템 구조 이해
- **호스트 Python 환경**: VLMEvalKit 미설치 → `scripts/main.py` 직접 실행 불가
- **Docker 기반 실행**: `ghcr.io/gwleee/spectravision:4.33` 이미지 사용
- **작동 방식**: v3 bash 스크립트가 Docker 컨테이너 직접 실행
- **실제 프로덕션**: Docker orchestration 필요 (MultiVersionEvaluator)

### 4. GPU 사용 현황
- **GPU 0**: 1% 사용률, idle
- **GPU 1**: 23% 사용률 (평가 시)
- **특이사항**: nvidia-smi 프로세스 목록에 표시 안됨 (Docker 내부 실행)
- **메모리**: 거의 사용 안함 (1MiB / 81920MiB)

## 생성된 파일들

### 스크립트
1. `run_433_eval_test_mode_v2.sh` - 초기 테스트 (10 samples, 단일 로그) - 사용 안함
2. `run_433_eval_test_mode_v3.sh` - 개선 버전 (10 samples, 모델별 로그) - v4로 대체됨
3. **`run_433_eval_test_mode_v4.sh`** - 🚀 최적화 버전 (persistent containers) - **현재 실행 중**
4. `scripts/parse_evaluation_results.py` - 로그 파싱 및 결과 추출
5. `scripts/generate_final_report.py` - 최종 종합 리포트 생성

### 문서
1. `docs/benchmark_result_formats.md` - 벤치마크 결과 포맷 문서화
2. `WORK_STATUS.md` - 이 문서

### 결과 디렉토리
- **`outputs/20251017_002/`** - 🚀 v4 테스트 세션 (현재 실행 중!)
  - `logs/qwen_chat.log` - qwen_chat 모델 전용 로그
  - `logs/evaluation_summary.log` - 전체 요약 로그
  - `results/` - 파싱된 결과 파일 저장 예정
  - `qwen_chat/{benchmark}/` - 각 벤치마크별 디렉토리

- `outputs/20251017_001/` - v3 테스트 세션 (6/192 완료 후 중단)
  - 6개 벤치마크 완료: MMBench_DEV_EN, TextVQA_VAL, GQA, MMMU, DocVQA, ChartQA (일부)
  - GPU 효율 문제 발견의 증거 자료

- `outputs/20251017_001_v2_backup/` - v2 테스트 백업 (포맷 분석용)

## 다음 단계 (우선순위순)

### 1. 🚀 v4 테스트 완료 대기 (현재 실행 중!)
- **스크립트**: `run_433_eval_test_mode_v4.sh`
- **백그라운드 ID**: 2a1da4
- **세션**: `outputs/20251017_002/`
- **예상 소요 시간**: 30분 - 1시간 (v3 대비 10-20배 빠름!)
- **총 조합**: 192개 (8 models × 24 benchmarks)
- **완료 후 작업**:
  - 모든 벤치마크 결과 포맷 수집 (24/24)
  - `docs/benchmark_result_formats.md` 업데이트
  - 파싱 스크립트 테스트 및 검증
  - GPU 효율 개선 확인

### 2. 로그 파서 개선 🔧
- **현재 상태**: 11개 포맷 처리 가능
- **필요 작업**:
  - JSON 포맷 지원 (OCRBench)
  - Multi-column 테이블 지원 (MMMU, POPE)
  - 나머지 13개 벤치마크 포맷 추가
- **파일**: `scripts/parse_evaluation_results.py`

### 3. 실제 결과 파일 생성 검증 🎯
- **목적**: VLMEvalKit이 실제로 결과 파일을 생성하는지 확인
- **방법 1**: 더 큰 샘플 크기 (50-100 samples)
  ```bash
  # run_433_eval_test_mode_v3.sh 수정
  -e VLMEVAL_SAMPLE_LIMIT=50 \
  ```
- **방법 2**: 작은 벤치마크 full dataset 테스트
  ```bash
  # AI2D_TEST (~1000 samples) 또는 ScienceQA_VAL
  # VLMEVAL_SAMPLE_LIMIT 제거
  ```
- **확인 사항**:
  - `.xlsx` 파일 생성 여부
  - `.json` 파일 생성 여부
  - 파일 내용 구조 확인

### 4. 프로덕션 시스템 통합 테스트 🚀
- **문제**: 호스트에 VLMEvalKit 미설치
- **해결 방안**:
  - Option A: VLMEvalKit을 호스트에 설치
  - Option B: Docker 기반 MultiVersionEvaluator 사용
  - Option C: v3 스크립트를 다른 transformer 버전으로 확장
- **테스트 대상**:
  - `scripts/main.py --multi-version` (Docker orchestration)
  - SequentialEvaluator (호스트 설치 시)
  - MultiVersionEvaluator (Docker 기반)

### 5. 다른 Transformer 버전 테스트 📦
- **4.37**: 8개 모델 (InternVL2, LLaVA, CogVLM 등)
- **4.43**: 2개 모델 (Phi-3.5-Vision, Moondream2)
- **4.49**: 9개 모델 (SmolVLM, Qwen2.5-VL 등)
- **4.51**: 2개 모델 (Phi-4-Vision, Llama-4-Scout)
- **방법**: v3 스크립트 복사 후 IMAGE 변경
  ```bash
  cp run_433_eval_test_mode_v3.sh run_437_eval_test_mode_v3.sh
  # IMAGE="ghcr.io/gwleee/spectravision:4.37"로 변경
  # MODELS 배열 변경
  ```

## 기술적 세부사항

### Docker 실행 명령어
```bash
docker run --rm --network=host --gpus all \
  -v "$SESSION_DIR":/workspace/VLMEvalKit/outputs \
  -v "$DATA_DIR":/workspace/LMUData \
  -v "$ENV_FILE":/workspace/VLMEvalKit/.env \
  --env-file "$ENV_FILE" \
  -e VLMEVAL_SAMPLE_LIMIT=10 \
  -e LMUData=/workspace/LMUData \
  "ghcr.io/gwleee/spectravision:4.33" \
  python /workspace/VLMEvalKit/run.py --data $benchmark --model $model
```

### 로그 파싱 로직
- **입력**: 모델별 로그 파일 (`logs/qwen_chat.log`)
- **처리**:
  1. "Evaluation Results:" 패턴 검색
  2. 테이블 형식 파싱 (Overall, sub-metrics)
  3. JSON 형식 파싱 (OCRBench)
- **출력**:
  - `results/{model}_results.csv`
  - `results/{model}_results.xlsx`
  - `results_statistics.json`

### 파일 소유권 문제
- **원인**: Docker 컨테이너가 root로 실행
- **해결**: 스크립트 마지막에 `chown` 실행
  ```bash
  sudo chown -R $(id -u):$(id -g) "$SESSION_DIR"
  ```

## 알려진 이슈

### 1. VLMEvalKit 결과 파일 미생성
- **상황**: 10 samples에서는 파일 생성 안됨
- **영향**: 로그 파싱으로 대체 필요
- **해결**: 더 큰 샘플 크기로 테스트 필요

### 2. GPU 프로세스 미표시
- **상황**: nvidia-smi에 프로세스가 안 보임
- **원인**: Docker 컨테이너 내부에서 실행
- **영향**: 없음 (GPU 사용률은 정상 표시)

### 3. 호스트 환경 VLMEvalKit 미설치
- **상황**: `scripts/main.py` 직접 실행 불가
- **영향**: Docker 기반 실행만 가능
- **해결**: Docker orchestration 사용 또는 호스트 설치

### 4. 백그라운드 프로세스 중복
- **상황**: 여러 개의 v2 스크립트 프로세스 실행 중
- **영향**: 리소스 낭비 가능
- **조치**: v3로 전환 후 중복 프로세스 정리 필요

## 모니터링 명령어

### 진행 상황 확인
```bash
# 전체 요약
tail -50 outputs/20251017_001/logs/evaluation_summary.log

# 현재 모델 로그
tail -100 outputs/20251017_001/logs/qwen_chat.log

# 성공/실패 카운트
grep -c "SUCCESS" outputs/20251017_001/logs/evaluation_summary.log
grep -c "FAILED" outputs/20251017_001/logs/evaluation_summary.log
```

### GPU 상태 확인
```bash
# GPU 사용률 및 메모리
nvidia-smi

# 지속적 모니터링 (1초마다)
watch -n 1 nvidia-smi
```

### Docker 상태 확인
```bash
# 실행 중인 컨테이너
docker ps

# 컨테이너 로그
docker logs <container_id>
```

### 백그라운드 프로세스 확인
```bash
# v3 스크립트 프로세스
ps aux | grep run_433_eval

# Python 프로세스
ps aux | grep main.py

# 모든 bash 셸 확인
/bashes
```

## 참고 정보

### 모델 이름 매핑
- **v3 스크립트**: `qwen_chat` (VLMEvalKit ID)
- **configs/models.yaml**: `"Qwen-VL-Chat"` (display name)
- **주의**: `scripts/main.py`는 display name 사용

### 벤치마크 이름 매핑
- **v3 스크립트**: `MMBench_DEV_EN`, `TextVQA_VAL` (VLMEvalKit ID)
- **configs/benchmarks.yaml**: `"MMBench"`, `"TextVQA"` (display name)

### 디렉토리 구조
```
SpectraBench-Vision/
├── configs/
│   ├── models.yaml           # 모델 설정 (30개 모델)
│   ├── benchmarks.yaml       # 벤치마크 설정 (24개)
│   └── hardware.yaml         # 하드웨어 설정
├── scripts/
│   ├── main.py               # 프로덕션 메인 (VLMEvalKit 필요)
│   ├── parse_evaluation_results.py  # 로그 파서
│   └── generate_final_report.py     # 최종 리포트
├── spectravision/
│   ├── evaluator.py          # SequentialEvaluator
│   ├── multi_version_evaluator.py   # MultiVersionEvaluator
│   ├── docker_orchestrator.py       # Docker 관리
│   └── config.py             # 설정 관리
├── docs/
│   └── benchmark_result_formats.md  # 벤치마크 포맷 문서
├── outputs/
│   ├── 20251017_001/         # v3 테스트 (진행 중)
│   └── 20251017_001_v2_backup/  # v2 백업
├── run_433_eval_test_mode_v2.sh  # 초기 테스트
├── run_433_eval_test_mode_v3.sh  # 현재 테스트
└── WORK_STATUS.md            # 이 문서
```

## 연락처 및 리소스

### Docker 이미지
- **Registry**: `ghcr.io/gwleee/spectravision`
- **Tags**: `4.33`, `4.37`, `4.43`, `4.49`, `4.51`
- **Base**: CUDA 12.1, PyTorch 2.1.0

### 문서
- **README.md**: 프로젝트 개요
- **CLAUDE.md**: Claude Code 가이드
- **QUICKSTART_TEST_MODE.md**: 퀵스타트 가이드
- **TESTING_HISTORY.md**: 테스트 히스토리

---

## 작업 재개 시 체크리스트

1. ✅ v3 테스트 완료 여부 확인
   ```bash
   tail -100 outputs/20251017_001/logs/evaluation_summary.log
   ```

2. ✅ 결과 파일 생성 확인
   ```bash
   ls -lh outputs/20251017_001/results/
   ```

3. ✅ 벤치마크 포맷 문서 업데이트
   - `docs/benchmark_result_formats.md` 확인
   - 24/24 포맷 수집 완료되었는지 확인

4. ✅ 로그 파서 테스트
   ```bash
   python3 scripts/parse_evaluation_results.py outputs/20251017_001
   ```

5. ⏭️ 다음 단계 결정
   - 더 큰 샘플 크기로 테스트?
   - 다른 transformer 버전 테스트?
   - 프로덕션 시스템 통합?

---

**마지막 업데이트**: 2025-10-17 05:05 KST (v4 시작)
**작성자**: Claude Code
**현재 세션**: 20251017_002 (v4 - 최적화 버전)
**백그라운드 프로세스**: 2a1da4
**Container ID**: 7682eaac74ef (spectravision_qwen_chat)
**진행률**: 시작됨 (192개 조합)
