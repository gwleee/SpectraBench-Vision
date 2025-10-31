# 빠른 참조 가이드 (Quick Reference)

**작성일**: 2025-10-17
**작성자**: Claude Code

---

## 📋 작업 재개 시 확인할 문서

다음에 돌아왔을 때 **이 순서대로** 확인하세요:

### 1️⃣ **이 문서** (`QUICK_REFERENCE.md`)
   - 빠른 상태 확인 및 명령어

### 2️⃣ **작업 현황** (`WORK_STATUS.md`)
   - 상세한 진행 상황, 발견사항, 다음 단계

### 3️⃣ **벤치마크 포맷** (`docs/benchmark_result_formats.md`)
   - 24개 벤치마크 결과 형식

### 4️⃣ **로그 파일들** (`outputs/20251017_002/logs/`)
   - 실제 실행 로그 및 결과

---

## 🚀 현재 상태 (2025-10-17 05:05 KST)

### ✅ v4 테스트 실행 중!
```bash
스크립트: run_433_eval_test_mode_v4.sh
백그라운드 프로세스: 2a1da4
세션 디렉토리: outputs/20251017_002/
Container ID: 7682eaac74ef (spectravision_qwen_chat)
예상 완료 시간: 30분 - 1시간
```

### 🔥 주요 개선사항 (v4)
**문제**: v3에서 GPU가 거의 사용되지 않음 (3-10%)
**원인**: 매 벤치마크마다 모델을 새로 다운로드/로딩 (84초 낭비)
**해결**: Persistent containers - 모델은 한 번만 로딩!
**결과**: 10-20배 속도 향상 예상 ✅

---

## 📊 진행 상황 확인 명령어

### 1. v4 테스트 상태 확인
```bash
# 전체 요약 로그 (마지막 50줄)
tail -50 outputs/20251017_002/logs/evaluation_summary.log

# 현재 모델 로그 (마지막 100줄)
tail -100 outputs/20251017_002/logs/qwen_chat.log

# 성공/실패 카운트
grep -c "SUCCESS" outputs/20251017_002/logs/evaluation_summary.log
grep -c "FAILED" outputs/20251017_002/logs/evaluation_summary.log

# 실시간 모니터링 (Ctrl+C로 종료)
tail -f outputs/20251017_002/logs/evaluation_summary.log
```

### 2. GPU 상태 확인
```bash
# 현재 GPU 사용률
nvidia-smi

# 실시간 모니터링 (1초마다)
watch -n 1 nvidia-smi

# v4에서는 GPU 효율이 크게 개선되어야 함!
# 추론 중에는 GPU 사용률 50-90% 예상
```

### 3. Docker 컨테이너 확인
```bash
# 실행 중인 컨테이너 (v4는 persistent container 사용)
docker ps

# Container ID: spectravision_qwen_chat_*
# v4에서는 하나의 컨테이너가 모든 벤치마크 실행!
```

### 4. 백그라운드 프로세스 확인
```bash
# v4 스크립트 프로세스
ps aux | grep run_433_eval

# 백그라운드 프로세스 ID: 2a1da4
```

---

## 🎯 완료 후 확인 사항

### v4 테스트가 완료되면:

1. **최종 결과 확인**
   ```bash
   # 전체 완료 여부
   tail -100 outputs/20251017_002/logs/evaluation_summary.log

   # 최종 리포트
   cat outputs/20251017_002/EVALUATION_REPORT.md
   ```

2. **벤치마크 포맷 수집 확인**
   ```bash
   # 모델별 로그에 24개 벤치마크 결과가 모두 있어야 함
   ls -lh outputs/20251017_002/logs/

   # 각 모델 로그에서 "Evaluation Results:" 카운트
   grep -c "Evaluation Results:" outputs/20251017_002/logs/qwen_chat.log
   # 결과: 24개여야 함
   ```

3. **GPU 효율 검증**
   ```bash
   # v3와 v4 로그 비교
   # v3: 6개 벤치마크에 약 25분 소요 (ChartQA 중단)
   # v4: 24개 벤치마크 예상 시간 30-60분 (4배 더 많은 작업을 2-4배 빠르게)
   ```

4. **다음 모델로 진행 확인**
   ```bash
   # v4는 8개 모델을 순차적으로 실행
   # qwen_chat → mPLUG-Owl2 → monkey-chat → ... (총 8개)

   # 현재 어떤 모델 실행 중인지 확인
   docker ps --format "{{.Names}}"
   ```

---

## 🛠️ 문제 발생 시

### v4 테스트가 멈췄거나 실패한 경우:

1. **백그라운드 프로세스 상태 확인**
   ```bash
   ps aux | grep 2a1da4  # 프로세스 ID
   ```

2. **컨테이너 로그 확인**
   ```bash
   docker logs 7682eaac74ef  # Container ID
   ```

3. **v4 다시 시작**
   ```bash
   # 기존 컨테이너 정리
   docker ps -a --filter "name=spectravision_" --format "{{.ID}}" | xargs -r docker rm -f

   # v4 재시작
   ./run_433_eval_test_mode_v4.sh
   ```

---

## 📁 디렉토리 구조

```
SpectraBench-Vision/
├── run_433_eval_test_mode_v4.sh  ← 🚀 현재 실행 중인 스크립트
├── run_433_eval_test_mode_v3.sh  ← (v4로 대체됨)
├── scripts/
│   ├── parse_evaluation_results.py   ← 로그 파싱 (결과 추출)
│   └── generate_final_report.py      ← 종합 리포트 생성
├── docs/
│   └── benchmark_result_formats.md   ← 벤치마크 포맷 문서
├── outputs/
│   ├── 20251017_002/  ← 🚀 v4 현재 세션
│   │   ├── logs/
│   │   │   ├── evaluation_summary.log
│   │   │   ├── qwen_chat.log
│   │   │   └── ... (8개 모델 로그)
│   │   ├── results/
│   │   └── EVALUATION_REPORT.md
│   └── 20251017_001/  ← v3 세션 (6/192 완료)
├── WORK_STATUS.md         ← 📄 상세 작업 현황
├── QUICK_REFERENCE.md     ← 📋 이 문서
└── CLAUDE.md              ← Claude Code 가이드
```

---

## 💡 핵심 요약

### v3 → v4 변경 사항
| 항목 | v3 | v4 |
|------|----|----|
| **컨테이너 전략** | `docker run --rm` (매번 삭제) | `docker run -d` (persistent) |
| **모델 로딩** | 24번 (벤치마크당 1번) | 1번 (모델당 1번) |
| **모델 다운로드 시간** | 84초 × 24 = 33분 | 84초 × 1 = 1.4분 |
| **GPU 효율** | 3-10% | 50-90% (예상) |
| **예상 완료 시간** | 2-3시간 | 30분-1시간 |

### 다음 작업
1. ✅ v4 테스트 완료 대기 (현재 실행 중)
2. 벤치마크 포맷 수집 (24/24)
3. 로그 파서 개선 (JSON, multi-column 지원)
4. 다른 transformer 버전 테스트 (4.37, 4.43, 4.49, 4.51)
5. 프로덕션 시스템 통합

---

## 📞 추가 정보

- **GPU**: 2× NVIDIA A100 80GB PCIe
- **메모리**: 444GB (436GB available)
- **Docker 이미지**: `ghcr.io/gwleee/spectravision:4.33`
- **Samples per benchmark**: 10 (test mode)

---

**마지막 업데이트**: 2025-10-17 05:10 KST
**백그라운드 프로세스**: 2a1da4
**세션**: outputs/20251017_002/
