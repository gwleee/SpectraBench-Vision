#!/usr/bin/env python3
"""
SpectraVision 종합 평가 실행기 (수정 버전)
30개 모델 × 24개 벤치마크 = 720개 조합 평가
올바른 VLMEvalKit 모델 이름 사용 + 정확도 정보 포함
"""
import os
import subprocess
import time
import json
import pandas as pd
import glob
from datetime import datetime
from pathlib import Path

# YAML에서 정의된 올바른 VLMEvalKit 모델 이름 사용
MODEL_GROUPS = {
    "transformers-4.33": {
        "container": "spectravision-4.33:latest",
        "models": [
            "qwen_chat", "VisualGLM_6b", "mPLUG-Owl2",
            "monkey", "XComposer2", "idefics_9b_instruct",
            "instructblip_13b", "PandaGPT_13B"
        ]
    },
    "transformers-4.37": {
        "container": "spectravision-4.37:latest", 
        "models": [
            "InternVL2-2B", "MiniCPM-V-2_6", "llava_v1.5_7b", "cogvlm-chat",
            "InternVL2-8B", "sharegpt4v_7b", "MMAlaya2", "llava_v1.5_13b"
        ]
    },
    "transformers-4.43": {
        "container": "spectravision-4.43:latest",
        "models": [
            "Phi-3.5-Vision", "Moondream2"
        ]
    },
    "transformers-4.49": {
        "container": "spectravision-4.49:latest",
        "models": [
            "SmolVLM-256M", "SmolVLM-500M", "SmolVLM", 
            "Qwen2-VL-2B-Instruct", "Qwen2.5-VL-3B-Instruct", "Aria",
            "Qwen2.5-VL-7B-Instruct", "Pixtral-12B", "Qwen2.5-VL-32B-Instruct"
        ]
    },
    "transformers-4.51": {
        "container": "spectravision-4.51:latest",
        "models": [
            "Phi-4-Vision", "Llama-4-Scout-17B-16E-Instruct"
        ]
    }
}

# 벤치마크 정의
BENCHMARKS = [
    "MMBench_DEV_EN", "MMBench_TEST_EN", "MMBench_DEV_CN", "MMBench_TEST_CN",
    "CCBench", "MME", "SEEDBench_IMG", "MMBench-Video",
    "AI2D", "ScienceQA_VAL", "ScienceQA_TEST", "ChartQA_TEST",
    "TextVQA_VAL", "DocVQA_VAL", "DocVQA_TEST", "InfoVQA_VAL",
    "OCRBench", "GQA_testdev", "VizWiz_VAL", "VizWiz_TEST", 
    "VQAv2_VAL", "VQAv2_TEST", "OKVQA_VAL", "HallusionBench"
]

class ComprehensiveEvaluator:
    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 실행 로그 파일
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"comprehensive_evaluation_{timestamp}.log"
        self.results_file = self.output_dir / f"evaluation_results_{timestamp}.json"
        
        self.results = {
            "start_time": datetime.now().isoformat(),
            "total_combinations": 0,
            "completed_combinations": 0,
            "failed_combinations": 0,
            "evaluations": []
        }
        
    def log(self, message):
        """로그 메시지 출력 및 파일 저장"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def parse_accuracy_from_results(self, model, benchmark, work_dir):
        """VLMEvalKit 결과 파일에서 정확도 추출"""
        try:
            # 결과 파일 경로 패턴들
            result_patterns = [
                f"{model}*{benchmark}*acc.csv",
                f"{model}*{benchmark}*.xlsx",
                f"{model}/{model}_{benchmark}_acc.csv",
                f"{model}/{model}_{benchmark}.xlsx"
            ]
            
            accuracy = None
            result_files = []
            
            for pattern in result_patterns:
                pattern_path = os.path.join(work_dir, pattern)
                matches = glob.glob(pattern_path, recursive=True)
                result_files.extend(matches)
            
            # CSV 파일에서 정확도 추출
            for result_file in result_files:
                if result_file.endswith('_acc.csv'):
                    try:
                        df = pd.read_csv(result_file)
                        if 'Overall' in df.columns:
                            accuracy = df['Overall'].iloc[0]
                        elif len(df.columns) > 1:
                            accuracy = df.iloc[0, -1]  # 마지막 열의 첫 번째 값
                        break
                    except Exception as e:
                        self.log(f"   ⚠️  CSV 파일 읽기 실패 {result_file}: {e}")
                        continue
                        
                elif result_file.endswith('.xlsx'):
                    try:
                        df = pd.read_excel(result_file)
                        if 'Overall' in df.columns:
                            accuracy = df['Overall'].iloc[0]
                        elif len(df.columns) > 1:
                            accuracy = df.iloc[0, -1]  # 마지막 열의 첫 번째 값
                        break
                    except Exception as e:
                        self.log(f"   ⚠️  Excel 파일 읽기 실패 {result_file}: {e}")
                        continue
            
            if accuracy is not None:
                try:
                    # 퍼센트 형태면 소수점으로 변환
                    if isinstance(accuracy, str) and '%' in accuracy:
                        accuracy = float(accuracy.replace('%', '')) / 100.0
                    else:
                        accuracy = float(accuracy)
                    return accuracy
                except (ValueError, TypeError):
                    self.log(f"   ⚠️  정확도 값 변환 실패: {accuracy}")
                    return None
            else:
                self.log(f"   ⚠️  정확도 값을 찾을 수 없음. 발견된 파일들: {result_files}")
                return None
                
        except Exception as e:
            self.log(f"   ⚠️  정확도 추출 중 오류: {e}")
            return None
    
    def calculate_total_combinations(self):
        """전체 조합 수 계산"""
        total = 0
        for group_name, group_data in MODEL_GROUPS.items():
            models_count = len(group_data["models"])
            benchmarks_count = len(BENCHMARKS)
            group_combinations = models_count * benchmarks_count
            total += group_combinations
            self.log(f"{group_name}: {models_count} models × {benchmarks_count} benchmarks = {group_combinations}")
        
        self.results["total_combinations"] = total
        self.log(f"📊 전체 조합 수: {total}")
        return total
    
    def run_single_evaluation(self, container, model, benchmark, combination_id):
        """단일 모델-벤치마크 조합 평가 실행"""
        self.log(f"🚀 [{combination_id}] {model} × {benchmark} 시작...")
        
        start_time = time.time()
        container_name = f"eval-{model.replace('/', '-').replace('_', '-').lower()}-{benchmark.lower()}-{combination_id}"
        
        cmd = [
            "docker", "run", "--gpus", "all",
            "-v", f"{os.getcwd()}/outputs:/workspace/outputs",
            "-e", "HUGGING_FACE_HUB_TOKEN",
            "-e", "HF_TOKEN", 
            "--name", container_name,
            container,
            "python", "/workspace/VLMEvalKit/run.py",
            "--model", model,
            "--data", benchmark,
            "--mode", "all",
            "--work-dir", "/workspace/outputs"
        ]
        
        accuracy = None
        try:
            # 컨테이너 실행
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1시간 타임아웃
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                self.log(f"✅ [{combination_id}] {model} × {benchmark} 완료 ({execution_time:.1f}초)")
                status = "success"
                error_message = None
                self.results["completed_combinations"] += 1
                
                # 정확도 추출 시도
                accuracy = self.parse_accuracy_from_results(model, benchmark, os.getcwd() + "/outputs")
                if accuracy is not None:
                    self.log(f"   📊 정확도: {accuracy:.4f}")
                else:
                    self.log(f"   📊 정확도: 추출 실패")
                    
            else:
                self.log(f"❌ [{combination_id}] {model} × {benchmark} 실패: {result.stderr[:200]}")
                status = "failed"
                error_message = result.stderr
                self.results["failed_combinations"] += 1
                
        except subprocess.TimeoutExpired:
            self.log(f"⏰ [{combination_id}] {model} × {benchmark} 타임아웃 (1시간)")
            status = "timeout"
            error_message = "Timeout after 1 hour"
            execution_time = 3600
            self.results["failed_combinations"] += 1
            
        except Exception as e:
            self.log(f"💥 [{combination_id}] {model} × {benchmark} 오류: {str(e)}")
            status = "error"
            error_message = str(e)
            execution_time = time.time() - start_time
            self.results["failed_combinations"] += 1
        
        # 컨테이너 정리
        try:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        except:
            pass
        
        # 결과 기록 (정확도 정보 포함)
        evaluation_result = {
            "combination_id": combination_id,
            "model": model,
            "benchmark": benchmark,
            "container": container,
            "status": status,
            "execution_time": execution_time,
            "accuracy": accuracy,  # 정확도 정보 추가
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results["evaluations"].append(evaluation_result)
        
        # 중간 결과 저장
        self.save_results()
        
        return status == "success"
    
    def save_results(self):
        """결과 저장"""
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
    
    def generate_summary_report(self):
        """요약 리포트 생성"""
        successful_evaluations = [e for e in self.results["evaluations"] if e["status"] == "success" and e["accuracy"] is not None]
        
        if successful_evaluations:
            summary = {
                "total_evaluations": len(self.results["evaluations"]),
                "successful_evaluations": len(successful_evaluations),
                "average_accuracy": sum(e["accuracy"] for e in successful_evaluations) / len(successful_evaluations),
                "accuracy_by_model": {},
                "accuracy_by_benchmark": {}
            }
            
            # 모델별 평균 정확도
            model_accuracies = {}
            for eval_result in successful_evaluations:
                model = eval_result["model"]
                if model not in model_accuracies:
                    model_accuracies[model] = []
                model_accuracies[model].append(eval_result["accuracy"])
            
            for model, accuracies in model_accuracies.items():
                summary["accuracy_by_model"][model] = {
                    "average": sum(accuracies) / len(accuracies),
                    "count": len(accuracies),
                    "min": min(accuracies),
                    "max": max(accuracies)
                }
            
            # 벤치마크별 평균 정확도
            benchmark_accuracies = {}
            for eval_result in successful_evaluations:
                benchmark = eval_result["benchmark"]
                if benchmark not in benchmark_accuracies:
                    benchmark_accuracies[benchmark] = []
                benchmark_accuracies[benchmark].append(eval_result["accuracy"])
            
            for benchmark, accuracies in benchmark_accuracies.items():
                summary["accuracy_by_benchmark"][benchmark] = {
                    "average": sum(accuracies) / len(accuracies),
                    "count": len(accuracies),
                    "min": min(accuracies),
                    "max": max(accuracies)
                }
                
            # 요약 리포트 저장
            summary_file = self.results_file.parent / f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.log(f"📋 요약 리포트 저장: {summary_file}")
            self.log(f"📊 전체 평균 정확도: {summary['average_accuracy']:.4f}")
            
            return summary
        else:
            self.log("⚠️  성공한 평가가 없어서 요약 리포트를 생성할 수 없습니다.")
            return None
    
    def run_comprehensive_evaluation(self):
        """종합 평가 실행"""
        self.log("🎯 SpectraVision 종합 평가 시작 (수정 버전)")
        self.log("=" * 50)
        
        # 전체 조합 수 계산
        total_combinations = self.calculate_total_combinations()
        
        combination_id = 0
        
        # 각 transformer 버전별로 실행
        for group_name, group_data in MODEL_GROUPS.items():
            self.log(f"\n📋 {group_name} 그룹 시작 ({len(group_data['models'])} 모델)")
            container = group_data["container"]
            
            for model in group_data["models"]:
                self.log(f"\n🤖 모델: {model}")
                
                for benchmark in BENCHMARKS:
                    combination_id += 1
                    progress = f"{combination_id}/{total_combinations}"
                    self.log(f"   📊 [{progress}] {benchmark}")
                    
                    success = self.run_single_evaluation(container, model, benchmark, combination_id)
                    
                    if not success:
                        # 실패 시 짧은 대기 후 계속
                        time.sleep(5)
                    
                    # 각 평가 후 짧은 대기 (GPU 메모리 정리)
                    time.sleep(2)
                
                # 모델 완료 후 중간 대기
                self.log(f"✅ {model} 모든 벤치마크 완료")
                time.sleep(10)
        
        # 최종 결과 요약
        self.results["end_time"] = datetime.now().isoformat()
        self.results["total_time_seconds"] = (
            datetime.fromisoformat(self.results["end_time"]) - 
            datetime.fromisoformat(self.results["start_time"])
        ).total_seconds()
        
        self.save_results()
        
        # 요약 리포트 생성
        summary = self.generate_summary_report()
        
        self.log("\n" + "=" * 50)
        self.log("🎉 종합 평가 완료!")
        self.log(f"📊 전체 조합: {self.results['total_combinations']}")
        self.log(f"✅ 성공: {self.results['completed_combinations']}")
        self.log(f"❌ 실패: {self.results['failed_combinations']}")
        self.log(f"⏱️  총 소요 시간: {self.results['total_time_seconds']:.1f}초")
        self.log(f"📁 결과 파일: {self.results_file}")
        
        if summary:
            self.log(f"📊 전체 평균 정확도: {summary['average_accuracy']:.4f}")

if __name__ == "__main__":
    evaluator = ComprehensiveEvaluator()
    evaluator.run_comprehensive_evaluation()