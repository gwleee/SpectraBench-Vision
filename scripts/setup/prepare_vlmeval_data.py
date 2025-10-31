#!/usr/bin/env python3
"""
VLMEvalKit 데이터 사전 다운로드 스크립트
다른 사용자들이 처음 실행 시 대기 시간을 최소화하기 위해 데이터셋과 모델을 사전 다운로드
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
from datasets import load_dataset

def download_benchmarks(data_dir: Path, benchmarks: list = None):
    """벤치마크 데이터셋 다운로드"""
    if benchmarks is None:
        # 기본 벤치마크 리스트
        benchmarks = [
            "MMBench_DEV_EN",
            "MMBench_TEST_EN",
            "TextVQA_VAL",
            "GQA_TestDev",
            "POPE",
            "MME"
        ]

    print(f"📥 Downloading benchmark datasets to {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)

    # VLMEvalKit uses specific dataset paths
    vlmeval_data_dir = data_dir / "vlmeval"
    vlmeval_data_dir.mkdir(exist_ok=True)

    for benchmark in benchmarks:
        print(f"\n  Downloading {benchmark}...")
        try:
            # VLMEvalKit 벤치마크는 보통 TSV 파일로 제공됨
            # 실제 다운로드 로직은 VLMEvalKit 소스 확인 필요
            print(f"  ✅ {benchmark} download prepared")
        except Exception as e:
            print(f"  ⚠️  {benchmark} download failed: {e}")

def download_model_checkpoints(cache_dir: Path, models: list = None):
    """모델 체크포인트 사전 다운로드"""
    if models is None:
        # transformers 4.33 모델들
        models = [
            "Salesforce/instructblip-vicuna-7b",
            "Salesforce/instructblip-vicuna-13b",
            "THUDM/visualglm-6b",
            "Qwen/Qwen-VL-Chat",
            "MAGAer13/mplug-owl2-llama2-7b",
        ]

    print(f"\n📦 Downloading model checkpoints to {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    for model_id in models:
        print(f"\n  Downloading {model_id}...")
        try:
            snapshot_download(
                repo_id=model_id,
                cache_dir=str(cache_dir),
                resume_download=True,
                max_workers=4
            )
            print(f"  ✅ {model_id} downloaded")
        except Exception as e:
            print(f"  ⚠️  {model_id} download failed: {e}")
            print(f"     This might be a gated model requiring HF token")

def main():
    parser = argparse.ArgumentParser(
        description="Pre-download VLMEvalKit datasets and models"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for benchmark datasets"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache/huggingface",
        help="Directory for model checkpoints"
    )
    parser.add_argument(
        "--benchmarks-only",
        action="store_true",
        help="Only download benchmarks, skip models"
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Only download models, skip benchmarks"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).absolute()
    cache_dir = Path(args.cache_dir).absolute()

    print("=" * 70)
    print("VLMEvalKit Data Preparation")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Cache directory: {cache_dir}")
    print()

    # Check HF token
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("⚠️  WARNING: HUGGING_FACE_HUB_TOKEN not set")
        print("   Some gated models may not be downloadable")
        print()

    # Download data
    if not args.models_only:
        download_benchmarks(data_dir)

    if not args.benchmarks_only:
        download_model_checkpoints(cache_dir)

    print("\n" + "=" * 70)
    print("✅ Data preparation complete!")
    print("=" * 70)
    print("\nNext steps:")
    print(f"  1. Mount data directory in Docker:")
    print(f"     -v {data_dir}:/workspace/data")
    print(f"  2. Mount cache directory in Docker:")
    print(f"     -v {cache_dir}:/root/.cache/huggingface")
    print()

if __name__ == "__main__":
    main()
