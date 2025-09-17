#!/bin/bash

echo "🤖 SpectraVision 완전 자동화 스크립트 시작"
echo "============================================"

# 빌드 완료 대기 함수
wait_for_builds() {
    echo "⏳ Gemini 최적화 빌드 완료 대기 중..."
    while ps aux | grep -q "[d]ocker build.*spectravision"; do
        sleep 60
        echo "🔄 빌드 진행 중... ($(date))"
    done
    echo "✅ 모든 빌드 완료!"
}

# 이미지 테스트 함수
test_images() {
    echo "🧪 Docker 이미지 기능 테스트 시작..."
    for version in "4.33" "4.37" "4.43" "4.49" "4.51"; do
        echo "Testing spectravision-$version:latest"
        timeout 30 docker run --rm spectravision-$version:latest python -c "import transformers; print(f'✅ Transformers {transformers.__version__} ready!')" || echo "❌ Test failed for $version"
    done
}

# 통합 이미지 빌드 함수
build_integrated() {
    echo "🔧 통합 이미지 (spectrabench-vision:latest) 빌드..."
    docker build -t spectrabench-vision:latest -f docker/integrated/Dockerfile .
}

# 태깅 및 푸시 함수
tag_and_push() {
    echo "🏷️ GitHub Container Registry 태깅 및 푸시..."
    docker login ghcr.io -u gwleee -p $GITHUB_TOKEN
    
    for version in "4.33" "4.37" "4.43" "4.49" "4.51"; do
        docker tag spectravision-$version:latest ghcr.io/gwleee/spectravision-$version:latest
        docker push ghcr.io/gwleee/spectravision-$version:latest
    done
    
    docker tag spectrabench-vision:latest ghcr.io/gwleee/spectrabench-vision:latest
    docker push ghcr.io/gwleee/spectrabench-vision:latest
    
    echo "✅ 모든 이미지 푸시 완료!"
}

# 전체 평가 시작 함수
start_evaluation() {
    echo "🚀 30개 모델 × 24개 벤치마크 전체 평가 시작..."
    docker run --gpus all \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v $(pwd)/outputs:/workspace/outputs \
        -e HF_TOKEN \
        -e HUGGING_FACE_HUB_TOKEN \
        --name spectravision-full-evaluation \
        spectrabench-vision:latest \
        python3 scripts/docker_main.py --mode comprehensive
}

# 메인 실행 순서
wait_for_builds
test_images
build_integrated
tag_and_push
start_evaluation

echo "🎉 모든 자동화 작업 완료! ($(date))"
