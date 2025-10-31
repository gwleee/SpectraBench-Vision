#!/bin/bash
# Transformers 4.37 LLaVA-Only TEST MODE Evaluation
# Tests LLaVA 2 models × 20 benchmarks × 10 samples = 40 combinations

set -e

BENCHMARK_DIR="/home/gwlee/Benchmark/SpectraBench-Vision"
OUTPUTS_BASE="$BENCHMARK_DIR/outputs"
ENV_FILE="$BENCHMARK_DIR/.env"
IMAGE="ghcr.io/gwleee/spectravision:4.37"
DATA_DIR="$BENCHMARK_DIR/data"
CACHE_DIR="$BENCHMARK_DIR/cache"

# Create directories if they don't exist
mkdir -p "$DATA_DIR"
mkdir -p "$CACHE_DIR/huggingface"
mkdir -p "$CACHE_DIR/matplotlib"

# Create session directory with date and auto-incrementing number
DATE_PREFIX=$(date +%Y%m%d)
SESSION_NUM=001
while [ -d "$OUTPUTS_BASE/${DATE_PREFIX}_${SESSION_NUM}_llava" ]; do
    SESSION_NUM=$(printf "%03d" $((10#$SESSION_NUM + 1)))
done
SESSION_DIR="$OUTPUTS_BASE/${DATE_PREFIX}_${SESSION_NUM}_llava"
mkdir -p "$SESSION_DIR"
mkdir -p "$SESSION_DIR/logs"
mkdir -p "$SESSION_DIR/results"

# CRITICAL: Set 777 permissions recursively
chmod -R 777 "$SESSION_DIR" 2>/dev/null || true
chmod -R 777 "$CACHE_DIR" 2>/dev/null || true

echo "========================================="
echo "Session Directory: $SESSION_DIR"
echo "========================================="

# 2 LLaVA models only
MODELS=(
  "llava_v1.5_7b"
  "llava_v1.5_13b"
)

# 20 benchmarks
BENCHMARKS=(
  "MMBench_DEV_EN"
  "TextVQA_VAL"
  "GQA_TestDev_Balanced"
  "MMMU_DEV_VAL"
  "DocVQA_VAL"
  "ChartQA_TEST"
  "InfoVQA_VAL"
  "OCRBench"
  "AI2D_TEST"
  "ScienceQA_VAL"
  "POPE"
  "HallusionBench"
  "MMStar"
  "RealWorldQA"
  "VisOnlyQA-VLMEvalKit"
  "VizWiz"
  "SEEDBench_IMG"
  "BLINK"
  "MMBench_DEV_KO"
  "SEEDBench_IMG_KO"
)

# Create summary log
SUMMARY_LOG="$SESSION_DIR/logs/evaluation_summary.log"

echo "=========================================" | tee "$SUMMARY_LOG"
echo "LLaVA-Only TEST MODE Evaluation" | tee -a "$SUMMARY_LOG"
echo "⚡ FAST MODE: 10 samples per benchmark" | tee -a "$SUMMARY_LOG"
echo "🚀 OPTIMIZED: Persistent containers" | tee -a "$SUMMARY_LOG"
echo "✅ ERROR DETECTION: Detects VLMEvalKit failures" | tee -a "$SUMMARY_LOG"
echo "📦 MODELS: 2 LLaVA models (7B, 13B)" | tee -a "$SUMMARY_LOG"
echo "📊 BENCHMARKS: 20 benchmarks" | tee -a "$SUMMARY_LOG"
echo "=========================================" | tee -a "$SUMMARY_LOG"
echo "Session: ${DATE_PREFIX}_${SESSION_NUM}_llava" | tee -a "$SUMMARY_LOG"
echo "Models: ${#MODELS[@]}" | tee -a "$SUMMARY_LOG"
echo "Benchmarks: ${#BENCHMARKS[@]}" | tee -a "$SUMMARY_LOG"
echo "Total combinations: $((${#MODELS[@]} * ${#BENCHMARKS[@]}))" | tee -a "$SUMMARY_LOG"
echo "Samples per benchmark: 10 (test mode)" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# GPU monitoring function
monitor_gpu() {
    local log_file=$1
    echo "=== GPU Status ===" | tee -a "$log_file"
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | tee -a "$log_file"
    echo "" | tee -a "$log_file"
}

echo "[INFO] Initial GPU status:" | tee -a "$SUMMARY_LOG"
monitor_gpu "$SUMMARY_LOG"

# Initialize result tracking
declare -A results
declare -A model_success_count
declare -A model_fail_count
declare -A containers
total=$((${#MODELS[@]} * ${#BENCHMARKS[@]}))
global_current=0
global_success=0
global_fail=0

# Initialize per-model counters
for model in "${MODELS[@]}"; do
    model_success_count[$model]=0
    model_fail_count[$model]=0
done

# Cleanup function
cleanup() {
    echo "" | tee -a "$SUMMARY_LOG"
    echo "=========================================" | tee -a "$SUMMARY_LOG"
    echo "Cleaning up containers..." | tee -a "$SUMMARY_LOG"
    for model in "${!containers[@]}"; do
        container_id="${containers[$model]}"
        if [ ! -z "$container_id" ]; then
            echo "Stopping container for $model: $container_id" | tee -a "$SUMMARY_LOG"
            docker stop "$container_id" 2>&1 | tee -a "$SUMMARY_LOG" || true
        fi
    done
    echo "Cleanup complete." | tee -a "$SUMMARY_LOG"
    echo "=========================================" | tee -a "$SUMMARY_LOG"
}

trap cleanup EXIT

# Run evaluations grouped by model
for model in "${MODELS[@]}"; do
    MODEL_LOG="$SESSION_DIR/logs/${model}.log"

    echo "=========================================" | tee -a "$SUMMARY_LOG" | tee "$MODEL_LOG"
    echo "Starting evaluation for model: $model" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
    echo "Benchmarks to test: ${#BENCHMARKS[@]}" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
    echo "🚀 OPTIMIZATION: Starting persistent container..." | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
    echo "=========================================" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
    echo "" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"

    model_start_time=$(date +%s)

    # Start persistent container
    echo "Starting persistent container for $model..." | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
    container_id=$(docker run -d --name "spectravision_437_llava_${model}_$$" \
        --gpus all \
        --network=host \
        -v "$SESSION_DIR":/workspace/VLMEvalKit/outputs \
        -v "$DATA_DIR":/workspace/LMUData \
        -v "$ENV_FILE":/workspace/VLMEvalKit/.env \
        -v "$CACHE_DIR/huggingface":/tmp/.cache/huggingface \
        -v "$CACHE_DIR/matplotlib":/tmp/.cache/matplotlib \
        --env-file "$ENV_FILE" \
        -e VLMEVAL_SAMPLE_LIMIT=10 \
        -e LMUData=/workspace/LMUData \
        -e MPLCONFIGDIR=/tmp/.cache/matplotlib \
        -e HF_HOME=/tmp/.cache/huggingface \
        -e TRANSFORMERS_CACHE=/tmp/.cache/huggingface \
        "$IMAGE" \
        tail -f /dev/null)

    containers[$model]="$container_id"
    echo "✅ Container started: $container_id" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
    echo "" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"

    # Run all benchmarks in this container
    for benchmark in "${BENCHMARKS[@]}"; do
        global_current=$((global_current + 1))

        echo "=========================================" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
        echo "[${global_current}/${total}] $model on $benchmark (10 samples)" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
        echo "🔄 Running in container: $container_id" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
        echo "=========================================" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"

        # Create model/benchmark directory
        RESULT_DIR="$SESSION_DIR/$model/$benchmark"
        mkdir -p "$RESULT_DIR"

        # Monitor GPU
        monitor_gpu "$MODEL_LOG"

        # Run evaluation using docker exec
        TEMP_OUTPUT=$(mktemp)
        docker exec "$container_id" \
            python /workspace/VLMEvalKit/run.py --data "$benchmark" --model "$model" \
            2>&1 | tee -a "$MODEL_LOG" | tee "$TEMP_OUTPUT"

        exit_code=${PIPESTATUS[0]}

        # Check for VLMEvalKit ERROR messages
        if grep -q "ERROR.*combination failed" "$TEMP_OUTPUT"; then
            echo "[✗ FAILED] $model on $benchmark (VLMEvalKit error)" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
            results["${model}_${benchmark}"]="FAILED"
            model_fail_count[$model]=$((${model_fail_count[$model]} + 1))
            global_fail=$((global_fail + 1))
        elif [ $exit_code -eq 0 ]; then
            echo "[✓ SUCCESS] $model on $benchmark" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
            results["${model}_${benchmark}"]="SUCCESS"
            model_success_count[$model]=$((${model_success_count[$model]} + 1))
            global_success=$((global_success + 1))
        else
            echo "[✗ FAILED] $model on $benchmark (exit: $exit_code)" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
            results["${model}_${benchmark}"]="FAILED"
            model_fail_count[$model]=$((${model_fail_count[$model]} + 1))
            global_fail=$((global_fail + 1))
        fi

        rm -f "$TEMP_OUTPUT"
        echo "" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
    done

    # Stop container
    echo "Stopping container for $model: $container_id" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
    docker stop "$container_id" 2>&1 | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
    echo "" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"

    # Model completion summary
    model_end_time=$(date +%s)
    model_duration=$((model_end_time - model_start_time))

    echo "=========================================" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
    echo "Model $model completed!" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
    echo "Duration: $((model_duration / 60)) minutes" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
    echo "Successful: ${model_success_count[$model]}" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
    echo "Failed: ${model_fail_count[$model]}" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
    echo "Success rate: $(awk "BEGIN {printf \"%.1f\", (${model_success_count[$model]}/${#BENCHMARKS[@]})*100}")%" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
    echo "=========================================" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"
    echo "" | tee -a "$SUMMARY_LOG" | tee -a "$MODEL_LOG"

    # Generate per-model report (CSV and XLSX)
    echo "Generating results for $model..." | tee -a "$SUMMARY_LOG"
    python3 "$BENCHMARK_DIR/scripts/parse_evaluation_results.py" \
        "$SESSION_DIR" "$MODEL_LOG" "$model" \
        2>&1 | tee -a "$SUMMARY_LOG"
done

# Generate final report
FINAL_REPORT="$SESSION_DIR/EVALUATION_REPORT.md"
cat > "$FINAL_REPORT" << EOF
# LLaVA-Only Evaluation Report: ${DATE_PREFIX}_${SESSION_NUM}_llava

## Summary
- **Date**: $(date '+%Y-%m-%d %H:%M:%S')
- **Transformer Version**: 4.37
- **Mode**: TEST MODE (10 samples, LLaVA-only)
- **Models**: ${#MODELS[@]} (LLaVA-1.5-7B, LLaVA-1.5-13B)
- **Benchmarks**: ${#BENCHMARKS[@]}
- **Total Combinations**: $total
- **Global Successful**: $global_success
- **Global Failed**: $global_fail
- **Global Success Rate**: $(awk "BEGIN {printf \"%.1f\", ($global_success/$total)*100}")%

## Per-Model Results

| Model | Successful | Failed | Success Rate |
|-------|-----------|--------|--------------|
EOF

for model in "${MODELS[@]}"; do
    success=${model_success_count[$model]}
    failed=${model_fail_count[$model]}
    rate=$(awk "BEGIN {printf \"%.1f\", ($success/${#BENCHMARKS[@]})*100}")
    echo "| $model | $success/${#BENCHMARKS[@]} | $failed | ${rate}% |" >> "$FINAL_REPORT"
done

cat >> "$FINAL_REPORT" << EOF

## Benchmarks Tested
EOF

for benchmark in "${BENCHMARKS[@]}"; do
    echo "- $benchmark" >> "$FINAL_REPORT"
done

cat >> "$FINAL_REPORT" << EOF

## Detailed Results

| Model | Benchmark | Status |
|-------|-----------|--------|
EOF

for model in "${MODELS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
        status="${results["${model}_${benchmark}"]}"
        echo "| $model | $benchmark | $status |" >> "$FINAL_REPORT"
    done
done

cat >> "$FINAL_REPORT" << EOF

## GPU Information
$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)
EOF

echo '```' >> "$FINAL_REPORT"

# Final summary
echo "=========================================" | tee -a "$SUMMARY_LOG"
echo "LLaVA-Only TEST MODE COMPLETED!" | tee -a "$SUMMARY_LOG"
echo "=========================================" | tee -a "$SUMMARY_LOG"
echo "Total combinations: $total" | tee -a "$SUMMARY_LOG"
echo "Successful: $global_success" | tee -a "$SUMMARY_LOG"
echo "Failed: $global_fail" | tee -a "$SUMMARY_LOG"
echo "Success rate: $(awk "BEGIN {printf \"%.1f\", ($global_success/$total)*100}")%" | tee -a "$SUMMARY_LOG"
echo "=========================================" | tee -a "$SUMMARY_LOG"
monitor_gpu "$SUMMARY_LOG"

echo "" | tee -a "$SUMMARY_LOG"
echo "📊 Results location: $SESSION_DIR" | tee -a "$SUMMARY_LOG"
echo "📝 Summary log: $SUMMARY_LOG" | tee -a "$SUMMARY_LOG"
echo "📄 Final report: $FINAL_REPORT" | tee -a "$SUMMARY_LOG"
echo ""
echo "========================================="
echo "Evaluation Report Generated!"
echo "View at: $FINAL_REPORT"
echo "========================================="
