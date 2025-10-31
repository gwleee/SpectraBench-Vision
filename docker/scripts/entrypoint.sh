#!/bin/bash
# Entrypoint script for SpectraBench-Vision Docker containers
# Automatically runs VLMEvalKit evaluation and generates parsed results
# Usage: docker run --gpus all -v $(pwd)/outputs:/workspace/outputs IMAGE [--data BENCHMARK --model MODEL]

set -e

# Default values
OUTPUT_DIR="/workspace/VLMEvalKit/outputs"
LOG_DIR="$OUTPUT_DIR/logs"
RESULTS_DIR="$OUTPUT_DIR/results"
TRANSFORMER_VERSION="${TRANSFORMERS_VERSION:-unknown}"

# Create output directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Parse arguments to detect model and benchmark
MODEL=""
BENCHMARK=""
ARGS=("$@")

# Extract model and benchmark from arguments
for i in "${!ARGS[@]}"; do
    if [[ "${ARGS[i]}" == "--model" && -n "${ARGS[i+1]}" ]]; then
        MODEL="${ARGS[i+1]}"
    elif [[ "${ARGS[i]}" == "--data" && -n "${ARGS[i+1]}" ]]; then
        BENCHMARK="${ARGS[i+1]}"
    fi
done

# Generate log file name
if [ -n "$MODEL" ] && [ -n "$BENCHMARK" ]; then
    LOG_FILE="$LOG_DIR/${MODEL}_${BENCHMARK}_$(date +%Y%m%d_%H%M%S).log"
    RESULT_NAME="${MODEL}_${BENCHMARK}"
else
    LOG_FILE="$LOG_DIR/evaluation_$(date +%Y%m%d_%H%M%S).log"
    RESULT_NAME="evaluation"
fi

echo "========================================="
echo "SpectraBench-Vision Entrypoint"
echo "========================================="
echo "Transformer Version: $TRANSFORMER_VERSION"
echo "Model: ${MODEL:-not specified}"
echo "Benchmark: ${BENCHMARK:-not specified}"
echo "Output Directory: $OUTPUT_DIR"
echo "Log File: $LOG_FILE"
echo "========================================="
echo ""

# Check if arguments are provided
if [ $# -eq 0 ]; then
    echo "No arguments provided. Starting default shell."
    echo ""
    echo "Usage examples:"
    echo "  docker run --gpus all IMAGE --data MMBench --model InternVL2-2B"
    echo "  docker run --gpus all IMAGE python /workspace/VLMEvalKit/run.py --data TextVQA --model llava_v1.5_7b"
    echo ""
    exec /bin/bash
fi

# Run VLMEvalKit evaluation with all arguments
echo "Running VLMEvalKit evaluation..."
echo "Command: python /workspace/VLMEvalKit/run.py $@"
echo ""

# Execute evaluation and capture output
python /workspace/VLMEvalKit/run.py "$@" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================="
echo "Evaluation completed with exit code: $EXIT_CODE"
echo "========================================="
echo ""

# Parse results if we have model and benchmark information
if [ -n "$MODEL" ] && [ -n "$BENCHMARK" ] && [ -f "$LOG_FILE" ]; then
    echo "Parsing evaluation results..."

    # Check if parse script exists in container
    PARSE_SCRIPT="/workspace/scripts/parse_evaluation_results.py"
    if [ -f "$PARSE_SCRIPT" ]; then
        python3 "$PARSE_SCRIPT" "$OUTPUT_DIR" "$LOG_FILE" "$MODEL" 2>&1 || {
            echo "Warning: Result parsing failed, but evaluation log is available at: $LOG_FILE"
        }

        echo ""
        echo "========================================="
        echo "Results Summary"
        echo "========================================="
        echo "Log file: $LOG_FILE"
        echo "Results location: $RESULTS_DIR"

        # List generated result files
        if [ -d "$RESULTS_DIR" ]; then
            RESULT_FILES=$(find "$RESULTS_DIR" -name "${MODEL}*" -type f 2>/dev/null || true)
            if [ -n "$RESULT_FILES" ]; then
                echo ""
                echo "Generated files:"
                echo "$RESULT_FILES" | while read -r file; do
                    echo "  - $file"
                done
            fi
        fi
    else
        echo "Warning: Parse script not found at $PARSE_SCRIPT"
        echo "Raw evaluation log saved to: $LOG_FILE"
    fi
else
    echo "Skipping result parsing (model or benchmark not specified)"
    echo "Raw output saved to: $LOG_FILE"
fi

echo ""
echo "========================================="
echo "Entrypoint completed"
echo "========================================="

exit $EXIT_CODE
