#!/usr/bin/env python3
"""
Parse VLMEvalKit evaluation logs and generate structured result files.

This script extracts evaluation results from log files and creates:
- CSV file with detailed results
- Summary report with statistics
"""

import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


def parse_log_file(log_path: Path) -> List[Dict]:
    """Parse evaluation log and extract results."""
    results = []
    current_test = {}

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Match test start: [1/192] qwen_chat on MMBench_DEV_EN (10 samples)
        # Also matches: [1/192] Testing qwen_chat on MMBench_DEV_EN (10 samples)
        test_match = re.match(r'\[(\d+)/(\d+)\] (?:Testing )?(\S+) on (\S+) \((\d+) samples\)', line)
        if test_match:
            current_test = {
                'index': int(test_match.group(1)),
                'total': int(test_match.group(2)),
                'model': test_match.group(3),
                'benchmark': test_match.group(4),
                'samples': int(test_match.group(5)),
                'status': 'unknown',
                'metrics': {}
            }

        # Match sample limit confirmation
        if '[SAMPLE LIMIT]' in line:
            sample_match = re.search(r"limited from (\d+) to (\d+) samples", line)
            if sample_match and current_test:
                current_test['original_size'] = int(sample_match.group(1))
                current_test['actual_samples'] = int(sample_match.group(2))

        # Match evaluation completion
        if 'The evaluation of model' in line and 'has finished!' in line:
            if current_test:
                current_test['status'] = 'completed'

        # Match SUCCESS/FAILED status
        # Matches both: "[✓ SUCCESS] qwen_chat on MMBench_DEV_EN" and "[✓ SUCCESS] Completed ..."
        success_match = re.match(r'\[✓ SUCCESS\] (?:Completed )?(\S+) on (\S+)', line)
        if success_match and current_test:
            current_test['status'] = 'SUCCESS'

        # Matches both: "[✗ FAILED] qwen_chat on MMBench_DEV_EN" and "[✗ FAILED] Failed ..."
        failed_match = re.match(r'\[✗ FAILED\] (?:Failed )?(\S+) on (\S+)', line)
        if failed_match and current_test:
            current_test['status'] = 'FAILED'
            error_match = re.search(r'exit code: (\d+)', line)
            if error_match:
                current_test['exit_code'] = int(error_match.group(1))

        # Match evaluation results table
        if 'Evaluation Results:' in line:
            # Skip to the actual results (usually 2-3 lines after due to INFO lines and empty lines)
            j = i + 1
            # First skip the timestamp line if present
            while j < len(lines) and ('INFO' in lines[j] or not lines[j].strip()):
                j += 1

            # Check if this is JSON format (OCRBench, etc.)
            json_lines = []
            json_start = j
            # Collect potential JSON lines
            while j < len(lines) and j < i + 50:
                result_line = lines[j].strip()
                if result_line.startswith('{'):
                    # Start collecting JSON
                    json_lines = [result_line]
                    j += 1
                    while j < len(lines) and not result_line.endswith('}'):
                        result_line = lines[j].strip()
                        json_lines.append(result_line)
                        j += 1
                        if result_line.endswith('}'):
                            break
                    # Try to parse JSON
                    try:
                        json_str = ' '.join(json_lines)
                        json_data = json.loads(json_str)
                        # Extract metrics from JSON
                        for key, value in json_data.items():
                            if isinstance(value, (int, float)):
                                # Map "Final Score Norm" to "Overall" for OCRBench
                                if key == "Final Score Norm":
                                    current_test['metrics']['Overall'] = float(value)
                                else:
                                    current_test['metrics'][key.replace(' ', '_')] = float(value)
                        break
                    except (json.JSONDecodeError, ValueError):
                        # Not valid JSON, fall back to table parsing
                        j = json_start
                        break
                j += 1

            # If no JSON found, parse as table
            j = json_start
            while j < len(lines) and j < i + 50:  # Look ahead up to 50 lines
                result_line = lines[j].strip()

                # Skip empty lines and separators
                if not result_line or result_line.startswith('-'):
                    j += 1
                    continue

                # Parse table-like results
                # Format 1: "Overall    0.9" or "Overall    50" (with label)
                overall_match = re.match(r'Overall\s+([\d.]+)', result_line)
                if overall_match:
                    current_test['metrics']['Overall'] = float(overall_match.group(1))

                # Format 2: "0  51" (just number without label - TextVQA format)
                # This appears after a numeric index like "0  51"
                number_only_match = re.match(r'^(\d+)\s+([\d.]+)$', result_line)
                if number_only_match and 'Overall' not in current_test['metrics']:
                    # This is likely the overall score without label
                    current_test['metrics']['Overall'] = float(number_only_match.group(2))

                # Format 3: Other metrics like "AR  0.9" or "choose  100"
                metric_match = re.match(r'([A-Za-z_]+)\s+([\d.]+)', result_line)
                if metric_match and result_line[0].isalpha():
                    metric_name = metric_match.group(1)
                    metric_value = float(metric_match.group(2))
                    # Skip header-like words
                    if metric_name not in ['split', 'validation', 'dev', 'test']:
                        current_test['metrics'][metric_name] = metric_value

                # Stop when we hit a separator or next section
                if result_line.startswith('===') or result_line.startswith('['):
                    break

                j += 1

        # When we find a SUCCESS/FAILED line, save the current test
        if (success_match or failed_match) and current_test and current_test.get('model'):
            # Special handling for HallusionBench: calculate Overall from aAcc, fAcc, qAcc
            if current_test.get('benchmark') == 'HallusionBench' and 'Overall' not in current_test['metrics']:
                metrics = current_test['metrics']
                if 'aAcc' in metrics and 'fAcc' in metrics and 'qAcc' in metrics:
                    overall = (metrics['aAcc'] + metrics['fAcc'] + metrics['qAcc']) / 3.0
                    current_test['metrics']['Overall'] = overall

            results.append(current_test.copy())
            current_test = {}

        i += 1

    return results


def create_summary_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Create a summary DataFrame from parsed results."""
    rows = []

    for result in results:
        row = {
            'Index': result.get('index', ''),
            'Model': result.get('model', ''),
            'Benchmark': result.get('benchmark', ''),
            'Status': result.get('status', 'unknown'),
            'Samples': result.get('samples', ''),
            'Overall_Score': result.get('metrics', {}).get('Overall', None),
        }

        # Add additional metrics as separate columns
        for metric, value in result.get('metrics', {}).items():
            if metric != 'Overall':
                row[f'Metric_{metric}'] = value

        rows.append(row)

    return pd.DataFrame(rows)


def generate_reports(session_dir: Path):
    """Generate result reports for a session directory."""
    log_file = session_dir / 'logs' / 'evaluation.log'

    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        return

    print(f"Parsing log file: {log_file}")
    results = parse_log_file(log_file)

    if not results:
        print("Warning: No results found in log file")
        return

    print(f"Found {len(results)} evaluation results")

    # Create DataFrame
    df = create_summary_dataframe(results)

    # Save as CSV
    csv_path = session_dir / 'results_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV: {csv_path}")

    # Save as Excel if openpyxl is available
    try:
        excel_path = session_dir / 'results_summary.xlsx'
        df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"✅ Saved Excel: {excel_path}")
    except ImportError:
        print("⚠️  openpyxl not available, skipping Excel export")

    # Generate statistics
    stats = {
        'total_tests': len(results),
        'successful': sum(1 for r in results if r['status'] == 'SUCCESS'),
        'failed': sum(1 for r in results if r['status'] == 'FAILED'),
        'success_rate': sum(1 for r in results if r['status'] == 'SUCCESS') / len(results) * 100 if results else 0,
        'models': sorted(set(r['model'] for r in results)),
        'benchmarks': sorted(set(r['benchmark'] for r in results)),
    }

    # Calculate average scores per model
    model_scores = {}
    for result in results:
        model = result['model']
        overall = result.get('metrics', {}).get('Overall')
        if overall is not None:
            if model not in model_scores:
                model_scores[model] = []
            model_scores[model].append(overall)

    stats['model_averages'] = {
        model: sum(scores) / len(scores)
        for model, scores in model_scores.items()
    }

    # Save statistics as JSON
    stats_path = session_dir / 'results_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Saved statistics: {stats_path}")

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total tests: {stats['total_tests']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    print(f"\nModels tested: {len(stats['models'])}")
    for model in stats['models']:
        avg = stats['model_averages'].get(model, 'N/A')
        if isinstance(avg, float):
            print(f"  - {model}: {avg:.2f}")
        else:
            print(f"  - {model}: {avg}")
    print(f"\nBenchmarks tested: {len(stats['benchmarks'])}")
    for benchmark in stats['benchmarks'][:10]:  # Show first 10
        print(f"  - {benchmark}")
    if len(stats['benchmarks']) > 10:
        print(f"  ... and {len(stats['benchmarks']) - 10} more")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python parse_evaluation_results.py <session_directory> [log_file] [model_name]")
        print("\nExamples:")
        print("  python parse_evaluation_results.py outputs/20251017_001")
        print("  python parse_evaluation_results.py outputs/20251017_001 outputs/20251017_001/logs/qwen_chat.log qwen_chat")
        sys.exit(1)

    session_dir = Path(sys.argv[1])

    if not session_dir.exists():
        print(f"Error: Directory not found: {session_dir}")
        sys.exit(1)

    # Check if log file and model name are provided (v4 style)
    if len(sys.argv) >= 3:
        log_file = Path(sys.argv[2])
        model_name = sys.argv[3] if len(sys.argv) >= 4 else None

        if not log_file.exists():
            print(f"Error: Log file not found: {log_file}")
            sys.exit(1)

        # Parse the specific model log file
        print(f"Parsing model-specific log: {log_file}")
        results = parse_log_file(log_file)

        if not results:
            print("Warning: No results found in log file")
            return

        print(f"Found {len(results)} evaluation results")

        # Create DataFrame
        df = create_summary_dataframe(results)

        # Save per-model results
        results_dir = session_dir / 'results'
        results_dir.mkdir(exist_ok=True)

        if model_name:
            csv_path = results_dir / f'{model_name}_results.csv'
            df.to_csv(csv_path, index=False)
            print(f"✅ Saved CSV: {csv_path}")

            try:
                excel_path = results_dir / f'{model_name}_results.xlsx'
                df.to_excel(excel_path, index=False, engine='openpyxl')
                print(f"✅ Saved Excel: {excel_path}")
            except ImportError:
                print("⚠️  openpyxl not available, skipping Excel export")
        else:
            csv_path = session_dir / 'results_summary.csv'
            df.to_csv(csv_path, index=False)
            print(f"✅ Saved CSV: {csv_path}")
    else:
        # Original behavior for backward compatibility
        generate_reports(session_dir)


if __name__ == '__main__':
    main()
