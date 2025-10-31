#!/usr/bin/env python3
"""
Generate final comprehensive report from all model results.

This script combines all per-model result files into a single comprehensive report.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd


def combine_model_results(session_dir: Path) -> pd.DataFrame:
    """Combine all model result CSVs into one DataFrame."""
    results_dir = session_dir / "results"

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return pd.DataFrame()

    all_dfs = []

    # Find all model result CSV files
    for csv_file in sorted(results_dir.glob("*_results.csv")):
        print(f"Loading: {csv_file.name}")
        df = pd.read_csv(csv_file)
        all_dfs.append(df)

    if not all_dfs:
        print("Warning: No result CSV files found")
        return pd.DataFrame()

    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df


def generate_summary_statistics(df: pd.DataFrame) -> Dict:
    """Generate summary statistics from combined results."""
    stats = {
        'total_tests': len(df),
        'models': sorted(df['Model'].unique().tolist()),
        'benchmarks': sorted(df['Benchmark'].unique().tolist()),
        'success_count': len(df[df['Status'] == 'SUCCESS']),
        'fail_count': len(df[df['Status'] == 'FAILED']),
    }

    stats['success_rate'] = (stats['success_count'] / stats['total_tests'] * 100) if stats['total_tests'] > 0 else 0

    # Per-model statistics
    stats['per_model'] = {}
    for model in stats['models']:
        model_df = df[df['Model'] == model]
        success = len(model_df[model_df['Status'] == 'SUCCESS'])
        total = len(model_df)
        stats['per_model'][model] = {
            'total': total,
            'success': success,
            'failed': total - success,
            'success_rate': (success / total * 100) if total > 0 else 0,
            'avg_score': model_df['Overall_Score'].mean() if 'Overall_Score' in model_df.columns else None
        }

    # Per-benchmark statistics
    stats['per_benchmark'] = {}
    for benchmark in stats['benchmarks']:
        bench_df = df[df['Benchmark'] == benchmark]
        success = len(bench_df[bench_df['Status'] == 'SUCCESS'])
        total = len(bench_df)
        stats['per_benchmark'][benchmark] = {
            'total': total,
            'success': success,
            'failed': total - success,
            'success_rate': (success / total * 100) if total > 0 else 0,
            'avg_score': bench_df['Overall_Score'].mean() if 'Overall_Score' in bench_df.columns else None
        }

    return stats


def create_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a pivot table with models as rows and benchmarks as columns."""
    if 'Overall_Score' not in df.columns:
        return pd.DataFrame()

    pivot = df.pivot_table(
        values='Overall_Score',
        index='Model',
        columns='Benchmark',
        aggfunc='mean'
    )

    # Add average column
    pivot['Average'] = pivot.mean(axis=1)

    return pivot


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python generate_final_report.py <session_directory>")
        print("\nExample:")
        print("  python generate_final_report.py outputs/20251017_001")
        sys.exit(1)

    session_dir = Path(sys.argv[1])

    if not session_dir.exists():
        print(f"Error: Directory not found: {session_dir}")
        sys.exit(1)

    print(f"Generating final report for: {session_dir}")
    print("="*60)

    # Combine all model results
    combined_df = combine_model_results(session_dir)

    if combined_df.empty:
        print("No results to process")
        return

    # Save combined results
    results_dir = session_dir / "results"
    results_dir.mkdir(exist_ok=True)

    combined_csv = results_dir / "FINAL_COMBINED_RESULTS.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"✅ Saved combined CSV: {combined_csv}")

    try:
        combined_excel = results_dir / "FINAL_COMBINED_RESULTS.xlsx"
        combined_df.to_excel(combined_excel, index=False, engine='openpyxl')
        print(f"✅ Saved combined Excel: {combined_excel}")
    except ImportError:
        print("⚠️  openpyxl not available, skipping combined Excel export")

    # Generate summary statistics
    stats = generate_summary_statistics(combined_df)

    # Save statistics
    stats_file = results_dir / "FINAL_STATISTICS.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Saved statistics: {stats_file}")

    # Create pivot table
    pivot_df = create_pivot_table(combined_df)
    if not pivot_df.empty:
        pivot_csv = results_dir / "FINAL_PIVOT_TABLE.csv"
        pivot_df.to_csv(pivot_csv)
        print(f"✅ Saved pivot table: {pivot_csv}")

        try:
            pivot_excel = results_dir / "FINAL_PIVOT_TABLE.xlsx"
            pivot_df.to_excel(pivot_excel, engine='openpyxl')
            print(f"✅ Saved pivot Excel: {pivot_excel}")
        except ImportError:
            pass

    # Print summary
    print("\n" + "="*60)
    print("FINAL EVALUATION SUMMARY")
    print("="*60)
    print(f"Total tests: {stats['total_tests']}")
    print(f"Models: {len(stats['models'])}")
    print(f"Benchmarks: {len(stats['benchmarks'])}")
    print(f"Successful: {stats['success_count']}")
    print(f"Failed: {stats['fail_count']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")

    print("\n" + "-"*60)
    print("Per-Model Summary:")
    print("-"*60)
    for model, model_stats in stats['per_model'].items():
        avg_score = model_stats['avg_score']
        avg_str = f"{avg_score:.2f}" if avg_score and not pd.isna(avg_score) else "N/A"
        print(f"  {model}:")
        print(f"    Success: {model_stats['success']}/{model_stats['total']} ({model_stats['success_rate']:.1f}%)")
        print(f"    Avg Score: {avg_str}")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
