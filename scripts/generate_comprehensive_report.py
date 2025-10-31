#!/usr/bin/env python3
"""
Generate comprehensive evaluation report combining all models and benchmarks.
"""

import sys
from pathlib import Path
import pandas as pd


def generate_comprehensive_report(session_dir: Path):
    """Generate comprehensive report from all model result files."""
    session_dir = Path(session_dir)
    results_dir = session_dir / 'results'

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Find all model result CSV files
    csv_files = sorted(results_dir.glob('*_results.csv'))

    if not csv_files:
        print(f"Error: No result CSV files found in {results_dir}")
        return

    print(f"Found {len(csv_files)} model result files")

    # Load all CSVs
    all_data = []
    for csv_file in csv_files:
        print(f"Loading {csv_file.name}")
        df = pd.read_csv(csv_file)
        all_data.append(df)

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Create pivot table: Models as rows, Benchmarks as columns
    pivot_df = combined_df.pivot_table(
        index='Model',
        columns='Benchmark',
        values='Overall_Score',
        aggfunc='first'
    )

    # Add average score per model
    pivot_df['Average'] = pivot_df.mean(axis=1, skipna=True)

    # Add average score per benchmark (at bottom)
    benchmark_avg = pivot_df.mean(axis=0, skipna=True)
    benchmark_avg.name = 'Average'
    pivot_df = pd.concat([pivot_df, benchmark_avg.to_frame().T])

    # Round all values to 2 decimal places
    pivot_df = pivot_df.round(2)

    # Save comprehensive report
    comprehensive_csv = results_dir / 'COMPREHENSIVE_REPORT.csv'
    pivot_df.to_csv(comprehensive_csv)
    print(f"✅ Saved comprehensive CSV: {comprehensive_csv}")

    try:
        comprehensive_excel = results_dir / 'COMPREHENSIVE_REPORT.xlsx'
        pivot_df.to_excel(comprehensive_excel, engine='openpyxl')
        print(f"✅ Saved comprehensive Excel: {comprehensive_excel}")
    except ImportError:
        print("⚠️  openpyxl not available, skipping Excel export")

    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*80)
    print(f"Total models: {len(csv_files)}")
    print(f"Total benchmarks: {len(pivot_df.columns) - 1}")  # -1 for Average column
    print(f"Total evaluations: {len(combined_df)}")

    print("\n" + "-"*80)
    print("Model Performance (Average Scores):")
    print("-"*80)
    for model in pivot_df.index[:-1]:  # Exclude the "Average" row
        avg_score = pivot_df.loc[model, 'Average']
        if pd.notna(avg_score):
            print(f"  {model:25s}: {avg_score:6.2f}")
        else:
            print(f"  {model:25s}: N/A")

    print("\n" + "-"*80)
    print("Benchmark Difficulty (Average Scores across all models):")
    print("-"*80)
    for benchmark in pivot_df.columns[:-1]:  # Exclude the "Average" column
        avg_score = pivot_df.loc['Average', benchmark]
        if pd.notna(avg_score):
            print(f"  {benchmark:30s}: {avg_score:6.2f}")
        else:
            print(f"  {benchmark:30s}: N/A")

    # Create detailed combined report (all data)
    combined_csv = results_dir / 'COMBINED_ALL_RESULTS.csv'
    combined_df.to_csv(combined_csv, index=False)
    print(f"\n✅ Saved combined detailed results: {combined_csv}")

    try:
        combined_excel = results_dir / 'COMBINED_ALL_RESULTS.xlsx'
        combined_df.to_excel(combined_excel, index=False, engine='openpyxl')
        print(f"✅ Saved combined detailed Excel: {combined_excel}")
    except ImportError:
        pass

    print("\n" + "="*80)
    print(f"Comprehensive reports saved to: {results_dir}")
    print("="*80)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python generate_comprehensive_report.py <session_directory>")
        print("\nExample:")
        print("  python generate_comprehensive_report.py outputs/20251021_003")
        sys.exit(1)

    session_dir = Path(sys.argv[1])

    if not session_dir.exists():
        print(f"Error: Directory not found: {session_dir}")
        sys.exit(1)

    generate_comprehensive_report(session_dir)


if __name__ == '__main__':
    main()
