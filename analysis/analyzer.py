"""
Performance Analyzer for SpectraVision
Analyzes evaluation results, identifies bottlenecks, and generates insights
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BottleneckAnalysis:
    """Container for bottleneck analysis results"""
    bottleneck_type: str  # "memory", "time", "failure"
    affected_combinations: List[str]
    severity: str  # "low", "medium", "high", "critical"
    description: str
    recommendations: List[str]

@dataclass
class PerformanceInsights:
    """Container for performance insights"""
    total_evaluations: int
    success_rate: float
    total_time_hours: float
    average_time_per_eval: float
    peak_memory_usage: float
    bottlenecks: List[BottleneckAnalysis]
    model_rankings: Dict[str, float]
    benchmark_difficulty: Dict[str, float]

class PerformanceAnalyzer:
    """Comprehensive performance analysis engine"""
    
    def __init__(self, results_dir: str):
        """
        Initialize Performance Analyzer
        
        Args:
            results_dir: Directory containing evaluation results
        """
        self.results_dir = Path(results_dir)
        
        # Handle relative paths from different locations
        if not self.results_dir.is_absolute():
            # If running from analysis/ directory, adjust path
            current_dir = Path.cwd()
            if current_dir.name == "analysis":
                # We're in analysis directory, go up one level
                self.results_dir = current_dir.parent / results_dir
            else:
                # We're in root directory
                self.results_dir = current_dir / results_dir
        
        self.results_dir = self.results_dir.resolve()
        
        # Try multiple potential file locations
        self.potential_results_files = [
            self.results_dir / "results" / "final_results.json",
            self.results_dir / "final_results.json",
            self.results_dir / "outputs" / "results" / "final_results.json",
            # Additional common locations
            Path("outputs/results/final_results.json"),
            Path("../outputs/results/final_results.json"),
            Path("../../outputs/results/final_results.json")
        ]
        self.monitoring_file = self.results_dir / "reports" / "performance_monitoring.json"
        
        # Load data
        self.results_data = self._load_results()
        self.monitoring_data = self._load_monitoring_data()
        
        # Analysis results
        self.insights = None
        
        logger.info(f"Performance Analyzer initialized")
        logger.info(f"   Working directory: {Path.cwd()}")
        logger.info(f"   Results directory: {self.results_dir}")
        if self.results_data:
            logger.info(f"   Results data loaded: {len(self.results_data.get('evaluations', []))} evaluations")
        logger.info(f"   Monitoring available: {self.monitoring_data is not None}")
    
    def _load_results(self) -> Optional[Dict[str, Any]]:
        """Load evaluation results from JSON file"""
        # First try to find the results file
        results_file = None
        for potential_file in self.potential_results_files:
            if potential_file.exists():
                results_file = potential_file
                break
        
        if not results_file:
            logger.warning(f"Results file not found in any of these locations:")
            for pf in self.potential_results_files:
                logger.warning(f"  {pf}")
            return None
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded results from: {results_file}")
            logger.info(f"Found {len(data.get('evaluations', []))} evaluations")
            return data
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to load results: {e}")
            return None
    
    def _load_monitoring_data(self) -> Optional[Dict[str, Any]]:
        """Load monitoring data from JSON file"""
        if not self.monitoring_file.exists():
            logger.warning(f"Monitoring file not found: {self.monitoring_file}")
            return None
        
        try:
            with open(self.monitoring_file, 'r') as f:
                data = json.load(f)
            logger.info("Loaded monitoring data")
            return data
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to load monitoring data: {e}")
            return None
    
    def load_from_json_string(self, json_data: Dict[str, Any]):
        """Load data from provided JSON dictionary"""
        self.results_data = json_data
        logger.info(f"Loaded data from provided JSON")
        logger.info(f"Total evaluations: {len(self.results_data.get('evaluations', []))}")
    
    def generate_accuracy_matrix(self) -> pd.DataFrame:
        """
        Generate model vs benchmark accuracy comparison matrix
        
        Returns:
            DataFrame with models as rows, benchmarks as columns, accuracy as values
        """
        if not self.results_data:
            return pd.DataFrame()
        
        evaluations = self.results_data.get('evaluations', [])
        
        # Get all models and benchmarks from configuration or evaluations
        all_models = set()
        all_benchmarks = set()
        
        # First, get all models and benchmarks from evaluations
        for eval in evaluations:
            all_models.add(eval['model_name'])
            all_benchmarks.add(eval['benchmark_name'])
        
        # Also try to get from config if available
        config = self.results_data.get('config', {})
        if config.get('models'):
            for model in config['models']:
                all_models.add(model['name'])
        if config.get('benchmarks'):
            for benchmark in config['benchmarks']:
                all_benchmarks.add(benchmark['name'])
        
        logger.info(f"Found {len(all_models)} models and {len(all_benchmarks)} benchmarks")
        logger.info(f"Models: {sorted(all_models)}")
        logger.info(f"Benchmarks: {sorted(all_benchmarks)}")
        
        # Create accuracy data structure - include all combinations
        accuracy_data = []
        eval_results = {}
        
        # First, organize all evaluation results
        for eval in evaluations:
            key = (eval['model_name'], eval['benchmark_name'])
            eval_results[key] = eval
        
        # Create data for all model-benchmark combinations
        for model in all_models:
            for benchmark in all_benchmarks:
                key = (model, benchmark)
                if key in eval_results:
                    eval = eval_results[key]
                    accuracy = None
                    
                    # Try to extract accuracy score
                    if eval.get('success', False) and eval.get('accuracy_score') is not None:
                        accuracy = eval['accuracy_score']
                        # Convert to percentage if needed, but handle special cases
                        if benchmark not in ['MME'] and accuracy <= 1.0:
                            accuracy = accuracy * 100
                        elif benchmark == 'MME':
                            # MME has special scoring, keep as is or convert to percentage differently
                            pass
                    
                    accuracy_data.append({
                        'model': model,
                        'benchmark': benchmark,
                        'accuracy': accuracy,
                        'success': eval.get('success', False),
                        'error': eval.get('error_message', None) if not eval.get('success', False) else None
                    })
                else:
                    # Missing evaluation - not run yet
                    accuracy_data.append({
                        'model': model,
                        'benchmark': benchmark,
                        'accuracy': None,
                        'success': False,
                        'error': 'Not evaluated'
                    })
        
        if not accuracy_data:
            logger.warning("No evaluation data found")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(accuracy_data)
        
        # Create accuracy matrix (only include rows with at least some accuracy data)
        accuracy_df = df[df['accuracy'].notna()]
        if accuracy_df.empty:
            logger.warning("No accuracy scores found")
            # Create empty matrix with all models and benchmarks
            models_list = sorted(all_models)
            benchmarks_list = sorted(all_benchmarks)
            empty_matrix = pd.DataFrame(index=models_list, columns=benchmarks_list)
            empty_matrix['Average'] = None
            return empty_matrix
        
        # Pivot to create accuracy matrix
        accuracy_matrix = accuracy_df.pivot(index='model', columns='benchmark', values='accuracy')
        
        # Add missing models/benchmarks as NaN
        for model in all_models:
            if model not in accuracy_matrix.index:
                accuracy_matrix.loc[model] = None
        
        for benchmark in all_benchmarks:
            if benchmark not in accuracy_matrix.columns:
                accuracy_matrix[benchmark] = None
        
        # Sort for consistent output
        accuracy_matrix = accuracy_matrix.sort_index()
        accuracy_matrix = accuracy_matrix.reindex(sorted(accuracy_matrix.columns), axis=1)
        
        # Add average column (exclude special benchmarks like MME from average)
        special_benchmarks = ['MME']  # Benchmarks to exclude from average
        regular_benchmarks = [col for col in accuracy_matrix.columns if col not in special_benchmarks]
        
        if regular_benchmarks:
            accuracy_matrix['Average'] = accuracy_matrix[regular_benchmarks].mean(axis=1, skipna=True)
        else:
            accuracy_matrix['Average'] = None
        
        # Add average row for regular benchmarks only
        if regular_benchmarks:
            avg_row = accuracy_matrix[regular_benchmarks].mean(axis=0, skipna=True)
            # Add average for the Average column
            avg_row['Average'] = accuracy_matrix['Average'].mean(skipna=True)
            accuracy_matrix.loc['Average'] = avg_row
        
        return accuracy_matrix

    def generate_accuracy_report(self) -> str:
        """
        Generate detailed accuracy comparison report
        
        Returns:
            Formatted accuracy report string
        """
        accuracy_matrix = self.generate_accuracy_matrix()
        
        if accuracy_matrix.empty:
            return "No accuracy data available for comparison."
        
        report_lines = [
            "=" * 120,
            "ACCURACY COMPARISON MATRIX",
            "=" * 120,
            "",
            "Model Performance Across Benchmarks (Accuracy %)",
            "",
        ]
        
        # Format the matrix for display
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 15)
        
        # Convert to formatted strings
        formatted_matrix = accuracy_matrix.round(1)
        
        # Create header
        benchmarks = [col for col in formatted_matrix.columns if col != 'Average']
        header = f"{'Model':<20}"
        for benchmark in benchmarks:
            header += f"{benchmark:<12}"
        header += f"{'Average':<12}"
        
        report_lines.append(header)
        report_lines.append("-" * len(header))
        
        # Add model rows
        models = [idx for idx in formatted_matrix.index if idx != 'Average']
        for model in models:
            row = f"{model:<20}"
            for benchmark in benchmarks:
                value = formatted_matrix.loc[model, benchmark]
                if pd.isna(value):
                    row += f"{'N/A':<12}"
                else:
                    row += f"{value:.1f}%{'':<7}"
            
            avg_value = formatted_matrix.loc[model, 'Average']
            if pd.isna(avg_value):
                row += f"{'N/A':<12}"
            else:
                row += f"{avg_value:.1f}%{'':<7}"
            
            report_lines.append(row)
        
        # Add average row
        if 'Average' in formatted_matrix.index:
            avg_row = f"{'Average':<20}"
            for benchmark in benchmarks:
                value = formatted_matrix.loc['Average', benchmark]
                if pd.isna(value):
                    avg_row += f"{'N/A':<12}"
                else:
                    avg_row += f"{value:.1f}%{'':<7}"
            
            avg_value = formatted_matrix.loc['Average', 'Average']
            if pd.isna(avg_value):
                avg_row += f"{'N/A':<12}"
            else:
                avg_row += f"{avg_value:.1f}%{'':<7}"
            
            report_lines.append("-" * len(header))
            report_lines.append(avg_row)
        
        report_lines.extend([
            "",
            "=" * 120,
            "TOP PERFORMERS BY CATEGORY",
            "=" * 120,
            "",
        ])
        
        # Best performing model per benchmark
        for benchmark in benchmarks:
            if benchmark in accuracy_matrix.columns:
                benchmark_scores = accuracy_matrix[benchmark].drop('Average', errors='ignore')
                if not benchmark_scores.empty:
                    best_model = benchmark_scores.idxmax()
                    best_score = benchmark_scores.max()
                    if not pd.isna(best_score):
                        report_lines.append(f"{benchmark:<20}: {best_model} ({best_score:.1f}%)")
        
        report_lines.extend([
            "",
            "OVERALL RANKINGS:",
            "",
        ])
        
        # Overall model ranking by average accuracy
        if 'Average' in accuracy_matrix.columns:
            avg_scores = accuracy_matrix['Average'].drop('Average', errors='ignore')
            sorted_models = avg_scores.sort_values(ascending=False)
            
            for i, (model, score) in enumerate(sorted_models.items(), 1):
                if not pd.isna(score):
                    report_lines.append(f"{i:2d}. {model}: {score:.1f}%")
        
        return "\n".join(report_lines)

    def generate_simple_accuracy_table(self) -> str:
        """
        Generate simple accuracy table matching the requested format exactly
        """
        accuracy_matrix = self.generate_accuracy_matrix()
        
        if accuracy_matrix.empty:
            return "No accuracy data available."
        
        # Remove rows where all benchmarks are NaN
        accuracy_matrix = accuracy_matrix.dropna(axis=0, how='all')
        # Remove average row for simple table
        if 'Average' in accuracy_matrix.index:
            accuracy_matrix = accuracy_matrix.drop('Average')
        
        if accuracy_matrix.empty:
            return "No successful evaluations found."
        
        # Get benchmarks (excluding Average)
        benchmarks = [col for col in accuracy_matrix.columns if col != 'Average']
        
        # Create header
        header_parts = []
        col_widths = [15]  # Model name column width
        
        # Calculate column widths
        for benchmark in benchmarks + ['Average']:
            col_widths.append(max(len(benchmark), 8))
        
        # Build header
        header = f"{'Model':<{col_widths[0]}}"
        for i, benchmark in enumerate(benchmarks + ['Average']):
            header += f"{benchmark:<{col_widths[i+1]}}"
        
        lines = [header]
        
        # Add data rows
        for model in accuracy_matrix.index:
            row = f"{model:<{col_widths[0]}}"
            
            # Add benchmark scores
            for i, benchmark in enumerate(benchmarks):
                value = accuracy_matrix.loc[model, benchmark]
                if pd.isna(value):
                    row += f"{'N/A':<{col_widths[i+1]}}"
                else:
                    row += f"{value:.1f}%{'':<{col_widths[i+1]-5}}"
            
            # Add average
            avg_value = accuracy_matrix.loc[model, 'Average']
            if pd.isna(avg_value):
                row += "N/A"
            else:
                row += f"{avg_value:.1f}%"
            
            lines.append(row)
        
        return '\n'.join(lines)
    
    def print_accuracy_summary(self):
        """Print accuracy summary to console"""
        print("\n" + "="*80)
        print("ACCURACY COMPARISON TABLE")
        print("="*80)
        print()
        print(self.generate_simple_accuracy_table())
        print()
        print("="*80)

    def save_accuracy_analysis(self, output_dir: Optional[str] = None):
        """Save accuracy analysis results to files"""
        if output_dir:
            analysis_dir = Path(output_dir)
        else:
            analysis_dir = self.results_dir / "analysis"
        
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate accuracy matrix
        accuracy_matrix = self.generate_accuracy_matrix()
        
        if not accuracy_matrix.empty:
            # Save CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = analysis_dir / f"accuracy_matrix_{timestamp}.csv"
            accuracy_matrix.to_csv(csv_file)
            
            # Save formatted report
            report_file = analysis_dir / f"accuracy_report_{timestamp}.txt"
            report_content = self.generate_accuracy_report()
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"Accuracy analysis saved:")
            print(f"   CSV Matrix: {csv_file}")
            print(f"   Report: {report_file}")
            
            return str(csv_file), str(report_file)
        else:
            print("No accuracy data available for analysis.")
            return None, None
    
    def analyze_performance(self) -> PerformanceInsights:
        """
        Perform comprehensive performance analysis
        
        Returns:
            PerformanceInsights object with all analysis results
        """
        if not self.results_data:
            raise ValueError("No results data available for analysis")
        
        logger.info("Starting performance analysis...")
        
        # Extract basic statistics
        evaluations = self.results_data.get('evaluations', [])
        summary = self.results_data.get('summary', {})
        
        total_evaluations = len(evaluations)
        successful_evals = sum(1 for eval in evaluations if eval.get('success', False))
        success_rate = (successful_evals / total_evaluations * 100) if total_evaluations > 0 else 0
        
        # Time analysis
        total_time_str = summary.get('total_time', '0s')
        total_time_hours = self._parse_time_string(total_time_str) / 3600
        avg_time_per_eval = sum(eval.get('execution_time', 0) for eval in evaluations) / len(evaluations) if evaluations else 0
        
        # Memory analysis
        peak_memory = max((eval.get('peak_memory', 0) for eval in evaluations), default=0)
        
        # Advanced analysis
        bottlenecks = self._identify_bottlenecks(evaluations)
        model_rankings = self._analyze_model_performance(evaluations)
        benchmark_difficulty = self._analyze_benchmark_difficulty(evaluations)
        
        # Create insights object
        self.insights = PerformanceInsights(
            total_evaluations=total_evaluations,
            success_rate=success_rate,
            total_time_hours=total_time_hours,
            average_time_per_eval=avg_time_per_eval,
            peak_memory_usage=peak_memory,
            bottlenecks=bottlenecks,
            model_rankings=model_rankings,
            benchmark_difficulty=benchmark_difficulty
        )
        
        logger.info("Performance analysis completed")
        return self.insights
    
    def _parse_time_string(self, time_str: str) -> float:
        """Parse time string (e.g., '123.4s', '2.5h') to seconds"""
        try:
            if 's' in time_str and 'h' in time_str:
                # Format like "84483.9s (23.47h)"
                import re
                seconds_match = re.search(r'([\d.]+)s', time_str)
                if seconds_match:
                    return float(seconds_match.group(1))
            elif time_str.endswith('s'):
                return float(time_str[:-1])
            elif time_str.endswith('h'):
                return float(time_str[:-1]) * 3600
            elif time_str.endswith('m'):
                return float(time_str[:-1]) * 60
            else:
                # Try to extract first number
                import re
                match = re.search(r'([\d.]+)', time_str)
                return float(match.group(1)) if match else 0
        except (ValueError, AttributeError):
            return 0
    
    def _identify_bottlenecks(self, evaluations: List[Dict[str, Any]]) -> List[BottleneckAnalysis]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Memory bottlenecks
        memory_issues = []
        for eval in evaluations:
            if not eval.get('success', False):
                error_msg = eval.get('error_message', '').lower()
                if 'memory' in error_msg or 'oom' in error_msg or 'cuda' in error_msg:
                    memory_issues.append(f"{eval['model_name']} on {eval['benchmark_name']}")
        
        if memory_issues:
            severity = "critical" if len(memory_issues) > len(evaluations) * 0.3 else "high"
            bottlenecks.append(BottleneckAnalysis(
                bottleneck_type="memory",
                affected_combinations=memory_issues,
                severity=severity,
                description=f"Memory-related failures detected in {len(memory_issues)} evaluations",
                recommendations=[
                    "Consider reducing batch sizes for affected models",
                    "Enable gradient checkpointing",
                    "Use mixed precision training",
                    "Upgrade to higher memory GPU if possible"
                ]
            ))
        
        # Time bottlenecks (extremely slow evaluations)
        execution_times = [eval.get('execution_time', 0) for eval in evaluations if eval.get('success', False)]
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            threshold = avg_time * 3  # 3x slower than average
            
            slow_evaluations = []
            for eval in evaluations:
                if eval.get('execution_time', 0) > threshold:
                    slow_evaluations.append(f"{eval['model_name']} on {eval['benchmark_name']}")
            
            if slow_evaluations:
                bottlenecks.append(BottleneckAnalysis(
                    bottleneck_type="time",
                    affected_combinations=slow_evaluations,
                    severity="medium",
                    description=f"Unusually slow evaluations detected ({len(slow_evaluations)} cases)",
                    recommendations=[
                        "Investigate model-specific optimization settings",
                        "Check for CPU bottlenecks during evaluation",
                        "Consider parallel processing where possible"
                    ]
                ))
        
        # Failure patterns
        failed_evaluations = [eval for eval in evaluations if not eval.get('success', False)]
        if failed_evaluations:
            failure_rate = len(failed_evaluations) / len(evaluations) * 100
            severity = "critical" if failure_rate > 20 else "high" if failure_rate > 10 else "medium"
            
            bottlenecks.append(BottleneckAnalysis(
                bottleneck_type="failure",
                affected_combinations=[f"{eval['model_name']} on {eval['benchmark_name']}" 
                                     for eval in failed_evaluations],
                severity=severity,
                description=f"High failure rate: {failure_rate:.1f}% of evaluations failed",
                recommendations=[
                    "Review error logs for common failure patterns",
                    "Verify model and benchmark configurations",
                    "Check hardware stability and cooling",
                    "Consider implementing retry mechanisms"
                ]
            ))
        
        return bottlenecks
    
    def _analyze_model_performance(self, evaluations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze relative performance of different models"""
        model_stats = {}
        
        for eval in evaluations:
            model_name = eval['model_name']
            if model_name not in model_stats:
                model_stats[model_name] = {
                    'total_evaluations': 0,
                    'successful_evaluations': 0,
                    'total_time': 0,
                    'total_memory': 0
                }
            
            stats = model_stats[model_name]
            stats['total_evaluations'] += 1
            
            if eval.get('success', False):
                stats['successful_evaluations'] += 1
                stats['total_time'] += eval.get('execution_time', 0)
                stats['total_memory'] += eval.get('peak_memory', 0)
        
        # Calculate efficiency scores (higher is better)
        model_rankings = {}
        for model_name, stats in model_stats.items():
            if stats['successful_evaluations'] > 0:
                success_rate = stats['successful_evaluations'] / stats['total_evaluations']
                avg_time = stats['total_time'] / stats['successful_evaluations']
                avg_memory = stats['total_memory'] / stats['successful_evaluations']
                
                # Efficiency score: success_rate / (normalized_time * normalized_memory)
                time_norm = avg_time / 1000  # Normalize to reasonable scale
                memory_norm = avg_memory / 10  # Normalize to reasonable scale
                
                efficiency_score = success_rate / max((time_norm * memory_norm), 0.01)
                model_rankings[model_name] = round(efficiency_score, 2)
            else:
                model_rankings[model_name] = 0.0
        
        return model_rankings
    
    def _analyze_benchmark_difficulty(self, evaluations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze relative difficulty of different benchmarks"""
        benchmark_stats = {}
        
        for eval in evaluations:
            benchmark_name = eval['benchmark_name']
            if benchmark_name not in benchmark_stats:
                benchmark_stats[benchmark_name] = {
                    'total_evaluations': 0,
                    'successful_evaluations': 0,
                    'total_time': 0
                }
            
            stats = benchmark_stats[benchmark_name]
            stats['total_evaluations'] += 1
            
            if eval.get('success', False):
                stats['successful_evaluations'] += 1
                stats['total_time'] += eval.get('execution_time', 0)
        
        # Calculate difficulty scores (higher = more difficult)
        benchmark_difficulty = {}
        for benchmark_name, stats in benchmark_stats.items():
            success_rate = stats['successful_evaluations'] / stats['total_evaluations']
            
            if stats['successful_evaluations'] > 0:
                avg_time = stats['total_time'] / stats['successful_evaluations']
                difficulty_score = (1 - success_rate) * 100 + (avg_time / 60)
            else:
                difficulty_score = 100
            
            benchmark_difficulty[benchmark_name] = round(difficulty_score, 2)
        
        return benchmark_difficulty
    
    def generate_summary_report(self) -> str:
        """
        Generate human-readable summary report with accuracy analysis
        
        Returns:
            Formatted summary report string
        """
        if not self.insights:
            self.analyze_performance()
        
        report_lines = [
            "=" * 80,
            "SPECTRAVISION PERFORMANCE ANALYSIS REPORT",
            "=" * 80,
            "",
            f"EVALUATION SUMMARY",
            f"   Total Evaluations: {self.insights.total_evaluations}",
            f"   Success Rate: {self.insights.success_rate:.1f}%",
            f"   Total Time: {self.insights.total_time_hours:.2f} hours",
            f"   Average Time per Evaluation: {self.insights.average_time_per_eval:.1f} seconds",
            f"   Peak Memory Usage: {self.insights.peak_memory_usage:.1f} GB",
            "",
            "=" * 80,
            "ACCURACY ANALYSIS",
            "=" * 80,
            "",
        ]
        
        # Add accuracy report
        accuracy_report = self.generate_accuracy_report()
        report_lines.append(accuracy_report)
        
        report_lines.extend([
            "",
            "=" * 80,
            "PERFORMANCE INSIGHTS",
            "=" * 80,
            "",
        ])
        
        # Model performance ranking
        if self.insights.model_rankings:
            report_lines.extend([
                "MODEL EFFICIENCY RANKING (Success Rate + Resource Usage)",
                "   Higher score = Better success rate + Lower resource usage",
            ])
            
            sorted_models = sorted(self.insights.model_rankings.items(), 
                                 key=lambda x: x[1], reverse=True)
            for i, (model, score) in enumerate(sorted_models, 1):
                report_lines.append(f"   {i:2d}. {model}: {score:.2f}")
            report_lines.append("")
        
        # Benchmark difficulty ranking
        if self.insights.benchmark_difficulty:
            report_lines.extend([
                "BENCHMARK DIFFICULTY RANKING",
                "   Higher score = More failures + Longer execution time",
            ])
            
            sorted_benchmarks = sorted(self.insights.benchmark_difficulty.items(),
                                     key=lambda x: x[1], reverse=True)
            for i, (benchmark, difficulty) in enumerate(sorted_benchmarks, 1):
                report_lines.append(f"   {i:2d}. {benchmark}: {difficulty:.2f}")
            report_lines.append("")
        
        # Bottleneck analysis
        if self.insights.bottlenecks:
            report_lines.extend([
                "IDENTIFIED BOTTLENECKS",
            ])
            
            for i, bottleneck in enumerate(self.insights.bottlenecks, 1):
                report_lines.extend([
                    f"   {i}. {bottleneck.bottleneck_type.upper()} BOTTLENECK ({bottleneck.severity.upper()})",
                    f"      Description: {bottleneck.description}",
                    f"      Affected: {len(bottleneck.affected_combinations)} combinations",
                    "      Recommendations:",
                ])
                for rec in bottleneck.recommendations:
                    report_lines.append(f"        - {rec}")
                report_lines.append("")
        
        report_lines.extend([
            "=" * 80,
            f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def save_analysis_results(self, output_file: Optional[str] = None):
        """Save analysis results to files"""
        if not self.insights:
            self.analyze_performance()
        
        # Create analysis directory
        analysis_dir = self.results_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary report
        if output_file:
            report_file = Path(output_file)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = analysis_dir / f"performance_analysis_{timestamp}.txt"
        
        report_content = self.generate_summary_report()
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Save detailed JSON analysis
        json_file = analysis_dir / f"detailed_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        detailed_data = {
            "analysis_timestamp": datetime.now().isoformat(),
            "insights": {
                "total_evaluations": self.insights.total_evaluations,
                "success_rate": self.insights.success_rate,
                "total_time_hours": self.insights.total_time_hours,
                "average_time_per_eval": self.insights.average_time_per_eval,
                "peak_memory_usage": self.insights.peak_memory_usage,
                "model_rankings": self.insights.model_rankings,
                "benchmark_difficulty": self.insights.benchmark_difficulty,
                "bottlenecks": [
                    {
                        "type": b.bottleneck_type,
                        "affected_combinations": b.affected_combinations,
                        "severity": b.severity,
                        "description": b.description,
                        "recommendations": b.recommendations
                    }
                    for b in self.insights.bottlenecks
                ]
            }
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2)
        
        # Also save accuracy analysis
        self.save_accuracy_analysis()
        
        logger.info(f"Analysis results saved:")
        logger.info(f"   Summary report: {report_file}")
        logger.info(f"   Detailed JSON: {json_file}")
        
        return str(report_file), str(json_file)


# Standalone execution for command line usage
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "outputs"
    
    print(f"Analyzing results from: {results_dir}")
    
    try:
        analyzer = PerformanceAnalyzer(results_dir)
        
        if analyzer.results_data is None:
            print("No results data found. Please ensure final_results.json exists.")
            print("Expected locations:")
            for pf in analyzer.potential_results_files:
                print(f"  {pf}")
            
            # Run demo with sample data
            print("\nRunning demo with sample data...")
            sample_data = {
                "summary": {
                    "total_evaluations": 10,
                    "successful_evaluations": 8,
                    "success_rate": 80.0,
                    "total_time": "1800s (0.5h)"
                },
                "evaluations": [
                    {"model_name": "Qwen2-VL-2B", "benchmark_name": "MMBench", "success": True, "accuracy_score": 0.7105},
                    {"model_name": "Qwen2-VL-2B", "benchmark_name": "TextVQA", "success": True, "accuracy_score": 0.7983},
                    {"model_name": "Qwen2-VL-2B", "benchmark_name": "GQA", "success": True, "accuracy_score": 0.6040},
                    {"model_name": "InternVL2-2B", "benchmark_name": "MMBench", "success": True, "accuracy_score": 0.678},
                    {"model_name": "InternVL2-2B", "benchmark_name": "TextVQA", "success": True, "accuracy_score": 0.703},
                    {"model_name": "InternVL2-2B", "benchmark_name": "GQA", "success": True, "accuracy_score": 0.712},
                    {"model_name": "LLaVA-1.5-7B", "benchmark_name": "MMBench", "success": True, "accuracy_score": 0.634},
                    {"model_name": "LLaVA-1.5-7B", "benchmark_name": "TextVQA", "success": True, "accuracy_score": 0.746},
                ]
            }
            analyzer.load_from_json_string(sample_data)
        
        # Generate accuracy analysis
        print("\nGenerating accuracy analysis...")
        analyzer.print_accuracy_summary()
        
        # Save results
        print("\nSaving analysis results...")
        csv_file, report_file = analyzer.save_accuracy_analysis()
        
        if analyzer.results_data:
            # Generate full performance analysis
            try:
                print("\nGenerating performance analysis...")
                analyzer.analyze_performance()
                analyzer.save_analysis_results()
            except Exception as e:
                print(f"Warning: Could not complete full performance analysis: {e}")
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


# Example of how to use with the provided JSON data
def analyze_from_json_data(json_data):
    """
    Convenience function to analyze results from JSON data directly
    
    Args:
        json_data: Dictionary containing the evaluation results
    """
    analyzer = PerformanceAnalyzer(".")  # Dummy path
    analyzer.load_from_json_string(json_data)
    
    print("ACCURACY ANALYSIS")
    print("="*50)
    analyzer.print_accuracy_summary()
    
    return analyzer