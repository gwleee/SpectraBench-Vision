"""
Sequential Evaluator for SpectraVision Phase 1
Handles VLMEvalKit integration and sequential model-benchmark evaluation
"""

import os
import sys
import json
import logging
import subprocess
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Container for single evaluation result"""
    model_name: str
    benchmark_name: str
    success: bool
    start_time: datetime
    end_time: datetime
    execution_time: float  # seconds
    peak_memory: float  # GB
    accuracy_score: Optional[float] = None  # Main accuracy score
    subtask_scores: Optional[Dict[str, float]] = None  # Subtask scores
    detailed_metrics: Optional[Dict[str, Any]] = None  # All detailed metrics
    metric_type: str = "accuracy"  # Type of main metric (accuracy, f1_score, hallucination_rate, etc.)
    metric_description: Optional[str] = None  # Description of what the metric represents
    error_message: Optional[str] = None
    result_file: Optional[str] = None
    vlmevalkit_output: Optional[Dict] = None

class SequentialEvaluator:
    """Sequential evaluation engine using VLMEvalKit"""
    
    def __init__(self, config: Dict[str, Any], monitor=None, verbose: bool = False, test_mode: bool = False):
        """
        Initialize Sequential Evaluator

        Args:
            config: Complete configuration dictionary
            monitor: Optional performance monitor
            verbose: Enable verbose logging
            test_mode: If True, run only 2 samples per benchmark for quick verification
        """
        self.config = config
        self.monitor = monitor
        self.verbose = verbose
        self.test_mode = test_mode
        self.results = []

        # Cache cleanup settings
        self.cleanup_cache = config.get("cleanup_cache", False)
        self.cleanup_level = config.get("cleanup_level", "light")
        
        # Convert to absolute path
        if Path(config["output_dir"]).is_absolute():
            base_output_dir = Path(config["output_dir"])
        else:
            # Resolve relative path based on project root
            base_output_dir = Path(__file__).parent.parent / config["output_dir"]

        base_output_dir = base_output_dir.resolve()

        # Create session-based directory: 연도날짜시간_숫자
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_counter = 1

        # Check if directory already exists and increment counter if needed
        while True:
            session_dir_name = f"{self.session_timestamp}_{session_counter:03d}"
            self.output_dir = base_output_dir / session_dir_name
            if not self.output_dir.exists():
                break
            session_counter += 1

        # Setup output directories within session folder
        self.results_dir = self.output_dir / "results"
        self.model_results_dir = self.results_dir / "model_results"
        self.logs_dir = self.output_dir / "logs"
        self.final_report_dir = self.output_dir / "final_report"
        # Legacy compatibility
        self.vlmevalkit_dir = self.output_dir / "vlmevalkit_results"
        self.reports_dir = self.output_dir / "reports"
        
        # VLMEvalKit installation path
        self.vlmevalkit_path = self._find_vlmevalkit_path()
        
        # Create directories
        for dir_path in [self.results_dir, self.model_results_dir, self.logs_dir,
                        self.final_report_dir, self.vlmevalkit_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Evaluation state
        self.current_evaluation = 0
        self.total_evaluations = len(config["models"]) * len(config["benchmarks"])
        
        logger.info(f"Sequential Evaluator initialized")
        logger.info(f"   Total evaluations: {self.total_evaluations}")
        logger.info(f"   Output directory: {self.output_dir}")
        logger.info(f"   VLMEvalKit results: {self.vlmevalkit_dir}")
        logger.info(f"   VLMEvalKit path: {self.vlmevalkit_path}")
        logger.info(f"   Cache cleanup: {self.cleanup_cache} (level: {self.cleanup_level})")
    
    def _find_vlmevalkit_path(self) -> Path:
        """
        Find VLMEvalKit installation path
        
        Returns:
            Path to VLMEvalKit directory
        """
        try:
            import vlmeval
            vlmevalkit_path = Path(vlmeval.__file__).parent.parent
            logger.debug(f"VLMEvalKit found at: {vlmevalkit_path}")
            return vlmevalkit_path
        except ImportError:
            # Fallback to common locations
            project_root = Path(__file__).parent.parent
            possible_paths = [
                project_root / "VLMEvalKit",
                project_root.parent / "VLMEvalKit",
                Path("./VLMEvalKit"),
                Path("../VLMEvalKit")
            ]
            
            for path in possible_paths:
                if path.exists() and (path / "vlmeval").exists():
                    logger.info(f"VLMEvalKit found at: {path}")
                    return path.resolve()
            
            raise FileNotFoundError("VLMEvalKit installation not found")
    
    def _cleanup_caches(self, level: str = "light"):
        """
        Clean up caches after evaluation
        
        Args:
            level: Cleanup level
                - light: PyTorch CUDA cache only
                - moderate: + transformers cache  
                - aggressive: + temporary files
        """
        if not self.cleanup_cache:
            return
            
        logger.info(f"Starting cache cleanup (level: {level})")
        
        try:
            # 1. PyTorch GPU cache cleanup (always performed)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("   PyTorch CUDA cache cleared")
            
            # 2. Transformers cache cleanup (moderate and above)
            if level in ["moderate", "aggressive"]:
                try:
                    import transformers
                    if hasattr(transformers, 'utils'):
                        transformers.utils.logging.set_verbosity_error()
                    logger.info("   Transformers cache cleared")
                except ImportError:
                    pass
            
            # 3. Temporary files cleanup (aggressive only)
            if level == "aggressive":
                cache_dirs = [
                    Path.home() / ".cache" / "huggingface" / "transformers",
                    Path.home() / ".cache" / "torch" / "sentence_transformers"
                ]
                
                for cache_dir in cache_dirs:
                    if cache_dir.exists():
                        try:
                            temp_patterns = ['*.tmp', '*.lock', '*incomplete*']
                            cleaned_files = 0
                            
                            for pattern in temp_patterns:
                                for temp_file in cache_dir.rglob(pattern):
                                    try:
                                        temp_file.unlink()
                                        cleaned_files += 1
                                    except OSError:
                                        pass
                            
                            if cleaned_files > 0:
                                logger.info(f"   Cleaned {cleaned_files} temporary files from {cache_dir.name}")
                                
                        except Exception as e:
                            logger.warning(f"   Could not clean {cache_dir}: {e}")
            
            logger.info("Cache cleanup completed")
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    def _setup_vlmevalkit_environment(self, model_config: Dict[str, Any]):
        """
        Setup environment for specific model's transformer requirements
        
        Args:
            model_config: Model configuration dictionary
        """
        # Set model-specific environment variables (simplified)
        env_vars = {
            "CUDA_VISIBLE_DEVICES": "0",  # Use first GPU by default
            "VLMEVAL_BATCH_SIZE": "1",    # Default batch size
            "HF_DATASETS_OFFLINE": "0",  # Allow dataset downloads
            "TRANSFORMERS_OFFLINE": "0",  # Allow model downloads
        }
        
        # Update current environment
        for key, value in env_vars.items():
            os.environ[key] = value
            logger.debug(f"Set {key}={value}")

    def _validate_model_availability(self, model_id: str) -> bool:
        """
        Validate that model is available in VLMEvalKit
        
        Args:
            model_id: VLMEvalKit model identifier
            
        Returns:
            True if model is available
        """
        try:
            # Change to VLMEvalKit directory and check model availability
            original_cwd = os.getcwd()
            os.chdir(self.vlmevalkit_path)
            
            # Import and check supported models
            sys.path.insert(0, str(self.vlmevalkit_path))
            from vlmeval.config import supported_VLM
            
            available = model_id in supported_VLM
            logger.debug(f"Model {model_id} available: {available}")
            
            if not available:
                # Log available models for debugging
                available_models = list(supported_VLM.keys())
                logger.warning(f"Available models: {available_models[:10]}...")  # Show first 10
                
                # Try to find similar models
                similar_models = [m for m in available_models if any(part in m.lower() for part in model_id.lower().split('_'))]
                if similar_models:
                    logger.info(f"Similar models found: {similar_models[:5]}")
            
            return available
            
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
        finally:
            os.chdir(original_cwd)
            if str(self.vlmevalkit_path) in sys.path:
                sys.path.remove(str(self.vlmevalkit_path))

    def _parse_vlmevalkit_results(self, result_file: str, benchmark_name: str) -> tuple:
        """
        Parse VLMEvalKit result file to extract accuracy scores
        
        Args:
            result_file: Path to VLMEvalKit result CSV file
            benchmark_name: Name of the benchmark
            
        Returns:
            Tuple of (main_accuracy, subtask_scores, detailed_metrics, metric_type, metric_description)
        """
        try:
            if not result_file or not Path(result_file).exists():
                logger.warning(f"Result file not found: {result_file}")
                return None, None, None, "accuracy", None
            
            logger.info(f"Reading result file: {result_file}")
            file_path = Path(result_file)
            
            # Read different file formats
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(result_file)
            elif file_path.suffix.lower() == '.xlsx':
                df = pd.read_excel(result_file)
            elif file_path.suffix.lower() == '.json':
                # Try to read JSON as records
                import json
                with open(result_file, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    logger.warning(f"Unsupported JSON format in {result_file}")
                    return None, None, None, "accuracy", None
            else:
                logger.warning(f"Unsupported file format: {file_path.suffix}")
                return None, None, None, "accuracy", None
                
            logger.info(f"Loaded {file_path.suffix} with shape: {df.shape}, columns: {list(df.columns)}")
            
            # Display DataFrame structure for debugging
            if len(df) > 0:
                logger.info(f"DataFrame columns: {list(df.columns)}")
                logger.info(f"DataFrame dtypes: {dict(df.dtypes)}")
                logger.info(f"First row data: {df.iloc[0].to_dict()}")
                if len(df) > 1:
                    logger.info(f"Last row data: {df.iloc[-1].to_dict()}")
            
            if df.empty:
                logger.warning(f"Result file is empty: {result_file}")
                return None, None, None, "accuracy", None
            
            # Safe data processing
            main_accuracy = None
            subtask_scores = {}
            detailed_metrics = {}
            metric_type = "accuracy"  # Default metric type
            metric_description = None
            
            # Convert all columns to dict for detailed metrics
            try:
                last_row = df.iloc[-1].to_dict()
                # Safe dictionary creation - handle NaN and special characters
                for k, v in last_row.items():
                    if pd.notna(v):
                        if isinstance(v, (int, float)):
                            if not np.isnan(v):
                                detailed_metrics[str(k)] = float(v)
                        else:
                            # Handle % characters in strings
                            safe_value = str(v).replace('%', 'percent')
                            detailed_metrics[str(k)] = safe_value
            except Exception as e:
                logger.warning(f"Error processing detailed metrics: {e}")
                detailed_metrics = {}
            
            # Benchmark-specific parsing with error handling
            try:
                if "mmvet" in benchmark_name.lower():
                    # MMVet benchmark doesn't have accuracy scores in result files
                    # It only contains question-answer pairs for manual evaluation
                    logger.info(f"MMVet benchmark detected - no automatic accuracy scoring available")
                    main_accuracy = None
                    metric_type = "manual_evaluation"
                    metric_description = "Requires manual evaluation - no automatic scoring"
                elif "hallusion" in benchmark_name.lower() or "hallusionbench" in benchmark_name.lower():
                    # HallusionBench uses aAcc, fAcc, qAcc metrics
                    logger.info(f"HallusionBench detected - looking for aAcc, fAcc, qAcc metrics")
                    hallusion_cols = ['aAcc', 'fAcc', 'qAcc', 'hallucination_rate', 'hall_rate', 'error_rate']
                    for col in hallusion_cols:
                        if col in df.columns:
                            # Look for Overall row first
                            overall_row = df[df['split'].str.contains('Overall', na=False, case=False)] if 'split' in df.columns else df.iloc[0:1]
                            if not overall_row.empty:
                                val = overall_row[col].iloc[0]
                            else:
                                val = df[col].iloc[0]
                            
                            if pd.notna(val) and isinstance(val, (int, float)):
                                if col in ['aAcc', 'fAcc', 'qAcc']:
                                    # These are accuracy metrics, use directly
                                    main_accuracy = float(val) / 100.0 if val > 1 else float(val)
                                    logger.info(f"Found HallusionBench {col}: {main_accuracy}")
                                    metric_type = col.lower()
                                    metric_description = f"HallusionBench {col} (answer accuracy)"
                                else:
                                    # Convert hallucination rate to accuracy (1 - hallucination_rate)
                                    if 0 <= val <= 1:
                                        main_accuracy = 1.0 - float(val)
                                    elif 0 <= val <= 100:
                                        main_accuracy = 1.0 - (float(val) / 100.0)
                                    logger.info(f"Converted hallucination rate {val} to accuracy {main_accuracy}")
                                    metric_type = "hallucination_rate"
                                    metric_description = f"Accuracy derived from hallucination rate (1 - {val})"
                                break
                elif "pope" in benchmark_name.lower():
                    # POPE focuses on F1-score, Precision, Recall
                    logger.info(f"POPE benchmark detected - looking for F1-score or precision/recall")
                    pope_cols = ['f1', 'f1_score', 'f1-score', 'precision', 'recall', 'accuracy']
                    for col in pope_cols:
                        if col in df.columns:
                            val = df[col].iloc[-1]
                            if pd.notna(val) and isinstance(val, (int, float)):
                                main_accuracy = float(val)
                                logger.info(f"Found POPE {col}: {main_accuracy}")
                                metric_type = col.replace('_', '').replace('-', '')
                                metric_description = f"POPE {col.upper()} metric"
                                break
                elif "mme" in benchmark_name.lower():
                    # MME uses composite scores (Perception + Reasoning)
                    logger.info(f"MME benchmark detected - looking for perception and reasoning scores")
                    
                    # First, try to find total/overall scores
                    priority_cols = ['total_score', 'overall_score']
                    for col in priority_cols:
                        if col in df.columns:
                            val = df[col].iloc[0]
                            if pd.notna(val) and isinstance(val, (int, float)):
                                main_accuracy = float(val)
                                logger.info(f"Found MME {col}: {main_accuracy}")
                                metric_type = "composite_score"
                                metric_description = f"MME {col.replace('_', ' ')}"
                                break
                    
                    # Try to sum perception + reasoning directly
                    if main_accuracy is None and 'perception' in df.columns and 'reasoning' in df.columns:
                        perc_val = df['perception'].iloc[0]
                        reason_val = df['reasoning'].iloc[0]
                        if pd.notna(perc_val) and pd.notna(reason_val):
                            main_accuracy = float(perc_val) + float(reason_val)
                            logger.info(f"Computed MME total score: {perc_val} + {reason_val} = {main_accuracy}")
                            metric_type = "composite_score"
                            metric_description = "MME total score (Perception + Reasoning)"
                    
                    # If no perception+reasoning, try perception_score + cognition_score
                    elif main_accuracy is None and 'perception_score' in df.columns and 'cognition_score' in df.columns:
                        perc_val = df['perception_score'].iloc[0]
                        cogn_val = df['cognition_score'].iloc[0]
                        if pd.notna(perc_val) and pd.notna(cogn_val):
                            main_accuracy = float(perc_val) + float(cogn_val)
                            logger.info(f"Computed MME total score: {perc_val} + {cogn_val} = {main_accuracy}")
                            metric_type = "composite_score"
                            metric_description = "MME total score (Perception + Cognition)"
                    
                    # If still no score, try individual components
                    if main_accuracy is None:
                        component_cols = ['perception_score', 'cognition_score']
                        for col in component_cols:
                            if col in df.columns:
                                val = df[col].iloc[-1]
                                if pd.notna(val) and isinstance(val, (int, float)):
                                    main_accuracy = float(val)
                                    logger.info(f"Found MME {col}: {main_accuracy}")
                                    metric_type = "composite_score"
                                    metric_description = f"MME {col.replace('_', ' ')} (component score)"
                                    break
                elif "ocrbench" in benchmark_name.lower():
                    # OCRBench uses JSON score files with Final Score Norm
                    logger.info(f"OCRBench benchmark detected - looking for Final Score Norm")
                    if isinstance(df, pd.DataFrame) and len(df) == 1:
                        # Check if this is a score JSON converted to DataFrame
                        row = df.iloc[0]
                        if 'Final Score Norm' in row:
                            main_accuracy = float(row['Final Score Norm']) / 100.0  # Convert to 0-1 scale
                            logger.info(f"Found OCRBench Final Score Norm: {main_accuracy*100}%")
                            metric_type = "composite_score"
                            metric_description = "OCRBench Final Score Normalized"
                        elif 'final score norm' in row:
                            main_accuracy = float(row['final score norm']) / 100.0
                            logger.info(f"Found OCRBench final score norm: {main_accuracy*100}%")
                            metric_type = "composite_score"
                            metric_description = "OCRBench Final Score Normalized"
                elif "mmbench" in benchmark_name.lower():
                    # MMBench has overall score and category-wise scores
                    if 'Overall' in df.columns:
                        overall_val = df['Overall'].iloc[-1]
                        if pd.notna(overall_val) and isinstance(overall_val, (int, float)):
                            main_accuracy = float(overall_val)
                    
                    # Extract subtask scores for MMBench categories
                    category_cols = [col for col in df.columns if col not in ['Overall', 'split', 'version']]
                    for col in category_cols:
                        try:
                            val = df[col].iloc[-1]
                            if pd.notna(val) and isinstance(val, (int, float)):
                                subtask_scores[str(col)] = float(val)
                        except (ValueError, KeyError, IndexError):
                            pass
                else:
                    # Generic parsing for other benchmarks
                    possible_score_cols = ['Overall', 'Accuracy', 'accuracy', 'Score', 'score']
                    for col in possible_score_cols:
                        if col in df.columns:
                            try:
                                val = df[col].iloc[-1]
                                if pd.notna(val) and isinstance(val, (int, float)):
                                    main_accuracy = float(val)
                                    break
                            except (ValueError, IndexError):
                                continue
                    
                    # If still no score found, use whitelisted accuracy columns only
                    if main_accuracy is None:
                        # Define accuracy column whitelist (exclude metadata columns)
                        accuracy_whitelist = {
                            'overall', 'accuracy', 'acc', 'score', 'correct_rate', 
                            'f1_score', 'avg_score', 'mean_score', 'total_score'
                        }
                        # Also exclude common metadata columns
                        metadata_blacklist = {
                            'index', 'idx', 'id', 'question', 'prediction', 'answer',
                            'split', 'version', 'timestamp', 'file', 'path'
                        }
                        
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        valid_cols = []
                        
                        for col in numeric_cols:
                            col_lower = str(col).lower()
                            # Include if in whitelist or not in blacklist
                            if (col_lower in accuracy_whitelist or 
                                not any(blacklist_term in col_lower for blacklist_term in metadata_blacklist)):
                                valid_cols.append(col)
                        
                        logger.info(f"Numeric columns: {list(numeric_cols)}")
                        logger.info(f"Valid accuracy columns after filtering: {valid_cols}")
                        
                        if valid_cols:
                            try:
                                # Use the last valid column
                                col_to_use = valid_cols[-1]
                                val = df[col_to_use].iloc[-1]
                                logger.info(f"Using column '{col_to_use}' with value: {val}")
                                
                                if pd.notna(val):
                                    # Additional validation: accuracy should be reasonable
                                    val_float = float(val)
                                    if 0 <= val_float <= 1 or 0 <= val_float <= 100:
                                        main_accuracy = val_float
                                        logger.info(f"Accepted accuracy value: {main_accuracy}")
                                    else:
                                        logger.warning(f"Rejected unreasonable accuracy value: {val_float} from column '{col_to_use}'")
                            except (ValueError, IndexError) as e:
                                logger.warning(f"Error processing column '{col_to_use}': {e}")
                                pass
            except Exception as e:
                logger.warning(f"Error parsing benchmark results: {e}")
            
            # Convert percentages if needed (values > 1 but <= 100 are likely percentages)
            try:
                if main_accuracy:
                    # Special handling for composite scores (MME, etc.) - allow large values
                    if metric_type == "composite_score":
                        # Composite scores can be large (e.g., MME total ~2000), don't convert
                        logger.info(f"Keeping composite score as-is: {main_accuracy}")
                    elif 1 < main_accuracy <= 100:
                        logger.info(f"Converting main accuracy from percentage: {main_accuracy} -> {main_accuracy / 100.0}")
                        main_accuracy = main_accuracy / 100.0
                    elif main_accuracy > 100 and metric_type != "composite_score":
                        logger.warning(f"Rejecting unreasonable main accuracy value: {main_accuracy}")
                        main_accuracy = None
                    
                for key, value in list(subtask_scores.items()):
                    if isinstance(value, (int, float)):
                        if 1 < value <= 100:
                            logger.info(f"Converting subtask '{key}' from percentage: {value} -> {value / 100.0}")
                            subtask_scores[key] = value / 100.0
                        elif value > 100:
                            logger.warning(f"Rejecting unreasonable subtask '{key}' value: {value}")
                            del subtask_scores[key]
            except Exception as e:
                logger.warning(f"Error converting percentages: {e}")
            
            return main_accuracy, subtask_scores, detailed_metrics, metric_type, metric_description
            
        except Exception as e:
            logger.warning(f"Failed to parse results from {result_file}: {e}")
            return None, None, None, "accuracy", None
    
    def _generate_evaluation_report(self, result: EvaluationResult):
        """
        Generate detailed evaluation report
        
        Args:
            result: EvaluationResult object
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Safe filename creation
        safe_model_name = self._safe_filename(result.model_name)
        safe_benchmark_name = self._safe_filename(result.benchmark_name)
        report_filename = f"{safe_model_name}_{safe_benchmark_name}_report_{timestamp}.txt"
        report_path = self.reports_dir / report_filename
        
        # Generate report content
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SPECTRAVISION EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(f"Model: {result.model_name}")
        report_lines.append(f"Benchmark: {result.benchmark_name}")
        report_lines.append(f"Generated: {timestamp}")
        report_lines.append(f"Evaluation Status: {'SUCCESS' if result.success else 'FAILED'}")
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("SUMMARY")
        report_lines.append("=" * 80)
        
        if result.success:
            # Execution details with dynamic metric labeling
            if result.accuracy_score is not None:
                # Determine appropriate label based on metric type
                if result.metric_type == "manual_evaluation":
                    metric_label = "Evaluation Status"
                    metric_value = "Requires manual evaluation"
                elif result.metric_type == "hallucination_rate":
                    metric_label = "Accuracy (from Hallucination Rate)"
                    try:
                        metric_value = f"{result.accuracy_score:.2%}"
                    except (ValueError, TypeError):
                        metric_value = f"{result.accuracy_score}"
                elif result.metric_type == "composite_score":
                    metric_label = "Composite Score"
                    metric_value = f"{result.accuracy_score:.2f}"
                elif "f1" in result.metric_type.lower():
                    metric_label = "F1-Score"
                    try:
                        metric_value = f"{result.accuracy_score:.2%}"
                    except (ValueError, TypeError):
                        metric_value = f"{result.accuracy_score}"
                elif "precision" in result.metric_type.lower():
                    metric_label = "Precision"
                    try:
                        metric_value = f"{result.accuracy_score:.2%}"
                    except (ValueError, TypeError):
                        metric_value = f"{result.accuracy_score}"
                elif "recall" in result.metric_type.lower():
                    metric_label = "Recall"
                    try:
                        metric_value = f"{result.accuracy_score:.2%}"
                    except (ValueError, TypeError):
                        metric_value = f"{result.accuracy_score}"
                else:
                    # Default to accuracy
                    metric_label = "Overall Accuracy"
                    try:
                        metric_value = f"{result.accuracy_score:.2%}"
                    except (ValueError, TypeError):
                        metric_value = f"{result.accuracy_score}"
                
                report_lines.append(f"{metric_label}: {metric_value}")
                
                # Add metric description if available
                if result.metric_description:
                    report_lines.append(f"Metric Description: {result.metric_description}")
            else:
                report_lines.append("Score: Not available")
            
            # Time formatting - safe formatting
            try:
                hours = int(result.execution_time // 3600)
                minutes = int((result.execution_time % 3600) // 60)
                seconds = int(result.execution_time % 60)
                if hours > 0:
                    time_str = f"{hours}h {minutes}m {seconds}s"
                elif minutes > 0:
                    time_str = f"{minutes}m {seconds}s"
                else:
                    time_str = f"{seconds}s"
            except (ValueError, TypeError):
                time_str = f"{result.execution_time:.1f}s"
                
            report_lines.append(f"Execution Time: {time_str} ({result.execution_time:.1f}s)")
            report_lines.append(f"Start Time: {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"End Time: {result.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if result.peak_memory > 0:
                report_lines.append(f"Peak Memory Usage: {result.peak_memory:.2f}GB")
            
            # Detailed performance breakdown
            if result.subtask_scores:
                report_lines.append("")
                report_lines.append("=" * 80)
                report_lines.append("DETAILED PERFORMANCE BY CATEGORY")
                report_lines.append("=" * 80)
                
                # Sort categories by performance (descending)
                try:
                    valid_scores = [(k, v) for k, v in result.subtask_scores.items() 
                                  if isinstance(v, (int, float)) and not np.isnan(v)]
                    sorted_categories = sorted(valid_scores, key=lambda x: x[1], reverse=True)
                except Exception as e:
                    logger.warning(f"Error sorting categories: {e}")
                    sorted_categories = list(result.subtask_scores.items())
                
                # Create performance table
                report_lines.append(f"{'Category':<35} {'Accuracy':<12} {'Grade'}")
                report_lines.append("-" * 55)
                
                for category, score in sorted_categories:
                    try:
                        # Safe score processing
                        if isinstance(score, (int, float)) and not np.isnan(score):
                            # Determine grade based on score
                            if score >= 0.9:
                                grade = "Excellent"
                            elif score >= 0.8:
                                grade = "Very Good"
                            elif score >= 0.7:
                                grade = "Good"
                            elif score >= 0.6:
                                grade = "Fair"
                            else:
                                grade = "Needs Improvement"
                            
                            # Format category name (replace underscores with spaces, title case)
                            formatted_category = str(category).replace('_', ' ').title()
                            
                            report_lines.append(f"{formatted_category:<35} {score:.2%:<12} {grade}")
                        else:
                            formatted_category = str(category).replace('_', ' ').title()
                            report_lines.append(f"{formatted_category:<35} {'N/A':<12} {'No Data'}")
                    except Exception as e:
                        logger.warning(f"Error formatting score for {category}: {e}")
                        continue
                
                # Statistics - safe calculation
                try:
                    valid_scores = [s for s in result.subtask_scores.values() 
                                  if isinstance(s, (int, float)) and not np.isnan(s)]
                    if valid_scores:
                        report_lines.append("")
                        report_lines.append("-" * 55)
                        report_lines.append(f"{'Best Performance:':<35} {max(valid_scores):.2%}")
                        report_lines.append(f"{'Worst Performance:':<35} {min(valid_scores):.2%}")
                        report_lines.append(f"{'Average Performance:':<35} {sum(valid_scores)/len(valid_scores):.2%}")
                        report_lines.append(f"{'Categories >= 80%:':<35} {sum(1 for s in valid_scores if s >= 0.8)}/{len(valid_scores)}")
                except Exception as e:
                    logger.warning(f"Error calculating statistics: {e}")
        else:
            # Failure details
            try:
                report_lines.append(f"Execution Time: {result.execution_time:.1f}s (Failed)")
            except (ValueError, TypeError):
                report_lines.append("Execution Time: Unknown (Failed)")
            
            if result.error_message:
                # Safe error message processing
                safe_error_msg = str(result.error_message).replace('%', 'percent')
                report_lines.append(f"Error: {safe_error_msg}")
            else:
                report_lines.append("Error: Unknown")
        
        # Technical details
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("TECHNICAL DETAILS")
        report_lines.append("=" * 80)
        if result.result_file:
            report_lines.append(f"Result File: {result.result_file}")
        
        # Add detailed metrics if available
        if result.detailed_metrics:
            report_lines.append("")
            report_lines.append("Raw Metrics:")
            for key, value in result.detailed_metrics.items():
                try:
                    if isinstance(value, float):
                        if not np.isnan(value):
                            report_lines.append(f"  {key}: {value:.4f}")
                        else:
                            report_lines.append(f"  {key}: N/A")
                    else:
                        safe_value = str(value).replace('%', 'percent')
                        report_lines.append(f"  {key}: {safe_value}")
                except Exception as e:
                    logger.warning(f"Error formatting metric {key}: {e}")
                    continue
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append(f"Report generated by SpectraVision at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        
        # Save report
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"Evaluation report saved: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation report: {e}")
            # Retry with alternative filename
            try:
                safe_filename = f"report_{timestamp}.txt"
                safe_path = self.reports_dir / safe_filename
                with open(safe_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(report_lines))
                logger.info(f"Evaluation report saved to alternative path: {safe_path}")
            except Exception as e2:
                logger.error(f"Failed to save evaluation report to alternative path: {e2}")
    
    def _copy_vlmevalkit_outputs_to_spectravision(self, model_name: str, benchmark_name: str, eval_work_dir: Path):
        """Copy VLMEvalKit outputs to SpectraBench-Vision outputs directory"""
        try:
            import shutil

            # Find VLMEvalKit outputs directory
            vlmevalkit_outputs = self.vlmevalkit_path / "outputs"

            if not vlmevalkit_outputs.exists():
                logger.warning(f"VLMEvalKit outputs directory not found: {vlmevalkit_outputs}")
                return

            logger.info(f"Copying VLMEvalKit outputs to SpectraBench-Vision outputs directory...")

            # Look for model-specific outputs in VLMEvalKit
            model_patterns = [
                model_name,
                model_name.replace('-', '_'),
                model_name.replace('_', '-'),
                model_name.lower(),
                model_name.upper()
            ]

            copied_files_count = 0

            # Search for model directories and files
            for model_pattern in model_patterns:
                # Look for model directory in VLMEvalKit outputs
                model_dirs = list(vlmevalkit_outputs.glob(f"**/{model_pattern}"))
                model_dirs.extend(list(vlmevalkit_outputs.glob(f"**/*{model_pattern}*")))

                for model_dir in model_dirs:
                    if model_dir.is_dir():
                        # Create target directory in SpectraBench-Vision outputs (model-based structure)
                        target_model_dir = self.output_dir / model_name
                        target_model_dir.mkdir(parents=True, exist_ok=True)

                        # Copy all files from the model directory
                        for file_path in model_dir.rglob('*'):
                            if file_path.is_file():
                                # Create relative path structure
                                rel_path = file_path.relative_to(model_dir)
                                target_file = target_model_dir / rel_path
                                target_file.parent.mkdir(parents=True, exist_ok=True)

                                try:
                                    shutil.copy2(file_path, target_file)
                                    copied_files_count += 1
                                    logger.debug(f"Copied: {file_path} -> {target_file}")
                                except Exception as e:
                                    logger.warning(f"Failed to copy {file_path}: {e}")

            # Also look for files that match model and benchmark patterns
            benchmark_patterns = [
                benchmark_name,
                benchmark_name.replace('-', '_'),
                benchmark_name.replace('_', '-'),
                benchmark_name.lower(),
                benchmark_name.upper()
            ]

            # Search for result files containing model and benchmark names
            for model_pattern in model_patterns:
                for bench_pattern in benchmark_patterns:
                    # Look for files with both model and benchmark names
                    patterns_to_search = [
                        f"**/*{model_pattern}*{bench_pattern}*",
                        f"**/*{bench_pattern}*{model_pattern}*",
                        f"**/{model_pattern}_{bench_pattern}*",
                        f"**/{bench_pattern}_{model_pattern}*"
                    ]

                    for pattern in patterns_to_search:
                        matching_files = list(vlmevalkit_outputs.glob(pattern))

                        for file_path in matching_files:
                            if file_path.is_file() and file_path.suffix in ['.csv', '.xlsx', '.json', '.txt']:
                                # Create target directory structure (model/benchmark)
                                target_dir = self.output_dir / model_name / benchmark_name
                                target_dir.mkdir(parents=True, exist_ok=True)

                                target_file = target_dir / file_path.name

                                try:
                                    if not target_file.exists():  # Avoid duplicates
                                        shutil.copy2(file_path, target_file)
                                        copied_files_count += 1
                                        logger.debug(f"Copied result file: {file_path} -> {target_file}")
                                except Exception as e:
                                    logger.warning(f"Failed to copy result file {file_path}: {e}")

            if copied_files_count > 0:
                logger.info(f"Successfully copied {copied_files_count} files from VLMEvalKit outputs to {self.output_dir / model_name}")
            else:
                logger.warning(f"No VLMEvalKit output files found for model '{model_name}' and benchmark '{benchmark_name}'")

                # Copy any recent files as fallback
                recent_files = []
                import time
                current_time = time.time()

                for file_path in vlmevalkit_outputs.rglob('*'):
                    if (file_path.is_file() and
                        file_path.suffix in ['.csv', '.xlsx', '.json'] and
                        (current_time - file_path.stat().st_mtime) < 3600):  # Files modified in last hour
                        recent_files.append((file_path, file_path.stat().st_mtime))

                if recent_files:
                    # Sort by modification time (newest first)
                    recent_files.sort(key=lambda x: x[1], reverse=True)

                    # Copy the 3 most recent files
                    target_dir = self.output_dir / model_name / "recent_outputs"
                    target_dir.mkdir(parents=True, exist_ok=True)

                    for file_path, _ in recent_files[:3]:
                        target_file = target_dir / file_path.name
                        try:
                            shutil.copy2(file_path, target_file)
                            logger.info(f"Copied recent file as fallback: {file_path} -> {target_file}")
                        except Exception as e:
                            logger.warning(f"Failed to copy recent file {file_path}: {e}")

        except Exception as e:
            logger.error(f"Error copying VLMEvalKit outputs: {e}")

    def _safe_filename(self, name: str) -> str:
        """Create safe filename from string"""
        if not name:
            return "unknown"

        safe_name = str(name)
        # Replace problematic characters
        safe_name = safe_name.replace('%', 'percent')
        safe_name = safe_name.replace('/', '_')
        safe_name = safe_name.replace('\\', '_')
        safe_name = safe_name.replace(':', '_')
        safe_name = safe_name.replace('*', '_')
        safe_name = safe_name.replace('?', '_')
        safe_name = safe_name.replace('"', '_')
        safe_name = safe_name.replace('<', '_')
        safe_name = safe_name.replace('>', '_')
        safe_name = safe_name.replace('|', '_')
        safe_name = safe_name.replace(' ', '_')

        # Remove consecutive underscores
        import re
        safe_name = re.sub(r'_+', '_', safe_name)
        safe_name = safe_name.strip('_')

        return safe_name if safe_name else "unknown"
    
    def _run_vlmevalkit_evaluation(self, model_config: Dict[str, Any], 
                                 benchmark_config: Dict[str, Any]) -> EvaluationResult:
        """
        Run single model-benchmark evaluation using VLMEvalKit
        
        Args:
            model_config: Model configuration
            benchmark_config: Benchmark configuration
            
        Returns:
            EvaluationResult object
        """
        model_name = model_config["name"]
        benchmark_name = benchmark_config["name"]
        vlmevalkit_model = model_config["vlm_id"]
        vlmevalkit_benchmark = benchmark_config["vlm_name"]
        
        start_time = datetime.now()
        
        logger.info(f"Starting evaluation: {model_name} on {benchmark_name}")
        
        try:
            # Validate model availability first
            if not self._validate_model_availability(vlmevalkit_model):
                error_msg = f"Model '{vlmevalkit_model}' not available in VLMEvalKit"
                logger.error(error_msg)
                
                return EvaluationResult(
                    model_name=model_name,
                    benchmark_name=benchmark_name,
                    success=False,
                    start_time=start_time,
                    end_time=datetime.now(),
                    execution_time=0.0,
                    peak_memory=0.0,
                    error_message=error_msg
                )
            
            # Setup environment for this model
            self._setup_vlmevalkit_environment(model_config)
            
            # Store original working directory
            original_cwd = os.getcwd()
            
            try:
                # Change to VLMEvalKit directory for execution
                os.chdir(self.vlmevalkit_path)
                
                # Create model/benchmark directory structure for each evaluation
                model_benchmark_dir = self.output_dir / model_name / benchmark_name
                model_benchmark_dir.mkdir(parents=True, exist_ok=True)

                # Also create a work directory for VLMEvalKit
                eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                eval_work_dir = model_benchmark_dir / f"work_{eval_timestamp}"
                eval_work_dir.mkdir(parents=True, exist_ok=True)
                
                # Prepare VLMEvalKit command - use absolute paths
                cmd = [
                    sys.executable, "run.py",
                    "--model", vlmevalkit_model,
                    "--data", vlmevalkit_benchmark,
                    "--work-dir", str(eval_work_dir.absolute()),
                    "--mode", "all"
                ]

                # Test mode: limit to 2 samples for quick verification
                if self.test_mode:
                    cmd.extend(["--nproc", "2"])
                    logger.info("Test mode enabled: evaluating only 2 samples")

                if self.verbose:
                    cmd.append("--verbose")
                
                logger.info(f"VLMEvalKit command: {' '.join(cmd)}")
                logger.info(f"Working directory: {os.getcwd()}")
                logger.info(f"Results will be saved to: {eval_work_dir}")
                
                # Run evaluation with real-time output
                if self.monitor:
                    self.monitor.start_evaluation(model_name, benchmark_name)
                
                # Create subprocess with real-time output
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    cwd=self.vlmevalkit_path
                )
                
                # Collect output while showing real-time progress
                output_lines = []
                try:
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            break
                        
                        line = line.strip()
                        if line:
                            print(line)  # Real-time console output
                            output_lines.append(line)
                            
                            # Log important progress lines
                            if any(keyword in line for keyword in ['it/s', 'Infer', 'Progress', '%']):
                                logger.info(f"Progress: {line}")
                    
                    # Wait for process to complete
                    return_code = process.wait()
                    
                except subprocess.TimeoutExpired:
                    process.kill()
                    return_code = -1
                    output_lines.append("Process killed due to timeout")
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                if self.monitor:
                    peak_memory = self.monitor.end_evaluation()
                else:
                    peak_memory = 0.0
                
                # Check if evaluation succeeded
                if return_code == 0:
                    logger.info(f"Completed: {model_name} on {benchmark_name} ({execution_time:.1f}s)")
                    
                    # Find result files - look in unique work directory (CSV, XLSX, JSON)
                    result_files = []
                    for pattern in ["**/*.csv", "**/*.xlsx", "**/*.json"]:
                        result_files.extend(list(eval_work_dir.glob(pattern)))
                    logger.info(f"Found {len(result_files)} result files in work directory: {eval_work_dir}")
                    if result_files:
                        logger.info(f"Result files found: {[str(f) for f in result_files]}")
                        
                        # Prioritize accuracy files first, then other CSV/XLSX files
                        def get_file_priority(file_path):
                            name = file_path.name.lower()
                            # Highest priority: _acc.csv files
                            if name.endswith('_acc.csv'):
                                return (0, file_path.stat().st_mtime)
                            # Special priority for OCRBench score JSON files
                            elif name.endswith('_score.json') and 'ocrbench' in name:
                                return (0.5, file_path.stat().st_mtime)
                            # Second priority: other CSV files
                            elif name.endswith('.csv'):
                                return (1, file_path.stat().st_mtime)
                            # Third priority: XLSX files (but not openai_result.xlsx which is raw data)
                            elif name.endswith('.xlsx') and 'openai_result' not in name and 'results' not in name:
                                return (2, file_path.stat().st_mtime)
                            # Fourth priority: other XLSX files
                            elif name.endswith('.xlsx'):
                                return (3, file_path.stat().st_mtime)
                            # Lowest priority: other JSON files
                            else:
                                return (4, file_path.stat().st_mtime)
                        
                        result_files.sort(key=get_file_priority)
                    
                    # If no results in work dir, check VLMEvalKit's default outputs
                    if not result_files:
                        vlmevalkit_outputs = self.vlmevalkit_path / "outputs"
                        logger.info(f"Searching VLMEvalKit outputs directory: {vlmevalkit_outputs}")
                        if vlmevalkit_outputs.exists():
                            result_files = []
                            for pattern in ["**/*.csv", "**/*.xlsx", "**/*.json"]:
                                result_files.extend(list(vlmevalkit_outputs.glob(pattern)))
                            logger.info(f"Found {len(result_files)} result files in VLMEvalKit outputs")
                            # Copy most recent file to current evaluation
                            if result_files:
                                latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
                                target_file = eval_work_dir / latest_result.name
                                import shutil
                                shutil.copy2(latest_result, target_file)
                                result_files = [target_file]
                                logger.info(f"Copied result file from VLMEvalKit outputs: {latest_result} -> {target_file}")
                        else:
                            logger.warning(f"VLMEvalKit outputs directory does not exist: {vlmevalkit_outputs}")

                    # Copy all VLMEvalKit outputs to SpectraBench-Vision outputs directory
                    self._copy_vlmevalkit_outputs_to_spectravision(model_name, benchmark_name, eval_work_dir)

                    result_file = str(result_files[0]) if result_files else None
                    
                    if result_file:
                        logger.info(f"Found result file: {result_file}")
                    else:
                        logger.warning("No result file found")
                    
                    # Parse accuracy scores from result file
                    accuracy_score, subtask_scores, detailed_metrics, metric_type, metric_description = self._parse_vlmevalkit_results(
                        result_file, benchmark_name
                    )
                    
                    if accuracy_score is not None:
                        logger.info(f"Accuracy: {accuracy_score:.1%}")
                    else:
                        logger.warning("Could not parse accuracy score from results")
                    
                    evaluation_result = EvaluationResult(
                        model_name=model_name,
                        benchmark_name=benchmark_name,
                        success=True,
                        start_time=start_time,
                        end_time=end_time,
                        execution_time=execution_time,
                        peak_memory=peak_memory,
                        accuracy_score=accuracy_score,
                        subtask_scores=subtask_scores,
                        detailed_metrics=detailed_metrics,
                        metric_type=metric_type,
                        metric_description=metric_description,
                        result_file=result_file
                    )
                    
                    # Save model-specific results immediately
                    self._save_model_results(evaluation_result)
                    
                    # Generate detailed report
                    self._generate_evaluation_report(evaluation_result)
                    
                    # Cache cleanup after successful evaluation
                    self._cleanup_caches(self.cleanup_level)
                    
                    return evaluation_result
                else:
                    logger.error(f"Failed: {model_name} on {benchmark_name}")
                    logger.error(f"   Exit code: {return_code}")
                    
                    # Collect stderr information
                    stderr_info = '\n'.join(output_lines[-10:])  # Last 10 lines of output
                    
                    evaluation_result = EvaluationResult(
                        model_name=model_name,
                        benchmark_name=benchmark_name,
                        success=False,
                        start_time=start_time,
                        end_time=end_time,
                        execution_time=execution_time,
                        peak_memory=peak_memory,
                        error_message=stderr_info[:1000] if stderr_info else "Unknown error"
                    )
                    
                    # Save failed result and generate report
                    self._save_model_results(evaluation_result)
                    self._generate_evaluation_report(evaluation_result)
                    
                    # Cache cleanup even after failed evaluation (light cleanup)
                    self._cleanup_caches("light")
                    
                    return evaluation_result
                    
            finally:
                # Always restore working directory
                os.chdir(original_cwd)
                
        except subprocess.TimeoutExpired:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.error(f"Timeout: {model_name} on {benchmark_name} after {execution_time:.1f}s")
            
            evaluation_result = EvaluationResult(
                model_name=model_name,
                benchmark_name=benchmark_name,
                success=False,
                start_time=start_time,
                end_time=end_time,
                execution_time=execution_time,
                peak_memory=0.0,
                error_message="Evaluation timeout (4 hours)"
            )
            
            self._generate_evaluation_report(evaluation_result)
            self._cleanup_caches("light")
            return evaluation_result
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.error(f"Exception: {model_name} on {benchmark_name}: {str(e)}")
            
            evaluation_result = EvaluationResult(
                model_name=model_name,
                benchmark_name=benchmark_name,
                success=False,
                start_time=start_time,
                end_time=end_time,
                execution_time=execution_time,
                peak_memory=0.0,
                error_message=str(e)
            )
            
            self._generate_evaluation_report(evaluation_result)
            self._cleanup_caches("light")
            return evaluation_result
    
    def _save_model_results(self, result: EvaluationResult):
        """Save model-specific results in structured JSON format"""
        # Save individual model result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"{result.model_name}_{result.benchmark_name}_{timestamp}.json"
        result_file = self.model_results_dir / result_filename
        
        result_data = {
            "model_name": result.model_name,
            "benchmark_name": result.benchmark_name,
            "success": result.success,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat(),
            "execution_time": result.execution_time,
            "peak_memory": result.peak_memory,
            "accuracy_score": result.accuracy_score,
            "subtask_scores": result.subtask_scores,
            "detailed_metrics": result.detailed_metrics,
            "error_message": result.error_message,
            "result_file": result.result_file
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        logger.debug(f"Saved model result to: {result_file}")
    
    def _save_intermediate_results(self):
        """Save results incrementally during evaluation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_file = self.results_dir / f"intermediate_results_{timestamp}.json"
        
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "completed_evaluations": len(self.results),
            "total_evaluations": self.total_evaluations,
            "results": [
                {
                    "model_name": r.model_name,
                    "benchmark_name": r.benchmark_name,
                    "success": r.success,
                    "execution_time": r.execution_time,
                    "accuracy_score": r.accuracy_score,
                    "error_message": r.error_message
                }
                for r in self.results
            ]
        }
        
        with open(intermediate_file, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def run(self) -> Dict[str, Any]:
        """
        Run complete sequential evaluation
        
        Returns:
            Dictionary containing all results and summary statistics
        """
        logger.info(f"Starting sequential evaluation of {self.total_evaluations} combinations")
        
        overall_start = datetime.now()
        
        # Run all model-benchmark combinations
        for model_config in self.config["models"]:
            for benchmark_config in self.config["benchmarks"]:
                self.current_evaluation += 1
                
                progress = f"[{self.current_evaluation}/{self.total_evaluations}]"
                logger.info(f"{progress} Evaluating {model_config['name']} on {benchmark_config['name']}")
                
                # Run single evaluation (cache cleanup happens inside)
                result = self._run_vlmevalkit_evaluation(model_config, benchmark_config)
                self.results.append(result)
                
                # Save intermediate results
                self._save_intermediate_results()
                
                # Optional: Add delay between evaluations for cooling
                cooling_delay = self.config.get("hardware", {}).get("cooling_delay")
                if cooling_delay:
                    import time
                    time.sleep(cooling_delay)
        
        overall_end = datetime.now()
        total_time = (overall_end - overall_start).total_seconds()
        
        # Calculate summary statistics
        successful_evals = sum(1 for r in self.results if r.success)
        failed_evals = len(self.results) - successful_evals
        
        # Save final results
        final_results = {
            "timestamp": overall_end.isoformat(),
            "config": self.config,
            "summary": {
                "total_evaluations": len(self.results),
                "successful_evaluations": successful_evals,
                "failed_evaluations": failed_evals,
                "success_rate": successful_evals / len(self.results) * 100 if self.results else 0,
                "total_time": f"{total_time:.1f}s ({total_time/3600:.2f}h)",
                "average_time_per_eval": total_time / len(self.results) if self.results else 0
            },
            "evaluations": [
                {
                    "model_name": r.model_name,
                    "benchmark_name": r.benchmark_name,
                    "success": r.success,
                    "start_time": r.start_time.isoformat(),
                    "end_time": r.end_time.isoformat(),
                    "execution_time": r.execution_time,
                    "peak_memory": r.peak_memory,
                    "accuracy_score": r.accuracy_score,
                    "subtask_scores": r.subtask_scores,
                    "detailed_metrics": r.detailed_metrics,
                    "error_message": r.error_message,
                    "result_file": r.result_file
                }
                for r in self.results
            ]
        }
        
        # Save final results file
        final_results_file = self.results_dir / "final_results.json"
        with open(final_results_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        logger.info(f"Final results saved to: {final_results_file}")

        # Generate final report with accuracy tables
        self._generate_final_report(self.results)

        summary = {
            "total_evaluations": len(self.results),
            "successful_evaluations": successful_evals,
            "failed_evaluations": failed_evals,
            "success_rate": successful_evals / len(self.results) * 100 if self.results else 0,
            "total_time": f"{total_time:.1f}s ({total_time/3600:.2f}h)",
        }

        return {"summary": summary, "evaluations": self.results, "config": self.config}

    def _generate_final_report(self, results: List[EvaluationResult]):
        """Generate comprehensive final report with accuracy tables"""
        try:
            logger.info("Generating final report with accuracy tables...")

            # Prepare data for tables
            successful_results = [r for r in results if r.success and r.accuracy_score is not None]

            if not successful_results:
                logger.warning("No successful results with accuracy scores to generate final report")
                return

            # Create accuracy summary table
            accuracy_data = []
            model_benchmark_matrix = {}

            for result in successful_results:
                model = result.model_name
                benchmark = result.benchmark_name
                accuracy = result.accuracy_score

                # Handle different metric types
                if result.metric_type == "composite_score":
                    # For composite scores like MME, don't convert to percentage
                    formatted_accuracy = f"{accuracy:.2f}"
                else:
                    # Convert to percentage for regular accuracy metrics
                    if 0 <= accuracy <= 1:
                        formatted_accuracy = f"{accuracy:.1%}"
                    elif 1 < accuracy <= 100:
                        formatted_accuracy = f"{accuracy:.1f}%"
                    else:
                        formatted_accuracy = f"{accuracy:.2f}"

                accuracy_data.append({
                    'Model': model,
                    'Benchmark': benchmark,
                    'Accuracy': formatted_accuracy,
                    'Metric_Type': result.metric_type,
                    'Description': result.metric_description or result.metric_type,
                    'Execution_Time': f"{result.execution_time:.1f}s"
                })

                # Build matrix for pivot table
                if model not in model_benchmark_matrix:
                    model_benchmark_matrix[model] = {}
                model_benchmark_matrix[model][benchmark] = formatted_accuracy

            # Save accuracy summary CSV
            import pandas as pd

            accuracy_df = pd.DataFrame(accuracy_data)
            accuracy_summary_file = self.final_report_dir / "accuracy_summary.csv"
            accuracy_df.to_csv(accuracy_summary_file, index=False)
            logger.info(f"Accuracy summary saved to: {accuracy_summary_file}")

            # Create pivot table (Model vs Benchmark matrix)
            pivot_data = []
            all_models = sorted(model_benchmark_matrix.keys())
            all_benchmarks = sorted(set(benchmark for model_data in model_benchmark_matrix.values()
                                      for benchmark in model_data.keys()))

            for model in all_models:
                row = {'Model': model}
                for benchmark in all_benchmarks:
                    row[benchmark] = model_benchmark_matrix[model].get(benchmark, 'N/A')
                pivot_data.append(row)

            pivot_df = pd.DataFrame(pivot_data)
            pivot_table_file = self.final_report_dir / "accuracy_matrix.csv"
            pivot_df.to_csv(pivot_table_file, index=False)
            logger.info(f"Accuracy matrix saved to: {pivot_table_file}")

            # Generate detailed results with all metrics
            detailed_data = []
            for result in results:
                detailed_entry = {
                    'Model': result.model_name,
                    'Benchmark': result.benchmark_name,
                    'Status': 'Success' if result.success else 'Failed',
                    'Accuracy': result.accuracy_score if result.success else None,
                    'Metric_Type': result.metric_type if result.success else None,
                    'Execution_Time_s': result.execution_time,
                    'Peak_Memory_GB': result.peak_memory,
                    'Start_Time': result.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'End_Time': result.end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'Error_Message': result.error_message if not result.success else None
                }

                # Add subtask scores if available
                if result.subtask_scores:
                    for subtask, score in result.subtask_scores.items():
                        col_name = f"Subtask_{subtask}"
                        detailed_entry[col_name] = f"{score:.1%}" if isinstance(score, (int, float)) and 0 <= score <= 1 else score

                detailed_data.append(detailed_entry)

            detailed_df = pd.DataFrame(detailed_data)
            detailed_results_file = self.final_report_dir / "detailed_results.csv"
            detailed_df.to_csv(detailed_results_file, index=False)
            logger.info(f"Detailed results saved to: {detailed_results_file}")

            # Generate HTML performance report
            self._generate_html_report(accuracy_df, pivot_df, detailed_df, successful_results)

            # Generate summary statistics
            self._generate_summary_statistics(results)

            logger.info(f"Final report generated successfully in: {self.final_report_dir}")

        except Exception as e:
            logger.error(f"Error generating final report: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _generate_html_report(self, accuracy_df, pivot_df, detailed_df, successful_results):
        """Generate HTML performance report"""
        try:
            session_name = self.output_dir.name
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SpectraBench-Vision Performance Report - {session_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .success {{ color: #27ae60; }}
        .error {{ color: #e74c3c; }}
        .summary-stats {{ background: #ecf0f1; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>SpectraBench-Vision Performance Report</h1>
        <p>Session: {session_name} | Generated: {timestamp}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div class="summary-stats">
            <p><strong>Total Evaluations:</strong> {len(detailed_df)}</p>
            <p><strong>Successful:</strong> <span class="success">{len(successful_results)}</span></p>
            <p><strong>Failed:</strong> <span class="error">{len(detailed_df) - len(successful_results)}</span></p>
            <p><strong>Success Rate:</strong> {len(successful_results)/len(detailed_df)*100:.1f}%</p>
            <p><strong>Models Tested:</strong> {len(accuracy_df['Model'].unique())}</p>
            <p><strong>Benchmarks Tested:</strong> {len(accuracy_df['Benchmark'].unique())}</p>
        </div>
    </div>

    <div class="section">
        <h2>Accuracy Matrix (Model × Benchmark)</h2>
        {pivot_df.to_html(index=False, escape=False, classes='accuracy-matrix')}
    </div>

    <div class="section">
        <h2>Top Performing Models</h2>
        {self._get_top_models_html(accuracy_df)}
    </div>

    <div class="section">
        <h2>Benchmark Difficulty Ranking</h2>
        {self._get_benchmark_difficulty_html(accuracy_df)}
    </div>

    <div class="section">
        <h2>Detailed Accuracy Results</h2>
        {accuracy_df.to_html(index=False, escape=False)}
    </div>

</body>
</html>
"""

            html_file = self.final_report_dir / "performance_report.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"HTML performance report saved to: {html_file}")

        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")

    def _get_top_models_html(self, accuracy_df):
        """Generate top models HTML table"""
        try:
            # Calculate average accuracy per model (only for percentage metrics)
            numeric_df = accuracy_df.copy()
            numeric_df['Accuracy_Numeric'] = numeric_df['Accuracy'].str.rstrip('%').astype(float, errors='ignore')

            model_avg = numeric_df.groupby('Model')['Accuracy_Numeric'].mean().sort_values(ascending=False)
            top_models = model_avg.head(10)

            html = "<table><tr><th>Rank</th><th>Model</th><th>Average Accuracy</th></tr>"
            for rank, (model, avg_acc) in enumerate(top_models.items(), 1):
                html += f"<tr><td>{rank}</td><td>{model}</td><td>{avg_acc:.1f}%</td></tr>"
            html += "</table>"
            return html
        except:
            return "<p>Unable to calculate top models</p>"

    def _get_benchmark_difficulty_html(self, accuracy_df):
        """Generate benchmark difficulty HTML table"""
        try:
            # Calculate average accuracy per benchmark (only for percentage metrics)
            numeric_df = accuracy_df.copy()
            numeric_df['Accuracy_Numeric'] = numeric_df['Accuracy'].str.rstrip('%').astype(float, errors='ignore')

            benchmark_avg = numeric_df.groupby('Benchmark')['Accuracy_Numeric'].mean().sort_values(ascending=True)
            difficult_benchmarks = benchmark_avg.head(10)

            html = "<table><tr><th>Rank</th><th>Benchmark (Hardest First)</th><th>Average Accuracy</th></tr>"
            for rank, (benchmark, avg_acc) in enumerate(difficult_benchmarks.items(), 1):
                html += f"<tr><td>{rank}</td><td>{benchmark}</td><td>{avg_acc:.1f}%</td></tr>"
            html += "</table>"
            return html
        except:
            return "<p>Unable to calculate benchmark difficulty</p>"

    def _generate_summary_statistics(self, results):
        """Generate summary statistics file"""
        try:
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]

            stats = {
                'session_info': {
                    'session_name': self.output_dir.name,
                    'timestamp': datetime.now().isoformat(),
                    'total_evaluations': len(results),
                    'successful_evaluations': len(successful_results),
                    'failed_evaluations': len(failed_results),
                    'success_rate': len(successful_results) / len(results) * 100 if results else 0
                },
                'execution_stats': {
                    'total_execution_time_seconds': sum(r.execution_time for r in results),
                    'average_execution_time_seconds': sum(r.execution_time for r in results) / len(results) if results else 0,
                    'max_execution_time_seconds': max(r.execution_time for r in results) if results else 0,
                    'min_execution_time_seconds': min(r.execution_time for r in results) if results else 0,
                },
                'memory_stats': {
                    'max_memory_usage_gb': max(r.peak_memory for r in results) if results else 0,
                    'average_memory_usage_gb': sum(r.peak_memory for r in results) / len(results) if results else 0
                },
                'model_stats': {
                    'total_models': len(set(r.model_name for r in results)),
                    'models_list': sorted(list(set(r.model_name for r in results)))
                },
                'benchmark_stats': {
                    'total_benchmarks': len(set(r.benchmark_name for r in results)),
                    'benchmarks_list': sorted(list(set(r.benchmark_name for r in results)))
                }
            }

            stats_file = self.final_report_dir / "summary_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)

            logger.info(f"Summary statistics saved to: {stats_file}")

        except Exception as e:
            logger.error(f"Error generating summary statistics: {e}")