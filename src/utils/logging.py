"""Logging configuration and utilities.

This module provides centralized logging setup for the entire pipeline,
ensuring consistent log formatting and handling across all components.

Example:
    >>> from src.utils.logging import setup_logging
    >>> setup_logging(log_level="INFO", log_file="training.log")
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


def setup_logging(
    log_level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None,
    include_timestamp: bool = True,
    console_output: bool = True
) -> logging.Logger:
    """Setup centralized logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs
        log_format: Custom log format string
        include_timestamp: Whether to include timestamp in logs
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logging("INFO", "training.log")
        >>> logger.info("Training started")
    """
    # Convert string log level to logging constant
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())
    
    # Default log format
    if log_format is None:
        if include_timestamp:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            log_format = "%(name)s - %(levelname)s - %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(
        fmt=log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Get root logger
    logger = logging.getLogger()
    
    # Clear existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.setLevel(log_level)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Set specific logger levels for external libraries
    _configure_external_loggers()
    
    return logger


def _configure_external_loggers() -> None:
    """Configure logging levels for external libraries."""
    # Reduce verbosity of external libraries
    external_loggers = {
        "transformers": logging.WARNING,
        "datasets": logging.WARNING,
        "tokenizers": logging.WARNING,
        "urllib3": logging.WARNING,
        "requests": logging.WARNING,
        "huggingface_hub": logging.WARNING,
        "torch": logging.WARNING,
        "matplotlib": logging.WARNING,
    }
    
    for logger_name, level in external_loggers.items():
        logging.getLogger(logger_name).setLevel(level)


def create_experiment_logger(
    experiment_name: str,
    output_dir: Union[str, Path],
    log_level: str = "INFO"
) -> logging.Logger:
    """Create a logger specific to an experiment.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Output directory for logs
        log_level: Logging level
        
    Returns:
        Experiment-specific logger
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for unique log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_path / f"{experiment_name}_{timestamp}.log"
    
    # Setup logging
    logger = setup_logging(
        log_level=log_level,
        log_file=log_file,
        console_output=True
    )
    
    logger.info(f"Started experiment: {experiment_name}")
    logger.info(f"Log file: {log_file}")
    
    return logger


class ProgressLogger:
    """Logger for training progress with rate limiting."""
    
    def __init__(self, logger: logging.Logger, log_every_n: int = 10):
        """Initialize progress logger.
        
        Args:
            logger: Base logger instance
            log_every_n: Log every N steps
        """
        self.logger = logger
        self.log_every_n = log_every_n
        self.step_count = 0
    
    def log_step(self, message: str, level: int = logging.INFO) -> None:
        """Log message only every N steps.
        
        Args:
            message: Message to log
            level: Logging level
        """
        self.step_count += 1
        if self.step_count % self.log_every_n == 0:
            self.logger.log(level, f"Step {self.step_count}: {message}")
    
    def log_always(self, message: str, level: int = logging.INFO) -> None:
        """Log message regardless of step count.
        
        Args:
            message: Message to log
            level: Logging level
        """
        self.logger.log(level, message)


class MetricsLogger:
    """Logger for training and evaluation metrics."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize metrics logger.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
        self.metrics_history = []
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None, prefix: str = "") -> None:
        """Log metrics with optional step information.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
            prefix: Optional prefix for metric names
        """
        # Format metrics for logging
        formatted_metrics = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_metrics.append(f"{prefix}{key}: {value:.4f}")
            else:
                formatted_metrics.append(f"{prefix}{key}: {value}")
        
        # Create log message
        if step is not None:
            message = f"Step {step} - " + ", ".join(formatted_metrics)
        else:
            message = ", ".join(formatted_metrics)
        
        self.logger.info(message)
        
        # Store in history
        metric_entry = {
            "step": step,
            "metrics": metrics.copy(),
            "timestamp": datetime.now()
        }
        self.metrics_history.append(metric_entry)
    
    def log_epoch_summary(self, epoch: int, train_metrics: dict, eval_metrics: dict) -> None:
        """Log summary of epoch results.
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            eval_metrics: Evaluation metrics
        """
        self.logger.info(f"=== Epoch {epoch} Summary ===")
        
        if train_metrics:
            self.log_metrics(train_metrics, prefix="train_")
        
        if eval_metrics:
            self.log_metrics(eval_metrics, prefix="eval_")
        
        self.logger.info("=" * 30)
    
    def get_best_metric(self, metric_name: str, higher_is_better: bool = True) -> dict:
        """Get the best value for a specific metric.
        
        Args:
            metric_name: Name of the metric
            higher_is_better: Whether higher values are better
            
        Returns:
            Dictionary with best metric information
        """
        relevant_entries = [
            entry for entry in self.metrics_history
            if metric_name in entry["metrics"]
        ]
        
        if not relevant_entries:
            return {}
        
        if higher_is_better:
            best_entry = max(relevant_entries, key=lambda x: x["metrics"][metric_name])
        else:
            best_entry = min(relevant_entries, key=lambda x: x["metrics"][metric_name])
        
        return {
            "value": best_entry["metrics"][metric_name],
            "step": best_entry["step"],
            "timestamp": best_entry["timestamp"]
        }


def log_system_info(logger: logging.Logger) -> None:
    """Log system information for debugging.
    
    Args:
        logger: Logger instance
    """
    import platform
    import psutil
    import torch
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # GPU information
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.info("CUDA not available")
    
    logger.info("=" * 30)