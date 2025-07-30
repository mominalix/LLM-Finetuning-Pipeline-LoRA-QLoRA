"""Custom trainer with enhanced functionality for LLM fine-tuning.

This module provides a custom trainer that extends the Hugging Face Trainer
with additional features for LoRA/QLoRA training, advanced logging, and
memory optimization.

Example:
    >>> from src.training import CustomTrainer
    >>> trainer = CustomTrainer(model=model, args=args, ...)
    >>> trainer.train()
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizer
)
from transformers.trainer_callback import TrainerCallback
from torch.utils.data import Dataset

import mlflow
from src.utils.gpu import log_gpu_memory, cleanup_gpu_memory


logger = logging.getLogger(__name__)


class CustomTrainer(Trainer):
    """Enhanced trainer with additional functionality for LLM fine-tuning.
    
    This trainer extends the base Hugging Face Trainer with:
    - Enhanced memory management
    - Custom logging and monitoring
    - LoRA-specific optimizations
    - Advanced evaluation metrics
    - MLflow integration
    
    Args:
        model: The model to train
        args: Training arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        tokenizer: Tokenizer for the model
        compute_metrics: Function to compute metrics
        callbacks: List of trainer callbacks
        **kwargs: Additional arguments passed to base Trainer
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        compute_metrics: Optional[callable] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        **kwargs
    ):
        """Initialize the custom trainer."""
        
        # Add custom callbacks if not provided
        if callbacks is None:
            callbacks = []
        
        # Add memory monitoring callback
        callbacks.append(MemoryMonitoringCallback())
        
        # Add MLflow logging callback
        callbacks.append(MLflowLoggingCallback())
        
        # Initialize parent trainer
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            **kwargs
        )
        
        # Store additional information
        self.custom_metrics_history = []
        self.best_metrics = {}
        
        logger.info("Initialized CustomTrainer with enhanced functionality")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute training loss with optional custom loss functions.
        
        Args:
            model: The model
            inputs: Input batch
            return_outputs: Whether to return model outputs
            
        Returns:
            Loss tensor and optionally model outputs
        """
        # Use parent implementation with potential for customization
        return super().compute_loss(model, inputs, return_outputs)
    
    def log(self, logs: Dict[str, float]) -> None:
        """Enhanced logging with custom metrics tracking.
        
        Args:
            logs: Dictionary of metrics to log
        """
        # Call parent logging
        super().log(logs)
        
        # Store metrics history
        step = self.state.global_step
        epoch = self.state.epoch
        
        log_entry = {
            "step": step,
            "epoch": epoch,
            "logs": logs.copy()
        }
        self.custom_metrics_history.append(log_entry)
        
        # Track best metrics
        for key, value in logs.items():
            if "eval_" in key and isinstance(value, (int, float)):
                metric_name = key.replace("eval_", "")
                if metric_name not in self.best_metrics:
                    self.best_metrics[metric_name] = {"value": value, "step": step}
                else:
                    # Assume higher is better for most metrics
                    if value > self.best_metrics[metric_name]["value"]:
                        self.best_metrics[metric_name] = {"value": value, "step": step}
        
        # Log GPU memory if available
        if torch.cuda.is_available() and step % 50 == 0:
            log_gpu_memory(logger, f"Step {step} - ")
    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """Enhanced evaluation with additional metrics.
        
        Args:
            eval_dataset: Dataset to evaluate on
            ignore_keys: Keys to ignore in metrics
            metric_key_prefix: Prefix for metric names
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Clean up GPU memory before evaluation
        if torch.cuda.is_available():
            cleanup_gpu_memory()
        
        # Run standard evaluation
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        # Add custom metrics if needed
        if hasattr(self, "_add_custom_eval_metrics"):
            custom_metrics = self._add_custom_eval_metrics(metrics)
            metrics.update(custom_metrics)
        
        return metrics
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Enhanced model saving with metadata.
        
        Args:
            output_dir: Directory to save the model
            _internal_call: Whether this is an internal call
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # Call parent save method
        super().save_model(output_dir, _internal_call)
        
        # Save additional metadata
        self._save_training_metadata(output_dir)
        
        logger.info(f"Model saved to: {output_dir}")
    
    def _save_training_metadata(self, output_dir: str) -> None:
        """Save additional training metadata.
        
        Args:
            output_dir: Directory to save metadata
        """
        metadata = {
            "training_completed": True,
            "total_steps": self.state.global_step,
            "best_metrics": self.best_metrics,
            "final_epoch": self.state.epoch,
            "model_name": getattr(self.model, "name_or_path", "unknown"),
        }
        
        # Add LoRA information if available
        if hasattr(self.model, "peft_config"):
            peft_config = self.model.peft_config
            if isinstance(peft_config, dict):
                peft_info = {}
                for key, config in peft_config.items():
                    peft_info[key] = {
                        "peft_type": config.peft_type.value if hasattr(config.peft_type, 'value') else str(config.peft_type),
                        "r": getattr(config, 'r', None),
                        "lora_alpha": getattr(config, 'lora_alpha', None),
                        "target_modules": getattr(config, 'target_modules', None),
                    }
                metadata["peft_config"] = peft_info
        
        # Save metadata to JSON file
        import json
        metadata_path = Path(output_dir) / "training_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Training metadata saved to: {metadata_path}")
    
    def get_train_dataloader(self):
        """Get training dataloader with memory optimizations."""
        dataloader = super().get_train_dataloader()
        
        # Apply memory optimizations if needed
        if torch.cuda.is_available() and hasattr(self.args, 'dataloader_pin_memory'):
            # Pin memory is already handled by TrainingArguments
            pass
        
        return dataloader
    
    def prediction_step(
        self,
        model,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        """Enhanced prediction step with memory management.
        
        Args:
            model: The model
            inputs: Input batch
            prediction_loss_only: Whether to return only the loss
            ignore_keys: Keys to ignore
            
        Returns:
            Prediction step outputs
        """
        # Memory cleanup for long evaluation runs
        if hasattr(self, '_prediction_step_count'):
            self._prediction_step_count += 1
        else:
            self._prediction_step_count = 1
        
        # Clean memory every 100 prediction steps
        if self._prediction_step_count % 100 == 0 and torch.cuda.is_available():
            cleanup_gpu_memory()
        
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


class MemoryMonitoringCallback(TrainerCallback):
    """Callback for monitoring GPU memory usage during training."""
    
    def __init__(self):
        self.memory_history = []
    
    def on_step_end(self, args, state, control, **kwargs):
        """Log memory usage at the end of each step."""
        if torch.cuda.is_available() and state.global_step % 100 == 0:
            from src.utils.gpu import monitor_gpu_memory
            memory_stats = monitor_gpu_memory()
            
            if memory_stats:
                self.memory_history.append({
                    "step": state.global_step,
                    "memory_allocated_gb": memory_stats["allocated_gb"],
                    "memory_utilization": memory_stats["utilization_percent"]
                })
                
                # Log warning if memory usage is high
                if memory_stats["utilization_percent"] > 90:
                    logger.warning(f"High GPU memory usage: {memory_stats['utilization_percent']:.1f}%")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Clean up memory at the end of each epoch."""
        if torch.cuda.is_available():
            cleanup_gpu_memory()
            logger.info(f"Cleaned GPU memory after epoch {state.epoch}")


class MLflowLoggingCallback(TrainerCallback):
    """Callback for enhanced MLflow logging."""
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log additional metrics to MLflow."""
        if not mlflow.active_run():
            return
        
        if logs:
            # Log all metrics
            for key, value in logs.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    mlflow.log_metric(key, value, step=state.global_step)
            
            # Log learning rate if available
            if "learning_rate" in logs:
                mlflow.log_metric("learning_rate", logs["learning_rate"], step=state.global_step)
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Log training start information."""
        if mlflow.active_run():
            mlflow.log_param("total_train_batch_size", args.train_batch_size)
            mlflow.log_param("num_train_epochs", args.num_train_epochs)
            mlflow.log_param("max_steps", args.max_steps)
            
            if model and hasattr(model, "config"):
                mlflow.log_param("model_type", model.config.model_type)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Log training completion information."""
        if mlflow.active_run():
            mlflow.log_metric("final_train_loss", state.log_history[-1].get("train_loss", 0))
            mlflow.log_metric("total_training_steps", state.global_step)


class EarlyStoppingCallback(TrainerCallback):
    """Enhanced early stopping callback with additional features."""
    
    def __init__(
        self,
        early_stopping_patience: int = 1,
        early_stopping_threshold: float = 0.0,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False
    ):
        """Initialize early stopping callback.
        
        Args:
            early_stopping_patience: Number of evaluations to wait
            early_stopping_threshold: Minimum improvement threshold
            metric_for_best_model: Metric to monitor
            greater_is_better: Whether higher values are better
        """
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        
        self.early_stopping_patience_counter = 0
        self.best_metric = None
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """Check for early stopping condition."""
        if logs is None:
            return
        
        current_metric = logs.get(self.metric_for_best_model)
        if current_metric is None:
            return
        
        if self.best_metric is None:
            self.best_metric = current_metric
        else:
            if self.greater_is_better:
                is_improvement = current_metric > self.best_metric + self.early_stopping_threshold
            else:
                is_improvement = current_metric < self.best_metric - self.early_stopping_threshold
            
            if is_improvement:
                self.best_metric = current_metric
                self.early_stopping_patience_counter = 0
            else:
                self.early_stopping_patience_counter += 1
        
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            logger.info(f"Early stopping triggered after {self.early_stopping_patience} evaluations")
            control.should_training_stop = True