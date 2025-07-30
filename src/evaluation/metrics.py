"""Metrics computation for various NLP tasks.

This module provides comprehensive metrics computation for different task types
including classification, token classification, question answering, and generation tasks.

Example:
    >>> from src.evaluation import MetricsComputer
    >>> computer = MetricsComputer(task_type="sequence_classification", metrics=["accuracy", "f1"])
    >>> metrics = computer.compute_metrics(eval_pred)
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass

import evaluate
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef
)
from transformers import EvalPrediction

from src.core.config import TaskType


logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Container for metric computation results."""
    name: str
    value: Union[float, Dict[str, float]]
    description: str


class MetricsComputer:
    """Comprehensive metrics computer for various NLP tasks.
    
    This class provides a unified interface for computing evaluation metrics
    across different task types with support for both standard and custom metrics.
    
    Args:
        task_type: Type of task (classification, token_classification, etc.)
        metrics: List of metric names to compute
        num_labels: Number of labels for classification tasks
        label_names: Names of the labels
        average: Averaging strategy for multi-class metrics
        
    Example:
        >>> computer = MetricsComputer(
        ...     task_type="sequence_classification",
        ...     metrics=["accuracy", "f1", "precision", "recall"]
        ... )
        >>> metrics = computer.compute_metrics(eval_pred)
    """
    
    def __init__(
        self,
        task_type: Union[str, TaskType],
        metrics: List[str] = None,
        num_labels: Optional[int] = None,
        label_names: Optional[List[str]] = None,
        average: str = "weighted",
        custom_metrics: Optional[Dict[str, Callable]] = None
    ):
        """Initialize the metrics computer.
        
        Args:
            task_type: Task type for metric computation
            metrics: List of metrics to compute
            num_labels: Number of labels for classification
            label_names: Label names for classification
            average: Averaging strategy for multi-class metrics
            custom_metrics: Dictionary of custom metric functions
        """
        self.task_type = task_type if isinstance(task_type, str) else task_type.value
        self.metrics = metrics or self._get_default_metrics()
        self.num_labels = num_labels
        self.label_names = label_names
        self.average = average
        self.custom_metrics = custom_metrics or {}
        
        # Initialize metric functions
        self._metric_functions = {}
        self._load_metric_functions()
        
        logger.info(f"Initialized MetricsComputer for {self.task_type}")
        logger.info(f"Computing metrics: {self.metrics}")
    
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute all configured metrics.
        
        Args:
            eval_pred: Evaluation predictions and labels
            
        Returns:
            Dictionary of computed metrics
            
        Example:
            >>> metrics = computer.compute_metrics(eval_pred)
            >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
        """
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        
        # Process predictions based on task type
        if self.task_type == "sequence_classification":
            predictions = self._process_classification_predictions(predictions)
        elif self.task_type == "token_classification":
            predictions, labels = self._process_token_classification_predictions(predictions, labels)
        
        # Compute all metrics
        computed_metrics = {}
        
        for metric_name in self.metrics:
            try:
                if metric_name in self.custom_metrics:
                    # Use custom metric function
                    metric_value = self.custom_metrics[metric_name](predictions, labels)
                elif metric_name in self._metric_functions:
                    # Use built-in metric function
                    metric_value = self._metric_functions[metric_name](predictions, labels)
                else:
                    # Try to load from HuggingFace evaluate library
                    metric_value = self._compute_hf_metric(metric_name, predictions, labels)
                
                if isinstance(metric_value, dict):
                    computed_metrics.update(metric_value)
                else:
                    computed_metrics[metric_name] = float(metric_value)
                    
            except Exception as e:
                logger.warning(f"Failed to compute metric {metric_name}: {str(e)}")
                computed_metrics[metric_name] = 0.0
        
        # Add task-specific metrics
        if self.task_type == "sequence_classification":
            computed_metrics.update(self._compute_classification_metrics(predictions, labels))
        
        return computed_metrics
    
    def _get_default_metrics(self) -> List[str]:
        """Get default metrics for the task type.
        
        Returns:
            List of default metric names
        """
        defaults = {
            "sequence_classification": ["accuracy", "f1", "precision", "recall"],
            "token_classification": ["accuracy", "f1", "precision", "recall"],
            "question_answering": ["exact_match", "f1"],
            "text_generation": ["bleu", "rouge"],
            "summarization": ["rouge"],
            "translation": ["bleu", "sacrebleu"]
        }
        
        return defaults.get(self.task_type, ["accuracy"])
    
    def _load_metric_functions(self) -> None:
        """Load metric computation functions."""
        self._metric_functions = {
            "accuracy": self._compute_accuracy,
            "f1": self._compute_f1,
            "precision": self._compute_precision,
            "recall": self._compute_recall,
            "mcc": self._compute_mcc,
            "auc": self._compute_auc,
        }
    
    def _process_classification_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Process predictions for classification tasks.
        
        Args:
            predictions: Raw model predictions
            
        Returns:
            Processed predictions
        """
        if predictions.ndim > 1:
            # Take argmax for multi-class classification
            return np.argmax(predictions, axis=1)
        return predictions
    
    def _process_token_classification_predictions(
        self, 
        predictions: np.ndarray, 
        labels: np.ndarray
    ) -> tuple:
        """Process predictions for token classification tasks.
        
        Args:
            predictions: Raw model predictions
            labels: True labels
            
        Returns:
            Tuple of processed (predictions, labels)
        """
        if predictions.ndim > 2:
            predictions = np.argmax(predictions, axis=2)
        
        # Flatten and remove padding tokens (label = -100)
        flat_predictions = []
        flat_labels = []
        
        for pred_seq, label_seq in zip(predictions, labels):
            for pred, label in zip(pred_seq, label_seq):
                if label != -100:  # Ignore padding tokens
                    flat_predictions.append(pred)
                    flat_labels.append(label)
        
        return np.array(flat_predictions), np.array(flat_labels)
    
    def _compute_accuracy(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute accuracy score.
        
        Args:
            predictions: Model predictions
            labels: True labels
            
        Returns:
            Accuracy score
        """
        return accuracy_score(labels, predictions)
    
    def _compute_f1(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute F1 score.
        
        Args:
            predictions: Model predictions
            labels: True labels
            
        Returns:
            F1 score
        """
        if self.num_labels == 2:
            # Binary classification
            _, _, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='binary', zero_division=0
            )
        else:
            # Multi-class classification
            _, _, f1, _ = precision_recall_fscore_support(
                labels, predictions, average=self.average, zero_division=0
            )
        
        return float(f1)
    
    def _compute_precision(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute precision score.
        
        Args:
            predictions: Model predictions
            labels: True labels
            
        Returns:
            Precision score
        """
        if self.num_labels == 2:
            precision, _, _, _ = precision_recall_fscore_support(
                labels, predictions, average='binary', zero_division=0
            )
        else:
            precision, _, _, _ = precision_recall_fscore_support(
                labels, predictions, average=self.average, zero_division=0
            )
        
        return float(precision)
    
    def _compute_recall(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute recall score.
        
        Args:
            predictions: Model predictions
            labels: True labels
            
        Returns:
            Recall score
        """
        if self.num_labels == 2:
            _, recall, _, _ = precision_recall_fscore_support(
                labels, predictions, average='binary', zero_division=0
            )
        else:
            _, recall, _, _ = precision_recall_fscore_support(
                labels, predictions, average=self.average, zero_division=0
            )
        
        return float(recall)
    
    def _compute_mcc(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute Matthews Correlation Coefficient.
        
        Args:
            predictions: Model predictions
            labels: True labels
            
        Returns:
            MCC score
        """
        return matthews_corrcoef(labels, predictions)
    
    def _compute_auc(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute AUC score (for binary classification).
        
        Args:
            predictions: Model predictions (probabilities)
            labels: True labels
            
        Returns:
            AUC score
        """
        if self.num_labels != 2:
            logger.warning("AUC metric only applicable for binary classification")
            return 0.0
        
        try:
            # If predictions are class indices, convert to probabilities
            if predictions.max() <= 1 and predictions.min() >= 0:
                return roc_auc_score(labels, predictions)
            else:
                # Assume predictions are logits or probabilities
                return 0.0
        except ValueError:
            logger.warning("Could not compute AUC score")
            return 0.0
    
    def _compute_hf_metric(
        self, 
        metric_name: str, 
        predictions: np.ndarray, 
        labels: np.ndarray
    ) -> float:
        """Compute metric using HuggingFace evaluate library.
        
        Args:
            metric_name: Name of the metric
            predictions: Model predictions
            labels: True labels
            
        Returns:
            Metric value
        """
        try:
            metric = evaluate.load(metric_name)
            result = metric.compute(predictions=predictions, references=labels)
            
            # Extract the main metric value
            if isinstance(result, dict):
                # Try common key names
                for key in [metric_name, f"{metric_name}_score", "score"]:
                    if key in result:
                        return result[key]
                # Return the first numeric value
                for value in result.values():
                    if isinstance(value, (int, float)):
                        return value
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to compute HF metric {metric_name}: {str(e)}")
            return 0.0
    
    def _compute_classification_metrics(
        self, 
        predictions: np.ndarray, 
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute comprehensive classification metrics.
        
        Args:
            predictions: Model predictions
            labels: True labels
            
        Returns:
            Dictionary of additional metrics
        """
        additional_metrics = {}
        
        try:
            # Per-class metrics
            if self.num_labels and self.num_labels > 2:
                precision, recall, f1, support = precision_recall_fscore_support(
                    labels, predictions, average=None, zero_division=0
                )
                
                for i in range(len(precision)):
                    label_name = self.label_names[i] if self.label_names else f"class_{i}"
                    additional_metrics[f"precision_{label_name}"] = precision[i]
                    additional_metrics[f"recall_{label_name}"] = recall[i]
                    additional_metrics[f"f1_{label_name}"] = f1[i]
            
            # Macro and micro averages
            if self.num_labels and self.num_labels > 2:
                for avg_type in ["macro", "micro"]:
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        labels, predictions, average=avg_type, zero_division=0
                    )
                    additional_metrics[f"precision_{avg_type}"] = precision
                    additional_metrics[f"recall_{avg_type}"] = recall
                    additional_metrics[f"f1_{avg_type}"] = f1
            
        except Exception as e:
            logger.warning(f"Failed to compute additional classification metrics: {str(e)}")
        
        return additional_metrics
    
    def get_classification_report(
        self, 
        predictions: np.ndarray, 
        labels: np.ndarray
    ) -> str:
        """Generate detailed classification report.
        
        Args:
            predictions: Model predictions
            labels: True labels
            
        Returns:
            Classification report string
        """
        target_names = self.label_names if self.label_names else None
        
        return classification_report(
            labels, 
            predictions, 
            target_names=target_names,
            digits=4,
            zero_division=0
        )
    
    def get_confusion_matrix(
        self, 
        predictions: np.ndarray, 
        labels: np.ndarray
    ) -> np.ndarray:
        """Generate confusion matrix.
        
        Args:
            predictions: Model predictions
            labels: True labels
            
        Returns:
            Confusion matrix array
        """
        return confusion_matrix(labels, predictions)
    
    def add_custom_metric(self, name: str, metric_function: Callable) -> None:
        """Add a custom metric function.
        
        Args:
            name: Name of the metric
            metric_function: Function that takes (predictions, labels) and returns a float
            
        Example:
            >>> def custom_accuracy(preds, labels):
            ...     return (preds == labels).mean()
            >>> computer.add_custom_metric("custom_acc", custom_accuracy)
        """
        self.custom_metrics[name] = metric_function
        if name not in self.metrics:
            self.metrics.append(name)
        
        logger.info(f"Added custom metric: {name}")


class RegressionMetricsComputer(MetricsComputer):
    """Specialized metrics computer for regression tasks."""
    
    def __init__(self, metrics: List[str] = None):
        """Initialize regression metrics computer.
        
        Args:
            metrics: List of metrics to compute
        """
        if metrics is None:
            metrics = ["mse", "mae", "r2"]
        
        super().__init__(task_type="regression", metrics=metrics)
        
        # Add regression-specific metrics
        self._metric_functions.update({
            "mse": self._compute_mse,
            "mae": self._compute_mae,
            "r2": self._compute_r2,
            "rmse": self._compute_rmse
        })
    
    def _compute_mse(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute Mean Squared Error."""
        return float(np.mean((predictions - labels) ** 2))
    
    def _compute_mae(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute Mean Absolute Error."""
        return float(np.mean(np.abs(predictions - labels)))
    
    def _compute_r2(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute R-squared score."""
        ss_res = np.sum((labels - predictions) ** 2)
        ss_tot = np.sum((labels - np.mean(labels)) ** 2)
        return float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
    
    def _compute_rmse(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute Root Mean Squared Error."""
        return float(np.sqrt(np.mean((predictions - labels) ** 2)))


def create_metrics_computer(
    task_type: str,
    num_labels: Optional[int] = None,
    metrics: Optional[List[str]] = None,
    **kwargs
) -> MetricsComputer:
    """Factory function to create appropriate metrics computer.
    
    Args:
        task_type: Type of task
        num_labels: Number of labels for classification
        metrics: List of metrics to compute
        **kwargs: Additional arguments
        
    Returns:
        Configured metrics computer
        
    Example:
        >>> computer = create_metrics_computer(
        ...     task_type="sequence_classification",
        ...     num_labels=3,
        ...     metrics=["accuracy", "f1"]
        ... )
    """
    if task_type == "regression":
        return RegressionMetricsComputer(metrics=metrics)
    else:
        return MetricsComputer(
            task_type=task_type,
            num_labels=num_labels,
            metrics=metrics,
            **kwargs
        )