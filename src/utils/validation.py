"""Data validation utilities for ensuring dataset quality.

This module provides functions to validate datasets and ensure they meet
the requirements for training and evaluation.

Example:
    >>> from src.utils.validation import validate_dataset
    >>> from datasets import DatasetDict
    >>> 
    >>> validate_dataset(datasets, config)
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
from datasets import Dataset, DatasetDict

from src.core.config import DatasetConfig, TaskType


logger = logging.getLogger(__name__)


def validate_dataset(datasets: DatasetDict, config: DatasetConfig) -> None:
    """Validate dataset structure and content.
    
    Args:
        datasets: DatasetDict to validate
        config: Dataset configuration
        
    Raises:
        ValueError: If dataset validation fails
    """
    logger.info("Validating dataset...")
    
    # Check if datasets is not empty
    if not datasets:
        raise ValueError("Dataset is empty")
    
    # Validate each split
    for split_name, dataset in datasets.items():
        _validate_split(dataset, split_name, config)
    
    # Validate split consistency
    _validate_split_consistency(datasets, config)
    
    # Task-specific validation
    _validate_task_requirements(datasets, config)
    
    logger.info("Dataset validation passed")


def _validate_split(dataset: Dataset, split_name: str, config: DatasetConfig) -> None:
    """Validate individual dataset split.
    
    Args:
        dataset: Dataset split to validate
        split_name: Name of the split
        config: Dataset configuration
        
    Raises:
        ValueError: If split validation fails
    """
    # Check if split is not empty
    if len(dataset) == 0:
        raise ValueError(f"Split '{split_name}' is empty")
    
    # Check required columns
    required_columns = {"text"}
    if config.label_column and config.label_column != "text":
        required_columns.add("label")
    
    missing_columns = required_columns - set(dataset.column_names)
    if missing_columns:
        raise ValueError(f"Split '{split_name}' missing required columns: {missing_columns}")
    
    # Validate text column
    _validate_text_column(dataset, split_name)
    
    # Validate label column if present
    if "label" in dataset.column_names:
        _validate_label_column(dataset, split_name, config)


def _validate_text_column(dataset: Dataset, split_name: str) -> None:
    """Validate text column content.
    
    Args:
        dataset: Dataset to validate
        split_name: Name of the split
        
    Raises:
        ValueError: If text validation fails
    """
    text_column = dataset["text"]
    
    # Check for None values
    none_count = sum(1 for text in text_column if text is None)
    if none_count > 0:
        logger.warning(f"Split '{split_name}' contains {none_count} None text values")
    
    # Check for empty strings
    empty_count = sum(1 for text in text_column if isinstance(text, str) and not text.strip())
    if empty_count > 0:
        logger.warning(f"Split '{split_name}' contains {empty_count} empty text values")
    
    # Check text length distribution
    text_lengths = [len(str(text).split()) if text else 0 for text in text_column[:1000]]
    if text_lengths:
        mean_length = np.mean(text_lengths)
        max_length = np.max(text_lengths)
        
        logger.info(f"Split '{split_name}' text stats: "
                   f"mean_length={mean_length:.1f} words, max_length={max_length} words")
        
        # Warn about very short or very long texts
        if mean_length < 3:
            logger.warning(f"Split '{split_name}' has very short texts (mean: {mean_length:.1f} words)")
        
        if max_length > 1000:
            logger.warning(f"Split '{split_name}' has very long texts (max: {max_length} words)")


def _validate_label_column(dataset: Dataset, split_name: str, config: DatasetConfig) -> None:
    """Validate label column content.
    
    Args:
        dataset: Dataset to validate
        split_name: Name of the split
        config: Dataset configuration
        
    Raises:
        ValueError: If label validation fails
    """
    labels = dataset["label"]
    
    # Check for None values
    none_count = sum(1 for label in labels if label is None)
    if none_count > 0:
        raise ValueError(f"Split '{split_name}' contains {none_count} None label values")
    
    # Get unique labels
    unique_labels = list(set(labels))
    unique_labels = [label for label in unique_labels if label is not None]
    
    logger.info(f"Split '{split_name}' contains {len(unique_labels)} unique labels: {unique_labels}")
    
    # Validate label types
    label_types = set(type(label).__name__ for label in unique_labels)
    if len(label_types) > 1:
        logger.warning(f"Split '{split_name}' has mixed label types: {label_types}")
    
    # Check if labels are integers for classification
    if config.label_column == "label":
        non_int_labels = [label for label in unique_labels if not isinstance(label, (int, np.integer))]
        if non_int_labels:
            logger.warning(f"Split '{split_name}' has non-integer labels: {non_int_labels}")
    
    # Check label range for classification
    if all(isinstance(label, (int, np.integer)) for label in unique_labels):
        min_label = min(unique_labels)
        max_label = max(unique_labels)
        
        if min_label < 0:
            logger.warning(f"Split '{split_name}' has negative labels (min: {min_label})")
        
        # Check for gaps in label sequence
        expected_labels = set(range(min_label, max_label + 1))
        missing_labels = expected_labels - set(unique_labels)
        if missing_labels:
            logger.warning(f"Split '{split_name}' missing labels: {sorted(missing_labels)}")


def _validate_split_consistency(datasets: DatasetDict, config: DatasetConfig) -> None:
    """Validate consistency across dataset splits.
    
    Args:
        datasets: All dataset splits
        config: Dataset configuration
        
    Raises:
        ValueError: If consistency validation fails
    """
    if len(datasets) < 2:
        return  # Nothing to compare
    
    # Check column consistency
    column_sets = [set(dataset.column_names) for dataset in datasets.values()]
    if not all(cols == column_sets[0] for cols in column_sets):
        logger.warning("Dataset splits have different columns")
        for split_name, dataset in datasets.items():
            logger.warning(f"  {split_name}: {dataset.column_names}")
    
    # Check label consistency
    if "label" in datasets[list(datasets.keys())[0]].column_names:
        _validate_label_consistency(datasets)


def _validate_label_consistency(datasets: DatasetDict) -> None:
    """Validate label consistency across splits.
    
    Args:
        datasets: All dataset splits with labels
    """
    all_labels = set()
    split_labels = {}
    
    for split_name, dataset in datasets.items():
        labels = set(label for label in dataset["label"] if label is not None)
        split_labels[split_name] = labels
        all_labels.update(labels)
    
    # Check if all splits have all labels (important for classification)
    for split_name, labels in split_labels.items():
        missing_labels = all_labels - labels
        if missing_labels and split_name != "test":  # Test set might not have all labels
            logger.warning(f"Split '{split_name}' missing labels: {sorted(missing_labels)}")
    
    # Log label distribution
    for split_name, dataset in datasets.items():
        if len(dataset) > 0:
            labels = [label for label in dataset["label"] if label is not None]
            if labels:
                unique, counts = np.unique(labels, return_counts=True)
                label_dist = dict(zip(unique, counts))
                logger.info(f"Split '{split_name}' label distribution: {label_dist}")


def _validate_task_requirements(datasets: DatasetDict, config: DatasetConfig) -> None:
    """Validate task-specific requirements.
    
    Args:
        datasets: All dataset splits
        config: Dataset configuration
    """
    # This function can be extended for specific task validations
    # For now, just log the task type
    logger.info(f"Dataset validated for task type: {config.name}")


def validate_model_config(model_config: Any, datasets: DatasetDict) -> None:
    """Validate model configuration against dataset.
    
    Args:
        model_config: Model configuration
        datasets: Dataset to validate against
        
    Raises:
        ValueError: If model config is incompatible with dataset
    """
    # Check if num_labels matches dataset
    if hasattr(model_config, 'num_labels') and model_config.num_labels:
        if "label" in datasets[list(datasets.keys())[0]].column_names:
            dataset_labels = set()
            for dataset in datasets.values():
                dataset_labels.update(label for label in dataset["label"] if label is not None)
            
            max_label = max(dataset_labels) if dataset_labels else 0
            expected_num_labels = max_label + 1 if isinstance(max_label, int) else len(dataset_labels)
            
            if model_config.num_labels != expected_num_labels:
                logger.warning(f"Model num_labels ({model_config.num_labels}) doesn't match "
                             f"dataset labels ({expected_num_labels})")


def check_dataset_balance(datasets: DatasetDict) -> Dict[str, Dict[str, float]]:
    """Check dataset balance across labels.
    
    Args:
        datasets: Dataset splits to check
        
    Returns:
        Dictionary with balance statistics per split
    """
    balance_stats = {}
    
    for split_name, dataset in datasets.items():
        if "label" in dataset.column_names:
            labels = [label for label in dataset["label"] if label is not None]
            if labels:
                unique, counts = np.unique(labels, return_counts=True)
                total = sum(counts)
                
                # Calculate balance ratio (min/max class ratio)
                if len(counts) > 1:
                    balance_ratio = min(counts) / max(counts)
                    
                    # Calculate entropy (higher = more balanced)
                    proportions = counts / total
                    entropy = -sum(p * np.log2(p) for p in proportions if p > 0)
                    max_entropy = np.log2(len(proportions))
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    
                    balance_stats[split_name] = {
                        "balance_ratio": balance_ratio,
                        "entropy": normalized_entropy,
                        "num_classes": len(unique),
                        "total_samples": total,
                        "class_distribution": dict(zip(unique, counts))
                    }
                    
                    # Log warnings for imbalanced datasets
                    if balance_ratio < 0.1:
                        logger.warning(f"Split '{split_name}' is highly imbalanced "
                                     f"(balance ratio: {balance_ratio:.3f})")
                    elif balance_ratio < 0.5:
                        logger.warning(f"Split '{split_name}' is moderately imbalanced "
                                     f"(balance ratio: {balance_ratio:.3f})")
    
    return balance_stats