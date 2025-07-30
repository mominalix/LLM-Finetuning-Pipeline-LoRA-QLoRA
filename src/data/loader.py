"""Data loading functionality for various dataset sources.

This module provides a unified interface for loading datasets from different sources
including HuggingFace Hub, local files, and custom datasets. It includes validation,
preprocessing, and splitting capabilities.

Example:
    >>> from src.data import DataLoader
    >>> from src.core.config import DatasetConfig
    >>> 
    >>> config = DatasetConfig(name="imdb", train_split="train", test_split="test")
    >>> loader = DataLoader(config)
    >>> datasets = loader.load()
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np

from datasets import (
    Dataset,
    DatasetDict,
    load_dataset,
    load_from_disk,
    concatenate_datasets
)
from sklearn.model_selection import train_test_split

from src.core.config import DatasetConfig
from src.utils.validation import validate_dataset


logger = logging.getLogger(__name__)


class DataLoader:
    """Unified data loader for various dataset sources.
    
    Supports loading from:
    - HuggingFace Hub datasets
    - Local datasets (CSV, JSON, Parquet)
    - Custom datasets with preprocessing
    - Cached datasets
    
    Args:
        config: Dataset configuration
        
    Example:
        >>> config = DatasetConfig(name="imdb", train_split="train")
        >>> loader = DataLoader(config)
        >>> datasets = loader.load()
        >>> print(len(datasets["train"]))
    """
    
    def __init__(self, config: DatasetConfig):
        """Initialize the data loader.
        
        Args:
            config: Dataset configuration object
        """
        self.config = config
        self.datasets = None
        
        logger.info(f"Initialized DataLoader for dataset: {config.name}")
    
    def load(self) -> DatasetDict:
        """Load and prepare the dataset.
        
        Returns:
            DatasetDict containing train/validation/test splits
            
        Raises:
            ValueError: If dataset cannot be loaded or is invalid
            FileNotFoundError: If local dataset file not found
        """
        logger.info(f"Loading dataset: {self.config.name}")
        
        try:
            # Try loading from cache first
            if self._has_cached_dataset():
                logger.info("Loading from cache...")
                datasets = self._load_from_cache()
            else:
                # Load from source
                datasets = self._load_from_source()
                
                # Cache the dataset if cache directory is specified
                if self.config.cache_dir:
                    self._cache_dataset(datasets)
            
            # Validate the loaded dataset
            validate_dataset(datasets, self.config)
            
            # Apply preprocessing
            datasets = self._preprocess_datasets(datasets)
            
            # Apply data filtering if needed
            datasets = self._filter_datasets(datasets)
            
            # Log dataset statistics
            self._log_dataset_stats(datasets)
            
            self.datasets = datasets
            return datasets
            
        except Exception as e:
            logger.error(f"Failed to load dataset {self.config.name}: {str(e)}")
            raise
    
    def _load_from_source(self) -> DatasetDict:
        """Load dataset from the original source.
        
        Returns:
            Raw DatasetDict from source
        """
        if self._is_huggingface_dataset():
            return self._load_huggingface_dataset()
        elif self._is_local_file():
            return self._load_local_dataset()
        else:
            raise ValueError(f"Unsupported dataset source: {self.config.name}")
    
    def _is_huggingface_dataset(self) -> bool:
        """Check if dataset is from HuggingFace Hub."""
        # Simple heuristic: if it doesn't contain file extensions or paths, assume HF Hub
        return not any(ext in self.config.name.lower() 
                      for ext in ['.csv', '.json', '.parquet', '.txt', '/', '\\'])
    
    def _is_local_file(self) -> bool:
        """Check if dataset is a local file."""
        return os.path.exists(self.config.name) or '/' in self.config.name or '\\' in self.config.name
    
    def _load_huggingface_dataset(self) -> DatasetDict:
        """Load dataset from HuggingFace Hub.
        
        Returns:
            DatasetDict from HuggingFace Hub
        """
        logger.info(f"Loading HuggingFace dataset: {self.config.name}")
        
        try:
            # Determine which splits to load
            splits_to_load = []
            if self.config.train_split:
                splits_to_load.append(self.config.train_split)
            if self.config.validation_split:
                splits_to_load.append(self.config.validation_split)
            if self.config.test_split:
                splits_to_load.append(self.config.test_split)
            
            # Load dataset
            if splits_to_load:
                datasets = load_dataset(
                    self.config.name,
                    split=splits_to_load,
                    cache_dir=self.config.cache_dir,
                    streaming=self.config.streaming
                )
                
                # Convert to DatasetDict if loading specific splits
                if isinstance(datasets, list):
                    dataset_dict = {}
                    for split_name, dataset in zip(splits_to_load, datasets):
                        dataset_dict[split_name] = dataset
                    datasets = DatasetDict(dataset_dict)
                elif not isinstance(datasets, DatasetDict):
                    # Single split loaded
                    datasets = DatasetDict({splits_to_load[0]: datasets})
            else:
                # Load all available splits
                datasets = load_dataset(
                    self.config.name,
                    cache_dir=self.config.cache_dir,
                    streaming=self.config.streaming
                )
            
            # Handle missing validation split
            if self.config.validation_split and self.config.validation_split not in datasets:
                if self.config.train_split in datasets:
                    logger.info("Creating validation split from training data")
                    datasets = self._create_validation_split(datasets)
            
            return datasets
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset: {str(e)}")
            raise ValueError(f"Could not load dataset {self.config.name} from HuggingFace Hub") from e
    
    def _load_local_dataset(self) -> DatasetDict:
        """Load dataset from local files.
        
        Returns:
            DatasetDict created from local files
        """
        logger.info(f"Loading local dataset: {self.config.name}")
        
        dataset_path = Path(self.config.name)
        
        if dataset_path.is_file():
            # Single file - determine format and load
            return self._load_single_file(dataset_path)
        elif dataset_path.is_dir():
            # Directory - look for split files
            return self._load_directory(dataset_path)
        else:
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    def _load_single_file(self, file_path: Path) -> DatasetDict:
        """Load dataset from a single file.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            DatasetDict with the loaded data
        """
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension == '.json':
            df = pd.read_json(file_path)
        elif file_extension == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Convert to Dataset
        dataset = Dataset.from_pandas(df)
        
        # Create train/validation split if needed
        if self.config.validation_size > 0:
            datasets = dataset.train_test_split(
                test_size=self.config.validation_size,
                seed=self.config.seed,
                shuffle=self.config.shuffle
            )
            return DatasetDict({
                "train": datasets["train"],
                "validation": datasets["test"]
            })
        else:
            return DatasetDict({"train": dataset})
    
    def _load_directory(self, dir_path: Path) -> DatasetDict:
        """Load dataset from a directory with split files.
        
        Args:
            dir_path: Path to the directory containing split files
            
        Returns:
            DatasetDict with loaded splits
        """
        datasets = {}
        
        # Look for common split file patterns
        split_patterns = {
            "train": ["train.*", "training.*"],
            "validation": ["val.*", "valid.*", "validation.*", "dev.*"],
            "test": ["test.*", "testing.*"]
        }
        
        for split_name, patterns in split_patterns.items():
            for pattern in patterns:
                matching_files = list(dir_path.glob(pattern))
                if matching_files:
                    file_path = matching_files[0]  # Take first match
                    datasets[split_name] = self._load_single_file(file_path)[split_name]
                    break
        
        if not datasets:
            raise ValueError(f"No recognizable split files found in {dir_path}")
        
        return DatasetDict(datasets)
    
    def _create_validation_split(self, datasets: DatasetDict) -> DatasetDict:
        """Create validation split from training data.
        
        Args:
            datasets: Original datasets
            
        Returns:
            Updated datasets with validation split
        """
        train_dataset = datasets[self.config.train_split]
        
        # Split training data
        split_datasets = train_dataset.train_test_split(
            test_size=self.config.validation_size,
            seed=self.config.seed,
            shuffle=self.config.shuffle,
            stratify_by_column=self.config.label_column if self.config.label_column in train_dataset.column_names else None
        )
        
        # Update datasets
        updated_datasets = DatasetDict(datasets)
        updated_datasets[self.config.train_split] = split_datasets["train"]
        updated_datasets[self.config.validation_split] = split_datasets["test"]
        
        logger.info(f"Created validation split: {len(split_datasets['test'])} samples")
        
        return updated_datasets
    
    def _preprocess_datasets(self, datasets: DatasetDict) -> DatasetDict:
        """Apply preprocessing to all datasets.
        
        Args:
            datasets: Raw datasets
            
        Returns:
            Preprocessed datasets
        """
        processed_datasets = DatasetDict()
        
        for split_name, dataset in datasets.items():
            logger.info(f"Preprocessing {split_name} split...")
            
            # Apply column renaming if needed
            if self.config.text_column != "text" and self.config.text_column in dataset.column_names:
                dataset = dataset.rename_column(self.config.text_column, "text")
            
            if self.config.label_column != "label" and self.config.label_column in dataset.column_names:
                dataset = dataset.rename_column(self.config.label_column, "label")
            
            # Remove duplicates if enabled
            if self.config.remove_duplicates:
                original_length = len(dataset)
                dataset = dataset.unique("text")
                removed_count = original_length - len(dataset)
                if removed_count > 0:
                    logger.info(f"Removed {removed_count} duplicate samples from {split_name}")
            
            # Shuffle if enabled
            if self.config.shuffle:
                dataset = dataset.shuffle(seed=self.config.seed)
            
            # Limit samples if specified
            if self.config.max_samples and len(dataset) > self.config.max_samples:
                dataset = dataset.select(range(self.config.max_samples))
                logger.info(f"Limited {split_name} to {self.config.max_samples} samples")
            
            processed_datasets[split_name] = dataset
        
        return processed_datasets
    
    def _filter_datasets(self, datasets: DatasetDict) -> DatasetDict:
        """Apply filtering to datasets (e.g., remove empty texts).
        
        Args:
            datasets: Datasets to filter
            
        Returns:
            Filtered datasets
        """
        filtered_datasets = DatasetDict()
        
        for split_name, dataset in datasets.items():
            original_length = len(dataset)
            
            # Filter out empty or very short texts
            dataset = dataset.filter(
                lambda example: example.get("text") and len(str(example["text"]).strip()) > 3
            )
            
            # Filter out missing labels if label column exists
            if "label" in dataset.column_names:
                dataset = dataset.filter(
                    lambda example: example.get("label") is not None
                )
            
            filtered_count = original_length - len(dataset)
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} invalid samples from {split_name}")
            
            filtered_datasets[split_name] = dataset
        
        return filtered_datasets
    
    def _has_cached_dataset(self) -> bool:
        """Check if cached dataset exists.
        
        Returns:
            True if cached dataset exists
        """
        if not self.config.cache_dir:
            return False
        
        cache_path = Path(self.config.cache_dir) / f"{self.config.name.replace('/', '_')}_processed"
        return cache_path.exists()
    
    def _load_from_cache(self) -> DatasetDict:
        """Load dataset from cache.
        
        Returns:
            Cached DatasetDict
        """
        cache_path = Path(self.config.cache_dir) / f"{self.config.name.replace('/', '_')}_processed"
        return load_from_disk(str(cache_path))
    
    def _cache_dataset(self, datasets: DatasetDict) -> None:
        """Cache the processed dataset.
        
        Args:
            datasets: Datasets to cache
        """
        if not self.config.cache_dir:
            return
        
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_path = cache_dir / f"{self.config.name.replace('/', '_')}_processed"
        datasets.save_to_disk(str(cache_path))
        
        logger.info(f"Cached processed dataset to {cache_path}")
    
    def _log_dataset_stats(self, datasets: DatasetDict) -> None:
        """Log dataset statistics.
        
        Args:
            datasets: Datasets to analyze
        """
        logger.info("Dataset Statistics:")
        
        total_samples = 0
        for split_name, dataset in datasets.items():
            split_size = len(dataset)
            total_samples += split_size
            logger.info(f"  {split_name}: {split_size:,} samples")
            
            # Log label distribution if label column exists
            if "label" in dataset.column_names:
                try:
                    labels = dataset["label"]
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    logger.info(f"    Label distribution: {dict(zip(unique_labels, counts))}")
                except Exception:
                    pass  # Skip if labels are not numeric
        
        logger.info(f"  Total: {total_samples:,} samples")
        
        # Log text length statistics for a sample
        if "text" in datasets[list(datasets.keys())[0]].column_names:
            try:
                sample_dataset = datasets[list(datasets.keys())[0]]
                sample_texts = sample_dataset.select(range(min(1000, len(sample_dataset))))["text"]
                text_lengths = [len(str(text).split()) for text in sample_texts]
                
                logger.info(f"  Text length stats (words): "
                           f"mean={np.mean(text_lengths):.1f}, "
                           f"std={np.std(text_lengths):.1f}, "
                           f"min={np.min(text_lengths)}, "
                           f"max={np.max(text_lengths)}")
            except Exception:
                pass  # Skip if text analysis fails