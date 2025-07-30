"""Main fine-tuning pipeline orchestrating the entire process.

This module contains the main FineTuningPipeline class that coordinates data loading,
model preparation, training, evaluation, and experiment tracking.

Example:
    >>> from src import FineTuningPipeline, load_config
    >>> 
    >>> config = load_config("configs/roberta_sentiment.yaml")
    >>> pipeline = FineTuningPipeline(config)
    >>> results = pipeline.run()
    >>> print(f"Best F1 Score: {results['best_f1']:.4f}")
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import random
import numpy as np

import mlflow
import mlflow.transformers
from mlflow.tracking import MlflowClient

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from datasets import Dataset, DatasetDict

from src.core.config import Config
from src.data.loader import DataLoader
from src.models.factory import ModelFactory
from src.training.trainer import CustomTrainer
from src.evaluation.metrics import MetricsComputer
from src.utils.logging import setup_logging
from src.utils.gpu import get_gpu_info, optimize_gpu_memory


logger = logging.getLogger(__name__)


class FineTuningPipeline:
    """Main pipeline for fine-tuning LLMs with LoRA/QLoRA.
    
    This class orchestrates the entire fine-tuning process including:
    - Data loading and preprocessing
    - Model and tokenizer setup with LoRA/QLoRA
    - Training with MLflow tracking
    - Evaluation and metrics computation
    - Model saving and registration
    
    Args:
        config: Configuration object containing all settings
        
    Example:
        >>> config = load_config("configs/roberta_sentiment.yaml")
        >>> pipeline = FineTuningPipeline(config)
        >>> results = pipeline.run()
    """
    
    def __init__(self, config: Config):
        """Initialize the pipeline with configuration.
        
        Args:
            config: Validated configuration object
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.datasets = None
        self.trainer = None
        
        # Setup logging
        setup_logging(
            log_level=config.training.log_level,
            log_file=getattr(config.training, 'log_file', None)
        )
        
        # Set random seeds for reproducibility
        self._set_seed()
        
        # Setup MLflow
        self._setup_mlflow()
        
        logger.info("Pipeline initialized successfully")
        logger.info(f"Configuration: {config.model.name} on {config.dataset.name}")
    
    def run(self) -> Dict[str, Any]:
        """Run the complete fine-tuning pipeline.
        
        Returns:
            Dictionary containing training results and metrics
            
        Raises:
            RuntimeError: If any step of the pipeline fails
        """
        start_time = time.time()
        
        try:
            with mlflow.start_run(run_name=self.config.mlflow.run_name) as run:
                logger.info(f"Started MLflow run: {run.info.run_id}")
                
                # Log configuration
                self._log_config()
                
                # Step 1: Load and prepare data
                logger.info("Loading and preparing dataset...")
                self.datasets = self._load_data()
                self._log_dataset_info()
                
                # Step 2: Setup model and tokenizer
                logger.info("Setting up model and tokenizer...")
                self.model, self.tokenizer = self._setup_model()
                self._log_model_info()
                
                # Step 3: Prepare training data
                logger.info("Preparing training data...")
                tokenized_datasets = self._prepare_training_data()
                
                # Step 4: Setup trainer
                logger.info("Setting up trainer...")
                self.trainer = self._setup_trainer(tokenized_datasets)
                
                # Step 5: Train model
                logger.info("Starting training...")
                train_results = self._train()
                
                # Step 6: Evaluate model
                logger.info("Evaluating model...")
                eval_results = self._evaluate()
                
                # Step 7: Save results
                logger.info("Saving results...")
                self._save_results(train_results, eval_results)
                
                # Calculate total time
                total_time = time.time() - start_time
                
                # Compile final results
                results = {
                    **train_results,
                    **eval_results,
                    "total_time": total_time,
                    "run_id": run.info.run_id,
                    "config": self.config.dict()
                }
                
                # Log final metrics
                mlflow.log_metric("total_time", total_time)
                
                logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
                return results
                
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Pipeline execution failed: {str(e)}") from e
    
    def _set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        seed = self.config.seed
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Additional deterministic settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        logger.info(f"Set random seed to {seed}")
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.set_experiment(self.config.mlflow.experiment_name)
            logger.info(f"Using MLflow experiment: {experiment.name}")
        except Exception as e:
            logger.warning(f"Failed to set MLflow experiment: {e}")
            mlflow.set_experiment("default")
        
        # Configure autologging if enabled
        if self.config.mlflow.autolog:
            mlflow.transformers.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=self.config.mlflow.log_model
            )
    
    def _log_config(self) -> None:
        """Log configuration parameters to MLflow."""
        # Log all configuration as parameters
        config_dict = self.config.dict()
        
        # Flatten nested configuration
        flat_config = self._flatten_dict(config_dict)
        
        for key, value in flat_config.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(key, value)
        
        # Log full config as artifact
        config_path = "config.yaml"
        with open(config_path, "w") as f:
            import yaml
            yaml.dump(config_dict, f, default_flow_style=False)
        mlflow.log_artifact(config_path)
        os.remove(config_path)
    
    def _load_data(self) -> DatasetDict:
        """Load and prepare the dataset.
        
        Returns:
            DatasetDict containing train/validation/test splits
        """
        data_loader = DataLoader(self.config.dataset)
        datasets = data_loader.load()
        
        logger.info(f"Loaded dataset: {self.config.dataset.name}")
        for split, dataset in datasets.items():
            logger.info(f"  {split}: {len(dataset)} samples")
        
        return datasets
    
    def _setup_model(self) -> Tuple[Any, Any]:
        """Setup model and tokenizer with LoRA/QLoRA.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        model_factory = ModelFactory(self.config)
        model, tokenizer = model_factory.create_model_and_tokenizer()
        
        # Log GPU info
        if torch.cuda.is_available():
            gpu_info = get_gpu_info()
            logger.info(f"Using GPU: {gpu_info}")
            
            # Optimize GPU memory if needed
            optimize_gpu_memory()
        
        return model, tokenizer
    
    def _prepare_training_data(self) -> DatasetDict:
        """Prepare tokenized datasets for training.
        
        Returns:
            Tokenized datasets ready for training
        """
        def tokenize_function(examples):
            """Tokenize examples for the model."""
            text_column = self.config.dataset.text_column
            
            # Handle different input formats
            if isinstance(examples[text_column], str):
                texts = [examples[text_column]]
            else:
                texts = examples[text_column]
            
            # Tokenize
            return self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.config.model.max_length,
                return_tensors=None  # Keep as lists for datasets
            )
        
        # Tokenize all splits
        tokenized_datasets = self.datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in self.datasets["train"].column_names 
                          if col != self.config.dataset.label_column]
        )
        
        # Rename label column to 'labels' for Trainer compatibility
        if self.config.dataset.label_column != "labels":
            tokenized_datasets = tokenized_datasets.rename_column(
                self.config.dataset.label_column, "labels"
            )
        
        # Set format for PyTorch
        tokenized_datasets.set_format("torch")
        
        logger.info("Tokenized datasets prepared for training")
        return tokenized_datasets
    
    def _setup_trainer(self, tokenized_datasets: DatasetDict) -> Trainer:
        """Setup the Trainer with all configurations.
        
        Args:
            tokenized_datasets: Tokenized datasets
            
        Returns:
            Configured Trainer instance
        """
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            adam_beta1=self.config.training.adam_beta1,
            adam_beta2=self.config.training.adam_beta2,
            adam_epsilon=self.config.training.adam_epsilon,
            max_grad_norm=self.config.training.max_grad_norm,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            warmup_ratio=self.config.training.warmup_ratio,
            warmup_steps=self.config.training.warmup_steps,
            evaluation_strategy=self.config.training.evaluation_strategy,
            eval_steps=self.config.training.eval_steps,
            save_strategy=self.config.training.save_strategy,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            greater_is_better=self.config.training.greater_is_better,
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            dataloader_pin_memory=self.config.training.dataloader_pin_memory,
            dataloader_num_workers=self.config.training.dataloader_num_workers,
            logging_strategy=self.config.training.logging_strategy,
            logging_steps=self.config.training.logging_steps,
            report_to="mlflow" if self.config.mlflow.log_metrics else None,
            run_name=self.config.mlflow.run_name,
        )
        
        # Setup metrics computer
        metrics_computer = MetricsComputer(
            task_type=self.config.model.task_type,
            metrics=self.config.evaluation.metrics,
            num_labels=self.config.model.num_labels
        )
        
        # Setup callbacks
        callbacks = []
        if self.config.training.early_stopping_patience:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.training.early_stopping_patience,
                    early_stopping_threshold=self.config.training.early_stopping_threshold
                )
            )
        
        # Create trainer
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets.get("validation") or tokenized_datasets.get("test"),
            tokenizer=self.tokenizer,
            compute_metrics=metrics_computer.compute_metrics,
            callbacks=callbacks,
        )
        
        return trainer
    
    def _train(self) -> Dict[str, Any]:
        """Train the model.
        
        Returns:
            Training results and metrics
        """
        # Resume from checkpoint if specified
        resume_from_checkpoint = self.config.training.resume_from_checkpoint
        
        # Start training
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save the final model
        self.trainer.save_model()
        
        # Log training results
        train_metrics = train_result.metrics
        for key, value in train_metrics.items():
            mlflow.log_metric(f"train_{key}", value)
        
        logger.info("Training completed successfully")
        return train_metrics
    
    def _evaluate(self) -> Dict[str, Any]:
        """Evaluate the trained model.
        
        Returns:
            Evaluation results and metrics
        """
        eval_results = {}
        
        # Evaluate on validation set if available
        if "validation" in self.datasets:
            val_metrics = self.trainer.evaluate(eval_dataset=None)  # Uses configured eval_dataset
            eval_results["validation"] = val_metrics
            
            for key, value in val_metrics.items():
                mlflow.log_metric(f"val_{key}", value)
        
        # Evaluate on test set if available and different from validation
        if "test" in self.datasets and "validation" in self.datasets:
            test_metrics = self.trainer.evaluate(eval_dataset=self.datasets["test"])
            eval_results["test"] = test_metrics
            
            for key, value in test_metrics.items():
                mlflow.log_metric(f"test_{key}", value)
        
        logger.info("Evaluation completed successfully")
        return eval_results
    
    def _save_results(self, train_results: Dict[str, Any], eval_results: Dict[str, Any]) -> None:
        """Save model and results.
        
        Args:
            train_results: Training metrics
            eval_results: Evaluation metrics
        """
        # Log model to MLflow if enabled
        if self.config.mlflow.log_model:
            model_info = mlflow.transformers.log_model(
                transformers_model={
                    "model": self.model,
                    "tokenizer": self.tokenizer,
                },
                artifact_path="model",
                task="text-classification",
            )
            
            # Register model if enabled
            if self.config.mlflow.register_model and self.config.mlflow.model_name:
                client = MlflowClient()
                model_version = client.create_model_version(
                    name=self.config.mlflow.model_name,
                    source=model_info.model_uri,
                    run_id=mlflow.active_run().info.run_id
                )
                logger.info(f"Registered model version: {model_version.version}")
        
        # Save predictions if enabled
        if self.config.evaluation.save_predictions:
            self._save_predictions()
    
    def _save_predictions(self) -> None:
        """Save model predictions for analysis."""
        prediction_dir = Path(self.config.evaluation.prediction_output_dir)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate predictions for test set
        if "test" in self.datasets:
            test_dataset = self.datasets["test"]
            predictions = self.trainer.predict(test_dataset)
            
            # Save predictions and labels
            import pandas as pd
            
            pred_df = pd.DataFrame({
                "predictions": predictions.predictions.argmax(axis=1),
                "labels": predictions.label_ids,
                "text": test_dataset[self.config.dataset.text_column]
            })
            
            pred_file = prediction_dir / "test_predictions.csv"
            pred_df.to_csv(pred_file, index=False)
            
            # Log as artifact
            mlflow.log_artifact(str(pred_file))
            
            logger.info(f"Saved predictions to {pred_file}")
    
    def _log_dataset_info(self) -> None:
        """Log dataset information to MLflow."""
        for split_name, dataset in self.datasets.items():
            mlflow.log_metric(f"dataset_{split_name}_size", len(dataset))
        
        # Log dataset configuration
        mlflow.log_param("dataset_name", self.config.dataset.name)
        mlflow.log_param("text_column", self.config.dataset.text_column)
        mlflow.log_param("label_column", self.config.dataset.label_column)
    
    def _log_model_info(self) -> None:
        """Log model information to MLflow."""
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        mlflow.log_metric("total_parameters", total_params)
        mlflow.log_metric("trainable_parameters", trainable_params)
        mlflow.log_metric("trainable_percentage", 100 * trainable_params / total_params)
        
        # Log model configuration
        mlflow.log_param("model_name", self.config.model.name)
        mlflow.log_param("task_type", self.config.model.task_type)
        mlflow.log_param("max_length", self.config.model.max_length)
        
        logger.info(f"Model parameters: {trainable_params:,} / {total_params:,} "
                   f"({100 * trainable_params / total_params:.2f}% trainable)")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flatten nested dictionary for MLflow logging.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)