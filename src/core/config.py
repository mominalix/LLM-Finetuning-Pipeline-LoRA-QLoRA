"""Configuration management for the LLM fine-tuning pipeline.

This module provides a comprehensive configuration system using Pydantic for validation
and type safety. It supports loading configurations from YAML files with intelligent
defaults and validation.

Example:
    >>> config = load_config("configs/roberta_sentiment.yaml")
    >>> print(config.model.name)
    'roberta-base'
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Literal
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum


class TaskType(str, Enum):
    """Supported task types."""
    SEQUENCE_CLASSIFICATION = "sequence_classification"
    TOKEN_CLASSIFICATION = "token_classification"
    QUESTION_ANSWERING = "question_answering"
    TEXT_GENERATION = "text_generation"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"


class LoRAConfig(BaseModel):
    """LoRA configuration parameters."""
    
    r: int = Field(8, ge=1, le=256, description="LoRA rank")
    alpha: int = Field(16, ge=1, description="LoRA alpha parameter")
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="LoRA dropout rate")
    bias: Literal["none", "all", "lora_only"] = Field("none", description="Bias configuration")
    target_modules: Optional[List[str]] = Field(None, description="Target modules for LoRA")
    use_rslora: bool = Field(False, description="Use Rank-Stabilized LoRA")
    use_dora: bool = Field(False, description="Use Weight-Decomposed Low-Rank Adaptation")
    
    @validator("target_modules", pre=True, always=True)
    def set_default_target_modules(cls, v, values):
        """Set default target modules based on model type if not provided."""
        if v is None:
            # Will be set dynamically based on model architecture
            return ["q_proj", "v_proj"]
        return v


class QLoRAConfig(BaseModel):
    """QLoRA configuration parameters."""
    
    load_in_4bit: bool = Field(True, description="Load model in 4-bit precision")
    bnb_4bit_compute_dtype: str = Field("bfloat16", description="Compute dtype for 4-bit base models")
    bnb_4bit_quant_type: str = Field("nf4", description="Quantization type (fp4 or nf4)")
    bnb_4bit_use_double_quant: bool = Field(True, description="Use double quantization")
    bnb_4bit_quant_storage: str = Field("uint8", description="Storage type to pack the quants")


class ModelConfig(BaseModel):
    """Model configuration."""
    
    name: str = Field(..., description="Model name or path")
    task_type: TaskType = Field(TaskType.SEQUENCE_CLASSIFICATION, description="Task type")
    num_labels: Optional[int] = Field(None, description="Number of labels for classification")
    max_length: int = Field(512, ge=1, le=8192, description="Maximum sequence length")
    use_fast_tokenizer: bool = Field(True, description="Use fast tokenizer")
    trust_remote_code: bool = Field(False, description="Trust remote code")
    cache_dir: Optional[str] = Field(None, description="Model cache directory")
    
    # Advanced model parameters
    torch_dtype: str = Field("auto", description="PyTorch dtype for model weights")
    device_map: Union[str, Dict[str, Any]] = Field("auto", description="Device mapping strategy")
    low_cpu_mem_usage: bool = Field(True, description="Enable low CPU memory usage")
    load_in_8bit: bool = Field(False, description="Load model in 8-bit precision")
    load_in_4bit: bool = Field(False, description="Load model in 4-bit precision")


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    
    name: str = Field(..., description="Dataset name or path")
    train_split: str = Field("train", description="Training split name")
    validation_split: Optional[str] = Field("validation", description="Validation split name")
    test_split: Optional[str] = Field("test", description="Test split name")
    
    # Data processing
    text_column: str = Field("text", description="Text column name")
    label_column: str = Field("label", description="Label column name")
    max_samples: Optional[int] = Field(None, description="Maximum number of samples")
    validation_size: float = Field(0.1, ge=0.0, le=0.5, description="Validation size if no validation split")
    
    # Data preprocessing
    remove_duplicates: bool = Field(True, description="Remove duplicate samples")
    shuffle: bool = Field(True, description="Shuffle the dataset")
    seed: int = Field(42, description="Random seed for shuffling")
    
    # Caching
    cache_dir: Optional[str] = Field(None, description="Dataset cache directory")
    streaming: bool = Field(False, description="Use streaming dataset")


class TrainingConfig(BaseModel):
    """Training configuration."""
    
    # Basic training parameters
    output_dir: str = Field("./outputs", description="Output directory")
    num_train_epochs: int = Field(3, ge=1, description="Number of training epochs")
    per_device_train_batch_size: int = Field(8, ge=1, description="Training batch size per device")
    per_device_eval_batch_size: int = Field(8, ge=1, description="Evaluation batch size per device")
    gradient_accumulation_steps: int = Field(1, ge=1, description="Gradient accumulation steps")
    
    # Optimization
    learning_rate: float = Field(2e-5, gt=0, description="Learning rate")
    weight_decay: float = Field(0.01, ge=0.0, description="Weight decay")
    adam_beta1: float = Field(0.9, ge=0.0, le=1.0, description="Adam beta1")
    adam_beta2: float = Field(0.999, ge=0.0, le=1.0, description="Adam beta2")
    adam_epsilon: float = Field(1e-8, gt=0, description="Adam epsilon")
    max_grad_norm: float = Field(1.0, gt=0, description="Maximum gradient norm")
    
    # Learning rate scheduling
    lr_scheduler_type: str = Field("linear", description="Learning rate scheduler type")
    warmup_ratio: float = Field(0.1, ge=0.0, le=1.0, description="Warmup ratio")
    warmup_steps: Optional[int] = Field(None, description="Warmup steps")
    
    # Evaluation and saving
    evaluation_strategy: str = Field("epoch", description="Evaluation strategy")
    eval_steps: Optional[int] = Field(None, description="Evaluation steps")
    save_strategy: str = Field("epoch", description="Save strategy")
    save_steps: Optional[int] = Field(None, description="Save steps")
    save_total_limit: int = Field(2, ge=1, description="Maximum number of checkpoints to keep")
    load_best_model_at_end: bool = Field(True, description="Load best model at end")
    metric_for_best_model: str = Field("eval_f1", description="Metric for best model selection")
    greater_is_better: bool = Field(True, description="Whether greater metric is better")
    
    # Performance optimizations
    fp16: bool = Field(False, description="Use FP16 mixed precision")
    bf16: bool = Field(False, description="Use BF16 mixed precision")
    gradient_checkpointing: bool = Field(True, description="Use gradient checkpointing")
    dataloader_pin_memory: bool = Field(True, description="Pin memory in data loader")
    dataloader_num_workers: int = Field(0, ge=0, description="Number of data loader workers")
    
    # Logging
    logging_strategy: str = Field("steps", description="Logging strategy")
    logging_steps: int = Field(10, ge=1, description="Logging steps")
    log_level: str = Field("info", description="Log level")
    
    # Early stopping
    early_stopping_patience: Optional[int] = Field(None, description="Early stopping patience")
    early_stopping_threshold: Optional[float] = Field(None, description="Early stopping threshold")
    
    # Resuming
    resume_from_checkpoint: Optional[str] = Field(None, description="Resume from checkpoint path")
    
    @validator("bf16", "fp16")
    def validate_precision(cls, v, values, field):
        """Ensure only one precision type is enabled."""
        if field.name == "bf16" and v and values.get("fp16"):
            raise ValueError("Cannot use both fp16 and bf16 at the same time")
        if field.name == "fp16" and v and values.get("bf16"):
            raise ValueError("Cannot use both fp16 and bf16 at the same time")
        return v


class MLflowConfig(BaseModel):
    """MLflow configuration."""
    
    tracking_uri: str = Field("http://localhost:5000", description="MLflow tracking URI")
    experiment_name: str = Field("default", description="MLflow experiment name")
    run_name: Optional[str] = Field(None, description="MLflow run name")
    artifact_location: Optional[str] = Field(None, description="Artifact location")
    
    # Logging configuration
    log_model: bool = Field(True, description="Log model to MLflow")
    log_predictions: bool = Field(True, description="Log predictions")
    log_metrics: bool = Field(True, description="Log metrics")
    log_artifacts: bool = Field(True, description="Log artifacts")
    
    # Auto-logging
    autolog: bool = Field(False, description="Enable MLflow autologging")
    log_every_n_epoch: int = Field(1, ge=1, description="Log metrics every N epochs")
    
    # Model registry
    register_model: bool = Field(False, description="Register model in MLflow")
    model_name: Optional[str] = Field(None, description="Model name for registry")


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    
    # Basic metrics
    compute_metrics: bool = Field(True, description="Compute evaluation metrics")
    metrics: List[str] = Field(["accuracy", "f1", "precision", "recall"], description="Metrics to compute")
    
    # Custom evaluation
    custom_metrics: List[str] = Field([], description="Custom metric functions")
    evaluation_datasets: List[str] = Field([], description="Additional evaluation datasets")
    
    # Prediction saving
    save_predictions: bool = Field(True, description="Save predictions to file")
    prediction_output_dir: str = Field("./predictions", description="Prediction output directory")
    
    # Benchmarking
    run_benchmark: bool = Field(False, description="Run benchmark evaluation")
    benchmark_datasets: List[str] = Field([], description="Benchmark datasets")


class Config(BaseModel):
    """Main configuration class."""
    
    # Core configurations
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    
    # PEFT configurations
    use_lora: bool = Field(True, description="Use LoRA fine-tuning")
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    
    use_qlora: bool = Field(False, description="Use QLoRA fine-tuning")
    qlora: QLoRAConfig = Field(default_factory=QLoRAConfig)
    
    # Experiment tracking
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    
    # Evaluation
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    
    # Environment
    seed: int = Field(42, description="Random seed")
    cuda_device: Optional[int] = Field(None, description="CUDA device ID")
    
    @root_validator
    def validate_config(cls, values):
        """Validate configuration consistency."""
        if values.get("use_lora") and values.get("use_qlora"):
            # QLoRA implies LoRA
            values["use_lora"] = True
            
        # Set model-specific defaults
        model_config = values.get("model")
        if model_config and values.get("use_qlora"):
            model_config.load_in_4bit = True
            
        return values
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Prevent additional fields
        validate_assignment = True  # Validate on assignment


def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Validated configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
        pydantic.ValidationError: If configuration is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    
    # Environment variable substitution
    config_data = _substitute_env_vars(config_data)
    
    return Config(**config_data)


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        config_path: Path where to save the configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config.dict(),
            f,
            default_flow_style=False,
            sort_keys=False,
            indent=2
        )


def _substitute_env_vars(data: Any) -> Any:
    """Recursively substitute environment variables in configuration data.
    
    Environment variables should be specified as ${VAR_NAME} or ${VAR_NAME:default_value}.
    
    Args:
        data: Configuration data (dict, list, or string)
        
    Returns:
        Data with environment variables substituted
    """
    if isinstance(data, dict):
        return {key: _substitute_env_vars(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_substitute_env_vars(item) for item in data]
    elif isinstance(data, str):
        # Simple environment variable substitution
        if data.startswith("${") and data.endswith("}"):
            var_expr = data[2:-1]  # Remove ${ and }
            if ":" in var_expr:
                var_name, default_value = var_expr.split(":", 1)
                return os.getenv(var_name, default_value)
            else:
                return os.getenv(var_expr, data)
    
    return data