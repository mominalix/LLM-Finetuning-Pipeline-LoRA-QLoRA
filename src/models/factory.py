"""Model factory for creating models with LoRA/QLoRA configurations.

This module provides a unified interface for creating and configuring different
model architectures with Parameter-Efficient Fine-Tuning (PEFT) techniques.

Example:
    >>> from src.models import ModelFactory
    >>> from src.core.config import Config
    >>> 
    >>> config = load_config("configs/roberta_sentiment.yaml")
    >>> factory = ModelFactory(config)
    >>> model, tokenizer = factory.create_model_and_tokenizer()
"""

import logging
from typing import Tuple, Any, List, Dict, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)

from src.core.config import Config, TaskType as ConfigTaskType


logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating models with LoRA/QLoRA configurations.
    
    This factory handles the complexity of setting up different model architectures
    with the appropriate PEFT configurations, quantization settings, and task-specific
    adaptations.
    
    Args:
        config: Configuration object containing model and training settings
        
    Example:
        >>> factory = ModelFactory(config)
        >>> model, tokenizer = factory.create_model_and_tokenizer()
        >>> print(f"Created model with {model.num_parameters()} parameters")
    """
    
    def __init__(self, config: Config):
        """Initialize the model factory.
        
        Args:
            config: Complete configuration object
        """
        self.config = config
        self.model_config = config.model
        self.lora_config = config.lora if config.use_lora else None
        self.qlora_config = config.qlora if config.use_qlora else None
        
        logger.info(f"Initialized ModelFactory for {self.model_config.name}")
    
    def create_model_and_tokenizer(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Create model and tokenizer with PEFT configurations.
        
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            ValueError: If model configuration is invalid
            RuntimeError: If model creation fails
        """
        try:
            # Create tokenizer first
            tokenizer = self._create_tokenizer()
            
            # Create base model
            model = self._create_base_model()
            
            # Apply quantization if needed
            if self.config.use_qlora:
                model = self._prepare_for_quantization(model)
            
            # Apply LoRA if enabled
            if self.config.use_lora:
                model = self._apply_lora(model, tokenizer)
            
            # Log model information
            self._log_model_info(model)
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to create model and tokenizer: {str(e)}")
            raise RuntimeError(f"Model creation failed: {str(e)}") from e
    
    def _create_tokenizer(self) -> PreTrainedTokenizer:
        """Create and configure tokenizer.
        
        Returns:
            Configured tokenizer
        """
        logger.info(f"Loading tokenizer: {self.model_config.name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.name,
            use_fast=self.model_config.use_fast_tokenizer,
            trust_remote_code=self.model_config.trust_remote_code,
            cache_dir=self.model_config.cache_dir
        )
        
        # Handle missing pad token (common in decoder-only models)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            else:
                tokenizer.add_special_tokens({'pad_token': '<pad>'})
                logger.info("Added new pad token")
        
        # Set model max length if specified
        if hasattr(tokenizer, 'model_max_length'):
            tokenizer.model_max_length = self.model_config.max_length
        
        logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
        return tokenizer
    
    def _create_base_model(self) -> PreTrainedModel:
        """Create base model based on task type.
        
        Returns:
            Base model instance
        """
        logger.info(f"Loading model: {self.model_config.name}")
        
        # Prepare model arguments
        model_kwargs = {
            "pretrained_model_name_or_path": self.model_config.name,
            "trust_remote_code": self.model_config.trust_remote_code,
            "cache_dir": self.model_config.cache_dir,
            "device_map": self.model_config.device_map,
            "low_cpu_mem_usage": self.model_config.low_cpu_mem_usage,
        }
        
        # Set torch dtype
        if self.model_config.torch_dtype != "auto":
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            model_kwargs["torch_dtype"] = dtype_map.get(
                self.model_config.torch_dtype, torch.float32
            )
        
        # Add quantization config if using QLoRA
        if self.config.use_qlora:
            model_kwargs["quantization_config"] = self._create_quantization_config()
        elif self.model_config.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif self.model_config.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        
        # Create model based on task type
        task_type = self.model_config.task_type
        
        if task_type == ConfigTaskType.SEQUENCE_CLASSIFICATION:
            model_kwargs["num_labels"] = self.model_config.num_labels
            model = AutoModelForSequenceClassification.from_pretrained(**model_kwargs)
        elif task_type == ConfigTaskType.TOKEN_CLASSIFICATION:
            model_kwargs["num_labels"] = self.model_config.num_labels
            model = AutoModelForTokenClassification.from_pretrained(**model_kwargs)
        elif task_type == ConfigTaskType.QUESTION_ANSWERING:
            model = AutoModelForQuestionAnswering.from_pretrained(**model_kwargs)
        elif task_type in [ConfigTaskType.TEXT_GENERATION, ConfigTaskType.SUMMARIZATION, ConfigTaskType.TRANSLATION]:
            model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Resize token embeddings if needed (for new pad token)
        if hasattr(model, 'resize_token_embeddings'):
            # This will be set later when we have the tokenizer
            pass
        
        return model
    
    def _create_quantization_config(self) -> BitsAndBytesConfig:
        """Create quantization configuration for QLoRA.
        
        Returns:
            BitsAndBytesConfig for quantization
        """
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        
        compute_dtype = dtype_map.get(
            self.qlora_config.bnb_4bit_compute_dtype,
            torch.bfloat16
        )
        
        config = BitsAndBytesConfig(
            load_in_4bit=self.qlora_config.load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.qlora_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.qlora_config.bnb_4bit_use_double_quant,
        )
        
        logger.info("Created 4-bit quantization config")
        logger.info(f"  Compute dtype: {compute_dtype}")
        logger.info(f"  Quant type: {self.qlora_config.bnb_4bit_quant_type}")
        logger.info(f"  Double quant: {self.qlora_config.bnb_4bit_use_double_quant}")
        
        return config
    
    def _prepare_for_quantization(self, model: PreTrainedModel) -> PreTrainedModel:
        """Prepare model for quantized training.
        
        Args:
            model: Base model
            
        Returns:
            Model prepared for quantized training
        """
        logger.info("Preparing model for quantized training")
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=self.config.training.gradient_checkpointing
        )
        
        return model
    
    def _apply_lora(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        """Apply LoRA configuration to the model.
        
        Args:
            model: Base model
            tokenizer: Tokenizer (needed for embedding resizing)
            
        Returns:
            Model with LoRA applied
        """
        logger.info("Applying LoRA configuration")
        
        # Resize embeddings if tokenizer was modified
        original_vocab_size = model.get_input_embeddings().num_embeddings
        if len(tokenizer) != original_vocab_size:
            model.resize_token_embeddings(len(tokenizer))
            logger.info(f"Resized embeddings from {original_vocab_size} to {len(tokenizer)}")
        
        # Determine target modules if not specified
        target_modules = self.lora_config.target_modules
        if not target_modules or target_modules == ["q_proj", "v_proj"]:
            target_modules = self._get_target_modules(model)
        
        # Determine task type for PEFT
        peft_task_type = self._get_peft_task_type()
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            task_type=peft_task_type,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            bias=self.lora_config.bias,
            target_modules=target_modules,
            inference_mode=False,  # Training mode
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        logger.info(f"Applied LoRA with configuration:")
        logger.info(f"  Rank (r): {self.lora_config.r}")
        logger.info(f"  Alpha: {self.lora_config.alpha}")
        logger.info(f"  Dropout: {self.lora_config.dropout}")
        logger.info(f"  Target modules: {target_modules}")
        
        return model
    
    def _get_target_modules(self, model: PreTrainedModel) -> List[str]:
        """Automatically determine target modules for LoRA based on model architecture.
        
        Args:
            model: Model to analyze
            
        Returns:
            List of target module names
        """
        model_name = self.model_config.name.lower()
        
        # Architecture-specific target modules
        if "roberta" in model_name or "bert" in model_name:
            return ["query", "value"]
        elif "llama" in model_name or "alpaca" in model_name:
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif "mistral" in model_name:
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif "falcon" in model_name:
            return ["query_key_value", "dense"]
        elif "gpt" in model_name:
            return ["c_attn", "c_proj"]
        elif "t5" in model_name:
            return ["q", "v", "k", "o"]
        else:
            # Default: try to find attention layers
            target_modules = []
            for name, module in model.named_modules():
                if any(layer_name in name.lower() for layer_name in ["attention", "attn"]):
                    if any(proj_name in name.lower() for proj_name in ["query", "value", "q_proj", "v_proj"]):
                        # Extract the last part of the module name
                        module_name = name.split(".")[-1]
                        if module_name not in target_modules:
                            target_modules.append(module_name)
            
            if target_modules:
                logger.info(f"Auto-detected target modules: {target_modules}")
                return target_modules
            else:
                # Fallback to common names
                logger.warning("Could not auto-detect target modules, using default")
                return ["q_proj", "v_proj"]
    
    def _get_peft_task_type(self) -> TaskType:
        """Convert config task type to PEFT task type.
        
        Returns:
            PEFT TaskType enum value
        """
        task_mapping = {
            ConfigTaskType.SEQUENCE_CLASSIFICATION: TaskType.SEQ_CLS,
            ConfigTaskType.TOKEN_CLASSIFICATION: TaskType.TOKEN_CLS,
            ConfigTaskType.QUESTION_ANSWERING: TaskType.QUESTION_ANS,
            ConfigTaskType.TEXT_GENERATION: TaskType.CAUSAL_LM,
            ConfigTaskType.SUMMARIZATION: TaskType.CAUSAL_LM,
            ConfigTaskType.TRANSLATION: TaskType.CAUSAL_LM,
        }
        
        return task_mapping.get(self.model_config.task_type, TaskType.CAUSAL_LM)
    
    def _log_model_info(self, model: PreTrainedModel) -> None:
        """Log information about the created model.
        
        Args:
            model: Model to analyze
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info("Model Information:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Trainable percentage: {100 * trainable_params / total_params:.4f}%")
        
        # Log memory footprint if on GPU
        if next(model.parameters()).is_cuda:
            try:
                from src.utils.gpu import monitor_gpu_memory
                memory_stats = monitor_gpu_memory()
                if memory_stats:
                    logger.info(f"  GPU memory usage: {memory_stats['allocated_gb']:.2f} GB")
            except ImportError:
                pass
        
        # Log model architecture info
        if hasattr(model, 'config'):
            config = model.config
            if hasattr(config, 'num_hidden_layers'):
                logger.info(f"  Hidden layers: {config.num_hidden_layers}")
            if hasattr(config, 'hidden_size'):
                logger.info(f"  Hidden size: {config.hidden_size}")
            if hasattr(config, 'num_attention_heads'):
                logger.info(f"  Attention heads: {config.num_attention_heads}")
        
        # Print trainable parameters summary if using PEFT
        if hasattr(model, 'print_trainable_parameters'):
            model.print_trainable_parameters()


def get_model_size_category(model_name: str) -> str:
    """Categorize model by size for optimization suggestions.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Size category string
    """
    model_name = model_name.lower()
    
    if any(size in model_name for size in ["7b", "6.7b", "7-b"]):
        return "large"
    elif any(size in model_name for size in ["13b", "12b", "11b"]):
        return "extra_large"
    elif any(size in model_name for size in ["30b", "33b", "34b", "65b", "70b"]):
        return "xxl"
    elif "base" in model_name:
        return "base"
    elif "large" in model_name:
        return "large"
    elif "small" in model_name or "mini" in model_name:
        return "small"
    else:
        return "medium"


def suggest_training_config(model_name: str, available_memory_gb: float) -> Dict[str, Any]:
    """Suggest training configuration based on model size and available memory.
    
    Args:
        model_name: Name of the model
        available_memory_gb: Available GPU memory in GB
        
    Returns:
        Dictionary with suggested configuration
    """
    size_category = get_model_size_category(model_name)
    
    suggestions = {
        "small": {
            "batch_size": 32,
            "gradient_accumulation_steps": 1,
            "use_qlora": False,
            "lora_r": 8,
            "fp16": True,
        },
        "base": {
            "batch_size": 16,
            "gradient_accumulation_steps": 2,
            "use_qlora": False,
            "lora_r": 16,
            "fp16": True,
        },
        "large": {
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "use_qlora": available_memory_gb < 16,
            "lora_r": 16,
            "fp16": True,
        },
        "extra_large": {
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "use_qlora": True,
            "lora_r": 32,
            "bf16": True,
        },
        "xxl": {
            "batch_size": 2,
            "gradient_accumulation_steps": 16,
            "use_qlora": True,
            "lora_r": 64,
            "bf16": True,
        }
    }
    
    config = suggestions.get(size_category, suggestions["base"])
    
    # Adjust based on available memory
    if available_memory_gb < 8:
        config["batch_size"] = max(1, config["batch_size"] // 2)
        config["gradient_accumulation_steps"] *= 2
        config["use_qlora"] = True
    elif available_memory_gb > 40:
        config["batch_size"] = min(32, config["batch_size"] * 2)
        config["gradient_accumulation_steps"] = max(1, config["gradient_accumulation_steps"] // 2)
    
    logger.info(f"Suggested config for {model_name} ({size_category}) with {available_memory_gb}GB:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    return config