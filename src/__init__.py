"""LLM Fine-tuning Pipeline with LoRA/QLoRA support.

A comprehensive pipeline for fine-tuning Large Language Models using Parameter-Efficient
Fine-Tuning (PEFT) techniques with MLflow experiment tracking.

Key Features:
- LoRA and QLoRA support for efficient fine-tuning
- Multiple model architectures (RoBERTa, Llama, Mistral, etc.)
- MLflow integration for experiment tracking
- Flexible configuration system
- Production-ready deployment options

Example:
    >>> from src import FineTuningPipeline
    >>> from src.config import load_config
    >>> 
    >>> config = load_config("configs/roberta_sentiment.yaml")
    >>> pipeline = FineTuningPipeline(config)
    >>> results = pipeline.run()
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from src.core.pipeline import FineTuningPipeline
from src.core.config import load_config, Config

__all__ = [
    "FineTuningPipeline",
    "load_config",
    "Config",
]