#!/usr/bin/env python3
"""
RoBERTa Sentiment Analysis Fine-tuning Example

This example demonstrates how to fine-tune RoBERTa for sentiment analysis
using the LLM fine-tuning pipeline with LoRA. It includes data loading,
model training, evaluation, and inference.

Usage:
    python examples/roberta_sentiment.py

Requirements:
    - GPU with at least 8GB memory (recommended)
    - MLflow server running (optional)
    - Internet connection for downloading model and dataset

Example Output:
    Training completed! Best F1 Score: 0.9342
    Model saved to: ./outputs/roberta_sentiment_imdb
"""

import sys
import os
from pathlib import Path
import logging

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import FineTuningPipeline, load_config
from src.utils.logging import setup_logging
from src.utils.gpu import get_gpu_info, check_gpu_requirements


def main():
    """Main function to run RoBERTa sentiment analysis fine-tuning."""
    
    # Setup logging
    logger = setup_logging(log_level="INFO", console_output=True)
    logger.info("Starting RoBERTa Sentiment Analysis Fine-tuning Example")
    
    # Check GPU availability
    gpu_info = get_gpu_info()
    if gpu_info["cuda_available"]:
        logger.info(f"CUDA available with {gpu_info['device_count']} GPU(s)")
        for device in gpu_info["devices"]:
            logger.info(f"  GPU {device['id']}: {device['name']} "
                       f"({device['total_memory_gb']:.1f} GB)")
    else:
        logger.warning("CUDA not available, training will use CPU (slow)")
    
    # Check GPU requirements
    if not check_gpu_requirements(required_memory_gb=6.0):
        logger.warning("GPU memory might be insufficient for optimal training")
    
    try:
        # Load configuration
        config_path = "configs/experiments/roberta_sentiment_analysis.yaml"
        
        # Create config if it doesn't exist
        if not os.path.exists(config_path):
            logger.info("Creating example configuration...")
            create_example_config(config_path)
        
        logger.info(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        
        # Log key configuration details
        logger.info("Configuration Summary:")
        logger.info(f"  Model: {config.model.name}")
        logger.info(f"  Dataset: {config.dataset.name}")
        logger.info(f"  Task: {config.model.task_type}")
        logger.info(f"  Use LoRA: {config.use_lora}")
        logger.info(f"  Use QLoRA: {config.use_qlora}")
        logger.info(f"  Epochs: {config.training.num_train_epochs}")
        logger.info(f"  Batch size: {config.training.per_device_train_batch_size}")
        
        # Initialize pipeline
        logger.info("Initializing fine-tuning pipeline...")
        pipeline = FineTuningPipeline(config)
        
        # Run training
        logger.info("Starting training process...")
        results = pipeline.run()
        
        # Display results
        logger.info("Training completed successfully!")
        logger.info("Final Results:")
        
        # Extract key metrics
        best_f1 = results.get("eval_f1", results.get("test_f1", "N/A"))
        best_accuracy = results.get("eval_accuracy", results.get("test_accuracy", "N/A"))
        training_time = results.get("total_time", "N/A")
        
        logger.info(f"  Best F1 Score: {best_f1}")
        logger.info(f"  Best Accuracy: {best_accuracy}")
        logger.info(f"  Training Time: {training_time:.2f} seconds" if training_time != "N/A" else "  Training Time: N/A")
        logger.info(f"  Model saved to: {config.training.output_dir}")
        logger.info(f"  MLflow Run ID: {results.get('run_id', 'N/A')}")
        
        # Run inference example
        if os.path.exists(config.training.output_dir):
            logger.info("Running inference example...")
            run_inference_example(config, results)
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return None


def create_example_config(config_path: str):
    """Create an example configuration file if it doesn't exist.
    
    Args:
        config_path: Path where to create the configuration
    """
    from src.core.config import Config, ModelConfig, DatasetConfig, TrainingConfig, LoRAConfig, MLflowConfig, EvaluationConfig
    
    # Create example configuration
    config = Config(
        model=ModelConfig(
            name="roberta-base",
            task_type="sequence_classification",
            num_labels=2,
            max_length=512,
            use_fast_tokenizer=True,
            trust_remote_code=False,
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True
        ),
        dataset=DatasetConfig(
            name="imdb",
            train_split="train",
            test_split="test",
            text_column="text",
            label_column="label",
            validation_size=0.1,
            remove_duplicates=True,
            shuffle=True,
            seed=42
        ),
        use_lora=True,
        lora=LoRAConfig(
            r=8,
            alpha=16,
            dropout=0.1,
            bias="none",
            target_modules=["query", "value"]
        ),
        use_qlora=False,
        training=TrainingConfig(
            output_dir="./outputs/roberta_sentiment_imdb",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=100
        ),
        mlflow=MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="roberta_sentiment_analysis",
            log_model=True,
            log_predictions=True
        ),
        evaluation=EvaluationConfig(
            compute_metrics=True,
            metrics=["accuracy", "f1", "precision", "recall"],
            save_predictions=True
        ),
        seed=42
    )
    
    # Save configuration
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    from src.core.config import save_config
    save_config(config, config_path)


def run_inference_example(config, training_results):
    """Run inference example with the trained model.
    
    Args:
        config: Training configuration
        training_results: Results from training
    """
    logger = logging.getLogger(__name__)
    
    try:
        from transformers import pipeline
        import torch
        
        # Load the trained model
        model_path = config.training.output_dir
        
        # Create inference pipeline
        classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Test examples
        test_examples = [
            "I absolutely loved this movie! The acting was phenomenal and the story was captivating.",
            "This was the worst film I've ever seen. Terrible acting and boring plot.",
            "The movie was okay, nothing special but not terrible either.",
            "An outstanding masterpiece! Brilliant direction and amazing performances.",
            "I fell asleep halfway through. Very disappointing and slow-paced."
        ]
        
        logger.info("Inference Examples:")
        logger.info("-" * 50)
        
        for i, text in enumerate(test_examples, 1):
            try:
                result = classifier(text)
                prediction = result[0]
                
                # Convert label to sentiment
                sentiment = "Positive" if prediction["label"] == "LABEL_1" else "Negative"
                confidence = prediction["score"]
                
                logger.info(f"Example {i}:")
                logger.info(f"  Text: {text}")
                logger.info(f"  Prediction: {sentiment} (confidence: {confidence:.4f})")
                logger.info("")
                
            except Exception as e:
                logger.error(f"Failed to run inference on example {i}: {e}")
        
    except Exception as e:
        logger.error(f"Failed to run inference examples: {e}")


def benchmark_performance():
    """Benchmark the performance of different configurations."""
    logger = logging.getLogger(__name__)
    logger.info("Running performance benchmark...")
    
    # This could be extended to compare different configurations
    # For now, just log system information
    from src.utils.logging import log_system_info
    log_system_info(logger)


if __name__ == "__main__":
    # Set environment variables for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"
    
    results = main()
    
    if results:
        print(f"\n{'='*60}")
        print("🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
        print(f"{'='*60}")
        print(f"Best F1 Score: {results.get('eval_f1', 'N/A')}")
        print(f"Best Accuracy: {results.get('eval_accuracy', 'N/A')}")
        print(f"Training Time: {results.get('total_time', 'N/A'):.2f} seconds")
        print(f"MLflow Run: {results.get('run_id', 'N/A')}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("❌ TRAINING FAILED")
        print(f"{'='*60}")
        sys.exit(1)