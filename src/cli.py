#!/usr/bin/env python3
"""Command-line interface for the LLM fine-tuning pipeline.

This module provides a command-line interface for running fine-tuning experiments
with various configurations and options.

Usage:
    python -m src.cli train --config configs/roberta_sentiment.yaml
    python -m src.cli evaluate --model outputs/roberta_sentiment --dataset imdb
    python -m src.cli list-models
    python -m src.cli benchmark --config configs/roberta_sentiment.yaml

Example:
    $ python -m src.cli train --config configs/roberta_sentiment.yaml --gpu 0
    $ python -m src.cli train --model roberta-base --dataset imdb --epochs 3
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from src import FineTuningPipeline, load_config
from src.core.config import Config, save_config, ModelConfig, DatasetConfig, TrainingConfig
from src.utils.logging import setup_logging
from src.utils.gpu import get_gpu_info, check_gpu_requirements


app = typer.Typer(
    name="llm-finetune",
    help="LLM Fine-tuning Pipeline with LoRA/QLoRA support",
    no_args_is_help=True
)
console = Console()


@app.command()
def train(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name or path"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Dataset name"),
    epochs: Optional[int] = typer.Option(None, "--epochs", "-e", help="Number of training epochs"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Training batch size"),
    learning_rate: Optional[float] = typer.Option(None, "--lr", help="Learning rate"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    gpu: Optional[int] = typer.Option(None, "--gpu", help="GPU device ID"),
    use_qlora: bool = typer.Option(False, "--qlora", help="Use QLoRA for memory-efficient training"),
    lora_r: Optional[int] = typer.Option(None, "--lora-r", help="LoRA rank parameter"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show configuration without training"),
):
    """Train a model with the specified configuration.
    
    You can either provide a configuration file or specify parameters directly.
    Command-line parameters override configuration file values.
    """
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level=log_level)
    logger = logging.getLogger(__name__)
    
    console.print(Panel.fit("🚀 LLM Fine-tuning Pipeline", style="bold blue"))
    
    try:
        # Load or create configuration
        if config:
            if not os.path.exists(config):
                console.print(f"❌ Configuration file not found: {config}", style="bold red")
                raise typer.Exit(1)
            
            console.print(f"📁 Loading configuration from: {config}")
            config_obj = load_config(config)
        else:
            # Create configuration from command-line arguments
            if not model or not dataset:
                console.print("❌ Either --config or both --model and --dataset must be specified", style="bold red")
                raise typer.Exit(1)
            
            console.print("🔧 Creating configuration from command-line arguments")
            config_obj = _create_config_from_args(
                model=model,
                dataset=dataset,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                output_dir=output_dir,
                use_qlora=use_qlora,
                lora_r=lora_r
            )
        
        # Override with command-line arguments
        config_obj = _override_config(
            config_obj,
            model=model,
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=output_dir,
            use_qlora=use_qlora,
            lora_r=lora_r
        )
        
        # Set GPU device
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            config_obj.cuda_device = gpu
        
        # Display configuration
        _display_config(config_obj)
        
        if dry_run:
            console.print("✅ Dry run completed - configuration validated", style="bold green")
            return
        
        # Check GPU requirements
        _check_gpu_requirements()
        
        # Initialize and run pipeline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Initializing pipeline...", total=None)
            pipeline = FineTuningPipeline(config_obj)
            
            progress.update(task, description="Running training...")
            results = pipeline.run()
            
            progress.update(task, description="Training completed!", completed=True)
        
        # Display results
        _display_results(results)
        
    except Exception as e:
        console.print(f"❌ Training failed: {str(e)}", style="bold red")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def evaluate(
    model: str = typer.Argument(..., help="Path to trained model"),
    dataset: str = typer.Argument(..., help="Dataset to evaluate on"),
    output_dir: Optional[str] = typer.Option("./evaluation_results", "--output", "-o", help="Output directory"),
    batch_size: Optional[int] = typer.Option(32, "--batch-size", "-b", help="Evaluation batch size"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Evaluate a trained model on a dataset."""
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level=log_level)
    
    console.print(Panel.fit("📊 Model Evaluation", style="bold blue"))
    console.print(f"Model: {model}")
    console.print(f"Dataset: {dataset}")
    
    # TODO: Implement evaluation logic
    console.print("⚠️  Evaluation functionality coming soon!", style="bold yellow")


@app.command("list-models")
def list_models():
    """List available pre-trained models."""
    
    console.print(Panel.fit("📋 Available Models", style="bold blue"))
    
    # Create table of popular models
    table = Table(title="Popular Models for Fine-tuning")
    table.add_column("Model Name", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Parameters", style="green")
    table.add_column("Tasks", style="blue")
    table.add_column("Memory (GB)", style="red")
    
    models = [
        ("roberta-base", "Encoder", "125M", "Classification, NER", "~1"),
        ("roberta-large", "Encoder", "355M", "Classification, NER", "~2"),
        ("bert-base-uncased", "Encoder", "110M", "Classification, NER, QA", "~1"),
        ("distilbert-base-uncased", "Encoder", "66M", "Classification, NER", "<1"),
        ("meta-llama/Llama-2-7b-hf", "Decoder", "7B", "Generation, Chat", "~14"),
        ("mistralai/Mistral-7B-v0.1", "Decoder", "7B", "Generation, Chat", "~14"),
        ("google/flan-t5-base", "Enc-Dec", "250M", "Generation, QA", "~1"),
        ("google/flan-t5-large", "Enc-Dec", "780M", "Generation, QA", "~3"),
    ]
    
    for model_name, model_type, params, tasks, memory in models:
        table.add_row(model_name, model_type, params, tasks, memory)
    
    console.print(table)
    
    console.print("\n💡 Tips:")
    console.print("• Use smaller models (RoBERTa, BERT) for classification tasks")
    console.print("• Use larger models (Llama, Mistral) with QLoRA for generation tasks")
    console.print("• Check memory requirements against your available GPU memory")


@app.command()
def benchmark(
    config: str = typer.Argument(..., help="Configuration file to benchmark"),
    runs: int = typer.Option(3, "--runs", "-r", help="Number of benchmark runs"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Benchmark training performance with different configurations."""
    
    console.print(Panel.fit("⚡ Performance Benchmark", style="bold blue"))
    console.print("⚠️  Benchmarking functionality coming soon!", style="bold yellow")


@app.command()
def create_config(
    model: str = typer.Argument(..., help="Model name"),
    dataset: str = typer.Argument(..., help="Dataset name"),
    output: str = typer.Option("config.yaml", "--output", "-o", help="Output configuration file"),
    task: str = typer.Option("sequence_classification", "--task", help="Task type"),
    epochs: int = typer.Option(3, "--epochs", help="Number of epochs"),
    batch_size: int = typer.Option(16, "--batch-size", help="Batch size"),
):
    """Create a configuration file with the specified parameters."""
    
    console.print(Panel.fit("📝 Configuration Creator", style="bold blue"))
    
    try:
        config_obj = _create_config_from_args(
            model=model,
            dataset=dataset,
            task=task,
            epochs=epochs,
            batch_size=batch_size
        )
        
        save_config(config_obj, output)
        console.print(f"✅ Configuration saved to: {output}", style="bold green")
        
        # Display created configuration
        _display_config(config_obj)
        
    except Exception as e:
        console.print(f"❌ Failed to create configuration: {str(e)}", style="bold red")
        raise typer.Exit(1)


def _create_config_from_args(**kwargs) -> Config:
    """Create configuration object from command-line arguments."""
    
    # Extract arguments
    model_name = kwargs.get("model", "roberta-base")
    dataset_name = kwargs.get("dataset", "imdb")
    task = kwargs.get("task", "sequence_classification")
    epochs = kwargs.get("epochs", 3)
    batch_size = kwargs.get("batch_size", 16)
    learning_rate = kwargs.get("learning_rate", 2e-5)
    output_dir = kwargs.get("output_dir", f"./outputs/{model_name.replace('/', '_')}_{dataset_name}")
    use_qlora = kwargs.get("use_qlora", False)
    lora_r = kwargs.get("lora_r", 8)
    
    # Determine number of labels based on dataset
    num_labels = _get_dataset_num_labels(dataset_name, task)
    
    # Create configuration
    from src.core.config import (
        Config, ModelConfig, DatasetConfig, TrainingConfig, 
        LoRAConfig, QLoRAConfig, MLflowConfig, EvaluationConfig
    )
    
    config = Config(
        model=ModelConfig(
            name=model_name,
            task_type=task,
            num_labels=num_labels,
            max_length=512
        ),
        dataset=DatasetConfig(
            name=dataset_name,
            train_split="train",
            test_split="test",
            validation_size=0.1
        ),
        use_lora=True,
        lora=LoRAConfig(r=lora_r),
        use_qlora=use_qlora,
        qlora=QLoRAConfig() if use_qlora else QLoRAConfig(load_in_4bit=False),
        training=TrainingConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate
        ),
        mlflow=MLflowConfig(),
        evaluation=EvaluationConfig(),
        seed=42
    )
    
    return config


def _override_config(config: Config, **kwargs) -> Config:
    """Override configuration with command-line arguments."""
    
    for key, value in kwargs.items():
        if value is not None:
            if key == "model":
                config.model.name = value
            elif key == "dataset":
                config.dataset.name = value
            elif key == "epochs":
                config.training.num_train_epochs = value
            elif key == "batch_size":
                config.training.per_device_train_batch_size = value
            elif key == "learning_rate":
                config.training.learning_rate = value
            elif key == "output_dir":
                config.training.output_dir = value
            elif key == "use_qlora":
                config.use_qlora = value
            elif key == "lora_r":
                config.lora.r = value
    
    return config


def _get_dataset_num_labels(dataset_name: str, task: str) -> Optional[int]:
    """Get number of labels for common datasets."""
    
    dataset_labels = {
        "imdb": 2,
        "ag_news": 4,
        "yelp_polarity": 2,
        "sst2": 2,
        "cola": 2,
        "mrpc": 2,
        "qnli": 2,
        "qqp": 2,
        "rte": 2,
        "wnli": 2,
        "banking77": 77,
    }
    
    if task == "sequence_classification":
        return dataset_labels.get(dataset_name, 2)
    
    return None


def _display_config(config: Config) -> None:
    """Display configuration in a formatted table."""
    
    table = Table(title="Training Configuration")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    
    # Key configuration items
    table.add_row("Model", config.model.name)
    table.add_row("Dataset", config.dataset.name)
    table.add_row("Task Type", config.model.task_type)
    table.add_row("Epochs", str(config.training.num_train_epochs))
    table.add_row("Batch Size", str(config.training.per_device_train_batch_size))
    table.add_row("Learning Rate", str(config.training.learning_rate))
    table.add_row("Use LoRA", "✅" if config.use_lora else "❌")
    table.add_row("Use QLoRA", "✅" if config.use_qlora else "❌")
    table.add_row("LoRA Rank", str(config.lora.r))
    table.add_row("Output Dir", config.training.output_dir)
    
    console.print(table)


def _check_gpu_requirements() -> None:
    """Check and display GPU requirements."""
    
    gpu_info = get_gpu_info()
    
    if gpu_info["cuda_available"]:
        console.print(f"🎮 GPU Available: {gpu_info['device_count']} device(s)")
        
        for device in gpu_info["devices"]:
            memory_gb = device["total_memory_gb"]
            status = "✅" if memory_gb >= 8 else "⚠️"
            console.print(f"   {status} GPU {device['id']}: {device['name']} ({memory_gb:.1f} GB)")
    else:
        console.print("⚠️  No GPU available - training will use CPU (very slow)", style="bold yellow")


def _display_results(results: Dict[str, Any]) -> None:
    """Display training results."""
    
    console.print(Panel.fit("🎉 Training Results", style="bold green"))
    
    # Create results table
    table = Table(title="Final Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Display key metrics
    metrics_to_show = [
        ("eval_accuracy", "Accuracy"),
        ("eval_f1", "F1 Score"),
        ("eval_precision", "Precision"),
        ("eval_recall", "Recall"),
        ("total_time", "Training Time (s)"),
    ]
    
    for key, display_name in metrics_to_show:
        if key in results:
            value = results[key]
            if isinstance(value, float):
                if key == "total_time":
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            table.add_row(display_name, formatted_value)
    
    console.print(table)
    
    # Show additional information
    if "run_id" in results:
        console.print(f"📊 MLflow Run ID: {results['run_id']}")


def main():
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()