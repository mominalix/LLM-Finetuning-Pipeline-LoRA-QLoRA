#!/usr/bin/env python3
"""Streamlit web application for LLM fine-tuning pipeline.

This module provides a user-friendly web interface for configuring and running
LLM fine-tuning experiments using the pipeline.

Usage:
    streamlit run src/gui/app.py
    python -m src.gui.app

Features:
    - Model and dataset selection
    - Interactive configuration builder
    - Real-time training monitoring
    - Results visualization
    - MLflow integration
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import os
import sys
import time
import logging
from typing import Dict, Any, Optional, List
import threading
import queue

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import FineTuningPipeline, load_config
from src.core.config import (
    Config, ModelConfig, DatasetConfig, TrainingConfig,
    LoRAConfig, QLoRAConfig, MLflowConfig, EvaluationConfig,
    save_config
)
from src.utils.gpu import get_gpu_info
from src.utils.logging import setup_logging


# Page configuration
st.set_page_config(
    page_title="LLM Fine-tuning Pipeline",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
    }
    .success-message {
        color: #4CAF50;
        font-weight: bold;
    }
    .error-message {
        color: #F44336;
        font-weight: bold;
    }
    .warning-message {
        color: #FF9800;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # Initialize session state
    if 'training_status' not in st.session_state:
        st.session_state.training_status = 'idle'
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None
    if 'config' not in st.session_state:
        st.session_state.config = None
    
    # Header
    st.markdown('<h1 class="main-header">🚀 LLM Fine-tuning Pipeline</h1>', unsafe_allow_html=True)
    st.markdown("**Fine-tune Large Language Models with LoRA/QLoRA using a simple web interface**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Configuration", "Training", "Results", "Monitoring", "Help"]
    )
    
    # Route to appropriate page
    if page == "Configuration":
        show_configuration_page()
    elif page == "Training":
        show_training_page()
    elif page == "Results":
        show_results_page()
    elif page == "Monitoring":
        show_monitoring_page()
    elif page == "Help":
        show_help_page()


def show_configuration_page():
    """Show the configuration page."""
    
    st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
    
    # Configuration tabs
    tab1, tab2, tab3 = st.tabs(["🤖 Model & Dataset", "🎯 Training", "🔧 Advanced"])
    
    with tab1:
        config = configure_model_and_dataset()
    
    with tab2:
        if 'temp_config' in locals():
            config = configure_training(config)
        else:
            config = configure_training()
    
    with tab3:
        if 'config' in locals():
            config = configure_advanced(config)
        else:
            config = configure_advanced()
    
    # Save configuration
    if st.button("💾 Save Configuration", type="primary"):
        if config:
            st.session_state.config = config
            
            # Save to file
            config_path = "current_config.yaml"
            save_config(config, config_path)
            
            st.success(f"✅ Configuration saved to {config_path}")
            
            # Show preview
            with st.expander("📋 Configuration Preview"):
                st.json(config.dict())
        else:
            st.error("❌ Please complete the configuration")


def configure_model_and_dataset() -> Optional[Config]:
    """Configure model and dataset settings."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Model Selection")
        
        model_type = st.selectbox(
            "Model Type",
            ["Encoder (Classification)", "Decoder (Generation)", "Encoder-Decoder"],
            help="Choose the type of model architecture"
        )
        
        if model_type == "Encoder (Classification)":
            model_options = [
                "roberta-base",
                "roberta-large",
                "bert-base-uncased",
                "bert-large-uncased",
                "distilbert-base-uncased",
            ]
            default_task = "sequence_classification"
        elif model_type == "Decoder (Generation)":
            model_options = [
                "meta-llama/Llama-2-7b-hf",
                "mistralai/Mistral-7B-v0.1",
                "microsoft/DialoGPT-medium",
                "gpt2",
                "gpt2-medium",
            ]
            default_task = "text_generation"
        else:  # Encoder-Decoder
            model_options = [
                "google/flan-t5-base",
                "google/flan-t5-large",
                "facebook/bart-base",
                "t5-base",
            ]
            default_task = "summarization"
        
        model_name = st.selectbox("Model", model_options)
        
        task_type = st.selectbox(
            "Task Type",
            ["sequence_classification", "token_classification", "question_answering", 
             "text_generation", "summarization", "translation"],
            index=0 if default_task == "sequence_classification" else 
                  3 if default_task == "text_generation" else 4
        )
        
        if task_type in ["sequence_classification", "token_classification"]:
            num_labels = st.number_input("Number of Labels", min_value=2, max_value=1000, value=2)
        else:
            num_labels = None
        
        max_length = st.number_input("Max Sequence Length", min_value=128, max_value=8192, value=512)
    
    with col2:
        st.subheader("📊 Dataset Selection")
        
        dataset_source = st.radio(
            "Dataset Source",
            ["HuggingFace Hub", "Local File", "Custom"]
        )
        
        if dataset_source == "HuggingFace Hub":
            dataset_options = [
                "imdb", "ag_news", "yelp_polarity", "sst2", "cola",
                "banking77", "emotion", "tweet_eval"
            ]
            dataset_name = st.selectbox("Dataset", dataset_options)
        elif dataset_source == "Local File":
            dataset_name = st.text_input("Dataset Path", placeholder="/path/to/dataset.csv")
        else:
            dataset_name = st.text_input("Custom Dataset", placeholder="your-dataset-name")
        
        text_column = st.text_input("Text Column", value="text")
        label_column = st.text_input("Label Column", value="label")
        
        col2a, col2b = st.columns(2)
        with col2a:
            validation_size = st.slider("Validation Split", 0.0, 0.5, 0.1)
        with col2b:
            max_samples = st.number_input("Max Samples (0 = all)", min_value=0, value=0)
        
        shuffle_data = st.checkbox("Shuffle Data", value=True)
        remove_duplicates = st.checkbox("Remove Duplicates", value=True)
    
    # Create configuration
    try:
        model_config = ModelConfig(
            name=model_name,
            task_type=task_type,
            num_labels=num_labels,
            max_length=max_length
        )
        
        dataset_config = DatasetConfig(
            name=dataset_name,
            text_column=text_column,
            label_column=label_column,
            validation_size=validation_size,
            max_samples=max_samples if max_samples > 0 else None,
            shuffle=shuffle_data,
            remove_duplicates=remove_duplicates
        )
        
        # Store in session state for next tabs
        st.session_state.temp_model_config = model_config
        st.session_state.temp_dataset_config = dataset_config
        
        return model_config, dataset_config
        
    except Exception as e:
        st.error(f"Configuration error: {str(e)}")
        return None


def configure_training(prev_config=None) -> Optional[Config]:
    """Configure training settings."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Training Parameters")
        
        num_epochs = st.number_input("Number of Epochs", min_value=1, max_value=100, value=3)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=16)
        learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=2e-5, format="%.0e")
        
        weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=1.0, value=0.01)
        warmup_ratio = st.slider("Warmup Ratio", 0.0, 0.5, 0.1)
        
        scheduler_type = st.selectbox(
            "LR Scheduler",
            ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"]
        )
    
    with col2:
        st.subheader("🔧 LoRA/QLoRA Settings")
        
        use_lora = st.checkbox("Use LoRA", value=True, help="Enable Parameter-Efficient Fine-Tuning")
        
        if use_lora:
            lora_r = st.number_input("LoRA Rank (r)", min_value=1, max_value=256, value=8)
            lora_alpha = st.number_input("LoRA Alpha", min_value=1, max_value=256, value=16)
            lora_dropout = st.slider("LoRA Dropout", 0.0, 0.5, 0.1)
            
            use_qlora = st.checkbox("Use QLoRA", value=False, help="Enable 4-bit quantization for memory efficiency")
            
            if use_qlora:
                st.info("🎯 QLoRA will use 4-bit quantization to reduce memory usage")
        else:
            use_qlora = False
            lora_r, lora_alpha, lora_dropout = 8, 16, 0.1
    
    # Performance settings
    st.subheader("⚡ Performance")
    col3, col4 = st.columns(2)
    
    with col3:
        use_bf16 = st.checkbox("Use BF16", value=True, help="Mixed precision training")
        gradient_checkpointing = st.checkbox("Gradient Checkpointing", value=True)
    
    with col4:
        gradient_accumulation = st.number_input("Gradient Accumulation Steps", min_value=1, max_value=32, value=1)
        eval_strategy = st.selectbox("Evaluation Strategy", ["epoch", "steps"])
    
    # Output settings
    output_dir = st.text_input("Output Directory", value="./outputs/model_training")
    
    # Create training configuration
    try:
        training_config = TrainingConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=scheduler_type,
            bf16=use_bf16,
            gradient_checkpointing=gradient_checkpointing,
            gradient_accumulation_steps=gradient_accumulation,
            evaluation_strategy=eval_strategy
        )
        
        lora_config = LoRAConfig(
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout
        )
        
        # Store in session state
        st.session_state.temp_training_config = training_config
        st.session_state.temp_lora_config = lora_config
        st.session_state.temp_use_lora = use_lora
        st.session_state.temp_use_qlora = use_qlora
        
        return training_config, lora_config, use_lora, use_qlora
        
    except Exception as e:
        st.error(f"Training configuration error: {str(e)}")
        return None


def configure_advanced(prev_config=None) -> Optional[Config]:
    """Configure advanced settings."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 MLflow Settings")
        
        mlflow_uri = st.text_input("Tracking URI", value="http://localhost:5000")
        experiment_name = st.text_input("Experiment Name", value="llm_finetuning")
        log_model = st.checkbox("Log Model", value=True)
        log_predictions = st.checkbox("Log Predictions", value=True)
    
    with col2:
        st.subheader("📈 Evaluation Settings")
        
        metrics = st.multiselect(
            "Metrics",
            ["accuracy", "f1", "precision", "recall", "mcc", "auc"],
            default=["accuracy", "f1", "precision", "recall"]
        )
        
        save_predictions = st.checkbox("Save Predictions", value=True)
        
    # Environment settings
    st.subheader("🌍 Environment")
    seed = st.number_input("Random Seed", min_value=0, max_value=99999, value=42)
    
    # Create final configuration if all components are ready
    if all(key in st.session_state for key in [
        'temp_model_config', 'temp_dataset_config', 'temp_training_config',
        'temp_lora_config', 'temp_use_lora', 'temp_use_qlora'
    ]):
        
        try:
            mlflow_config = MLflowConfig(
                tracking_uri=mlflow_uri,
                experiment_name=experiment_name,
                log_model=log_model,
                log_predictions=log_predictions
            )
            
            eval_config = EvaluationConfig(
                metrics=metrics,
                save_predictions=save_predictions
            )
            
            # Create complete configuration
            config = Config(
                model=st.session_state.temp_model_config,
                dataset=st.session_state.temp_dataset_config,
                training=st.session_state.temp_training_config,
                use_lora=st.session_state.temp_use_lora,
                lora=st.session_state.temp_lora_config,
                use_qlora=st.session_state.temp_use_qlora,
                qlora=QLoRAConfig() if st.session_state.temp_use_qlora else QLoRAConfig(load_in_4bit=False),
                mlflow=mlflow_config,
                evaluation=eval_config,
                seed=seed
            )
            
            return config
            
        except Exception as e:
            st.error(f"Advanced configuration error: {str(e)}")
            return None
    
    return None


def show_training_page():
    """Show the training page."""
    
    st.markdown('<h2 class="section-header">🚀 Training</h2>', unsafe_allow_html=True)
    
    if st.session_state.config is None:
        st.warning("⚠️ Please configure your experiment first")
        if st.button("Go to Configuration"):
            st.rerun()
        return
    
    # Display current configuration summary
    with st.expander("📋 Current Configuration"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model", st.session_state.config.model.name)
            st.metric("Dataset", st.session_state.config.dataset.name)
        
        with col2:
            st.metric("Epochs", st.session_state.config.training.num_train_epochs)
            st.metric("Batch Size", st.session_state.config.training.per_device_train_batch_size)
        
        with col3:
            st.metric("LoRA", "✅" if st.session_state.config.use_lora else "❌")
            st.metric("QLoRA", "✅" if st.session_state.config.use_qlora else "❌")
    
    # GPU information
    gpu_info = get_gpu_info()
    if gpu_info["cuda_available"]:
        st.success(f"🎮 GPU Available: {gpu_info['device_count']} device(s)")
        for device in gpu_info["devices"]:
            st.info(f"GPU {device['id']}: {device['name']} ({device['total_memory_gb']:.1f} GB)")
    else:
        st.warning("⚠️ No GPU detected - training will be slow")
    
    # Training controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Training", type="primary", disabled=st.session_state.training_status == 'running'):
            start_training()
    
    with col2:
        if st.button("⏹️ Stop Training", disabled=st.session_state.training_status != 'running'):
            stop_training()
    
    with col3:
        if st.button("🔄 Reset"):
            reset_training()
    
    # Training status and progress
    if st.session_state.training_status == 'running':
        show_training_progress()
    elif st.session_state.training_status == 'completed':
        st.success("✅ Training completed!")
        if st.session_state.training_results:
            show_training_summary()
    elif st.session_state.training_status == 'failed':
        st.error("❌ Training failed!")


def start_training():
    """Start the training process."""
    
    st.session_state.training_status = 'running'
    
    # Show training progress
    progress_container = st.container()
    
    with progress_container:
        st.info("🚀 Starting training...")
        
        # Create progress bars
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize pipeline
            status_text.text("Initializing pipeline...")
            pipeline = FineTuningPipeline(st.session_state.config)
            
            # This is a simplified version - in a real app, you'd want to run this in a separate thread
            # and update progress asynchronously
            status_text.text("Training in progress...")
            results = pipeline.run()
            
            # Training completed
            progress_bar.progress(100)
            status_text.text("Training completed!")
            
            st.session_state.training_status = 'completed'
            st.session_state.training_results = results
            
            st.success("🎉 Training completed successfully!")
            st.rerun()
            
        except Exception as e:
            st.session_state.training_status = 'failed'
            st.error(f"Training failed: {str(e)}")


def stop_training():
    """Stop the training process."""
    st.session_state.training_status = 'stopped'
    st.warning("⏹️ Training stopped by user")


def reset_training():
    """Reset the training state."""
    st.session_state.training_status = 'idle'
    st.session_state.training_results = None
    st.success("🔄 Training state reset")


def show_training_progress():
    """Show real-time training progress."""
    
    st.subheader("📈 Training Progress")
    
    # Placeholder for real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Epoch", "2/3")
    with col2:
        st.metric("Loss", "0.234")
    with col3:
        st.metric("Accuracy", "0.892")
    with col4:
        st.metric("F1 Score", "0.887")
    
    # Progress chart (placeholder)
    chart_data = pd.DataFrame({
        'epoch': [1, 2, 3],
        'train_loss': [0.8, 0.4, 0.2],
        'val_loss': [0.7, 0.5, 0.3],
        'accuracy': [0.6, 0.8, 0.9]
    })
    
    st.line_chart(chart_data.set_index('epoch')[['train_loss', 'val_loss']])


def show_training_summary():
    """Show training results summary."""
    
    results = st.session_state.training_results
    
    st.subheader("📊 Training Results")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("Accuracy", results.get("eval_accuracy", 0)),
        ("F1 Score", results.get("eval_f1", 0)),
        ("Precision", results.get("eval_precision", 0)),
        ("Recall", results.get("eval_recall", 0))
    ]
    
    for i, (name, value) in enumerate(metrics):
        with [col1, col2, col3, col4][i]:
            st.metric(name, f"{value:.4f}" if value else "N/A")
    
    # Training time
    if "total_time" in results:
        st.metric("Training Time", f"{results['total_time']:.2f}s")
    
    # Download model button
    if st.button("📥 Download Model"):
        st.info("Model download functionality would be implemented here")


def show_results_page():
    """Show the results and analysis page."""
    
    st.markdown('<h2 class="section-header">📊 Results & Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.training_results is None:
        st.info("No training results available. Please run a training experiment first.")
        return
    
    results = st.session_state.training_results
    
    # Results overview
    st.subheader("📈 Performance Overview")
    
    # Create metrics visualization
    metrics_data = {
        'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall'],
        'Value': [
            results.get('eval_accuracy', 0),
            results.get('eval_f1', 0),
            results.get('eval_precision', 0),
            results.get('eval_recall', 0)
        ]
    }
    
    fig = px.bar(
        metrics_data, 
        x='Metric', 
        y='Value',
        title='Model Performance Metrics',
        color='Value',
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Training history (placeholder)
    st.subheader("📉 Training History")
    
    # Generate sample training history
    epochs = list(range(1, st.session_state.config.training.num_train_epochs + 1))
    train_loss = [0.8 - i*0.2 for i in range(len(epochs))]
    val_loss = [0.7 - i*0.15 for i in range(len(epochs))]
    
    history_df = pd.DataFrame({
        'Epoch': epochs,
        'Training Loss': train_loss,
        'Validation Loss': val_loss
    })
    
    fig2 = px.line(
        history_df, 
        x='Epoch', 
        y=['Training Loss', 'Validation Loss'],
        title='Training Progress'
    )
    st.plotly_chart(fig2, use_container_width=True)


def show_monitoring_page():
    """Show the monitoring and MLflow integration page."""
    
    st.markdown('<h2 class="section-header">📊 Monitoring</h2>', unsafe_allow_html=True)
    
    # MLflow integration
    st.subheader("🔍 MLflow Integration")
    
    if st.session_state.config:
        mlflow_uri = st.session_state.config.mlflow.tracking_uri
        experiment_name = st.session_state.config.mlflow.experiment_name
        
        st.info(f"MLflow Server: {mlflow_uri}")
        st.info(f"Experiment: {experiment_name}")
        
        if st.button("🔗 Open MLflow UI"):
            st.markdown(f"[Open MLflow UI]({mlflow_uri})")
    
    # System monitoring
    st.subheader("💻 System Status")
    
    gpu_info = get_gpu_info()
    
    if gpu_info["cuda_available"]:
        for device in gpu_info["devices"]:
            with st.expander(f"GPU {device['id']}: {device['name']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Memory", f"{device['total_memory_gb']:.1f} GB")
                    if 'memory_allocated_gb' in device:
                        st.metric("Used Memory", f"{device['memory_allocated_gb']:.1f} GB")
                
                with col2:
                    if 'memory_free_gb' in device:
                        st.metric("Free Memory", f"{device['memory_free_gb']:.1f} GB")
                    if 'utilization_percent' in device:
                        st.progress(device['utilization_percent'] / 100)
    else:
        st.warning("No GPU detected")


def show_help_page():
    """Show the help and documentation page."""
    
    st.markdown('<h2 class="section-header">❓ Help & Documentation</h2>', unsafe_allow_html=True)
    
    # Quick start guide
    with st.expander("🚀 Quick Start Guide"):
        st.markdown("""
        1. **Configuration**: Set up your model, dataset, and training parameters
        2. **Training**: Start the fine-tuning process and monitor progress
        3. **Results**: Analyze performance metrics and training history
        4. **Monitoring**: Track experiments with MLflow integration
        
        ### Recommended Settings:
        - **Small models** (RoBERTa, BERT): Use LoRA without QLoRA
        - **Large models** (Llama, Mistral): Use QLoRA for memory efficiency
        - **GPU memory < 8GB**: Enable QLoRA and reduce batch size
        """)
    
    # Model recommendations
    with st.expander("🤖 Model Recommendations"):
        st.markdown("""
        | Task | Recommended Models | Memory (GB) |
        |------|-------------------|-------------|
        | Text Classification | RoBERTa-base, BERT-base | 2-4 |
        | Named Entity Recognition | RoBERTa-large, BERT-large | 4-8 |
        | Text Generation | Llama-2-7B, Mistral-7B | 14-28 |
        | Question Answering | RoBERTa-large, BERT-large | 4-8 |
        | Summarization | FLAN-T5-base, BART-base | 2-4 |
        """)
    
    # LoRA/QLoRA explanation
    with st.expander("🔧 LoRA & QLoRA Explained"):
        st.markdown("""
        ### LoRA (Low-Rank Adaptation)
        - Reduces trainable parameters by 99%+
        - Maintains model performance
        - Faster training and lower memory usage
        
        ### QLoRA (Quantized LoRA)
        - Combines LoRA with 4-bit quantization
        - Enables fine-tuning of 7B+ models on consumer GPUs
        - Reduces memory usage by ~65%
        
        ### Parameter Guidelines:
        - **LoRA Rank (r)**: 8-16 for most tasks, 32-64 for complex tasks
        - **LoRA Alpha**: Usually 2x the rank value
        - **LoRA Dropout**: 0.05-0.1 for regularization
        """)
    
    # Troubleshooting
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        ### Common Issues:
        
        **Out of Memory Error:**
        - Enable QLoRA
        - Reduce batch size
        - Enable gradient checkpointing
        - Reduce sequence length
        
        **Slow Training:**
        - Enable mixed precision (BF16/FP16)
        - Increase batch size if memory allows
        - Use gradient accumulation
        
        **Poor Performance:**
        - Increase LoRA rank
        - Adjust learning rate
        - Add more training epochs
        - Check data quality
        """)
    
    # Links and resources
    st.subheader("🔗 Useful Links")
    
    links = [
        ("📚 Documentation", "https://github.com/your-repo/docs"),
        ("🐛 Report Issues", "https://github.com/your-repo/issues"),
        ("💬 Community", "https://discord.gg/your-community"),
        ("📄 Research Papers", "https://arxiv.org/abs/2106.09685"),
    ]
    
    for title, url in links:
        st.markdown(f"- [{title}]({url})")


def main_cli():
    """CLI entry point for the Streamlit app."""
    import subprocess
    import sys
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", __file__,
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        main_cli()
    else:
        main()