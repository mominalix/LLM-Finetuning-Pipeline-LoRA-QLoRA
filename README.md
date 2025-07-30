# LLM Fine-tuning Pipeline with LoRA/QLoRA

A comprehensive, production-ready pipeline for fine-tuning Large Language Models using Parameter-Efficient Fine-Tuning (PEFT) techniques with MLflow experiment tracking and a user-friendly GUI.

## Features

- **Parameter-Efficient Fine-tuning**: Support for LoRA and QLoRA techniques
- **Multiple Model Support**: RoBERTa, Llama 2, Mistral, GPT models, and more
- **Experiment Tracking**: Full MLflow integration for reproducible experiments
- **Flexible Configuration**: YAML-based configuration system for easy customization
- **Data Pipeline**: Robust data loading, preprocessing, and validation
- **Model Evaluation**: Comprehensive metrics and custom evaluation functions
- **Web GUI**: Simple interface for model selection and training configuration
- **Production Ready**: Docker support, CI/CD pipelines, and deployment scripts
- **Resume-Friendly**: Industry best practices and professional documentation

## Supported Models

| Model Type      | Model Names               | Use Cases                    |
|-----------------|---------------------------|------------------------------|
| **Encoder Models**  | RoBERTa, BERT, DistilBERT   | Classification, NER, QA      |
| **Decoder Models**  | Llama 2, Mistral, Falcon    | Text generation, Chat        |
| **Encoder-Decoder** | T5, BART, Flan-T5         | Summarization, Translation |

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Pipeline │    │  Training Core  │    │   Evaluation    │
│                 │    │                 │    │                 │
│ • Data Loading  │ -> │ • LoRA/QLoRA    │ -> │ • Metrics       │
│ • Preprocessing │    │ • MLflow Track. │    │ • Benchmarking  │
│ • Validation    │    │ • Model Manag.  │    │ • Reporting     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                     │
         ▼                       ▼                     ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web GUI       │    │  Configuration  │    │   Deployment    │
│                 │    │                 │    │                 │
│ • Model Select  │    │ • YAML Configs  │    │ • Docker        │
│ • Training UI   │    │ • Hyperparams   │    │ • API Server    │
│ • Monitoring    │    │ • Data Splits   │    │ • Model Serving │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Project Structure

```
llm-finetuning-pipeline/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
├── docker-compose.yml
├── Dockerfile
│
├── configs/                    # Configuration files
│   ├── models/                # Model-specific configs
│   ├── datasets/              # Dataset-specific configs
│   └── experiments/           # Experiment configs
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── core/                  # Core pipeline components
│   ├── models/                # Model implementations
│   ├── data/                  # Data processing
│   ├── training/              # Training logic
│   ├── evaluation/            # Evaluation metrics
│   └── utils/                 # Utility functions
│
├── gui/                       # Web interface
│   ├── app.py                # Main Flask/Streamlit app
│   ├── components/           # UI components
│   └── static/               # Static assets
│
├── notebooks/                 # Jupyter notebooks
│   ├── examples/             # Usage examples
│   └── experiments/          # Research notebooks
│
├── tests/                     # Unit tests
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── docs/                      # Documentation
│   ├── api/                  # API documentation
│   ├── guides/               # User guides
│   └── best_practices/       # Best practices
│
├── scripts/                   # Utility scripts
│   ├── setup_env.sh
│   ├── download_models.py
│   └── benchmark.py
│
└── deployments/              # Deployment configurations
    ├── kubernetes/
    ├── docker/
    └── cloud/
```

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional but recommended)
- Docker (optional, for containerized deployment)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/mominalix/LLM-Finetuning-Pipeline-LoRA-QLoRA.git
cd LLM-Finetuning-Pipeline-LoRA-QLoRA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start MLflow Server (Optional)

```bash
# Start MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000
```

### 3. Run Your First Experiment

**Option A: Using the Web GUI (Recommended for beginners)**

```bash
# Start the web interface
streamlit run src/gui/app.py

# Open your browser to http://localhost:8501
```

**Option B: Using the Command Line**

```bash
# Run RoBERTa sentiment analysis example
python examples/roberta_sentiment.py

# Or use the CLI directly
python -m src.cli train --model roberta-base --dataset imdb --epochs 3
```

**Option C: Using Python API**

```python
from src import FineTuningPipeline, load_config

# Load configuration
config = load_config("configs/experiments/roberta_sentiment_analysis.yaml")

# Run training
pipeline = FineTuningPipeline(config)
results = pipeline.run()

print(f"Training completed! F1 Score: {results['eval_f1']:.4f}")
```

## Step-by-Step Tutorial

### Step 1: Understanding the Configuration System

The pipeline uses YAML configuration files for maximum flexibility:

```yaml
# Basic configuration structure
model:
  name: "roberta-base"
  task_type: "sequence_classification"
  num_labels: 2

dataset:
  name: "imdb"
  train_split: "train"
  test_split: "test"

training:
  num_train_epochs: 3
  per_device_train_batch_size: 16
  learning_rate: 2e-5

lora:
  r: 8
  alpha: 16
  dropout: 0.1
```

### Step 2: Choosing the Right Model and Configuration

#### For Text Classification (Recommended: RoBERTa + LoRA)

```yaml
model:
  name: "roberta-base"  # or "roberta-large" for better performance
  task_type: "sequence_classification"
  num_labels: 2  # Adjust based on your dataset

use_lora: true
use_qlora: false  # Not needed for RoBERTa base/large

lora:
  r: 8        # Good starting point
  alpha: 16   # Usually 2x the rank
  dropout: 0.1
```

#### For Text Generation (Recommended: Llama 2 + QLoRA)

```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"
  task_type: "text_generation"
  max_length: 2048

use_lora: true
use_qlora: true  # Essential for 7B+ models

lora:
  r: 16       # Higher rank for complex tasks
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

training:
  per_device_train_batch_size: 2  # Smaller due to memory constraints
  gradient_accumulation_steps: 8   # Maintain effective batch size
```

### Step 3: Data Preparation

#### Using HuggingFace Datasets (Easiest)

```yaml
dataset:
  name: "imdb"  # Any dataset from HuggingFace Hub
  text_column: "text"
  label_column: "label"
  validation_size: 0.1
```

#### Using Local CSV Files

```yaml
dataset:
  name: "/path/to/your/dataset.csv"
  text_column: "review_text"
  label_column: "sentiment"
  validation_size: 0.2
```

#### Using Custom Datasets

Create a CSV file with your data:

```csv
text,label
"I love this movie!",1
"This was terrible.",0
"Great acting and plot.",1
```

### Step 4: Training Configurations by Hardware

#### High-End GPU (24GB+ VRAM)

```yaml
training:
  per_device_train_batch_size: 32
  gradient_accumulation_steps: 1
  bf16: true
  gradient_checkpointing: false  # Can disable for speed

# Can use larger models without QLoRA
model:
  name: "roberta-large"  # or even "meta-llama/Llama-2-7b-hf" with QLoRA
```

#### Mid-Range GPU (8-16GB VRAM)

```yaml
training:
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 2
  bf16: true
  gradient_checkpointing: true

# Use QLoRA for models >1B parameters
use_qlora: true  # If using large models
```

#### Low-End GPU (4-8GB VRAM)

```yaml
training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  bf16: true
  gradient_checkpointing: true

# Must use QLoRA for any model >500M parameters
use_qlora: true
model:
  max_length: 256  # Reduce sequence length
```

## Real-World Examples

### Example 1: Customer Review Classification

```bash
# 1. Prepare your data (CSV format)
echo "text,label
Great product quality!,positive
Terrible customer service,negative
Amazing value for money,positive" > customer_reviews.csv

# 2. Create configuration
python -m src.cli create-config \
  --model roberta-base \
  --dataset customer_reviews.csv \
  --task sequence_classification \
  --output configs/customer_reviews.yaml

# 3. Train the model
python -m src.cli train --config configs/customer_reviews.yaml
```

### Example 2: Named Entity Recognition

```yaml
model:
  name: "roberta-base"
  task_type: "token_classification"
  num_labels: 9  # B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC, O

dataset:
  name: "conll2003"
  text_column: "tokens"
  label_column: "ner_tags"

lora:
  r: 16  # Higher rank for sequence labeling
  alpha: 32
  target_modules: ["query", "value", "key"]  # Include key projection
```

### Example 3: Question Answering with T5

```yaml
model:
  name: "google/flan-t5-base"
  task_type: "text_generation"
  max_length: 512

dataset:
  name: "squad"
  # Custom preprocessing function will handle QA format

training:
  num_train_epochs: 5
  learning_rate: 3e-4  # Higher for T5

lora:
  r: 32
  alpha: 64
  target_modules: ["q", "v", "k", "o", "wi_0", "wi_1", "wo"]  # T5 modules
```

## Configuration

The pipeline uses YAML configuration files for maximum flexibility:

```yaml
# configs/roberta_sentiment.yaml
model:
  name: "roberta-base"
  task_type: "sequence_classification"
  num_labels: 2
  
lora:
  r: 8
  alpha: 16
  dropout: 0.1
  target_modules: ["query", "value"]
  
dataset:
  name: "imdb"
  train_split: "train"
  test_split: "test"
  max_length: 512
  
training:
  epochs: 3
  batch_size: 16
  learning_rate: 2e-5
  warmup_steps: 100
  
mlflow:
  experiment_name: "roberta_sentiment_analysis"
  tracking_uri: "http://localhost:5000"
```

## Experiment Tracking

All experiments are automatically tracked with MLflow:

- **Parameters**: Model configs, hyperparameters, data splits
- **Metrics**: Training/validation loss, accuracy, F1-score, custom metrics
- **Artifacts**: Model checkpoints, training plots, evaluation reports
- **Models**: Versioned model registry with staging/production promotion

## Monitoring and Evaluation

### Using MLflow UI

1. Start MLflow: `mlflow server --host 0.0.0.0 --port 5000`
2. Open browser: `http://localhost:5000`
3. Browse experiments and compare runs
4. Download trained models

### Using the Web GUI

1. Start GUI: `streamlit run src/gui/app.py`
2. Open browser: `http://localhost:8501`
3. Navigate to "Monitoring" tab
4. View real-time training progress

### Programmatic Monitoring

```python
import mlflow

# Search for best run
best_run = mlflow.search_runs(
    experiment_ids=["1"],
    order_by=["metrics.eval_f1 DESC"],
    max_results=1
).iloc[0]

print(f"Best F1 Score: {best_run['metrics.eval_f1']}")
print(f"Best Run ID: {best_run['run_id']}")

# Load best model
model_uri = f"runs:/{best_run['run_id']}/model"
model = mlflow.transformers.load_model(model_uri)
```

## Deployment Options

### Local Development
```bash
python -m src.gui.app  # Start web interface
python -m src.train configs/roberta_sentiment.yaml  # CLI training
```

### Docker
```bash
docker-compose up  # Full stack with MLflow and GUI
```

### Cloud Deployment
- **AWS SageMaker**: Pre-configured training jobs
- **Google Cloud AI Platform**: Vertex AI integration
- **Azure ML**: Azure Machine Learning pipelines
- **Kubernetes**: Scalable distributed training

## Advanced Usage

### Custom Metrics

```python
from src.evaluation import MetricsComputer

def custom_accuracy(predictions, labels):
    """Custom accuracy that ignores label 0."""
    mask = labels != 0
    return (predictions[mask] == labels[mask]).mean()

# Add to metrics computer
computer = MetricsComputer(task_type="sequence_classification")
computer.add_custom_metric("custom_accuracy", custom_accuracy)
```

### Custom Data Loading

```python
from src.data import DataLoader
from datasets import Dataset
import pandas as pd

# Load custom data
def load_custom_dataset():
    df = pd.read_json("custom_data.jsonl", lines=True)
    return Dataset.from_pandas(df)

# Use in pipeline
from src.core.config import DatasetConfig
config = DatasetConfig(name="custom", ...)
loader = DataLoader(config)
# Override with custom loader
```

### Hyperparameter Sweeps

```python
import mlflow
from itertools import product

# Define parameter grid
param_grid = {
    'learning_rate': [1e-5, 2e-5, 5e-5],
    'lora_r': [8, 16, 32],
    'batch_size': [8, 16, 32]
}

# Run sweep
for lr, r, batch_size in product(*param_grid.values()):
    with mlflow.start_run():
        # Update config
        config.training.learning_rate = lr
        config.lora.r = r
        config.training.per_device_train_batch_size = batch_size
        
        # Run training
        pipeline = FineTuningPipeline(config)
        results = pipeline.run()
```

## Best Practices Implemented

### LoRA/QLoRA Optimization
- **Memory Efficient**: QLoRA with 4-bit quantization for large models
- **Optimal Parameters**: Research-backed default values for rank, alpha, dropout
- **Target Module Selection**: Automatic identification of optimal layers
- **Mixed Precision**: BF16/FP16 training for faster convergence

### Production Readiness
- **Type Safety**: Full type hints and Pydantic validation
- **Error Handling**: Comprehensive exception handling and logging
- **Testing**: Unit, integration, and end-to-end tests
- **Documentation**: Detailed API docs and user guides
- **Monitoring**: Real-time training metrics and alerts

### Data Management
- **Validation**: Automatic data quality checks and statistics
- **Caching**: Efficient data loading with HuggingFace datasets
- **Preprocessing**: Configurable tokenization and augmentation
- **Streaming**: Support for large datasets that don't fit in memory

## Common Issues and Solutions

### Memory Issues

**Problem**: CUDA out of memory error

**Solutions**:
1. Enable QLoRA: `use_qlora: true`
2. Reduce batch size: `per_device_train_batch_size: 1`
3. Increase gradient accumulation: `gradient_accumulation_steps: 16`
4. Enable gradient checkpointing: `gradient_checkpointing: true`
5. Reduce sequence length: `max_length: 256`

### Slow Training

**Problem**: Training is very slow

**Solutions**:
1. Enable mixed precision: `bf16: true` or `fp16: true`
2. Increase batch size if memory allows
3. Use gradient accumulation instead of small batches
4. Enable dataloader optimizations:
   ```yaml
   training:
     dataloader_pin_memory: true
     dataloader_num_workers: 4
   ```

### Poor Performance

**Problem**: Model accuracy is low

**Solutions**:
1. Increase LoRA rank: `r: 32` or higher
2. Adjust learning rate: try `1e-4` for LoRA
3. Add more epochs: `num_train_epochs: 5`
4. Check data quality and preprocessing
5. Try different target modules:
   ```yaml
   lora:
     target_modules: ["query", "value", "key", "dense"]
   ```

### MLflow Connection Issues

**Problem**: Cannot connect to MLflow server

**Solutions**:
1. Start MLflow server: `mlflow server --host 0.0.0.0 --port 5000`
2. Check firewall settings
3. Use local SQLite backend:
   ```yaml
   mlflow:
     tracking_uri: "sqlite:///mlflow.db"
   ```

## Next Steps

1. **Experiment with different models**: Try various architectures for your task
2. **Optimize hyperparameters**: Use grid search or Bayesian optimization
3. **Deploy to production**: Use MLflow model serving or containerization
4. **Scale up**: Use distributed training for very large datasets
5. **Custom extensions**: Add your own metrics, callbacks, and preprocessing

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for transformers and PEFT libraries
- [MLflow](https://mlflow.org/) for experiment tracking
- [LoRA Paper](https://arxiv.org/abs/2106.09685) by Hu et al.
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) by Dettmers et al.

## Support

- Email: support@example.com
- Discord: [Join our community](https://discord.gg/example)
- Documentation: [docs.example.com](https://docs.example.com)
- Issues: [GitHub Issues](https://github.com/your-username/llm-finetuning-pipeline/issues)

---

**Star this repository if you find it useful!**