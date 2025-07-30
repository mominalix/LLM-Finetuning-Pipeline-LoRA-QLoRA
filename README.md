# LLM Fine-tuning Pipeline with LoRA/QLoRA

A comprehensive, production-ready pipeline for fine-tuning Large Language Models using Parameter-Efficient Fine-Tuning (PEFT) techniques with MLflow experiment tracking and a user-friendly GUI.

## 🚀 Features

- **Parameter-Efficient Fine-tuning**: Support for LoRA and QLoRA techniques
- **Multiple Model Support**: RoBERTa, Llama 2, Mistral, GPT models, and more
- **Experiment Tracking**: Full MLflow integration for reproducible experiments
- **Flexible Configuration**: YAML-based configuration system for easy customization
- **Data Pipeline**: Robust data loading, preprocessing, and validation
- **Model Evaluation**: Comprehensive metrics and custom evaluation functions
- **Web GUI**: Simple interface for model selection and training configuration
- **Production Ready**: Docker support, CI/CD pipelines, and deployment scripts
- **Resume-Friendly**: Industry best practices and professional documentation

## 📊 Supported Models

| Model Type | Model Names | Use Cases |
|------------|-------------|-----------|
| **Encoder Models** | RoBERTa, BERT, DistilBERT | Classification, NER, QA |
| **Decoder Models** | Llama 2, Mistral, Falcon | Text generation, Chat |
| **Encoder-Decoder** | T5, BART, Flan-T5 | Summarization, Translation |

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Pipeline │    │  Training Core  │    │   Evaluation    │
│                 │    │                 │    │                 │
│ • Data Loading  │───▶│ • LoRA/QLoRA    │───▶│ • Metrics       │
│ • Preprocessing │    │ • MLflow Track. │    │ • Benchmarking  │
│ • Validation    │    │ • Model Manag.  │    │ • Reporting     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web GUI       │    │  Configuration  │    │   Deployment    │
│                 │    │                 │    │                 │
│ • Model Select  │    │ • YAML Configs  │    │ • Docker        │
│ • Training UI   │    │ • Hyperparams   │    │ • API Server    │
│ • Monitoring    │    │ • Data Splits   │    │ • Model Serving │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional but recommended)
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/llm-finetuning-pipeline.git
cd llm-finetuning-pipeline
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up MLflow:**
```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000
```

### Example: Fine-tune RoBERTa for Sentiment Analysis

```python
from pipeline import FineTuningPipeline
from configs import load_config

# Load configuration
config = load_config("configs/roberta_sentiment.yaml")

# Initialize pipeline
pipeline = FineTuningPipeline(config)

# Run training
results = pipeline.run()

print(f"Training completed! Best F1 Score: {results['best_f1']:.4f}")
```

## 📁 Project Structure

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

## 🔧 Configuration

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

## 📈 Experiment Tracking

All experiments are automatically tracked with MLflow:

- **Parameters**: Model configs, hyperparameters, data splits
- **Metrics**: Training/validation loss, accuracy, F1-score, custom metrics
- **Artifacts**: Model checkpoints, training plots, evaluation reports
- **Models**: Versioned model registry with staging/production promotion

## 🎯 Best Practices Implemented

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

## 🚀 Deployment Options

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

## 📚 Examples

- **Text Classification**: [RoBERTa Sentiment Analysis](examples/roberta_sentiment.md)
- **Named Entity Recognition**: [BERT NER](examples/bert_ner.md)
- **Text Generation**: [Llama 2 Chat Fine-tuning](examples/llama2_chat.md)
- **Question Answering**: [T5 QA](examples/t5_qa.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for transformers and PEFT libraries
- [MLflow](https://mlflow.org/) for experiment tracking
- [LoRA Paper](https://arxiv.org/abs/2106.09685) by Hu et al.
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) by Dettmers et al.

## 📞 Support

- 📧 Email: support@example.com
- 💬 Discord: [Join our community](https://discord.gg/example)
- 📖 Documentation: [docs.example.com](https://docs.example.com)
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/llm-finetuning-pipeline/issues)

---

⭐ **Star this repository if you find it useful!** ⭐