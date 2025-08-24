# Personalized Federated Learning with Meta-Learning (FedMAML vs FedAvg)

This repository implements a complete federated learning system that compares **Federated Averaging (FedAvg)** with **Federated MAML (FedMAML)** for personalized federated learning. The system is designed to handle non-IID data distributions and provides comprehensive evaluation metrics.

## 🎯 Project Overview

The project addresses the challenge of training machine learning models in federated settings where:
- Data is distributed across multiple clients
- Data distributions are non-IID (non-independent and identically distributed)
- Personalization is important for individual client performance
- Privacy must be maintained through federated learning

## 🏗️ Architecture

### Core Components

1. **`model.py`** - Neural network architectures for MNIST and CIFAR-10
2. **`data_loader.py`** - Non-IID data partitioning using Dirichlet distribution
3. **`maml.py`** - First-Order MAML implementation for meta-learning
4. **`fedavg.py`** - Standard federated averaging algorithm
5. **`fedmaml.py`** - FedMAML implementation integrating MAML into FL rounds
6. **`utils.py`** - Utility functions for model operations and evaluation
7. **`train.py`** - Main training script comparing both approaches
8. **`run.py`** - Simple command runner for common tasks

### Key Features

- **Non-IID Data Distribution**: Realistic federated scenarios using Dirichlet distribution
- **Differential Privacy**: Optional Gaussian noise addition to updates
- **Personalization Evaluation**: Tests model adaptation on individual client data
- **Comprehensive Logging**: Tracks global accuracy, personalized accuracy, and training loss
- **Configurable Parameters**: YAML-based configuration for easy experimentation

## 🚀 Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.1+
- CUDA (optional, for GPU acceleration)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd applicative-project-1/fedmaml_project
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   
   # On Windows:
   .venv\Scripts\activate
   
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   python run.py install
   ```

## 📊 Usage

### Quick Commands

Use the `run.py` script for common tasks:

```bash
# Install dependencies
python run.py install

# Run tests
python run.py test

# Run training
python run.py train

# Clean output files
python run.py clean

# Show project status
python run.py status

# Show help
python run.py help
```

### Basic Training

Run the training script with default configuration:

```bash
python train.py
```

### Custom Configuration

Create a custom `config.yaml` file or modify the existing one:

```yaml
dataset: MNIST           # MNIST or CIFAR10
num_clients: 10          # Number of federated clients
dirichlet_alpha: 0.3     # Non-IID degree (smaller = more non-IID)
rounds: 50               # Number of federated rounds
local_epochs: 1          # Local training epochs per round
batch_size: 64           # Training batch size
lr_global: 0.001         # Global learning rate (FedAvg)
lr_inner: 0.01           # Inner learning rate (MAML)
maml_meta_lr: 0.001      # Meta learning rate (FedMAML)
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dataset` | Dataset to use (MNIST/CIFAR10) | MNIST |
| `num_clients` | Number of federated clients | 10 |
| `dirichlet_alpha` | Non-IID degree (lower = more non-IID) | 0.3 |
| `clients_per_round` | Clients selected per round | 5 |
| `rounds` | Total federated rounds | 50 |
| `local_epochs` | Local training epochs | 1 |
| `batch_size` | Training batch size | 64 |
| `lr_global` | FedAvg learning rate | 0.001 |
| `lr_inner` | MAML inner learning rate | 0.01 |
| `maml_meta_lr` | FedMAML meta learning rate | 0.001 |
| `inner_steps` | MAML inner adaptation steps | 1 |
| `dp_noise_std` | Differential privacy noise | 0.0 |

## 📈 Results

The training script generates several outputs:

- **`loss_vs_rounds.png`** - Training loss comparison
- **`global_accuracy_vs_rounds.png`** - Global test accuracy comparison
- **`personalized_accuracy_vs_rounds.png`** - Personalized accuracy comparison
- **`results.csv`** - Summary of final results
- **`per_client_personalized.csv`** - Per-client personalized performance

## 🔬 Algorithm Details

### FedAvg (Federated Averaging)
- Standard federated learning approach
- Clients train locally and send model updates
- Server averages client updates
- No personalization mechanism

### FedMAML (Federated MAML)
- Integrates Model-Agnostic Meta-Learning into federated learning
- Each client adapts the global model to their local data
- Meta-learning optimizes for fast adaptation
- Enables personalization through few-shot learning

### Non-IID Data Distribution
- Uses Dirichlet distribution to create realistic federated scenarios
- `dirichlet_alpha` controls the degree of non-IIDness
- Lower values create more heterogeneous client distributions

## 🧪 Experiments

### Supported Datasets
- **MNIST**: 28x28 grayscale digits (0-9)
- **CIFAR-10**: 32x32 color images across 10 classes

### Evaluation Metrics
- **Global Accuracy**: Performance on held-out test set
- **Personalized Accuracy**: Performance after client-specific adaptation
- **Training Loss**: Convergence behavior across rounds

## 🔒 Privacy Features

- **Differential Privacy**: Optional Gaussian noise addition
- **Federated Learning**: Data never leaves client devices
- **Secure Aggregation**: Model updates are aggregated securely

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or number of clients
2. **Poor FedMAML Performance**: Check learning rates and inner steps
3. **Data Loading Errors**: Ensure dataset is properly downloaded

### Performance Tips

- Use GPU acceleration when available
- Adjust `dirichlet_alpha` for different non-IID scenarios
- Tune learning rates based on dataset and model architecture

## 📚 References

- [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)
- [Federated Learning with Personalization Layers](https://arxiv.org/abs/1912.00818)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
