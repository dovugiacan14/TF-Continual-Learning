# TF-Continual-Learning

A PyTorch-based framework for Continual Learning research, implementing various state-of-the-art algorithms for Class-Incremental Learning (CIL) and Task-Incremental Learning (TIL) scenarios. The framework also includes neural architecture search capabilities using genetic algorithms.

## Overview

This repository provides a unified implementation of multiple continual learning approaches, enabling researchers to:
- Train and evaluate various continual learning algorithms
- Experiment with different neural network architectures
- Perform neural architecture search using genetic evolution
- Benchmark performance on standard datasets (CIFAR-100, ImageNet-100)

## Project Structure

```
TF-Continual-Learning/
├── class_il/              # Class-Incremental Learning implementations
│   ├── models/           # Continual learning algorithms
│   ├── convs/            # CNN architectures
│   ├── genetic/          # Genetic algorithm components
│   ├── utils/            # Utility functions
│   ├── trainer.py        # Training script
│   ├── test.py           # Evaluation script
│   └── evolve.py         # Neural architecture search
├── task_il/              # Task-Incremental Learning implementations
│   ├── approaches/       # Task-IL algorithms
│   ├── networks/         # Network architectures
│   ├── dataloaders/      # Data loading utilities
│   └── genetic/          # Genetic algorithm components
├── requirements.txt      # Python dependencies
└── LICENSE              # MIT License
```

## Supported Algorithms

### Class-Incremental Learning
- **EWC** (Elastic Weight Consolidation)
- **LwF** (Learning without Forgetting)
- **iCaRL** (Incremental Classifier and Representation Learning)
- **GEM** (Gradient Episodic Memory)
- **A-GEM** (Averaged Gradient Episodic Memory)
- **Replay** (Experience Replay)
- **DER** (Dark Experience Replay)
- **PODNet** (Pods for Continual Learning)
- **SimpleCIL** (Simple Class-Incremental Learning)
- **BIC** (Boundary-balancing Incremental Learning)
- **Foster** (Forgetting-free Continual Learning)
- **CoIL** (Continual Learning with Injected Features)
- **SSRE** (Self-supervised Replay)
- **RMM** (Remanent Memories)
- **WA** (Warp Architecture)
- **IL2A** (Incremental Learning with Augmented Data)
- **PA2S** (Prototype Augmentation)
- **BEef-ISO** (Benchmarking)

### Task-Incremental Learning
- **EWC** (Elastic Weight Consolidation)
- **SGD** (Standard Stochastic Gradient Descent)

## Neural Architecture Search

The framework includes genetic algorithm-based neural architecture search:
- Automated architecture evolution
- Population-based optimization
- Fitness evaluation with continual learning metrics
- Mutation operations for architecture variation

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.0+
- CUDA (recommended for GPU training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/dovugiacan14/TF-Continual-Learning.git
cd TF-Continual-Learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

- numpy
- Pillow
- scikit-learn
- scipy
- torch
- torchvision
- tqdm

## Usage

### Class-Incremental Learning Training

Create a configuration file (JSON format) and run:

```bash
cd class_il
python template.py --config exps/replay_for_evolve.json
```

Or use the trainer directly:

```bash
cd class_il
python trainer.py
```

### Task-Incremental Learning Training

```bash
cd task_il
python test.py
```

### Neural Architecture Search

Run genetic evolution for architecture search:

```bash
cd class_il
python evolve.py
```

Configuration parameters (in [evolve.py](class_il/evolve.py)):
- `pop_size`: Population size (default: 2)
- `max_gen`: Maximum generations (default: 3)

## Configuration

Models are configured via JSON files. Key parameters include:

- `model_name`: Algorithm to use (e.g., "ewc", "icarl", "replay")
- `dataset`: Dataset name ("cifar100", "imagenet100")
- `init_cls`: Number of initial classes
- `increment`: Classes added per task
- `network`: Network architecture (e.g., "resnet18", "resnet32")
- `seed`: Random seed for reproducibility
- `device`: GPU device ID(s)

## Supported Datasets

- **CIFAR-100**: 100 classes, 32x32 images
- **ImageNet-100**: Subset of ImageNet with 100 classes
- **ImageNet-1000**: Full ImageNet dataset

## Metrics

The framework evaluates continual learning using:

- **Average Accuracy (AIA)**: Mean accuracy across all tasks
- **Forgetting Measure**: Performance degradation on previous tasks
- **Accuracy Matrix**: Per-task, per-class accuracy
- **Top-1/Top-5 Accuracy**: Standard classification metrics

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
